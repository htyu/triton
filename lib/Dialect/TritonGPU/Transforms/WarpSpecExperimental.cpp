#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"
#include <unordered_set>

#define DEBUG_TYPE "triton-warp-spec-experimental"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

int getWarpGroupId(Operation *op) {
  if (!op->hasAttr("tt.warp_group_id"))
    return -1;
  return mlir::cast<IntegerAttr>(op->getAttr("tt.warp_group_id")).getInt();
}

static bool hasWarpGroupId(Operation *op, int warpGroupId) {
  auto id = getWarpGroupId(op);
  return id == warpGroupId;
}

void setWarpGroupId(MLIRContext *ctx, Operation *op, int WGId) {
  op->setAttr("tt.warp_group_id",
              IntegerAttr::get(IntegerType::get(ctx, 32), WGId));
}

SmallVector<int> getNestedWarpGroupIds(Operation *op) {
  SetVector<int> wgIds;
  op->walk([&](Operation *curOp) { wgIds.insert(getWarpGroupId(curOp)); });
  SmallVector<int> res(wgIds.begin(), wgIds.end());
  llvm::sort(res);
  return res;
}

class OpBuilderWithWarpGroupIds : public OpBuilder {
public:
  OpBuilderWithWarpGroupIds(MLIRContext *context) : OpBuilder(context) {}

  void setWarpGroupId(int wgId) { warpGroupId = wgId; }
  void setWarpGroupIdFromOp(Operation *op) {
    setWarpGroupId(getWarpGroupId(op));
  }

  template <typename OpTy, typename... Args>
  OpTy createWithWarpGroupId(Args &&...args) {
    OpTy op = create<OpTy>(std::forward<Args>(args)...);
    mlir::triton::gpu::setWarpGroupId(context, op, warpGroupId);
    return op;
  }

private:
  // a list of warp group ids?
  // SmallVector<int> warpGroupIds;
  int warpGroupId;
};

SmallVector<unsigned> checkDependencyAndCollectUsedArgs(
    scf::ForOp forOp, int warpGroupId,
    DenseMap<BlockArgument, Value> &blockArgToYieldOperand) {

  std::unordered_set<Operation *> visited;
  SetVector<unsigned> argSet;

  // DFS
  std::function<void(Operation *)> dfs = [&](Operation *op) {
    if (visited.find(op) != visited.end())
      return;
    visited.insert(op);
    for (Value operand : op->getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (!blockArgToYieldOperand[blockArg])
          continue;
        argSet.insert(blockArg.getArgNumber() - forOp.getNumInductionVars());
        operand = blockArgToYieldOperand[blockArg];
      }
      Operation *depOp = operand.getDefiningOp();
      assert(depOp && "Unexpected Value with no defining op");
      if (depOp->getBlock() != forOp.getBody())
        continue;
      // assert(hasWarpGroupId(depOp, warpGroupId) && "Dependency error");
      dfs(depOp);
    }
  };

  // Start from operations that are marked with this warpGroupId explicitly and
  // check dependency with DFS traversal
  forOp.walk([&](Operation *op) {
    if (hasWarpGroupId(op, warpGroupId) && !isa<scf::YieldOp>(op))
      dfs(op);
  });

  // Collect used block args
  SmallVector<unsigned> args(argSet.begin(), argSet.end());
  llvm::sort(args);
  return args;
}

DenseMap<int, scf::ForOp> createForOpsForEachAgent(scf::ForOp forOp) {
  DenseMap<int, scf::ForOp> warpGroupsToForOp;
  // Collect operation list for each warpGroupId
  DenseMap<int, SmallVector<Operation *>> opList;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    int warpGroupId = getWarpGroupId(&op);
    if (warpGroupId == -1)
      continue;
    opList[warpGroupId].push_back(&op);
  }

  if (opList.size() <= 1)
    return warpGroupsToForOp;

  // Figure out dependency between agentLoop operations that belong to different
  // warp groups.

  // Treat ops without an explicit warpGroupId as belonging to all warpGroups.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    int warpGroupId = getWarpGroupId(&op);
    if (warpGroupId == -1) {
      for (auto &warpGroup : opList) {
        warpGroup.second.push_back(&op);
      }
    }
  }

  // Prepare blockArgToYieldOperand mapping
  DenseMap<BlockArgument, Value> blockArgToYieldOperand;
  auto yieldOp = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  assert(yieldOp.getNumOperands() == forOp.getNumRegionIterArgs());
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
    blockArgToYieldOperand[forOp.getRegionIterArg(i)] = yieldOp.getOperand(i);

  auto loc = forOp.getLoc();
  OpBuilderWithWarpGroupIds builder(forOp.getContext());
  auto ctx = forOp.getContext();

  // Create newForOp for each warp group
  for (int warpGroupId : getNestedWarpGroupIds(forOp)) {
    if (warpGroupId == -1)
      continue;
    LDBG("warpGroupId " << warpGroupId);
    auto usedArgs = checkDependencyAndCollectUsedArgs(forOp, warpGroupId,
                                                      blockArgToYieldOperand);

    // Prepare newLoopArgs
    SmallVector<Value> newLoopArgs;
    for (unsigned argNumber : usedArgs)
      newLoopArgs.push_back(forOp.getInitArgs()[argNumber]);

    // Create newForOp
    builder.setWarpGroupId(warpGroupId);
    builder.setInsertionPoint(forOp);
    auto newForOp = builder.createWithWarpGroupId<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        newLoopArgs);

    // Initialize Value mapping from forOp to newForOp
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (unsigned i = 0; i < usedArgs.size(); ++i) {
      LDBG("-- argnum " << i);
      auto oldArg = forOp.getRegionIterArgs()[usedArgs[i]];
      auto newArg = newForOp.getRegionIterArgs()[i];
      mapping.map(oldArg, newArg);
    }

    // Clone all operations with this warpGroupId to newForOp
    builder.setInsertionPointToStart(newForOp.getBody());
    for (Operation *op : opList[warpGroupId]) {
      LLVM_DEBUG({
        LDBG("-- op ");
        op->dump();
      });
      Operation *newOp = builder.clone(*op, mapping);
      setWarpGroupId(ctx, newOp, warpGroupId);
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
    }

    // Create YieldOp for newForOp
    SmallVector<Value> newYieldOperands;
    for (unsigned i : usedArgs) {
      LDBG("lookup operand " << i);
      yieldOp.getOperand(i).dump();
      newYieldOperands.push_back(mapping.lookup(yieldOp.getOperand(i)));
    }
    bool createNewYield = true;
    if (newForOp.getBody()->mightHaveTerminator()) {
      auto initialYield =
          llvm::cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
      if (newYieldOperands.size() == 0) {
        setWarpGroupId(ctx, initialYield, warpGroupId);
        createNewYield = false;
      }
    }
    if (createNewYield) {
      auto newYieldOp =
          builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      setWarpGroupId(ctx, newYieldOp, warpGroupId);
    }

    // Replace results of forOp with results of newForOp
    for (unsigned i = 0; i < usedArgs.size(); ++i) {
      auto oldResult = forOp.getResult(usedArgs[i]);
      auto newResult = newForOp.getResult(i);
      oldResult.replaceUsesWithIf(newResult, [&](OpOperand &operand) -> bool {
        return hasWarpGroupId(operand.getOwner(), warpGroupId);
      });
    }

    warpGroupsToForOp[warpGroupId] = newForOp;
  }

  return warpGroupsToForOp;
}

DenseMap<int, scf::IfOp> SpecializeRegion(triton::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  auto loc = funcOp.getLoc();

  // Get block from funcOp
  Block *block = &funcOp.getBody().front();
  auto returnOp = llvm::cast<triton::ReturnOp>(block->getTerminator());

  // Collect original operations
  SmallVector<Operation *> opList;
  for (Operation &op : block->getOperations())
    opList.push_back(&op);

  // Get curWarpGroupId
  builder.setInsertionPoint(returnOp);
  Value curWarpGroupId = builder.create<ttng::GetCanonicalWarpIdOp>(loc);

  // Resources for each warpGroupId
  DenseMap<int, std::shared_ptr<OpBuilderWithWarpGroupIds>>
      warpGroupsToBuilders;
  DenseMap<int, scf::IfOp> warpGroupsToIfOp;
  DenseMap<int, IRMapping> warpGroupsToIRMappings;

  for (int warpGroupId : getNestedWarpGroupIds(funcOp)) {
    if (warpGroupId == -1)
      continue;
    // Create IfOp for each warpGroupId
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, curWarpGroupId,
        builder.create<arith::ConstantIntOp>(loc, warpGroupId, 32));

    auto ifOp = builder.create<scf::IfOp>(loc, cond);
    warpGroupsToIfOp[warpGroupId] = ifOp;
    setWarpGroupId(context, ifOp, warpGroupId);

    // Create OpBuilderWithWarpGroupIds for each agent
    auto warpGroupBuilder =
        std::make_shared<OpBuilderWithWarpGroupIds>(context);
    warpGroupsToBuilders[warpGroupId] = warpGroupBuilder;
    warpGroupBuilder->setWarpGroupId(warpGroupId);

    // Set insertion point before yieldOp
    auto yieldOp = ifOp.thenYield();
    setWarpGroupId(context, yieldOp, warpGroupId);
    warpGroupBuilder->setInsertionPoint(yieldOp);
  }

  // Clone all operations into corresponding if blocks
  SmallVector<Operation *> cloned;
  for (Operation *op : opList) {
    auto warpGroupId = getWarpGroupId(op);
    if (warpGroupId != -1) {
      cloned.push_back(op);
      IRMapping &mapping = warpGroupsToIRMappings[warpGroupId];
      LLVM_DEBUG({
        LDBG("clone op ");
        op->dump();
      });
      Operation *newOp = warpGroupsToBuilders[warpGroupId]->clone(*op, mapping);
      auto newForOp = dyn_cast<scf::ForOp>(newOp);
      if (newForOp) {
        for (Operation &opT : newForOp.getBody()->without_terminator()) {
          LLVM_DEBUG({
            LDBG("addr " << (&opT) << ": ");
            opT.dump();
          });
        }
      }
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
    }
  }

  // Remove original operations that have been cloned in reverse order
  for (auto it = cloned.rbegin(); it != cloned.rend(); ++it) {
    Operation *op = *it;
    LLVM_DEBUG({
      LDBG("erasing op ");
      op->dump();
    });
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (forOp) {
      for (Operation &opT : forOp.getBody()->without_terminator()) {
        LLVM_DEBUG({
          LDBG("addr " << (&opT) << ": ");
          opT.dump();
        });
      }
    }
    {
      bool hasUse = false;
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        for (Operation *user : op->getResult(i).getUsers()) {
          hasUse = true;
          LLVM_DEBUG({
            LDBG("op has use ");
            user->dump();
          });
        }
      }
      if (!hasUse)
        op->erase();
    }
  }

  LLVM_DEBUG({
    LDBG("created IfOps:");
    warpGroupsToIfOp[0]->dump();
    warpGroupsToIfOp[1]->dump();
  });
  return warpGroupsToIfOp;
}

struct Channel {
public:
  using Relation = std::pair<int, int>;

  Channel(int producer, int consumer, Operation *src, Operation *dst,
          Value srcOperand)
      : relation(producer, consumer), srcOp(src), dstOp(dst),
        srcOperand(srcOperand) {}

  bool operator==(const Channel &c) {
    return relation == c.relation && srcOp == c.srcOp && dstOp == c.dstOp;
  }

  Relation relation;
  Operation *srcOp;
  Operation *dstOp;
  Value srcOperand;
};

void collectAsyncChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                          triton::FuncOp &funcOp) {
  funcOp.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (result.use_empty() || !op->hasAttr("async_agent")) {
        continue;
      }
      auto producerAgent =
          op->getAttrOfType<DenseIntElementsAttr>("async_agent");
      if (producerAgent.getValues<int>().size() > 1) {
        continue;
      }
      for (Operation *userOp : result.getUsers()) {
        if (!userOp->hasAttr("async_agent") ||
            userOp->getAttrOfType<DenseIntElementsAttr>("async_agent")
                    .getValues<int>()
                    .size() > 1) {
          continue;
        }
        auto consumerAgentId =
            userOp->getAttrOfType<DenseIntElementsAttr>("async_agent")
                .getValues<int>()[0];
        auto producerAgentId = producerAgent.getValues<int>()[0];
        if (producerAgentId != consumerAgentId) {
          channels.push_back(std::make_unique<Channel>(
              producerAgentId, consumerAgentId, op, userOp, result));
        }
      }
    }
  });

  LLVM_DEBUG({
    LDBG("Aysnc channels:");
    for (auto &channel : channels) {
      LDBG("producer op: " << channel->relation.first);
      channel->srcOp->dump();
      channel->srcOperand.dump();
      LDBG("consumer: " << channel->relation.second);
      channel->dstOp->dump();
    }
  });
}

void reduceChannels(SmallVector<Channel *> &channels,
                    DenseMap<Operation *, SmallVector<Channel *>> &map) {
  // If producers or their consumers has the same convergent consumer,
  // and those producers, producers' consumers and the convergent consumer are
  // in the same block, They share the same token.
  auto checkConverge = [](Operation *op1, Operation *op2) -> Operation * {
    // Only check level-0 and level-1 convergence, e.g.
    // producer:       load0          load1
    //                   |              |
    // consumer:  convertLayout0  convertLayout1
    //                    \             /
    // consumer:                 dot
    // The example above is level-1 convergence.
    // If convertLayoutOps converge in deeper depth, this function will
    // fail to detect.
    // TODO: implement general level-N convergence.
    if (op1 == op2) {
      return op1;
    }
    if (op1->getBlock() == op2->getBlock() && op1->hasOneUse() &&
        op2->hasOneUse() &&
        *(op1->getUsers().begin()) == *(op2->getUsers().begin()) &&
        (*(op1->getUsers().begin()))->getBlock() == op1->getBlock()) {
      return *(op1->getUsers().begin());
    }
    return nullptr;
  };
  assert(channels.size() > 0 && "channel size is zero");
  // Compare with existing channels in map
  for (auto c0 = channels.begin(); c0 != channels.end(); ++c0) {
    bool isConvergent = false;
    for (auto &kv : map) {
      if (kv.second.size() > 0 &&
          (*c0)->srcOp->getBlock() == kv.second.front()->srcOp->getBlock()) {
        if (auto cvg = checkConverge((*c0)->dstOp, kv.second.front()->dstOp)) {
          kv.second.push_back(*c0);
          isConvergent = true;
          break;
        }
      }
    }
    if (!isConvergent) {
      map[(*c0)->dstOp].push_back(*c0);
    }
  }

  // Reorder channels and maps based on locations of producers
  for (auto &kv : map) {
    if (kv.second.size() > 1) {
      auto &allOps = kv.second.front()->srcOp->getBlock()->getOperations();
      std::sort(
          kv.second.begin(), kv.second.end(), [&](Channel *a, Channel *b) {
            auto itrA =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == a->srcOp;
                });
            auto itrB =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == b->srcOp;
                });
            assert(itrA != allOps.end() && itrB != allOps.end());
            return std::distance(itrA, itrB) < 0;
          });
    }
  }
}

scf::ForOp createNewLoop(scf::ForOp forOp, int numStages,
                         scf::ForOp &parentForOp) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();

  // The agentId set of pipelineIdx is the union of agentId sets of all ops in
  // the for loop
  OpBuilderWithAgentIds builder(forOp.getContext());
  builder.setAgentIdsFromArray(getNestedAgentIds(forOp));

  builder.setInsertionPoint(forOp);
  Value numStagesVal =
      builder.createWithAgentIds<arith::ConstantIntOp>(loc, numStages, 32);

  // 0. Append pipelineIdx to block arguments
  Value phase =
      body->insertArgument(body->getNumArguments(), builder.getI1Type(), loc);
  Value pipelineIdx =
      body->insertArgument(body->getNumArguments(), builder.getI32Type(), loc);

  // 1. prepare index and phase for next iteration
  // nextIdx = curIdx + 1
  // nextPhase = ((nextIdx < numStages && curPhase) || (nextIdx >= numStages &&
  // curPhase^1))
  // nextIdx = nextIdx >= numStages ? 0 : nextIdx
  auto yieldOp = llvm::cast<scf::YieldOp>(body->getTerminator());
  builder.setInsertionPoint(yieldOp);
  Value one = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 1, 32);
  Value zero = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 0, 32);
  Value _1_1b = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 1, 1);
  // generate index for next iter
  Value nextPipelineIdx =
      builder.createWithAgentIds<arith::AddIOp>(loc, pipelineIdx, one);
  Value pipelineGECond = builder.createWithAgentIds<arith::CmpIOp>(
      loc, arith::CmpIPredicate::uge, nextPipelineIdx, numStagesVal);
  Value pipelineLTCond = builder.createWithAgentIds<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, nextPipelineIdx, numStagesVal);
  Value cyclePipelineIdx = builder.createWithAgentIds<arith::SubIOp>(
      loc, nextPipelineIdx, numStagesVal);
  nextPipelineIdx = builder.createWithAgentIds<mlir::arith::SelectOp>(
      loc, pipelineGECond, cyclePipelineIdx, nextPipelineIdx);
  // generate phase for next iter
  Value flipPhase =
      builder.createWithAgentIds<mlir::arith::XOrIOp>(loc, phase, _1_1b);
  Value cond0 = builder.createWithAgentIds<mlir::arith::AndIOp>(
      loc, pipelineGECond, flipPhase);
  Value cond1 = builder.createWithAgentIds<mlir::arith::AndIOp>(
      loc, pipelineLTCond, phase);
  Value nextPhase =
      builder.createWithAgentIds<mlir::arith::OrIOp>(loc, cond0, cond1);

  // 2. Append pipelineIdx to yield operands
  yieldOp->insertOperands(yieldOp.getNumOperands(),
                          {nextPhase, nextPipelineIdx});

  // 3. create newLoopArgs
  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getInitArgs())
    newLoopArgs.push_back(operand);

  builder.setInsertionPoint(forOp);
  Value initPipelineIdx, initEmptyIdx, initPhase;
  zero = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 0, 32);
  if (parentForOp) {
    // Make sure prior pipelineIdx is inserted in the end of parentForOp
    initPipelineIdx = parentForOp.getBody()->getArguments().back();
    Value numSteps = builder.createWithAgentIds<arith::SubIOp>(
        loc, forOp.getUpperBound(), forOp.getLowerBound());
    numSteps = builder.createWithAgentIds<arith::AddIOp>(loc, numSteps,
                                                         forOp.getStep());
    Value one = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 1, 32);
    Value two = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 2, 32);
    numSteps = builder.createWithAgentIds<arith::SubIOp>(loc, numSteps, one);
    numSteps = builder.createWithAgentIds<arith::DivUIOp>(loc, numSteps,
                                                          forOp.getStep());
    // initPipelineIdx = (parentForOp.pipelineIdx * numSteps) % numStages
    // initPhase = ((parentForOp.pipelineIdx * numSteps) / numStages) & 1
    initPipelineIdx = builder.createWithAgentIds<arith::MulIOp>(
        loc, initPipelineIdx, numSteps);
    Value pipelineIdx = builder.createWithAgentIds<arith::DivUIOp>(
        loc, initPipelineIdx, numStagesVal);
    initPipelineIdx = builder.createWithAgentIds<arith::SubIOp>(
        loc, initPipelineIdx,
        builder.createWithAgentIds<arith::MulIOp>(loc, pipelineIdx,
                                                  numStagesVal));
    pipelineIdx =
        builder.createWithAgentIds<arith::AndIOp>(loc, pipelineIdx, one);
    initPhase = builder.createWithAgentIds<arith::TruncIOp>(
        loc, builder.getI1Type(), pipelineIdx);
  } else {
    // phase init to false and pipelineIdx init to 0
    initPipelineIdx = zero;
    initPhase = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 0, 1);
  }
  newLoopArgs.append({initPhase, initPipelineIdx});

  // 4. Create newForOp and take the region of forOp
  auto newForOp = builder.createWithAgentIds<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      newLoopArgs);
  newForOp.getRegion().takeBody(forOp.getRegion());

  // 5. Replace forOp with newForOp
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  return newForOp;
}

SmallVector<Operation *> getAgentLoop(triton::FuncOp funcOp,
                                      const SmallVector<Channel *> &channels) {
  // AgentLoop: outermost Ops with regions in funcOp which contain at least one
  // relation between producer and consumer. It assumes producer-consumer
  // relation going across two outermost Ops in funcOp is forbidden. For
  // example, In the example of runOnOperation(), only the outermost ForOp is
  // agentLoop, the inner ForOp is not.
  SmallVector<Operation *> agentLoopOps;
  auto isAgentLoop = [&](Operation *agentLoop) -> bool {
    for (auto c : channels) {
      Operation *producer = c->srcOp, *consumer = c->dstOp;
      while (producer && !isa<triton::FuncOp>(producer->getParentOp())) {
        producer = producer->getParentOp();
      }
      while (consumer && !isa<triton::FuncOp>(consumer->getParentOp())) {
        consumer = consumer->getParentOp();
      }
      if (producer == agentLoop && consumer == agentLoop) {
        return true;
      }
      assert((producer != agentLoop ||
              isa<triton::FuncOp>(producer->getParentOp())) &&
             (consumer != agentLoop ||
              isa<triton::FuncOp>(consumer->getParentOp())) &&
             "Error: producer and consumer belongs to different agentLoopOps");
    }
    return false;
  };
  Operation *op;
  for (Operation &bodyOp : funcOp.getBody().front().getOperations()) {
    op = &bodyOp;
    if (op->getNumRegions() > 0) {
      // If this op as a whole is a producer or consumer, continue
      if (getAgentIds(op).size() == 1) {
        continue;
      }
      if (isAgentLoop(op)) {
        agentLoopOps.push_back(op);
      }
    }
  }
  return agentLoopOps;
}

void appendPipelineIdxArgs(SmallVector<Operation *> &agentLoop) {

  SmallVector<scf::ForOp> orderedForOps;
  for (auto &op : agentLoop) {
    op->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
      if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
        orderedForOps.push_back(forOp);
      }
    });
  }

  for (auto &op : orderedForOps) {
    scf::ForOp parentForOp = op->getParentOfType<scf::ForOp>();
    scf::ForOp newForOp;
    // for(...) -> for(..., phase, pipelineIdx)
    newForOp = createNewLoop(op, 1, parentForOp);
    auto agentLoopForItr =
        std::find(agentLoop.begin(), agentLoop.end(), op.getOperation());
    if (agentLoopForItr != agentLoop.end()) {
      // Update agentLoop
      *agentLoopForItr = newForOp.getOperation();
    }
  }
}

DenseMap<Channel *, Value>
createToken(const DenseMap<Operation *, SmallVector<Channel *>> &map,
            triton::FuncOp funcOp, int numStages) {
  DenseMap<Channel *, Value> ret;
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  for (auto it = map.begin(); it != map.end(); ++it) {
    Value v;
    if (it->second.front()->srcOp->getParentOfType<scf::ForOp>()) {
      v = builder.create<ttng::CreateTokenOp>(funcOp.getLoc(), numStages);
    } else {
      // No need to pipeline
      v = builder.create<ttng::CreateTokenOp>(funcOp.getLoc(), 1);
    }
    for (auto &c : it->second) {
      ret[c] = v;
    }
  }
  return ret;
}

DenseMap<Channel *, Value> createBuffer(const SmallVector<Channel *> &channels,
                                        triton::FuncOp funcOp, int numStages) {
  DenseMap<Channel *, Value> bufferMap;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  for (const auto &c : channels) {

    if (auto tensorType = dyn_cast<RankedTensorType>(c->srcOperand.getType())) {
      // Get basic information from tensorType
      auto order = ttg::getOrder(tensorType.getEncoding());
      auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
      auto elemType = tensorType.getElementType();

      // Get shape, layout and type of a slice
      auto sliceShape = tensorType.getShape();
      auto sharedLayout = ttg::SharedEncodingAttr::get(
          context, sliceShape, order, CTALayout, elemType);
      auto sliceType =
          RankedTensorType::get(sliceShape, elemType, sharedLayout);

      // Get shape, layout and type of the complete buffer
      SmallVector<int64_t> bufferShape(sliceShape.begin(), sliceShape.end());
      bufferShape.insert(bufferShape.begin(), numStages);
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(context);
      auto bufferType =
          RankedTensorType::get(bufferShape, elemType, sharedLayout);
      Type memdescType =
          tt::MemDescType::get(bufferShape, elemType, sharedLayout,
                               sharedMemorySpace, /*mutableMemory*/ true);
      Value buffer;

      auto loadOp = dyn_cast<triton::LoadOp>(c->srcOp);
      if (loadOp) {
        buffer =
            builder.create<ttg::LocalAllocOp>(funcOp.getLoc(), memdescType);
      } else {
        buffer = builder.create<ttg::LocalAllocOp>(funcOp.getLoc(), memdescType,
                                                   c->srcOperand);
      }
      bufferMap[c] = buffer;
    } else {
      llvm_unreachable("Unexpected result type");
    }
  }
  return bufferMap;
}

void buildAsyncComm(const DenseMap<Operation *, SmallVector<Channel *>> &map,
                    const DenseMap<Channel *, Value> &tokenMap,
                    const DenseMap<Channel *, Value> &bufferMap,
                    int numStages) {

  auto getSameLevelOp = [](Operation *p, Operation *c) -> Operation * {
    while (!isa<triton::FuncOp>(c)) {
      if (c->getParentOp() == p->getParentOp()) {
        return c;
      }
      c = c->getParentOp();
    }
    llvm_unreachable("Failed to find consumer's same level Op with producer");
  };

  auto consumerReleaseHeutistic = [&](Operation *p,
                                      Operation *c) -> Operation * {
    if (c->getBlock() == p->getBlock()) {
      auto consumerAgentId =
          c->getAttrOfType<DenseIntElementsAttr>("async_agent")
              .getValues<int>()[0];
      for (auto it = c->getBlock()->rbegin(); it != c->getBlock()->rend();
           ++it) {
        if (!it->hasAttr("async_agent")) {
          continue;
        }
        auto asyncAttr = it->getAttrOfType<DenseIntElementsAttr>("async_agent")
                             .getValues<int>();
        if (asyncAttr.size() == 1 && asyncAttr[0] == consumerAgentId) {
          return &(*it);
        }
      }
      return nullptr;
    } else {
      return getSameLevelOp(p, c);
    }
  };

  auto getAgents = [&](Operation *p, Operation *c, SmallVector<AgentId> &agentP,
                       SmallVector<AgentId> &agentC,
                       SmallVector<AgentId> &agentsPC) -> void {
    agentP = getNestedAgentIds(p);
    agentC = getNestedAgentIds(c);
    agentsPC.reserve(agentP.size() + agentC.size());
    agentsPC.insert(agentsPC.end(), agentP.begin(), agentP.end());
    agentsPC.insert(agentsPC.end(), agentC.begin(), agentC.end());
  };

  // Don't pipeline dots that depend on ops other than scf.yield and scf.for.
  // Because the DotOp will be replaced by a DotAsyncOp, which will be issued in
  // iter_i but waited in iter_i+1. The use of DotAsyncOp should not be ops
  // other than scf.for and scf.yield because the result of DotAsyncOp is not
  // ready in iter_i.
  auto getValidDot = [&](const SmallVector<Channel *> &block) -> Operation * {
    Operation *headConsumer = block.front()->dstOp;
    if (block.size() == 2 &&
        isa<triton::DotOp>(*headConsumer->getUsers().begin()) &&
        headConsumer->getParentOfType<scf::ForOp>()) {
      auto dotOp = cast<triton::DotOp>(*headConsumer->getUsers().begin());
      auto dot = dotOp.getResult();
      auto resTy = dyn_cast<RankedTensorType>(dot.getType());
      auto cArg = dyn_cast<BlockArgument>(dotOp.getOperand(2));
      if (auto resEnc =
              dyn_cast<ttg::NvidiaMmaEncodingAttr>(resTy.getEncoding()))
        if (resEnc.isHopper() && dot.hasOneUse() &&
            isa<scf::YieldOp>(*dot.getUsers().begin()) && cArg &&
            cArg.hasOneUse() && numStages > 1)
          return dotOp.getOperation();
    }
    return nullptr;
  };

  // TODO: try to optimize locations of arriving and waiting token
  // for fused-attention
  for (auto kv : map) {
    /*****************Token related*****************/
    auto headProducer = kv.second.front()->srcOp;
    auto tailProducer = kv.second.back()->srcOp;
    auto headConsumer = kv.second.front()->dstOp;
    auto tailConsumer = kv.second.back()->dstOp;
    auto token = tokenMap.find(kv.second.front())->second;
    SmallVector<AgentId> agentP, agentC, agentsPC;
    getAgents(headProducer, headConsumer, agentP, agentC, agentsPC);
    OpBuilderWithAgentIds builder(headProducer->getContext());

    if (auto funcOp = dyn_cast<triton::FuncOp>(headProducer->getParentOp())) {
      builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    } else {
      builder.setInsertionPoint(headProducer->getParentOp());
    }
    builder.setAgentIdsFromArray(agentsPC);
    Value pipelineIdx;
    if (auto forOp = headProducer->getParentOfType<scf::ForOp>()) {
      pipelineIdx = forOp.getBody()->getArguments().back();
    } else {
      // existing");
      pipelineIdx = builder.createWithAgentIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 32);
    }

    if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(headProducer)) {
      // link local alloc to the consumer side
      // insert ProducerAcquireOp
      builder.setInsertionPoint(headProducer);
      builder.setAgentIdsFromArray(agentP);
      builder.createWithAgentIds<ttng::ProducerAcquireOp>(
          headProducer->getLoc(), token, pipelineIdx);

      // insert ProducerCommitOp
      builder.setInsertionPointAfter(tailProducer);
      builder.createWithAgentIds<ttng::ProducerCommitOp>(tailProducer->getLoc(),
                                                         token, pipelineIdx);
    } else {
      assert(false && "Not implemented yet");
    }

    builder.setAgentIdsFromArray(agentC);
    // insert ConsumerWaitOp
    auto consumerWaitPoint = getSameLevelOp(headProducer, headConsumer);
    builder.setInsertionPoint(consumerWaitPoint);
    builder.createWithAgentIds<ttng::ConsumerWaitOp>(headConsumer->getLoc(),
                                                     token, pipelineIdx);
    /// async launch dots
    if (auto cvg = getValidDot(kv.second)) {
      auto dotOp = cast<triton::DotOp>(cvg);
      auto dot = dotOp.getResult();
      auto loc = dot.getLoc();
      auto forOp = cvg->getParentOfType<scf::ForOp>();

      auto agentIds = getNestedAgentIds(dotOp);
      OpBuilderWithAgentIds builder(dotOp.getContext());
      builder.setAgentIdsFromArray(agentIds);
      builder.setInsertionPoint(dotOp);

      // 0. replace Dot with DotAsync
      auto dotAsync =
          builder.createWithAgentIds<triton::nvidia_gpu::WarpGroupDotOp>(
              loc, dotOp.getA(), dotOp.getB(), dotOp.getC(),
              dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
      dot.replaceAllUsesWith(dotAsync.getResult());
      builder.createWithAgentIds<triton::nvidia_gpu::WarpGroupDotWaitOp>(
          loc, dotAsync.getResult(), 1);

      // 1. insert ConsumerReleaseOp for DotAsyncOps
      Value cond = builder.createWithAgentIds<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, forOp.getInductionVar(),
          forOp.getLowerBound());
      auto ifOp =
          builder.createWithAgentIds<scf::IfOp>(loc, ArrayRef<Type>{}, cond,
                                                /*hasElse*/ false);
      setAgentIds(ifOp.thenYield().getOperation(), agentIds);
      builder.setInsertionPointToStart(ifOp.thenBlock());
      Value consumerReleaseIdx = forOp.getBody()->getArguments().back();
      Value zero = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 0, 32);
      Value one = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 1, 32);
      Value lastStage = builder.createWithAgentIds<arith::ConstantIntOp>(
          loc, numStages - 1, 32);
      Value consumerReleaseIdxMinusOne =
          builder.createWithAgentIds<arith::SubIOp>(loc, consumerReleaseIdx,
                                                    one);
      cond = builder.createWithAgentIds<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, consumerReleaseIdx, zero);
      consumerReleaseIdx = builder.createWithAgentIds<arith::SelectOp>(
          loc, cond, lastStage, consumerReleaseIdxMinusOne);
      builder.createWithAgentIds<ttng::ConsumerReleaseOp>(loc, token,
                                                          consumerReleaseIdx);

      // 2. If there's any outstanding DotAsyncOps, we need to wait for them.
      builder.setInsertionPointAfter(forOp);
      unsigned resultIndex = dotAsync->getUses().begin()->getOperandNumber();
      Value result = forOp->getResult(resultIndex);
      auto dotWait =
          builder.createWithAgentIds<triton::nvidia_gpu::WarpGroupDotWaitOp>(
              forOp.getLoc(), result, 0);
      result.replaceAllUsesExcept(dotWait.getResult(0), dotWait);

      // 3. insert ConsumerReleaseOp for outstanding DotAsyncOps
      zero = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 0, 32);
      one = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 1, 32);
      lastStage = builder.createWithAgentIds<arith::ConstantIntOp>(
          loc, numStages - 1, 32);
      consumerReleaseIdx = forOp.getResults().back();
      consumerReleaseIdxMinusOne = builder.createWithAgentIds<arith::SubIOp>(
          loc, consumerReleaseIdx, one);
      cond = builder.createWithAgentIds<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, consumerReleaseIdx, zero);
      consumerReleaseIdx = builder.createWithAgentIds<arith::SelectOp>(
          loc, cond, lastStage, consumerReleaseIdxMinusOne);
      builder.createWithAgentIds<ttng::ConsumerReleaseOp>(loc, token,
                                                          consumerReleaseIdx);
      dotOp->erase();
    } else {
      // insert ConsumerReleaseOp
      auto consumerReleasePoint =
          consumerReleaseHeutistic(tailProducer, tailConsumer);
      builder.setInsertionPointAfter(consumerReleasePoint);
      builder.createWithAgentIds<ttng::ConsumerReleaseOp>(
          consumerReleasePoint->getLoc(), token, pipelineIdx);
    }

    /*****************Buffer related*****************/
    /// splitLoadsInForLoop
    for (auto &c : kv.second) {
      assert(isa<triton::LoadOp>(c->srcOp) && "prodcuerOp is not tt.load");
      auto loadOp = cast<triton::LoadOp>(c->srcOp);
      auto buffer = bufferMap.find(c)->second;
      MLIRContext *context = loadOp->getContext();
      OpBuilderWithAgentIds builder(context);
      builder.setInsertionPoint(loadOp->getParentOp());
      builder.setAgentIdsFromArray(agentsPC);

      builder.setInsertionPoint(loadOp);
      Value loadResult = loadOp.getResult();
      if (auto tensorType = dyn_cast<RankedTensorType>(loadResult.getType())) {
        // Get basic information from tensorType
        auto order = ttg::getOrder(tensorType.getEncoding());
        auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
        auto elemType = tensorType.getElementType();

        // Get shape, layout and type of a slice
        auto sliceShape = tensorType.getShape();
        auto sharedLayout = ttg::SharedEncodingAttr::get(
            context, sliceShape, order, CTALayout, elemType);
        auto sliceType =
            RankedTensorType::get(sliceShape, elemType, sharedLayout);

        Attribute sharedMemorySpace =
            triton::gpu::SharedMemorySpaceAttr::get(context);
        tt::MemDescType subviewTy = tt::MemDescType::get(
            sliceType.getShape().drop_front(), sliceType.getElementType(),
            sliceType.getEncoding(), sharedMemorySpace,
            /*mutableMemory=*/true);
        Value zero =
            builder.create<arith::ConstantIntOp>(loadOp.getLoc(), 0, 32);
        Value one =
            builder.create<arith::ConstantIntOp>(loadOp.getLoc(), 1, 32);
        SmallVector<Value> copyOffsets(sliceType.getRank(), zero);
        copyOffsets[0] = pipelineIdx;
        auto view = builder.create<ttg::MemDescSubviewOp>(
            loadOp.getLoc(), subviewTy, buffer, copyOffsets);
        // Create cp.async
        builder.setAgentIdsFromOp(loadOp);
        builder.setInsertionPointAfter(loadOp);
        Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
            loadOp.getLoc(), loadOp.getPtr(), view, loadOp.getMask(),
            loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
            loadOp.getIsVolatile());

        // Extract part.
        builder.setAgentIdsFromValueUsers(loadResult);
        builder.setInsertionPoint(c->dstOp);
        SmallVector<Value> loadOffsets(sliceType.getRank(), zero);
        loadOffsets[0] = pipelineIdx;
        auto viewLoad = builder.create<ttg::MemDescSubviewOp>(
            loadOp.getLoc(), subviewTy, buffer, loadOffsets);

        // Replace all uses of loadResult
        loadResult.replaceAllUsesWith(viewLoad.getResult());
        loadOp.erase();
      }
    }
  }
}

DenseMap<AgentId, scf::ForOp> createForOpsForEachAgentId(scf::ForOp forOp) {
  // Collect operation list for each agentId
  DenseMap<AgentId, SmallVector<Operation *>> opList;
  for (Operation &op : forOp.getBody()->without_terminator())
    for (AgentId agentId : getAgentIds(&op))
      opList[agentId].push_back(&op);

  // Prepare blockArgToYieldOperand mapping
  DenseMap<BlockArgument, Value> blockArgToYieldOperand;
  auto yieldOp = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  assert(yieldOp.getNumOperands() == forOp.getNumRegionIterArgs());
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
    blockArgToYieldOperand[forOp.getRegionIterArg(i)] = yieldOp.getOperand(i);

  auto loc = forOp.getLoc();
  OpBuilderWithAgentIds builder(forOp.getContext());
  DenseMap<AgentId, scf::ForOp> agentsToForOp;

  // Create newForOp for each agent
  for (AgentId agentId : getNestedAgentIds(forOp)) {
    auto usedArgs = checkDependencyAndCollectUsedArgs(forOp, agentId,
                                                      blockArgToYieldOperand);

    // Prepare newLoopArgs
    SmallVector<Value> newLoopArgs;
    for (unsigned argNumber : usedArgs)
      newLoopArgs.push_back(forOp.getInitArgs()[argNumber]);

    // Create newForOp
    builder.setAgentIdsFromArray({agentId});
    builder.setInsertionPoint(forOp);
    auto newForOp = builder.createWithAgentIds<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        newLoopArgs);

    // Initialize Value mapping from forOp to newForOp
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (unsigned i = 0; i < usedArgs.size(); ++i) {
      auto oldArg = forOp.getRegionIterArgs()[usedArgs[i]];
      auto newArg = newForOp.getRegionIterArgs()[i];
      mapping.map(oldArg, newArg);
    }

    // Clone all operations with this agentId to newForOp
    builder.setInsertionPointToStart(newForOp.getBody());
    for (Operation *op : opList[agentId]) {
      Operation *newOp = builder.clone(*op, mapping);
      setAgentIds(newOp, {agentId});
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
    }

    // Create YieldOp for newForOp
    SmallVector<Value> newYieldOperands;
    for (unsigned i : usedArgs)
      newYieldOperands.push_back(mapping.lookup(yieldOp.getOperand(i)));
    auto newYieldOp =
        builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    setAgentIds(newYieldOp, {agentId});

    // Replace results of forOp with results of newForOp
    for (unsigned i = 0; i < usedArgs.size(); ++i) {
      auto oldResult = forOp.getResult(usedArgs[i]);
      auto newResult = newForOp.getResult(i);
      oldResult.replaceUsesWithIf(newResult, [&](OpOperand &operand) -> bool {
        return hasAgentId(operand.getOwner(), agentId);
      });
    }

    agentsToForOp[agentId] = newForOp;
  }

  return agentsToForOp;
}

DenseMap<AgentId, Operation *> agentDivision(Operation *agentLoop) {
  // A general agent division in agentLoop could be:
  // *  If opWithRegion has results, e.g. scf.for, this opWithRegion will be
  //    split into several new operations, each agent has one, which
  //    has the part of results related to this agent. One agent could own
  //    all original results or none of them, but one result must belong to
  //    one and only one agent.
  // *  if opWithRegions doesn't have result. Simply split for every agent.
  // *  So does operands of opWithRegions
  // However, current agentLoops are all ForOps and IfOps. So we customize
  // the implementation.
  DenseMap<AgentId, Operation *> agentBackbone;
  agentLoop->walk([&](Operation *op) {
    auto ids = getAgentIds(op);
    if (op->getNumRegions() > 0 && ids.size() > 1) {
      // ForOp: change iterArgs and yield results
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        auto forOps = createForOpsForEachAgentId(forOp);
        if (op == agentLoop) {
          for (auto kv : forOps) {
            auto f = kv.second;
            auto id = getAgentIds(f.getOperation());
            assert(id.size() == 1 &&
                   "generated ForOp doesn't have one and only one agentId");
            agentBackbone[id.front()] = f.getOperation();
          }
        }
        forOp.erase();
      } else {
        llvm_unreachable("Unexpected Op with regions");
      }
    }
  });
  assert(agentBackbone.size() > 0 && "Agent division failed");
  return agentBackbone;
}

void cloneAgentLoopForEachAgentId(SmallVector<Operation *> &agentLoop) {
  SmallVector<Operation *> newBackBone;

  for (Operation *op : agentLoop) {
    auto loc = op->getLoc();
    OpBuilderWithAgentIds builder(op->getContext());
    builder.setInsertionPoint(op);
    // First, agent division
    DenseMap<AgentId, Operation *> agentAgentLoop = agentDivision(op);

    // Second, remove irrelevant Ops
    for (auto kv : agentAgentLoop) {
      SmallVector<Operation *> deleteOps;
      AgentId targetId = kv.first;
      Operation *newAgentLoop = kv.second;
      newAgentLoop->walk([&](Operation *subOp) {
        auto ids = getAgentIds(subOp);
        if (std::find(ids.begin(), ids.end(), targetId) == ids.end()) {
          deleteOps.push_back(subOp);
        }
      });
      for (auto it = deleteOps.rbegin(); it != deleteOps.rend(); ++it) {
        (*it)->erase();
      }
    }
  }
}

#define GEN_PASS_DEF_TRITONGPUWARPSPECEXPERIMENTAL
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUWarpSpecExperimentalPass
    : public impl::TritonGPUWarpSpecExperimentalBase<
          TritonGPUWarpSpecExperimentalPass> {
public:
  using impl::TritonGPUWarpSpecExperimentalBase<
      TritonGPUWarpSpecExperimentalPass>::TritonGPUWarpSpecExperimentalBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    // First step: collect channels
    SmallVector<std::unique_ptr<Channel>> channelsOrigin;
    collectAsyncChannels(channelsOrigin, funcOp);
    SmallVector<Channel *> channels;
    for (const auto &c : channelsOrigin) {
      channels.push_back(c.get());
    }

    if (channels.empty()) {
      return;
    }

    // cvgOp-channels map
    DenseMap<Operation *, SmallVector<Channel *>> map;
    reduceChannels(channels, map);

    // Prepare phase, getAgentLoop, appendPipelineIdxArgs
    SmallVector<Operation *> agentLoop = getAgentLoop(funcOp, channels);
    appendPipelineIdxArgs(agentLoop);

    // Create token, buffer and data transfer between async agents
    DenseMap<Channel *, Value> tokenMap = createToken(map, funcOp, 1);
    DenseMap<Channel *, Value> bufferMap = createBuffer(channels, funcOp, 1);
    buildAsyncComm(map, tokenMap, bufferMap, 1);

    // Clone agentLoop, remove irrelevant blockArgument for {forOp, ifOp}
    cloneAgentLoopForEachAgentId(agentLoop);

    auto ret = SpecializeRegion(funcOp);
    LLVM_DEBUG({
      LDBG("with IfOps");
      funcOp.dump();
    });
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
    LLVM_DEBUG({
      LDBG("post pass");
      getOperation()->dump();
    });
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
