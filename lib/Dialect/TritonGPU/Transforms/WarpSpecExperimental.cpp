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
#include <unordered_set>

#define DEBUG_TYPE "triton-warp-spec-experimental"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
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

DenseMap<int, scf::ForOp> createForOpsForEachWarpGroupId(scf::ForOp forOp) {
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

#define GEN_PASS_DEF_TRITONGPUWARPSPECEXPERIMENTAL
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUWarpSpecExperimentalPass
    : public impl::TritonGPUWarpSpecExperimentalBase<
          TritonGPUWarpSpecExperimentalPass> {
public:
  using impl::TritonGPUWarpSpecExperimentalBase<
      TritonGPUWarpSpecExperimentalPass>::TritonGPUWarpSpecExperimentalBase;

  int getNumBuffersOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr("tt.num_buffers"))
      return numBuffers;
    return mlir::cast<IntegerAttr>(forOp->getAttr("tt.num_buffers")).getInt();
  }

  void runOnFuncOp(triton::FuncOp funcOp) {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_buffer <= 1.
      if (getNumBuffersOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    if (loops.empty())
      return;
    for (scf::ForOp forOp : loops) {
      createForOpsForEachWarpGroupId(forOp);
    }

    auto ret = SpecializeRegion(funcOp);
    LLVM_DEBUG({
      LDBG("with IfOps");
      funcOp.dump();
    });

#if 0
    // We can't erase the original forOp since there is a use
    // of the tl.load from the consumer warp group. We need
    // to fix the producer-consumer channel first.
    for (scf::ForOp forOp : loops) {
      bool hasUse = false;
      for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
        for (Operation *user : forOp.getResult(i).getUsers()) {
          hasUse = true;
          LLVM_DEBUG({LDBG("forOp has use "); user->dump(); });
        }
      }
      if (!hasUse) {
        LLVM_DEBUG({LDBG("erasing op "); forOp.dump(); });
        for (Operation &op : forOp.getBody()->without_terminator()) {
          LLVM_DEBUG({LDBG("addr " << (&op) << ": "); op.dump(); });
          for (Operation *user : op.getUsers()) {
            LDBG("-- user " << user);
          }
        }
        forOp.erase();
      }
    }
#endif
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
