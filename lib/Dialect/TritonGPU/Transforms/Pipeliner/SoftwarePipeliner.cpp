#include "PipelineExpander.h"
#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// Return true if the preconditions for pipelining the loop are met.
bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def;
                   }))
    return false;
  // Don't pipeline outer loops.
  if (forOp
          ->walk([&](Operation *op) {
            if (forOp.getOperation() == op)
              return WalkResult::advance();
            if (isa<scf::ForOp, scf::WhileOp>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  return true;
}

/// Collect loads to pipeline. Return success if we can pipeline this loop
static void collectLoadOpsToPipeline(scf::ForOp forOp,
                                     SmallVectorImpl<tt::LoadOp> &ops,
                                     bool &hasMMAV3) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(&op)) {
      bool candidate = false;
      if (isLoadFromTensorPtr(loadOp)) {
        // Map to TMA load.
        candidate = true;
      } else {
        auto ptr = loadOp.getPtr();
        unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
        if (auto mask = loadOp.getMask())
          vec =
              std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

        auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
        if (!tensorTy)
          continue;
        auto ty =
            tensorTy.getElementType().cast<tt::PointerType>().getPointeeType();
        unsigned width = vec * ty.getIntOrFloatBitWidth();
        // We do not pipeline all loads for the following reasons:
        // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8 and 16.
        // 2. It's likely that pipling small loads won't offer much performance
        //    improvement and may even hurt performance by increasing register
        //    pressure.
        if (width >= 32)
          candidate = true;
      }
      if (candidate)
        ops.push_back(loadOp);
    }
  }
}

// Create an allocation that can old distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, tt::LoadOp loadOp,
                         unsigned distance) {
  OpBuilder builder(forOp);
  auto ty = loadOp.getType().cast<RankedTensorType>();
  auto CTALayout = ttg::getCTALayout(ty.getEncoding());
  Attribute sharedEnc = ttg::SharedEncodingAttr::get(
      ty.getContext(), ty.getShape(), ttg::getOrder(ty.getEncoding()),
      CTALayout, ty.getElementType());
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type allocType =
      RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
  Value alloc = builder.create<mlir::triton::gpu::AllocTensorOp>(
      loadOp.getLoc(), allocType);
  return alloc;
}

void createAsyncLoad(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                     Value insertIdx, Value extractIdx, Value phase);

void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands);

void setWaitNum(Operation *op,
                mlir::triton::PipeliningOption::PipelinerPart part,
                unsigned iteration, unsigned numLoadsInStage);

// Convert load ops into their asyn version and apply multi-buffering based on
// the number of stages.
static SmallVector<Value> createAsynOps(scf::ForOp &forOp,
                                        ArrayRef<tt::LoadOp> loads,
                                        int numStages, bool hasMMAV3) {
  struct AsyncLoad {
    AsyncLoad(tt::LoadOp loadOp, Value alloc) : loadOp(loadOp), alloc(alloc) {}
    tt::LoadOp loadOp;
    Value alloc;
  };
  int numBuffers = numStages - 1;
  // For MMAv3 we need an extra buffer as this is assumed in the wgmma
  // pipelining post-processing.
  // TODO: Improve modeling of wgmma pipelining.
  if (hasMMAV3)
    numBuffers++;
  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> allocs;
  SmallVector<Value> newOperands;
  bool needsMbarrierPhase = false;
  bool needsAsyncWait = false;
  for (tt::LoadOp loadOp : loads) {
    Value alloc = createAlloc(forOp, loadOp, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    newOperands.push_back(alloc);
    allocs.push_back(alloc);
    asyncLoads.emplace_back(loadOp, alloc);
    if (isLoadFromTensorPtr(loadOp))
      needsMbarrierPhase = true;
    else
      needsAsyncWait = true;
  }

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  // Create two new counters to index into the allocs.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value insertIdx = minusOne;
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  Value phase;
  if (needsMbarrierPhase) {
    phase = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    newOperands.push_back(phase);
  }
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;
  for (int i = 0; i < asyncLoads.size(); i++) {
    asyncLoads[i].alloc = newForOp.getBody()->getArgument(newOperandIndex + i);
  }
  insertIdx =
      newForOp.getBody()->getArgument(newOperandIndex + asyncLoads.size());
  extractIdx =
      newForOp.getBody()->getArgument(newOperandIndex + asyncLoads.size() + 1);

  // Create two counters for the insert and extract indices to avoid creating
  // long liverange.
  builder.setInsertionPoint(asyncLoads.front().loadOp);
  insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
  Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               insertIdx, numBuffersVal);
  insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);

  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  if (needsMbarrierPhase) {
    phase = newForOp.getBody()->getArgument(newOperandIndex +
                                            asyncLoads.size() + 2);
    Value oneI1 = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    Value nextPhase = builder.create<arith::XOrIOp>(loc, phase, oneI1);
    phase = builder.create<arith::SelectOp>(loc, cndExt, phase, nextPhase);
  }

  bool firstLoad = true;
  for (AsyncLoad &asyncLoad : asyncLoads) {
    createAsyncLoad(forOp, asyncLoad.loadOp, asyncLoad.alloc, insertIdx,
                    extractIdx, phase);
    firstLoad = false;
  }
  // Insert a waitOp after the first async copy. This does make the assumption
  // that the wait will be scheduled in a different stage that all the async
  // copy but we cannot guarantee that one wait is enough otherwise.
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::InsertSliceAsyncOp>(op)) {
      OpBuilder builder(op.getContext());
      builder.setInsertionPointAfter(&op);
      builder.create<ttg::AsyncWaitOp>(op.getLoc(), 0);
      break;
    }
  }
  SmallVector<Value> newYieldOperands = {insertIdx, extractIdx};
  if (needsMbarrierPhase)
    newYieldOperands.push_back(phase);
  // Patch the yield with the updated counters.
  appendToYield(forOp, newYieldOperands);

  return allocs;
}

std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp forOp, int numStages, bool prefetchExtract);
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred);

bool getGeneralSchedule(scf::ForOp &forOp, int numStages,
                        mlir::triton::PipeliningOption &options) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  SmallVector<tt::LoadOp> loads;
  bool hasMMAV3 = false;
  collectLoadOpsToPipeline(forOp, loads, hasMMAV3);
  if (loads.empty())
    return false;
  bool hasAsynCp = llvm::any_of(
      loads, [](tt::LoadOp load) { return !isLoadFromTensorPtr(load); });

  // 2. Convert the loads into async loads and create the allocs.
  SmallVector<Value> allocs = createAsynOps(forOp, loads, numStages, hasMMAV3);

  // 3. Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages, /*prefetchExtract=*/!hasMMAV3);

  // 4. Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = predicateOp;
  options.supportDynamicLoops = true;
  unsigned numLoadsInStage = (numStages - 2) * loads.size();
  options.annotateFn =
      [numLoadsInStage](Operation *op,
                        mlir::triton::PipeliningOption::PipelinerPart part,
                        unsigned iteration) {
        return setWaitNum(op, part, iteration, numLoadsInStage);
      };

  if (hasAsynCp) {
    // Insert a wait 0 after the loop
    OpBuilder builder(forOp);
    builder.setInsertionPointAfter(forOp);
    builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), 0);
    // Explicitly deallocate allocated tensors after the wait op
    for (auto alloc : allocs)
      builder.create<ttg::DeallocTensorOp>(forOp.getLoc(), alloc);
  }
  return true;
}

static void pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::triton::PipeliningOption options;
  if (!preCondition(forOp))
    return;

  bool foundSchedule = false;
  foundSchedule = preProcessLoopAndGetSchedule(forOp, numStages, options);

  // TODO: add more pipelines strategy.
  if (!foundSchedule)
    foundSchedule = getGeneralSchedule(forOp, numStages, options);
  if (!foundSchedule)
    return;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);

  if (succeeded(newForOp))
    mlir::triton::asyncLaunchDots(newForOp.value());
}

namespace {
struct PipelinePass : public TritonGPUPipelineBase<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int numStages, int numWarps, int numCTAs,
               int computeCapability) {
    this->numStages = numStages;
    this->numWarps = numWarps;
    this->numCTAs = numCTAs;
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    if (this->numStages <= 1)
      return;
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops) {
      pipelineLoop(forOp, numStages);
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass>
mlir::triton::gpu::createPipelinePass(int numStages, int numWarps, int numCTAs,
                                      int computeCapability) {
  return std::make_unique<PipelinePass>(numStages, numWarps, numCTAs,
                                        computeCapability);
}
