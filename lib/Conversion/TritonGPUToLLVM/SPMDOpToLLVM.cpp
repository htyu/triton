#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  explicit GetProgramIdOpConversion(LLVMTypeConverter &typeConverter,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::GetProgramIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value programId = targetInfo.programId(rewriter, op->getLoc(),
                                           op->getParentOfType<ModuleOp>(),
                                           op.getAxisAsInt());
    rewriter.replaceOp(op, programId);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// TODO[goostavz]: GetThreadIdOp/GetClusterCTAIdOp is a temporary solution
// before async dialect is done. These concepts should appear in ttgpu
// level, and they are planned to be deprecated along with ttgpu.mbarrier_xxx
// ops.
struct GetThreadIdOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::GetThreadIdOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::GetThreadIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, getThreadId(rewriter, op->getLoc()));
    return success();
  }
};

struct GetClusterCTAIdOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::GetClusterCTAIdOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::GetClusterCTAIdOp>::ConvertOpToLLVMPattern;

  explicit GetClusterCTAIdOpConversion(LLVMTypeConverter &typeConverter,
                                       const TargetInfoBase &targetInfo,
                                       PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::nvidia_gpu::GetClusterCTAIdOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetClusterCTAIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, targetInfo.getClusterCTAId(rewriter, op->getLoc()));
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               const TargetInfoBase &targetInfo,
                                               PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<GetThreadIdOpConversion>(typeConverter, benefit);
  patterns.add<GetClusterCTAIdOpConversion>(typeConverter, targetInfo, benefit);
}
