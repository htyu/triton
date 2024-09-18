// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRITONNVIDIAGPU_OPS
#define TRITONNVIDIAGPU_OPS

include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUTypes.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonTypeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTNG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonNvidiaGPU_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

// --------------------------------------------------------------------------------------------------
// MBarrier related Ops:
// 1, These mbarrier commands are currently not needed, and not taken into consideration:
//    (1), mbarrier.expect_tx
//    (2), mbarrier.arrive_drop
//    (3), mbarrier.complete_tx
//    (4), mbarrier.inval
//
// 2, The mbarriers is supported to be created in vector, and accessed in separate via tensor.extract.
//    The mbarriers created in vector will have counters initialized in the same configuration. A
//    typical example to demonstrate this:
//
//    %1 = triton_nvidia_gpu.alloc_mbarrier { count = 1 } : tensor<4x!tt.ptr<i64>>
//    scf.for %iv = %lb to %ub step %step iter_args() -> () {
//      %buffer_id = arith.remi %iv, %c4 : i32
//      %2 = triton_nvidia_gpu.extract_mbarrier %1[%buffer_id] : tensor<4xi64>, i32 -> !tt.ptr<i64>
//      triton_nvidia_gpu.mbarrier_arrive %2 {expectTx = 2048} : !tt.ptr<i64> -> ()
//    }
//    ...
//    scf.for %iv = %lb to %ub step %step iter_args() -> () {
//      %buffer_id = arith.remi %iv, %c4 : i32
//      %2 = triton_nvidia_gpu.extract_mbarrier %1[%buffer_id] : tensor<4xi64>, i32 -> !tt.ptr<i64>
//      triton_nvidia_gpu.mbarrier_wait %2, %c0 : !tt.ptr<i64>, i1 -> ()
//    }

def TTNG_AllocMBarrierOp : TTNG_Op<"alloc_mbarrier", [MemoryEffects<[MemAlloc<SharedMemory>]>]> {
  let summary = "allocate a vector of mbarriers";

  let description = [{
    Allocate and initialize a vector of mbarriers. The size of the vector is implied in the returned type.
    Each mbarrier is initialized as:
    1, the current phase initialized to 0.
    2, the expected arrival count initialized to 'count'.
    3, the pending arrival count initialized to 'count'.
    4, the tx-count initialized to 0.

    Example:

    case a. when created in vector:
    %1 = triton_nvidia_gpu.alloc_mbarrier { count = 1 } : tensor<4xi64>

    case b. when created in scalar:
    %1 = triton_nvidia_gpu.alloc_mbarrier { count = 1 } : !tt.ptr<i64>

  }];

  let assemblyFormat = [{attr-dict `:` type($result)}];

  let arguments = (ins I32Attr:$count);

  let results = (outs AnyTypeOf<[TT_Ptr, I64Tensor]>:$result);
}

def TTNG_ExtractMBarrierOp : TTNG_Op<"extract_mbarrier", [Pure]> {
  let summary = "extract a mbarrier from a vector of mbarriers";

  let description = [{
    Extract a mbarrier from a vector of mbarriers

    Example:

    %1 = triton_nvidia_gpu.extract_mbarrier %mbarriers[%idx] : tensor<4xi64>, index -> !tt.ptr<i64>

  }];

  let assemblyFormat = "$tensor `[` $index `]` attr-dict `:` type($tensor) `,` type($index) `->` type($result)";

  let arguments = (ins I64Tensor:$tensor, I32:$index);

  let results = (outs TT_Ptr:$result);
}

def TTNG_MBarrierWaitOp : TTNG_Op<"mbarrier_wait", [MemoryEffects<[MemRead<SharedMemory>, MemWrite<SharedMemory>]>]> {
  let summary = "mbarrier wait";

  let description = [{
    This operation defining the waiting action for a mbarrier.
    The subsequent operations should not execute until this operation completes waiting.

    Example:

    triton_nvidia_gpu.mbarrier_wait %0, %1 : !tt.ptr<i64>

  }];

  let arguments = (ins TT_Ptr:$mbarrier, I1: $phase);

  let assemblyFormat = "$mbarrier `,` $phase attr-dict `:` type($mbarrier)";
}

def TTNG_MBarrierArriveOp : TTNG_Op<"mbarrier_arrive", [AttrSizedOperandSegments,
                                                      MemoryEffects<[MemWrite<SharedMemory>]>]> {
  let summary = "mbarrier arrive";

  let description = [{
    This operation defining the arriving action for a mbarrier.
    txCount:
        An optional attribute that set tx-count. This Op will be lowered into
        mbarrier.arrive.expect_tx if the optional attribute exist.
    trackAsyncOp:
        If true, this op will be lowered into cp.async.mbarrier.arrive.noinc.
    pred:
        Only perform arrive action when pred is true.
    remoteCtaId:
        if set, perform an remote arrive action.

    Example:

    triton_nvidia_gpu.mbarrier_arrive %0 {trackAsyncOp = false} : !tt.ptr<i64>

  }];

  let arguments = (ins TT_Ptr:$mbarrier,
                       Optional<I1>:$pred,
                       Optional<I32>:$remoteCtaId,
                       I1Attr: $trackAsyncOp,
                       DefaultValuedAttr<I32Attr, "0">: $txCount
                  );

  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

// TODO[goostavz]: ThreadId & ClusterCTAId should not be exposed to
//                 ttgpu level. Remove them when async dialect is ready.
def TTNG_GetThreadIdOp : TTNG_Op<"get_thread_id", [Pure]> {
  let description = [{
    Returns the one dimensional threadId.
  }];

  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNG_GetClusterCTAIdOp : TTNG_Op<"get_cluster_cta_id", [Pure]> {
  let description = [{
    Returns the one dimensional cluster_cta_id.
  }];

  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNG_GetCanonicalWarpIdOp : TTNG_Op<"get_canonical_warp_id", [Pure]> {
  let description = [{
    Returns the one dimensional warpId when it's used for producing warp uniform values.
  }];

  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}


def TTNG_FenceAsyncSharedOp : TTNG_Op<"fence_async_shared"> {
  let arguments = (ins BoolAttr:$bCluster);

  let summary = "fence proxy async";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 90;
    }
  }];
}

def TTNG_ClusterArriveOp : TTNG_Op<"cluster_arrive", []> {
  let arguments = (ins I1Attr:$relaxed);
  let assemblyFormat = "attr-dict";
}

def TTNG_ClusterWaitOp : TTNG_Op<"cluster_wait", []> {
  let assemblyFormat = "attr-dict";
}

//
// WarpGroupDot Op
//
def TTNG_WarpGroupDotOp : TTNG_Op<"warp_group_dot", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                                     DotLike,
                                                     TypesMatchWith<"result's type matches accumulator's type",
                                                                     "d", "c", "$_self">]> {
    let summary = "warp group dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c. For docs on InputPrecisionAttr, see TT_DotOp
    }];

    let arguments = (ins TT_TensorOrMemDesc:$a,
                         TT_TensorOrMemDesc:$b,
                         TT_FpIntTensor:$c,
                         DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
                         DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc,
                         DefaultValuedAttr<BoolAttr, "false">:$isAsync);

    let results = (outs TT_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c attr-dict `:` type($a) `*` type($b) `->` type($d)";
}

def TTNG_WarpGroupDotWaitOp : TTNG_Op<"warp_group_dot_wait", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                                              AllTypesMatch<["inputs", "outputs"]>]> {
  let summary = "warp group dot wait";
  let arguments = (ins Variadic<TT_TensorOrMemDesc>:$inputs, I32Attr:$pendings);
  let results = (outs Variadic<TT_TensorOrMemDesc>:$outputs);
  let description = [{
    Waits until there are $pendings or fewer outstanding async dot operations.

    $inputs must be the tensors corresponding to the async dot ops that we're
    waiting on.  For example, if there are N pending async dot ops and we call
    `warp_group_dot_wait 1`, then $inputs must be the result of the first dot op.
  }];

  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

def TTNG_InitBarrierOp : TTNG_Op<"init_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Initialize a barrier in the given shared memory allocation.";

    let description = [{
        Initializes a shared memory allocation with mbarrier information.
        `alloc` is a descriptor to the shared memory allocation. `count` is the
        number of arrives expected by the barrier.

        This lowers to PTX mbarrier.init.shared::cta.b64.
    }];

    let hasVerifier = 1;
    let arguments = (ins TT_MemDescType:$alloc,
                         I32Attr:$count);
    let assemblyFormat = "$alloc `,` $count attr-dict `:` type($alloc)";
}

def TTNG_InvalBarrierOp : TTNG_Op<"inval_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Invalidate a barrier allocation.";

    let description = [{
      Invalidate a barrier allocation so that it can be re-used. According to PTX
      spec this has to be done before any re-use of the memory used by mbarrier.

      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
    }];

    let hasVerifier = 1;
    let arguments = (ins TT_MemDescType:$alloc);
    let assemblyFormat = "$alloc attr-dict `:` type($alloc)";
}

def TTNG_BarrierExpectOp : TTNG_Op<"barrier_expect", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Signal a barrier of an expected number of bytes to be copied.";

  let description = [{
    This signal the barrier that `size` bytes are expected to be copied. The
    associated barrier wait will block until the expected number of bytes are copied.
  }];

  let hasVerifier = 1;
  let arguments = (
    ins TT_MemDescType:$alloc,
    I32Attr:$size,
    I1:$pred
  );

  let assemblyFormat = [{
    $alloc `,` $size `,` $pred  attr-dict `:` type($alloc)
  }];
}

def TTNG_WaitBarrierOp : TTNG_Op<"wait_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "wait until the mbarrier phase completes.";

    let description = [{
      Blocks the program progress until the mbarrier object in `alloc` completes
      its current phase.

      This lowers a waitloop using PTX instruction
      mbarrier.try_wait.parity.shared.b64.

      The barrier behavior is described here:
      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms
    }];

    let hasVerifier = 1;
    let arguments = (ins TT_MemDescType:$alloc,
                         I32:$phase);
    let assemblyFormat = "$alloc `,` $phase attr-dict `:` type($alloc)";
}


def TTNG_AsyncTMACopyGlobalToLocalOp : TTNG_Op<"async_tma_copy_global_to_local", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy data based on descriptor from global memory to local memory asynchronously";

  let description = [{
    This operation copies data from global memory to local memory
    asynchronously.  This is analogue to tt.load except the data are copied to
    local memory pointed by the memory descriptor instread of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc_ptr`.
  }];

  let hasVerifier = 1;
  let arguments = (
    ins TT_PtrType:$desc_ptr,
    Variadic<I32>:$coord,
    TT_MemDescType:$barrier,
    TT_MemDescType:$result,
    I1:$pred,
    DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile
  );

  let assemblyFormat = [{
    $desc_ptr `[` $coord `]` $result `,` $barrier `,` $pred
    oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
    attr-dict `:` type($desc_ptr) `,` type($barrier) `->` type($result)
  }];
}

def TTNG_AsyncTMACopyLocalToGlobalOp : TTNG_Op<"async_tma_copy_local_to_global", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy data based on descriptor from local memory to global memory asynchronously";

  let description = [{
    This operation copies data from local memory to global memory
    asynchronously.  This is analogue to tt.store except the data are copied from
    local memory pointed by the memory descriptor instread of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc_ptr`.
  }];

  let arguments = (
    ins TT_PtrType:$desc_ptr,
    Variadic<I32>:$coord,
    TT_MemDescType:$src);

  let assemblyFormat = [{
    $desc_ptr `[` $coord `]` $src
    attr-dict `:` type($desc_ptr) `,` type($src)
  }];
}

def TTNG_TMAStoreWait : TTNG_Op<"async_tma_store_wait"> {
  let summary = "wait until all the inputs are read.";
  let arguments = (ins I32Attr:$pendings);
  let description = [{
    Wait until all the read operations are done from the associated store operations.
    This is needed before the shared memory can be written to.
  }];

  let assemblyFormat = "attr-dict";
}

def TTNG_GetAgentIdOp : TTNG_Op<"get_agent_id", [Pure]> {
  let results = (outs I32:$result);

  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = "attr-dict `:` type($result)";
}

//
// Token
//

def TTNG_CreateTokenOp : TTNG_Op<"create_token"> {
  let results = (outs TensorOf<[TTNG_TokenType]>:$result);

  let arguments = (ins I32Attr:$num);

  let builders = [OpBuilder<(ins "uint32_t":$num)>];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNG_ProducerAcquireOp : TTNG_Op<"producer_acquire"> {
  let arguments = (ins TensorOf<[TTNG_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

def TTNG_ProducerCommitOp : TTNG_Op<"producer_commit"> {
  let arguments = (ins TensorOf<[TTNG_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

def TTNG_ConsumerWaitOp : TTNG_Op<"consumer_wait"> {
  let arguments = (ins TensorOf<[TTNG_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

def TTNG_ConsumerReleaseOp : TTNG_Op<"consumer_release"> {
  let arguments = (ins TensorOf<[TTNG_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

//
// Mutex
//

def TTNG_GetMutexRoleIdOp : TTNG_Op<"get_mutex_role_id"> {
  let results = (outs I32:$result);

  let arguments = (ins I32Attr:$num);

  let builders = [OpBuilder<(ins "uint32_t":$num)>];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNG_CreateMutexOp : TTNG_Op<"create_mutex"> {
  let results = (outs TTNG_MutexType:$result);

  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNG_LockOp : TTNG_Op<"lock"> {
  let arguments = (ins TTNG_MutexType:$mutex);

  let assemblyFormat = "$mutex attr-dict `:` type(operands)";
}

def TTNG_UnlockOp : TTNG_Op<"unlock"> {
  let arguments = (ins TTNG_MutexType:$mutex);

  let assemblyFormat = "$mutex attr-dict `:` type(operands)";
}

def TTNG_RegAllocOp : TTNG_Op<"reg_alloc", []> {
  let summary = "register allocation";

  let arguments = (ins I32Attr: $regCount);

  let assemblyFormat = "$regCount attr-dict";
}

def TTNG_RegDeallocOp : TTNG_Op<"reg_dealloc", []> {
  let summary = "register deallocation";

  let arguments = (ins I32Attr: $regCount);

  let assemblyFormat = "$regCount attr-dict";
}

#endif
