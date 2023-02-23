import argparse
import sys

import triton
import triton._C.libtriton.triton as libtriton
import triton.compiler as tc

if __name__ == '__main__':

    # valid source and target formats
    VALID_FORMATS = ['triton-ir', 'triton-gpu-ir', 'llvm-ir', 'ptx', 'amdgcn']

    # set up the argument parser
    # TODO: conditional requirements
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help="Source file to compile")
    parser.add_argument('--target', required=True,
                        help="Target format, one of: " + ', '.join(VALID_FORMATS))
    parser.add_argument('--sm', type=int, help="Compute capability to compile for")
    parser.add_argument('--ptx-version', type=int, help="PTX version to compile for")
    parser.add_argument('--gfx', type=str, help="AMDGPU target to compile for")
    parser.add_argument('--triple', type=str, help="target triple, for example: amdgcn-amd-amdhsa")
    parser.add_argument('--features', type=str, help="target features, for example: +sramecc,-xnack")

    # parse the args
    args = parser.parse_args()

    # TODO: clean-up and re-use triton.compiler primitive functions
    # check for validity of format arguments
    if args.target not in VALID_FORMATS:
        print("Invalid target format: " + args.target)
        sys.exit(0)

    # parse source file to MLIR module
    context = libtriton.ir.context()
    module = libtriton.ir.parse_mlir_module(args.src, context)
    module.context = context

    # optimizer triton-ir
    module = triton.compiler.optimize_triton_ir(module)
    if args.target == 'triton-ir':
        print(module.str())
        sys.exit(0)

    # llvm-ir -> amdgcn
    if args.target == 'amdgcn':
        # auto detect available architecture and features
        # if nothing detected, set with default values
        arch_details = tc.get_amdgpu_arch_fulldetails()
        if not arch_details:
            arch_name = ""
            arch_triple = "amdgcn-amd-amdhsa"
            arch_features = ""
        else:
            arch_triple, arch_name, arch_features = arch_details

        # stop processing if architecture name is not automatically detected and is not set manually
        if not args.gfx and not arch_name:
            raise argparse.ArgumentError(None, "Must specify --gfx for AMDGCN compilation")

        # rewrite default and automatically detected values with manually provided data
        if args.gfx:
            arch_name = args.gfx
        if args.triple:
            arch_triple = args.triple
        if args.features:
            arch_features = args.features

        # triton-ir -> triton-gpu-ir
        # use compute_capability == 80
        module = triton.compiler.ttir_to_ttgir(module, num_warps=4, num_stages=3, compute_capability=80)
        # triton-gpu-ir -> llvm-ir
        # use compute_capability == 80
        module = triton.compiler.ttgir_to_llir(module, extern_libs=None, compute_capability=80)
        # llvm-ir -> amdgcn asm, hsaco binary
        module, hsaco_path = triton.compiler.llir_to_amdgcn_and_hsaco(module, arch_name, arch_triple, arch_features)

        print(hsaco_path)
        print(module)
        sys.exit(0)

    if not args.sm:
        raise argparse.ArgumentError(None, "Must specify --sm for PTX compilation")

    # triton-ir -> triton-gpu-ir
    module = triton.compiler.ttir_to_ttgir(module, num_warps=4)
    module = triton.compiler.optimize_ttgir(module, num_stages=3, compute_capability=args.sm)
    if args.target == 'triton-gpu-ir':
        print(module.str())
        sys.exit(0)

    # triton-gpu-ir -> llvm-ir
    module = triton.compiler.ttgir_to_llir(module, extern_libs=None, compute_capability=args.sm)
    if args.target == 'llvm-ir':
        print(module)
        sys.exit(0)

    # llvm-ir -> ptx
    if args.target == 'ptx':
        if not args.ptx_version:
            raise argparse.ArgumentError(None, "Must specify --ptx-version for PTX compilation")
        module = triton.compiler.llir_to_ptx(module, compute_capability=args.sm, ptx_version=args.ptx_version)

    print(module)
