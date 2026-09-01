# XKOMP
An experimental OpenMP runtime-library implementation built on top of the XKaapi runtime system, and an extended LLVM's Clang ABI.

# Installation
You must have an installation of
- [XKRT](https://gitlab.inria.fr/xkaapi/dev-v2)
- [LLVM(ANL)](https://github.com/anlsys/llvm-project) - and compile your openmp program with it

Example of LLVM build
```
cmake ../llvm -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="openmp;offload" -DLLVM_TARGETS_TO_BUILD="NVPTX;X86" -DCMAKE_INSTALL_PREFIX=~/install/llvm/debug
```

Example of XKOMP build
```
cmake -DCMAKE_BUILD_TYPE=Debug ../
```

Example of application build
```
xkcc main.c -o main
```

# Remarks
- Pass `-fopenmp-task-jit-abi=[pointers|packed]` to `xkcxx` so that the LLVM-IR/PTX of tasks/targets is kept after compile-time, and passed to the XKOMP runtime for JIT during taskgraph optimizations. With `pointers`, all arguments (firstprivate, shared, etc.) are stored in a `void ** args` array, referenced within the outlined region. With `packed`, they are packed in a `void * args, size_t size` compact structure.
- The `OMP_TASKGRAPH_OPT=opt1,opt2,...,optn` can be used to enable `taskgraph` optimizations. Available optimizations are: `transitive-reduction`, `reduce-node`, `batch`, `prog-fuse`, `jit`, `copy-normalize`, `copy-fuse`, `sequence`

# Bits of history
XKRT is a fork of XKaapi, that it extended with support for task dependencies over intersecting regions of memory (https://gitlab.inria.fr/xkaapi/dev-v2)
This repo is a prototype implementation of OpenMP, with partial support for taskgraphs (6.0) and extensions for dataflow, on top of XKRT.

A previous implementation of OpenMP on top of XKaapi is available here:
- https://gitlab.inria.fr/openmp/kaapi
- https://gitlab.inria.fr/openmp/libkomp
