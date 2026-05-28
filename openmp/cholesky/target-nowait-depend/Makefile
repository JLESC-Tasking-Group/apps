SRC=main.cc
BIN=main

CFLAGS=-Wall -Wextra -fopenmp -O3
LDFLAGS=-lm

LLVM_CXX=clang++
LLVM_CFLAGS=-Wall -Wextra -fopenmp -O3
LLVM_CFLAGS+=-g
LLVM_CFLAGS+=-fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target
LLVM_CFLAGS+=-march=sm_80
LLVM_LDFLAGS=
LLVM_LDFLAGS+=-lcudart
#LLVM_LDFLAGS+=-L$(CLANG_DIR)/lib -lomptarget
#LLVM_LDFLAGS+=-L/ccc/products/nvhpc-22.2/system/default/Linux_x86_64/22.2/cuda/lib64 -lcudart

all: llvm

llvm:
	$(LLVM_CXX) $(CFLAGS) $(LLVM_CFLAGS) $(LLVM_LDFLAGS) $(LDFLAGS) $(SRC) -o $(BIN)-llvm

clean:
	rm -f $(BIN)-llvm

