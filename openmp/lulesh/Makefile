CLANG_DIR=$(shell which clang | rev | cut -d '/' -f 3- | rev)
GCC_DIR=$(shell which g++ | rev | cut -d '/' -f 3- | rev)
#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
#CC        = clang++
#CC        = nvc++

#===============================================================================
# Program name & source code list
#===============================================================================

program = lulesh

source = lulesh.cc \
        lulesh-viz.cc \
        lulesh-util.cc \
        lulesh-init.cc \
		energy.cc

obj = $(source:.cc=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
LDFLAGS=
CFLAGS=-O3
#CFLAGS=-O0 -g
CFLAGS+=-fopenmp -fopenmp-offload-mandatory
CFLAGS+=-foffload-lto
#CFLAGS+=-DALLOC_OMP=1

# XKOMP flags
# CFLAGS+=-DUSE_XKOMP_TASKGRAPH=1 -std=c++20
# CFLAGS+=-L$(XKOMP_HOME)/lib -lomp -Wl,-rpath,.
#LDFLAGS+=-L$(XKOMP_HOME)/lib -lomp -Wl,-rpath,.

# v100/a100/h100
CC=clang++
CFLAGS+=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80
CFLAGS+=-DALLOC_CUDA=1
CFLAGS+=-DCUDA_ENERGY=0
#LDFLAGS+=-lnvidia-ml
#LDFLAGS+=-lcuda
LDFLAGS+=-lcudart

# mi250, mi300x
# CC=clang++
# CFLAGS+=-I $(ROCM_PATH)/include
# CFLAGS+=-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx90a,gfx942
# CFLAGS+=-DALLOC_HIP=1
# CFLAGS+=-DROCM_ENERGY=1
# LDFLAGS+=-L $(ROCM_PATH)/lib/ -lamdhip64
# LDFLAGS+=-L $(ROCM_PATH)/lib/ -lrocm_smi64

#CFLAGS+=-fopenmp-cuda-mode -fopenmp-assume-no-thread-state -fopenmp-assume-no-nested-parallelism

# pvc
# CFLAGS+=-fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device pvc"
# CC = icpx
# CFLAGS+=-DZES_ENERGY=1
# CFLAGS+=-I /usr/include/level_zero
# LDFLAGS+=-lze_loader

# LDFLAGS=-L$(GCC_DIR)/lib64 -lstdc++
#LDFLAGS+=-L$(CLANG_DIR)/lib -lomptarget

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cc lulesh.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) mpc-omp-task-trace-* report*

test: $(program)
	./$(program) -i 1 -s 32 -r 11 -b 1 -c 1

run: $(program)
	./$(program) -i 100 -s 128 -r 11 -b 1 -c 1

