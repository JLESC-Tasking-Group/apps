#CC=clang
CC=xkcxx -DUSE_XKOMP=1

CFLAGS=
CFLAGS+=-fopenmp -fopenmp-version=60
CFLAGS+=-O3
#CFLAGS+=-ffast-math
CFLAGS+=-DNB_STEPS=10
#CFLAGS+=-DUSE_REPLAYABLE=1
CFLAGS+=-DUSE_TASKGRAPH=1
#CFLAGS+=-DBATCH_SIZE=8
#CFLAGS+=-DSEQUENCE_SIZE=128
#CFLAGS+=-DGRAN_TMP=8
#CFLAGS+=-DOC_SPLIT=6
#CFLAGS+=-DOC_BACK_SPLIT=10

LDFLAGS=
LDFLAGS+=-lm
#LDFLAGS+=-L$(POT)/lib -lpot_ompt

# OmpSs-2-specific
#CFLAGS+=-DUSE_OMPSS=1
#CFLAGS+=-fompss-2=libnodes
#LDFLAGS+=-lnuma

SRC=timer.c mpi_utils.c ompss-2_wrappers.c train_gpt2.c

# -----------------------------------------------------------------------------
# CPU build
# -----------------------------------------------------------------------------
cpu:
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRC) -o train_gpt2_cpu

# -----------------------------------------------------------------------------
# GPU build: OpenMP target offload to NVIDIA (sm_80), like ../lulesh-gpu.
# Selects the target backend in train_gpt2.c via -DUSE_TARGET=1. The host
# taskgraph/XKOMP runtime is not used here (USE_TASKGRAPH/USE_XKOMP default 0).
# -----------------------------------------------------------------------------

GPU_CFLAGS=
GPU_CFLAGS+=-DUSE_TARGET=1
GPU_CFLAGS+=-fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-offload-mandatory
GPU_CFLAGS+=-foffload-lto

# Auto-detect unique compute capabilities (e.g., "80", "86")
DETECTED_SMS := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sort -u | tr -d '.')
ifneq ($(DETECTED_SMS),)
    GPU_CFLAGS += $(foreach sm,$(DETECTED_SMS),--offload-arch=sm_$(sm))
else
    GPU_CFLAGS += --offload-arch=sm_80
endif

GPU_LDFLAGS=
GPU_LDFLAGS+=-lcudart

gpu:
	$(CC) $(CFLAGS) $(GPU_CFLAGS) $(SRC) -o train_gpt2_gpu $(LDFLAGS) $(GPU_LDFLAGS)

# -----------------------------------------------------------------------------
# Correctness test. test_gpt2.c does `#define TESTING` + `#include "train_gpt2.c"`
# (single translation unit) and checks logits / loss / gradients against the
# reference file. test_cpu (host backend) validates the shared algorithm.
# NOTE: test_gpu is provided for convenience but will report mismatches as-is:
# test_gpt2.c compares HOST copies of model.acts/model.grads/mean_loss, which are
# device-resident in the target backend (and it skips main()'s `target enter
# data`). Making it meaningful needs test_gpt2.c to map the buffers and
# `target update from` the compared tensors before each check.
# -----------------------------------------------------------------------------
TEST_SRC=timer.c mpi_utils.c ompss-2_wrappers.c test_gpt2.c

test_cpu:
	$(CC) $(CFLAGS) $(LDFLAGS) $(TEST_SRC) -o test_gpt2_cpu

test_gpu:
	$(CC) $(CFLAGS) $(GPU_CFLAGS) $(TEST_SRC) -o test_gpt2_gpu $(LDFLAGS) $(GPU_LDFLAGS)

clean:
	rm -rf train_gpt2_cpu train_gpt2_gpu test_gpt2_cpu test_gpt2_gpu train

.PHONY: cpu gpu test_cpu test_gpu clean
