# =============================================================================
# MNMGDatalog TC - build configuration
#
# One source (tc.cpp) expresses every backend/schedule via the shared toggles in
# ../common.mk. The two CUDA "versions" of tc_benchmark map to toggles:
#     make USE_TASKGRAPH=0          # v1_baseline  : plain target tasks, host loop
#     make USE_TASKGRAPH=1          # v2_cudagraph : record-once / replay (default)
# and the backend / schedule are orthogonal:
#     make USE_TARGET=1             # GPU target offload (default is CPU tasks)
#     make USE_SYNC=1               # synchronous blocking baseline
#
# Run: ./tc.x <data.bin> [capacity_mult] [frontier_slots]
#   capacity_mult is OPTIONAL and defaults to auto: the closure size is
#   discovered by an untimed probe solve and cached in tc_capacity.cache, so
#   `./tc.x <data.bin>` just works. Pass a value to reproduce the CUDA
#   reference's next_pow2(edges*mult) sizing instead.
#   TC_CACHE=<f>   capacity cache file (default ./tc_capacity.cache)
#   TC_WARMUP=<n>  untimed, ungraphed warm-up rounds before round 0 (default 3)
#   TC_WORKERS=<n> grid-stride worker count of the GPU fixpoint kernels
#   TC_VERIFY=1    read the edge table back and check the fill / build on device
#   TC_TRACE=1     one flushed stderr line per kernel dispatch (use with USE_SYNC=1
#                  to name a faulting kernel; see README "Debugging")
#   TC_WRITE=1 writes <input>_<version>_tc.bin ; TC_CSV=<f> writes the metric row.
#
# Debug builds:
#     make USE_TARGET=1 USE_SYNC=1                # blocking launches, no tasking
#     make USE_TARGET=1 TC_MAPPED_MEM=1           # krylov/lulesh memory model
#     make repro USE_TARGET=1                     # standalone reproducer
#
# The fixpoint runs ONCE and each round is timed: round 0 (record) / round 1 (1st
# replay) / rounds 2..N-1 (avg, stddev), the same shape the Krylov drivers use for
# iterations, plus the MNMGDatalog paper's end-to-end total and its per-phase
# breakdown (file IO, H2D, setup, compute, D2H). The round count is determined by
# the dataset, not by a command-line knob.
# =============================================================================
include ../common.mk

CFLAGS += -Wall -Wextra

# Memory model A/B (see README "Debugging"). 0: device-only buffers via
# omp_target_alloc + is_device_ptr (default, mirrors the CUDA reference).
# 1: pinned host buffers + map(present:), i.e. what krylov/lulesh/HPCCG do.
TC_MAPPED_MEM ?= 0
CFLAGS += -DTC_MAPPED_MEM=$(TC_MAPPED_MEM)

# Datasets live in the reference checkout inside this app directory.
DATA_DIR ?= MNMGDatalog-reference/data
DATA     ?= $(DATA_DIR)/data_7035.bin
MULT     ?=            # empty = auto-size (see tc_resolve_capacity in tc.cpp)

program = tc.x

all: tc

tc: tc.cpp ../tasking.h ../alloc.h ../alloc.c
	$(CC) $(CFLAGS) tc.cpp ../alloc.c -o $(program) $(LDFLAGS)

# data_7035.bin -> TC=146120 in 64 rounds (see the reference README).
test run: tc
	./$(program) $(DATA) $(MULT)

# Standalone minimal reproducer: omp_target_alloc + is_device_ptr + device
# atomics in k_expand's grid-stride shape, with every dimension on argv. Builds
# and runs without tasking.h / alloc.h / xkomp. See the header of tc_repro.cpp.
repro: tc_repro.cpp
	$(CC) $(CFLAGS) tc_repro.cpp -o tc_repro.x $(LDFLAGS)

# Same reproducer against the krylov/lulesh memory model.
repro-mapped: tc_repro.cpp
	$(CC) $(CFLAGS) -DREPRO_MAPPED tc_repro.cpp -o tc_repro_mapped.x $(LDFLAGS)

clean:
	rm -f $(program) tc_repro.x tc_repro_mapped.x *.o

.PHONY: all tc test run repro repro-mapped clean
