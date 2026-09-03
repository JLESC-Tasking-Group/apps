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
# Run: ./tc.x <data.bin> [capacity_mult] [repeats] [frontier_slots]
#   TC_WARMUP=<n>  untimed warm-up solves before the timed solve 0 (default 1)
#   TC_WORKERS=<n> grid-stride worker count of the GPU fixpoint kernels
#   TC_WRITE=1 writes <input>_<version>_tc.bin ; TC_CSV=<f> writes the metric row.
#
# Reports solve 0 (record) / solve 1 (1st replay) / solves 2..R-1 (avg, stddev)
# like the Krylov drivers, plus the MNMGDatalog paper's end-to-end total and its
# per-phase breakdown (file IO, H2D, setup, graph build, compute, D2H).
# =============================================================================
include ../common.mk

CFLAGS += -Wall -Wextra

# Datasets live in the reference checkout inside this app directory.
DATA_DIR ?= MNMGDatalog-reference/data
DATA     ?= $(DATA_DIR)/data_7035.bin
MULT     ?= 64
REPEATS  ?= 5

program = tc.x

all: tc

tc: tc.cpp ../tasking.h ../alloc.h ../alloc.c
	$(CC) $(CFLAGS) tc.cpp ../alloc.c -o $(program) $(LDFLAGS)

# data_7035.bin -> TC=146120 in 64 rounds (see the reference README). 3 timed
# solves is the minimum that reports record / 1st replay / steady state.
test: tc
	./$(program) $(DATA) $(MULT) 3

run: tc
	./$(program) $(DATA) $(MULT) $(REPEATS)

clean:
	rm -f $(program) *.o

.PHONY: all tc test run clean
