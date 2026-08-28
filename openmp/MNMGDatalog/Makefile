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
#   TC_WRITE=1 writes <input>_<version>_tc.bin ; TC_CSV=<f> writes the metric row.
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

# data_7035.bin -> TC=146120 in 64 iterations (see the reference README).
test: tc
	./$(program) $(DATA) $(MULT) 1

run: tc
	./$(program) $(DATA) $(MULT) $(REPEATS)

clean:
	rm -f $(program) *.o

.PHONY: all tc test run clean
