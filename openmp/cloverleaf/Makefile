# =============================================================================
# CloverLeaf - tasking.h port build configuration.
#
# In-place port of the UoB-HPC CloverLeaf C++ code to the shared apps/openmp
# tasking.h backends. It reuses the model-independent driver/ and a new model
# directory src/tasking/ (cloned from src/omp-target, with context.h routing the
# conditional-offload macro through tasking.h). The per-timestep kernel chain is
# wrapped in a record/replay taskgraph in driver/hydro.cpp (guarded by
# CLOVER_TASKING).
#
# This Makefile bypasses the upstream CMake build (kept as CMakeLists.txt) and
# compiles a single-node (NO_MPI) binary with the shared toggles from ../common.mk:
#     make                           # CPU tasks + taskgraph (defaults)
#     make USE_TARGET=1              # GPU target offload
#     make USE_SYNC=1               # synchronous blocking baseline (use -nb 1)
#     make USE_TASKGRAPH=0          # plain tasks/target, no record/replay
#
# Run with a CloverLeaf input deck, e.g.:
#     ./cloverleaf --file InputDecks/clover_bm16_short.in --device 0
# (--device 0 selects the host device under the conditional-target model.)
# =============================================================================
include ../common.mk

# C++17 + single-node + the tasking model hooks. CloverLeaf needs C++17.
CFLAGS += -std=c++17
CFLAGS += -Idriver -Isrc/tasking
CFLAGS += -DNO_MPI -DCLOVER_TASKING

program = cloverleaf

DRIVER_SRC  = $(wildcard driver/*.cpp)
KERNEL_SRC  = $(wildcard src/tasking/*.cpp)
source      = $(DRIVER_SRC) $(KERNEL_SRC)
obj         = $(source:.cpp=.o)

HDR = $(wildcard driver/*.h) src/tasking/context.h ../tasking.h

all: $(program)

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@

# Correctness reference: the canonical short deck (test_problem 4). Large (3840^2);
# use a smaller deck for a quick smoke test.
test: $(program)
	./$(program) --file InputDecks/clover_bm2_short.in --device 0

run: $(program)
	./$(program) --file InputDecks/clover_bm16_short.in --device 0

clean:
	rm -f $(program) $(obj)

.PHONY: all clean test run
