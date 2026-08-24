# =============================================================================
# LULESH - build configuration
#
# Build:                     make            (or: make lulesh)
# CPU vs GPU / schedule are chosen via the shared toggles in ../common.mk, e.g.:
#     make USE_TARGET=1              # GPU target offload
#     make USE_SYNC=1               # synchronous blocking baseline (use -nb 1)
#     make USE_TASKGRAPH=0          # plain tasks/target, no record/replay
#
# Task granularity (tasks per loop on CPU / offload blocks on GPU) is a runtime
# knob: -nb <n> (default 32).
# =============================================================================
include ../common.mk

# Optional energy measurement backend for the FOM/watt figure (see energy.cc).
# Leave off for a plain build (power is then reported as 0):
#CFLAGS  += -DCUDA_ENERGY=1
#LDFLAGS += -lnvidia-ml
#CFLAGS  += -DROCM_ENERGY=1
#LDFLAGS += -lrocm_smi64
#CFLAGS  += -DZES_ENERGY=1
#LDFLAGS += -lze_loader

# ---- Sources --------------------------------------------------------------
program = lulesh

source = lulesh.cc \
        lulesh-viz.cc \
        lulesh-util.cc \
        lulesh-init.cc \
        energy.cc

obj = $(source:.cc=.o)

HDR = lulesh.h energy.h ../tasking.h

# =============================================================================
# Targets to build
# =============================================================================
all: $(program)

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cc $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) mpc-omp-task-trace-* report*

test: $(program)
	./$(program) -i 1 -s 32 -r 11 -b 1 -c 1

run: $(program)
	./$(program) -i 100 -s 128 -r 11 -b 1 -c 1

.PHONY: all clean test run
