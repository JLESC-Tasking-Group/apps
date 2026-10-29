# =============================================================================
# miniWeather - build configuration (tasking.h port)
#
# Build:                     make            (or: make miniweather)
# CPU vs GPU / schedule are chosen via the shared toggles in ../common.mk, e.g.:
#     make USE_TARGET=1              # GPU target offload
#     make USE_SYNC=1               # synchronous blocking baseline (use -nb 1)
#     make USE_TASKGRAPH=0          # plain tasks/target, no record/replay
#
# Problem size is compile-time (rebuild per size); override via MW_DEFS, e.g.:
#     make MW_DEFS="-D_NX=200 -D_NZ=100 -D_SIM_TIME=200 -D_OUT_FREQ=-1 -D_DATA_SPEC=DATA_SPEC_THERMAL"
# Tasks per loop is the runtime knob -nb <n> (default 32); -i <n> caps the step
# count for the harness.
#
# The port is single-node (MPI stripped) and does no NetCDF output (_OUT_FREQ<0),
# so no MPI / pnetcdf dependency is needed.
# =============================================================================
include ../common.mk

# Default problem (overridden by MW_DEFS from the harness / command line).
MW_DEFS ?= -D_NX=100 -D_NZ=50 -D_SIM_TIME=400 -D_OUT_FREQ=-1 -D_DATA_SPEC=DATA_SPEC_THERMAL
CFLAGS  += $(MW_DEFS)

program = miniweather
source  = tasking/miniweather.cpp
HDR     = ../tasking.h

all: $(program)

miniweather: $(source) $(HDR)
	$(CC) $(CFLAGS) $(source) -o $(program) $(LDFLAGS)

# Correctness: reference conservation test (NX=100 NZ=50 SIM_TIME=400).
test: miniweather
	./$(program) -nb 8

run: miniweather
	./$(program) -nb 32

clean:
	rm -f $(program) *.o

.PHONY: all miniweather clean test run
