# =============================================================================
# sw4lite - tasking.h port (self-contained proxy) build configuration
#
# This Makefile builds the apps/openmp tasking.h port (tasking/sw4lite.cpp), a
# single-source proxy of sw4lite's numerical core (4th-order elastic stencil +
# predictor/corrector). The upstream sw4lite build (full MPI/BLAS code) is kept
# as Makefile.upstream / Makefile.cuda.
#
# Build:                     make            (or: make sw4lite)
# CPU vs GPU / schedule via the shared toggles in ../common.mk, e.g.:
#     make USE_TARGET=1          # GPU target offload
#     make USE_SYNC=1            # synchronous baseline (-nb 1)
#     make USE_TASKGRAPH=0       # plain tasks/target
#
# Problem size is compile-time (rebuild per size); override via SW4_DEFS, e.g.:
#     make SW4_DEFS="-DSW4_N=96 -DSW4_NZ=96"
# Tasks per loop is the runtime knob -nb <n> (default 32); -i <n> sets #steps.
# =============================================================================
include ../common.mk

SW4_DEFS ?= -DSW4_N=64 -DSW4_NZ=64
CFLAGS   += $(SW4_DEFS)

program = sw4lite
source  = tasking/sw4lite.cpp
HDR     = ../tasking.h

all: $(program)

sw4lite: $(source) $(HDR)
	$(CC) $(CFLAGS) $(source) -o $(program) $(LDFLAGS)

test: sw4lite
	./$(program) -i 20 -nb 8 -v

run: sw4lite
	./$(program) -i 100 -nb 32

clean:
	rm -f $(program) *.o

.PHONY: all sw4lite clean test run
