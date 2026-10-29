# =============================================================================
# PolyBench stencils - build configuration
#
# Build a kernel:            make jacobi-2d   (jacobi-1d / heat-3d / fdtd-2d)
# Build all:                 make all
# CPU vs GPU / schedule are chosen via the shared toggles in ../common.mk, e.g.:
#     make jacobi-2d USE_TARGET=1          # GPU target offload
#     make jacobi-2d USE_SYNC=1            # synchronous blocking baseline (-nb 1)
#     make jacobi-2d USE_TASKGRAPH=0       # plain tasks/target, no record/replay
#
# Problem size is compile-time (rebuild per size); override via PB_DEFS, e.g.:
#     make jacobi-2d PB_DEFS="-DPB_N=512 -DPB_TSTEPS=100"
# Tasks per loop is the runtime knob -nb <n> (default 32).
#
# Only the four fully-parallel stencils are in the harness. seidel-2d and adi
# (loop-carried wavefront / tridiagonal dependences) are intentionally excluded
# -- like Cholesky, they are not clean record/replay taskgraph examples. The
# upstream sources remain in ../polybench-stencils/ for reference.
# =============================================================================
include ../common.mk

CFLAGS += -Wall -Wextra
# Per-size recompile override (set by the evaluate.py harness).
PB_DEFS ?=
CFLAGS  += $(PB_DEFS)

HDR = pb_common.h ../tasking.h

KERNELS = jacobi-1d jacobi-2d heat-3d fdtd-2d

all: $(KERNELS)

jacobi-1d: jacobi-1d.c $(HDR)
	$(CC) $(CFLAGS) jacobi-1d.c -o jacobi-1d.x $(LDFLAGS)

jacobi-2d: jacobi-2d.c $(HDR)
	$(CC) $(CFLAGS) jacobi-2d.c -o jacobi-2d.x $(LDFLAGS)

heat-3d: heat-3d.c $(HDR)
	$(CC) $(CFLAGS) heat-3d.c -o heat-3d.x $(LDFLAGS)

fdtd-2d: fdtd-2d.c $(HDR)
	$(CC) $(CFLAGS) fdtd-2d.c -o fdtd-2d.x $(LDFLAGS)

test: jacobi-2d
	./jacobi-2d.x -i 11 -nb 8 -v

run: jacobi-2d
	./jacobi-2d.x -i 100 -nb 32

clean:
	rm -f *.x

.PHONY: all $(KERNELS) clean test run
