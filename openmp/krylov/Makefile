# =============================================================================
# Krylov solvers - build configuration
#
# Build a solver:            make cg          (cr / bicgstab / minres / gmres)
# Build all solvers:         make all
# CPU vs GPU / schedule are chosen via the shared toggles in ../common.mk, e.g.:
#     make cg USE_TARGET=1           # GPU target offload
#     make cg USE_SYNC=1             # synchronous blocking baseline
#     make cg USE_TASKGRAPH=0        # plain tasks, no record/replay
# =============================================================================
include ../common.mk

# ---- App-specific flags ---------------------------------------------------
CFLAGS += -Icommon
CFLAGS += -Wall -Wextra
# Compile-time problem/solver defaults (all also settable at runtime, see -h):
CFLAGS += -DGRID_N=64
CFLAGS += -DMAX_ITER=50

# ---- Sources --------------------------------------------------------------
COMMON_SRC = common/spmat.cpp common/kalloc.cpp common/kernels.cpp common/driver.cpp
COMMON_HDR = common/spmat.h common/kalloc.h common/kernels.h ../tasking.h common/driver.h

SOLVERS = cg cr bicgstab minres gmres

# =============================================================================
# Solvers (one target each). Add a new solver <name> as:
#   <name>: $(COMMON_SRC) <name>/<name>.cpp $(COMMON_HDR)
#           $(CC) $(CFLAGS) $(COMMON_SRC) <name>/<name>.cpp -o <name>.x $(LDFLAGS)
# =============================================================================
all: $(SOLVERS)

cg: $(COMMON_SRC) cg/cg.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) cg/cg.cpp -o cg.x $(LDFLAGS)

cr: $(COMMON_SRC) cr/cr.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) cr/cr.cpp -o cr.x $(LDFLAGS)

bicgstab: $(COMMON_SRC) bicgstab/bicgstab.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) bicgstab/bicgstab.cpp -o bicgstab.x $(LDFLAGS)

minres: $(COMMON_SRC) minres/minres.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) minres/minres.cpp -o minres.x $(LDFLAGS)

gmres: $(COMMON_SRC) gmres/gmres.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) gmres/gmres.cpp -o gmres.x $(LDFLAGS)

clean:
	rm -f *.x

.PHONY: all $(SOLVERS) clean
