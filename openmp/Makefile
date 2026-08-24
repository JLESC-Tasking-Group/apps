# =============================================================================
# apps/openmp - top-level build dispatcher.
#
# Thin wrapper that delegates to each app's own Makefile (which shares
# common.mk). Backend/schedule toggles propagate to the sub-makes, e.g.:
#     make                     # build everything, CPU tasks + taskgraph
#     make krylov              # just the krylov solvers
#     make USE_TARGET=1        # GPU target offload for all apps
#     make USE_SYNC=1          # synchronous blocking baseline
#
# Only build targets are exposed here; use each app's Makefile directly for its
# app-specific targets (e.g. `make -C lulesh run`, `make -C llm.c test`).
# Cholesky is intentionally not part of this harness (not a taskgraph example);
# build it standalone via `make -C cholesky`.
# =============================================================================

APPS = krylov lulesh llmc

all: $(APPS)

krylov:
	$(MAKE) -C krylov all

lulesh:
	$(MAKE) -C lulesh all

llmc:
	$(MAKE) -C llm.c all

clean:
	$(MAKE) -C krylov clean
	$(MAKE) -C lulesh clean
	$(MAKE) -C llm.c  clean

.PHONY: all $(APPS) clean
