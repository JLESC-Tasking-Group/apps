# =============================================================================
# llm.c (GPT-2 training) - build configuration
#
# Build:                     make            (or: make train_gpt2)
# CPU vs GPU / schedule are chosen via the shared toggles in ../common.mk, e.g.:
#     make USE_TARGET=1              # GPU target offload
#     make USE_SYNC=1               # synchronous blocking baseline
#     make USE_TASKGRAPH=0          # plain tasks/target, no record/replay
#
# The problem size and step count are COMPILE-TIME macros. Override them (e.g.
# from the evaluation harness) via LLMC_DEFS, e.g.:
#     make LLMC_DEFS='-DNB_STEPS=10 -DSEQUENCE_SIZE=256 -DBATCH_SIZE=4'
# =============================================================================
include ../common.mk

# ---- App-specific flags ---------------------------------------------------
CFLAGS += -DNB_STEPS=10
# Size / granularity overrides injected by the sweep (SEQUENCE_SIZE, BATCH_SIZE,
# NB_STEPS, GRAN_TMP, OC_SPLIT, ...). Empty by default.
LLMC_DEFS ?=
CFLAGS += $(LLMC_DEFS)

# OmpSs-2 host backend (mutually exclusive with USE_TARGET):
#CFLAGS  += -DUSE_OMPSS=1 -fompss-2=libnodes
#LDFLAGS += -lnuma

SRC = timer.c mpi_utils.c ompss-2_wrappers.c train_gpt2.c
TEST_SRC = timer.c mpi_utils.c ompss-2_wrappers.c test_gpt2.c

# =============================================================================
# Targets to build
# =============================================================================
all: train_gpt2

train_gpt2: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o train_gpt2 $(LDFLAGS)

# Correctness test (single TU: test_gpt2.c #includes train_gpt2.c). Meaningful on
# the CPU backend; on GPU it compares host copies (see test_gpt2.c notes).
test: $(TEST_SRC)
	$(CC) $(CFLAGS) $(TEST_SRC) -o test_gpt2 $(LDFLAGS)

clean:
	rm -rf train_gpt2 test_gpt2 train

.PHONY: all test clean
