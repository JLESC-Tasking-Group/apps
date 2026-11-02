# CloverLeaf - tasking.h port (tasking/cloverleaf.cpp). The vendored upstream
# tree (driver/, src/<model>/, CMakeLists.txt) is the reference it was ported from.
#
#     make                   # toggles from ../common.mk (USE_TARGET, USE_SYNC, ...)
#     ./cloverleaf -s 512 -i 50 -nb 32 -u 1
# -s cells per side, -i timestep PAIRS (one TASKGRAPH_LOOP iteration is an
# x-first + y-first pair), -nb tasks per loop, -u pairs per taskgraph instance.
include ../common.mk

program = cloverleaf
source  = tasking/cloverleaf.cpp
HDR     = ../tasking.h ../tg_report.h

all: $(program)

$(program): $(source) $(HDR)
	$(CC) $(CFLAGS) $(source) -o $(program) $(LDFLAGS)

run: $(program)
	./$(program) -s 512 -i 50 -nb 32

clean:
	rm -f $(program) *.o

.PHONY: all clean run
