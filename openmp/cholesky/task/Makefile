CC     = clang
CFLAGS = -Wall -O3
CFLAGS+=-Wno-vla-cxx-extension
#CFLAGS+= -I common/
#CFLAGS+= -I ./ -I $(MKLROOT)/include/ -I $(MKLROOT)/include/intel64/lp64/
LDFLAGS= -lm -llapack -lblas
#LDFLAGS+= -L $(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

all:
	$(CC) $(CFLAGS) -fopenmp main.c -o cholesky $(LDFLAGS)

clean:
	-rm -f cholesky

.PHONY: cholesky
