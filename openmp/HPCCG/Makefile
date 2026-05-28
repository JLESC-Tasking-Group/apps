# Common code
COMMON_CFLAGS=-I common/ -DWALL -O3 #-g
#COMMON_CFLAGS+=-Wno-unused-parameter -Wno-unused-variable
COMMON_CFLAGS+=-Wno-comment -Wno-cast-function-type -Wno-format-overflow
#COMMON_CFLAGS+=-march=native -mcpu=native -mtune=native
COMMON_CFLAGS+=-Wall -ffast-math
COMMON_LDFLAGS=
COMMON_SRC_FILES=compute_residual.cpp dump_matlab_matrix.cpp generate_matrix.cpp \
			HPC_Sparse_Matrix.cpp main.cpp  make_local_matrix.cpp mytimer.cpp \
			read_HPC_row.cpp YAML_Doc.cpp YAML_Element.cpp
COMMON_SRC=$(addprefix common/,$(COMMON_SRC_FILES))

help:
	@echo "usage: make [seq|omp|omp-mpi|omp-task|oss-mpi]"

##################################
# --------- SEQUENTIAL --------- #
##################################
SEQ_CXX=g++
SEQ_CFLAGS=-Wall -Wextra
SEQ_LDFLAGS=
SEQ_SRC=$(COMMON_SRC) omp/ddot.cpp omp/exchange_externals.cpp \
				omp/HPCCG.cpp omp/HPC_sparsemv.cpp omp/waxpby.cpp
SEQ_OBJ=$(SEQ_SRC:.cpp=.seq.o)
SEQ_APP=hpccg-seq

%.seq.o: %.cpp
	$(SEQ_CXX) $(COMMON_CFLAGS) $(SEQ_CFLAGS) -o $@ -c $<

$(SEQ_APP): $(SEQ_OBJ)
	$(SEQ_CXX) $(COMMON_CFLAGS) $(SEQ_CFLAGS) $(SEQ_OBJ) $(COMMON_LDFLAGS) $(SEQ_LDFLAGS) -o $(SEQ_APP)

seq: $(SEQ_APP)

##################################
# ----------- OMP WS ----------- #
##################################
OMP_CXX=clang++
OMP_CFLAGS=-Wall -Wextra -DUSING_OMP -fopenmp
OMP_LDFLAGS=
OMP_SRC=$(COMMON_SRC) omp/ddot.cpp omp/exchange_externals.cpp \
				omp/HPCCG.cpp omp/HPC_sparsemv.cpp omp/waxpby.cpp
OMP_OBJ=$(OMP_SRC:.cpp=.omp.o)
OMP_APP=hpccg-omp-$(OMP_CXX)

%.omp.o: %.cpp
	$(OMP_CXX) $(COMMON_CFLAGS) $(OMP_CFLAGS) -o $@ -c $<

$(OMP_APP): $(OMP_OBJ)
	$(OMP_CXX) $(COMMON_CFLAGS) $(OMP_CFLAGS) $(OMP_OBJ) $(COMMON_LDFLAGS) $(OMP_LDFLAGS) -o $(OMP_APP)

omp: $(OMP_APP)

##################################
# -------- OMP WS MPI ---------- #
##################################
OMP_MPI_CXX=g++
OMP_MPI_MPICXX=I_MPI_CXX=$(OMP_MPI_CXX) MPICH_CXX=$(OMP_MPI_CXX) OMPI_CXX=$(OMP_MPI_CXX) mpicxx
OMP_MPI_CFLAGS=-Wall -Wextra -DUSING_MPI -DUSING_OMP -fopenmp
OMP_MPI_LDFLAGS=
OMP_MPI_SRC=$(COMMON_SRC) omp/ddot.cpp omp/exchange_externals.cpp \
				omp/HPCCG.cpp omp/HPC_sparsemv.cpp omp/waxpby.cpp
OMP_MPI_OBJ=$(OMP_MPI_SRC:.cpp=.omp-mpi.o)
OMP_MPI_APP=hpccg-omp-mpi-$(OMP_MPI_CXX)

%.omp-mpi.o: %.cpp
	$(OMP_MPI_MPICXX) $(COMMON_CFLAGS) $(OMP_MPI_CFLAGS) -o $@ -c $<

$(OMP_MPI_APP): $(OMP_MPI_OBJ)
	$(OMP_MPI_MPICXX) $(COMMON_CFLAGS) $(OMP_MPI_CFLAGS) $(OMP_MPI_OBJ) $(COMMON_LDFLAGS) $(OMP_MPI_LDFLAGS) -o $(OMP_MPI_APP)

omp-mpi: $(OMP_MPI_APP)

##################################
# --------- OMP TASK ----------- #
##################################
OMP_TASK_CXX=clang++
OMP_TASK_CFLAGS=-Wall -Wextra -DUSING_OMP -fopenmp
OMP_TASK_LDFLAGS=
OMP_TASK_SRC=$(COMMON_SRC) omp-task/HPCCG.cpp
OMP_TASK_OBJ=$(OMP_TASK_SRC:.cpp=.omp-task.o)
OMP_TASK_APP=hpccg-omp-task-$(OMP_TASK_CXX)

%.omp-task.o: %.cpp
	$(OMP_TASK_CXX) $(COMMON_CFLAGS) $(OMP_TASK_CFLAGS) -o $@ -c $<

$(OMP_TASK_APP): $(OMP_TASK_OBJ)
	$(OMP_TASK_CXX) $(COMMON_CFLAGS) $(OMP_TASK_CFLAGS) $(OMP_TASK_OBJ) $(COMMON_LDFLAGS) $(OMP_TASK_LDFLAGS) -o $(OMP_TASK_APP)

omp-task: $(OMP_TASK_APP)

##################################
# ------- OMP TASK MPI --------- #
##################################
OMP_TASK_MPI_CXX=clang++
OMP_TASK_MPI_MPICXX=I_MPI_CXX=$(OMP_TASK_MPI_CXX) MPICH_CXX=$(OMP_TASK_MPI_CXX) OMPI_CXX=$(OMP_TASK_MPI_CXX) mpicxx
OMP_TASK_MPI_CFLAGS=-Wall -Wextra -DUSING_MPI -DUSING_OMP -DUSING_TAMPI -fopenmp -DUSING_OMPI
OMP_TASK_MPI_LDFLAGS=
OMP_TASK_MPI_SRC=$(COMMON_SRC) omp-task/HPCCG.cpp
OMP_TASK_MPI_OBJ=$(OMP_TASK_MPI_SRC:.cpp=.omp-task-mpi.o)
OMP_TASK_MPI_APP=hpccg-omp-task-mpi-$(OMP_TASK_MPI_CXX)

%.omp-task-mpi.o: %.cpp
	$(OMP_TASK_MPI_MPICXX) $(COMMON_CFLAGS) $(OMP_TASK_MPI_CFLAGS) -o $@ -c $<

$(OMP_TASK_MPI_APP): $(OMP_TASK_MPI_OBJ)
	$(OMP_TASK_MPI_MPICXX) $(COMMON_CFLAGS) $(OMP_TASK_MPI_CFLAGS) $(OMP_TASK_MPI_OBJ) $(COMMON_LDFLAGS) $(OMP_TASK_MPI_LDFLAGS) -o $(OMP_TASK_MPI_APP)

omp-task-mpi: $(OMP_TASK_MPI_APP)

##################################
# ---------- OSS TASK ---------- #
##################################
OSS_CXX=mcxx
OSS_CFLAGS=-Wall -Wextra -DUSING_OMPSS --ompss-2
OSS_CFLAGS+=-Wno-parentheses -Wno-unused-local-typedefs -Wno-unused-parameter
OSS_CFLAGS+=--keep-all-files
OSS_LDFLAGS=
OSS_SRC=$(COMMON_SRC) oss/HPCCG.cpp
OSS_OBJ=$(OSS_SRC:.cpp=.oss.o)
OSS_APP=hpccg-oss

%.oss.o: %.cpp
	$(OSS_CXX) $(COMMON_CFLAGS) $(OSS_CFLAGS) -o $@ -c $<

$(OSS_APP): $(OSS_OBJ)
	$(OSS_CXX) $(COMMON_CFLAGS) $(OSS_CFLAGS) $(OSS_OBJ) $(COMMON_LDFLAGS) $(OSS_LDFLAGS) -o $(OSS_APP)

oss: $(OSS_APP)

##################################
# ------- OSS_MPI TASK --------- #
##################################
OSS_MPI_CXX=mcxx
OSS_MPI_MPICXX=I_MPI_CXX=$(OSS_MPI_CXX) MPICH_CXX=$(OSS_MPI_CXX) OMPI_CXX=$(OSS_MPI_CXX) mpicxx
OSS_MPI_CFLAGS=-Wall -Wextra -DUSING_MPI -DUSING_OMPSS --ompss-2 -DUSING_OMPI
OSS_MPI_CFLAGS+=-Wno-parentheses -Wno-unused-local-typedefs -Wno-unused-parameter
OSS_MPI_CFLAGS+=-I $(TAMPI_HOME)/include
OSS_MPI_CFLAGS+=--keep-all-files
OSS_MPI_LDFLAGS=-L $(TAMPI_HOME)/lib -ltampi
OSS_MPI_SRC=$(COMMON_SRC) oss/HPCCG.cpp
OSS_MPI_OBJ=$(OSS_MPI_SRC:.cpp=.oss-mpi.o)
OSS_MPI_APP=hpccg-oss-mpi

%.oss-mpi.o: %.cpp
	$(OSS_MPI_MPICXX) $(COMMON_CFLAGS) $(OSS_MPI_CFLAGS) -o $@ -c $<

$(OSS_MPI_APP): $(OSS_MPI_OBJ)
	$(OSS_MPI_MPICXX) $(COMMON_CFLAGS) $(OSS_MPI_CFLAGS) $(OSS_MPI_OBJ) $(COMMON_LDFLAGS) $(OSS_MPI_LDFLAGS) -o $(OSS_MPI_APP)

oss-mpi: $(OSS_MPI_APP)

###################################
# ----------- utils ------------- #
###################################
clean:
	rm -rf $(SEQ_OBJ) $(OMP_OBJ) $(OMP_MPI_OBJ) $(OMP_TASK_OBJ) $(OMP_APP_OBJ) $(OMP_TASK_MPI_OBJ) $(OSS_OBJ) $(OSS_MPI_OBJ)

dclean:
	rm -rf *.yaml *.btr *.o *.json mcxx_*.cpp traces*

fclean: clean dclean
	rm -rf $(SEQ_APP) $(OMP_APP) $(OMP_MPI_APP) $(OMP_TASK_APP) $(OMP_TASK_MPI_APP) $(OSS_APP) $(OSS_MPI_APP)

re: fclean all
