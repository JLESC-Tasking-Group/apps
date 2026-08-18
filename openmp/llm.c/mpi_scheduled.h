#ifndef __MPI_SCHEDULED_H__
#define __MPI_SCHEDULED_H__

/*
Allow the function passed in parameters to use oss pragmas

This function should only be called inside an user-defined MPI_Op function.
The function passed in parameters will be put on nanos scheduler, and thus
will be able to use oss pragmas, which is not possible otherwise

@param function The function to be called on nanos scheduler
@param args The arguments to be passed to the given function
@param label Label to be used by paraver
*/
void mpi_launch_on_scheduler(void (*function)(void *), void *args, char *label);

#endif