#if defined(MPI) && defined(OSS)
#include <inttypes.h>
#include <nodes.h>
#include <nosv/alpi.h>
#include <pthread.h>

typedef struct {
    pthread_mutex_t _mutex;
    pthread_cond_t _cond;
    int _signaled;
} condition_variable_t;

// Blocking wait
void wait_condition_variable_block(condition_variable_t *cond_var) {
    pthread_mutex_lock(&cond_var->_mutex);
    while (cond_var->_signaled == 0) {
        pthread_mutex_unlock(&cond_var->_mutex);
        pthread_cond_wait(&cond_var->_cond, &cond_var->_mutex);
        pthread_mutex_lock(&cond_var->_mutex);
    }
    pthread_mutex_unlock(&cond_var->_mutex);
}

// Wait by yielding
void wait_condition_variable_yield(condition_variable_t *cond_var) {
    uint64_t target_ns = 0; // Better performances. However yielding is mandatory to avoid CPU starvation
    uint64_t actual_ns;
    pthread_mutex_lock(&cond_var->_mutex);
    while (cond_var->_signaled == 0) {
        pthread_mutex_unlock(&cond_var->_mutex);
        alpi_task_waitfor_ns(target_ns, &actual_ns);
        pthread_mutex_lock(&cond_var->_mutex);
    }
    pthread_mutex_unlock(&cond_var->_mutex);
}

void condition_variable_callback(void *untyped_arg) {
    condition_variable_t *cond_var = (condition_variable_t *)untyped_arg;
    pthread_mutex_lock(&cond_var->_mutex);
    cond_var->_signaled = 1;
    pthread_cond_signal(&cond_var->_cond);
    pthread_mutex_unlock(&cond_var->_mutex);
}

// This function can be called by 2 different thread, one that is on nosV runtime and the other that isn't
// If the thread is on nosV runtime, spawn the reduction function and yield, otherwise use a blocking wait
void mpi_launch_on_scheduler(void (*function)(void *), void *args, char *label) {
    condition_variable_t cond_var = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
    struct alpi_task *task = NULL;
    alpi_task_self(&task);
    nanos6_spawn_function(function, args, condition_variable_callback, &cond_var, label);
    if (task == NULL) // Current thread is not on nanos scheduler.
        wait_condition_variable_block(&cond_var);
    else // Current thread is on nanos scheduler
        wait_condition_variable_yield(&cond_var);
}
#endif