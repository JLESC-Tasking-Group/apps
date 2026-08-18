// #include <omp-tools.h>

#ifndef __TRACER_OMPT_H__
# define __TRACER_OMPT_H__

# ifdef NO_TRACER
#  define OMPT_SET_LABEL(...)
# else

# ifdef __cplusplus
extern "C" void ompt_set_task_name(const char * name);
# else
void ompt_set_task_name(const char * name);
# endif
# define OMPT_SET_LABEL(...)                            \
    do {                                                \
        char buffer[128];                               \
        snprintf(buffer, sizeof(buffer), __VA_ARGS__);  \
        ompt_set_task_name(buffer);                     \
    } while (0)

# endif

#endif /* __TRACER_OMPT_H__ */