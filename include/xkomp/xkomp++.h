#ifndef __XKOMP_PLUS_PLUS_H__
# define __XKOMP_PLUS_PLUS_H__

/**
 *  a C++ API for OpenMP, so we can fastly prototype APIs
 */

# include <xkomp/xkomp.h>
# include <functional>

static inline void
pragma_omp_taskgraph(
    xkomp_taskgraph_id_t graph_id,
    xkomp_taskgraph_flags_t flags,
    std::function<void(void)> f
) {
    xkomp_taskgraph_t * taskgraph = xkomp_taskgraph_begin(graph_id, flags);
    if (taskgraph->rc == 1)
        f();
    xkomp_taskgraph_end(taskgraph);

    # if 0
    {
        char fname[128];
        snprintf(fname, sizeof(fname), "taskgraph-%d.dot", graph_id);
        FILE * f = fopen(fname, "w");
        taskgraph->tdg.dump_tasks(f);
        fclose(f);
    }
    # endif
}

static inline void
pragma_omp_taskgraphloop(
    xkomp_taskgraph_id_t graph_id,
    xkomp_taskgraph_flags_t flags,
    std::function<void(void)> f,
    xkomp_taskgraphloop_event_handle_t evt
) {
    // TODO
}

#endif /* __XKOMP_PLUS_PLUS_H__ */
