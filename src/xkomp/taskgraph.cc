# include <xkomp/xkomp.h>

static inline void
coherence_checks(
    xkomp_taskgraph_id_t graph_id,
    xkomp_taskgraph_flags_t flags
) {
    // clauses are not supported yet
    if (flags & XKOMP_TASKGRAPH_FLAG_RESET)
        LOGGER_FATAL("Not supported");

    if (flags & XKOMP_TASKGRAPH_FLAG_IF)
        LOGGER_FATAL("Not supported");

    if (flags & XKOMP_TASKGRAPH_FLAG_NOGROUP)
        LOGGER_FATAL("Not supported");

    if (graph_id != 0)
        LOGGER_FATAL("Only supporting graph_id == 0, and a single taskgraph");
}

extern "C"
xkomp_taskgraph_t *
xkomp_taskgraph_begin(
    xkomp_taskgraph_id_t graph_id,
    xkomp_taskgraph_flags_t flags
) {
    coherence_checks(graph_id, flags);

    // retrieve tdg
    xkomp_t * xkomp = xkomp_get();
    xkomp_taskgraph_t & taskgraph = xkomp->taskgraphs[graph_id];
    ++taskgraph.rc;

    // first execution = record
    if (taskgraph.rc == 1)
    {
        constexpr bool execute_commands = true;
        xkomp->runtime.task_dependency_graph_record_start(&taskgraph.tdg, execute_commands);
    }
    else
    {
        // second or further executions

        // optimize on first replay
        if (taskgraph.rc == 2)
        {
            /* build a CG from a tdg */
            xkomp->runtime.command_graph_from_task_dependency_graph(&taskgraph.tdg, &taskgraph.cg);

            /* optimize the CG with the passes selected by OMP_TASKGRAPH_OPT */
            taskgraph.cg.optimize(xkomp->env.OMP_TASKGRAPH_OPT);
        }

        /* replay the CG */
        xkomp->runtime.command_graph_replay(&taskgraph.cg);
    }

    // TODO: if map is resized, address would change
    return &taskgraph;
}

extern "C"
void
xkomp_taskgraph_end(xkomp_taskgraph_t * taskgraph)
{
    xkomp_t * xkomp = xkomp_get();

    // stop recording
    if (taskgraph->rc == 1)
    {
        // implicit taskwait in XKRT
        xkomp->runtime.task_dependency_graph_record_stop();
    }
    else
    {
        // TODO
    }
}

extern "C"
xkomp_taskgraphloop_t *
xkomp_taskgraphloop_begin(
    xkomp_taskgraph_id_t graph_id,
    xkomp_taskgraph_flags_t flags
) {
    coherence_checks(graph_id, flags);
    return NULL;
}

extern "C"
void
xkomp_taskgraphloop_end(xkomp_taskgraphloop_t * loop)
{
}
