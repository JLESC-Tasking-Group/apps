
# include <xkomp/xkomp.h>

template <typename T>
T
xkomp_env_init_parse_convert(const char * const value)
{
    static_assert(sizeof(T) == 0, "Unsupported type for xkomp_env_init_parse");
}

template <>
int
xkomp_env_init_parse_convert<int>(const char * const value)
{
    return atoi(value);
}

template <>
char
xkomp_env_init_parse_convert<char>(const char * const value)
{
    return *value;
}

template <>
const char *
xkomp_env_init_parse_convert<const char *>(const char * const value)
{
    return strdup(value);
}

template <typename T>
void
xkomp_env_init_parse(
    T * var,
    const char * const name,
    const T default_value
) {
    const char * value = getenv(name);
    if (value && *value)
        *var = xkomp_env_init_parse_convert<T>(value);
    else
        *var = default_value;
}

/**
 *  OMP_TASKGRAPH_OPT: comma/space-separated list of cgir command-graph
 *  optimization passes applied when a recorded taskgraph is optimized.
 *      - unset      -> default trio (reduce-node, transitive-reduction, sequence)
 *      - empty ""   -> no passes (equivalent to "none")
 *      - "none"     -> no passes
 *      - otherwise  -> the named passes; unknown names are warned and ignored
 *  Accepted names are the cgir pass names (see command_graph_pass_to_str):
 *  "copy-normalize", "copy-fuse", "reduce-node", "transitive-reduction", "jit", "prog-fuse", "sequence", "batch".
 */
static cgir::command_graph_pass_set_t
xkomp_env_init_parse_taskgraph_opt(void)
{
    const char * value = getenv("OMP_TASKGRAPH_OPT");

    // unset -> default trio: reduce the graph, then collapse same-device OpenMP
    // task chains into is_sequence super-task batches (the batch pass is opt-in
    // via OMP_TASKGRAPH_OPT for vendor-graph capture).
    if (value == NULL)
        return cgir::COMMAND_GRAPH_PASS_REDUCE_NODE_BIT
             | cgir::COMMAND_GRAPH_PASS_TRANSITIVE_REDUCTION_BIT
             | cgir::COMMAND_GRAPH_PASS_SEQUENCE_BIT;

    // set (possibly empty) -> parse tokens; an empty string yields no tokens == "none"
    cgir::command_graph_pass_set_t passes = 0;
    char * dup = strdup(value);
    for (char * tok = strtok(dup, ", \t") ; tok != NULL ; tok = strtok(NULL, ", \t"))
    {
        if (strcmp(tok, "none") == 0)
            continue;

        const cgir::command_graph_pass_t pass = cgir::command_graph_pass_from_str(tok);
        if (pass == cgir::COMMAND_GRAPH_PASS_MAX)
            // command_graph_pass_names() ends with a trailing ", ", so "none"
            // appends cleanly as the final accepted value
            LOGGER_WARN("OMP_TASKGRAPH_OPT: unknown optimization pass '%s' (ignored). Available passes: %snone",
                        tok, cgir::command_graph_pass_names());
        else
            passes |= cgir::command_graph_pass_bit(pass);
    }
    free(dup);

    // prog-fuse only fuses LLVM-IR; the jit pass must follow it to compile the
    // fused program. Without jit, the fused launcher stays NULL and aborts at
    // run-time.
    if ((passes & cgir::COMMAND_GRAPH_PASS_PROG_FUSE_BIT) &&
       !(passes & cgir::COMMAND_GRAPH_PASS_JIT_BIT))
        LOGGER_WARN("OMP_TASKGRAPH_OPT: 'prog-fuse' is enabled without 'jit'; "
                    "fused programs will not be compiled and will abort at "
                    "run-time. Add 'jit' to OMP_TASKGRAPH_OPT.");

    return passes;
}

/**
 *  Spec 6.0 says
 *  Modifications to the environment variables after the program has started,
 *  even if modified by the program itself, are ignored by the OpenMP
 *  implementation. This routine load at the program starts.
 */
void
xkomp_env_init(xkomp_env_t * env)
{
    // parse env variables
    # define F(T, S, D) xkomp_env_init_parse<T>(&env->S, #S, D)
    F(char,         OMP_DISPLAY_ENV,    'f');   // true, false or verbose ('t', 'f' or 'v')
    F(int,          OMP_NUM_THREADS,    0);
    F(int,          OMP_THREAD_LIMIT,   INT_MAX);
    F(const char *, OMP_PLACES,         "cores");
    F(const char *, OMP_PROC_BIND,      "close");
    # undef F

    // OMP_TASKGRAPH_OPT needs custom parsing (pass-name list -> bitset, and an
    // empty value differs from unset), so it is handled outside the F() table.
    env->OMP_TASKGRAPH_OPT = xkomp_env_init_parse_taskgraph_opt();

    // maybe display
    switch (env->OMP_DISPLAY_ENV)
    {
        case ('f'):
            break ;

        case ('t'):
        case ('v'):
        {
            LOGGER_IMPL("-- XKOMP environment variables --");
            break ;
        }

        default:
            LOGGER_ERROR("Invalid `OMP_DISPLAY_ENV`");
    }
}
