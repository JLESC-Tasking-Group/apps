# include <omp.h>

# include <xkomp/xkomp.h>
# include <xkomp/kmp.h>

/* Scatter helper emitted by clang for an unpacked task: fills out[k] with the
 * address of the task's k-th captured (frozen firstprivate) value. See
 * emitTaskUnpackedScatter in clang's CGOpenMPRuntime.cpp. */
typedef void (*kmp_task_scatter_t)(void * /* kmp_task_t * */, void ** /* out */);

// task args
typedef struct  task_args_t
{
    task_t * task;

    /* The argument array for the JIT/prog-fuse path only: it feeds the forwarded
     * void(void**) kernel once the `jit` pass compiles it (recorded into a PROG
     * command's `args`; see body_omp_task). The ahead-of-time routine
     * `.omp_task_entry.(gtid, kmp_task_t*)` does NOT use `kargs` -- it recovers
     * shareds/privates from the kmp_task_t. `kargs` has two shapes, matching the
     * forwarded kernel the compiler emits:
     *
     *  - Unpacked form (fusable): `kargs` holds one &value slot per captured
     *    firstprivate scalar (kargs[k] == &value). The compiler emits the body
     *    as an unpacked kernel `void .omp_task_kernel.(v0, v1, ...)`.
     *    Exposing each captured value as its own slot is what lets prog-fuse
     *    deduplicate/align them across consecutive task bodies and fuse their
     *    loops. The slots are filled once by the scatter helper -- deferred to
     *    record time (see body_omp_task), when the captures are live: SHARED_PTR
     *    slots load a value from `shareds`, which the caller populates only AFTER
     *    task alloc returns. They point into the frozen data environment of THIS
     *    allocation, so a recorded taskgraph replays correctly.
     *
     *  - Packed form: `kargs` holds a single slot, kargs[0] == the task's
     *    kmp_task_t*; the packed kernel `void .omp_task_kernel.(void**)` reads it
     *    back to recover shareds/privates. Used for tasks that are not
     *    unpacked-eligible (taskloops, target, shared/by-ref captures, ...); fusion
     *    stays at the PROGRAM level.
     *
     * `kargs` points at storage reserved at the END of this task allocation
     * (after the kmp_task_t + shareds), so this struct stays a fixed size and
     * the kmp_task_t offset (see ktask_from_task) is independent of n_kargs. */
    void ** kargs;
    size_t  n_kargs;

    /* Compiler-emitted scatter (kmp_task_scatter_t) that fills `kargs`, stashed at
     * alloc and run at record time (see body_omp_task); NULL for the packed
     * void(void**) form (kargs[0] == tt, filled at alloc). */
    void *  scatter;

    // followed by the kmp task, its shareds, then the kargs slots
}               task_args_t;

/*
 * Number of access slots reserved for a task allocated through the stock-LLVM
 * ABI (__kmpc_omp_task_alloc / __kmpc_omp_target_task_alloc).
 *
 * The patched LLVM passes the dependence/access count at allocation time
 * (__kmpc_omp_task_alloc_with_deps), so XKOMP sizes the task exactly.  Stock
 * LLVM does not: it allocates first and only reveals the dependences later in
 * __kmpc_omp_task_with_deps.  An XKRT task stores its accesses inline *before*
 * the kmp_task_t region, so the kmp_task_t offset depends on the access count,
 * which must therefore be fixed at allocation time.  For the stock ABI we hence
 * reserve a FIXED number of slots and pad the unused ones with inert accesses
 * (keeping the kmp_task_t at a fixed offset); a task needing more dependences
 * than this is rejected.
 */
# define XKOMP_FIXED_ACCESSES   4

// see llvm impl
const size_t
round_up_to_val(size_t size, size_t val)
{
    if (size & (val - 1))
    {
        size &= ~(val - 1);
        if (size <= SIZE_MAX - val)
            size += val;
    }
    return size;
}

static inline kmp_task_t *
ktask_from_task(task_t * task)
{
    task_args_t * args = (task_args_t *) TASK_ARGS(task);
    assert(args);

    kmp_task_t * ktask = (kmp_task_t *) (args + 1);
    assert(ktask);

    return ktask;
}

/*
 * Return the task's argument array (the `void ** args` for the JIT'd void(void**)
 * kernel; see task_args_t). For an unpacked task the slots are the per-value
 * &value pointers (args[k] == &value); for a packed task args[0] is the
 * kmp_task_t*. The per-value slots are what let prog-fuse fuse the bodies' LOOPS.
 */
static inline void **
kargs_from_task(task_t * task)
{
    task_args_t * args = (task_args_t *) TASK_ARGS(task);
    assert(args);
    return args->kargs;
}

/* number of void* slots in the task's args array consumed by the routine */
static inline size_t
kargs_nslots(task_t * task)
{
    task_args_t * args = (task_args_t *) TASK_ARGS(task);
    assert(args);
    return args->n_kargs;
}

static inline task_t *
task_from_ktask(kmp_task_t * ktask)
{
    assert(ktask);

    task_args_t * args = (task_args_t *) (((char *) ktask) - sizeof(task_args_t));
    assert(args);
    assert((void *)(args + 1) == (void *) ktask);
    assert(args->task);
    // assert(TASK_ARGS(args->task) == (void *) args);

    return args->task;
}

/*
 * Pre-fill access slots [from, to) with inert "virtual" (ACCESS_MODE_V) accesses.
 * They carry no data movement and have no successors, so every runtime loop that
 * iterates over the task's reserved slots (data fetch, dependence release,
 * taskgraph walk) skips/ignores them.  This keeps acs->ac (and hence the
 * kmp_task_t offset used by ktask_from_task) fixed, while real dependences (if
 * any) overwrite the leading slots in __kmpc_omp_task_with_deps_v2.
 */
static inline void
task_pad_accesses(task_t * task, access_t * accesses, kmp_int32 from, kmp_int32 to)
{
    for (kmp_int32 i = from ; i < to ; ++i)
        new (accesses + i) access_t(task, ACCESS_MODE_V, ACCESS_CONCURRENCY_SEQUENTIAL, ACCESS_SCOPE_NONUNIFIED);
}

// Run an OpenMP task body directly via its ahead-of-time libomp routine
// `routine(gtid, kmp_task_t*)` (the compiler's `.omp_task_entry.` proxy). This is
// the non-JIT execution path; the JIT/fusion path instead compiles the forwarded
// void(void**) kernel and is launched by the runtime (see command_prog_run_host).
static inline void
body_omp_task_run(
    task_t * task
) {
    kmp_task_t * ktask = ktask_from_task(task);
    assert(ktask);
    assert(ktask->routine);
    thread_t * thread = thread_t::get_tls();
    const kmp_int32 gtid = thread ? (kmp_int32) thread->gtid : 0;
    ktask->routine(gtid, ktask);
}

// Called from a non-device thread
static inline void
body_omp_task(
    runtime_t * runtime,
    device_t * device,
    task_t * task
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);
    assert(thread->team);

    body_omp_task_run(task);

    // if recording, insert a new command for replay
    if (task->flags & TASK_FLAG_RECORD)
    {
        static bool warned = false;
        if (warned == false)
        {
            LOGGER_WARN("TODO: currently, assume XKRT tasks are never released, and so the OpenMP task and data environment can be passed as such... FIX ME! If the task gets released (it should), we should instead protect here by incrementing its ref counter, and when respawning the task during the replay, replicate its data environment ?");
            warned = true;
        }

        task_rec_info_t * rec = TASK_REC_INFO(task);
        assert(rec);

        /* Record a PROG command for replay. Its launch mode is TASK_SPAWN, so on
         * replay the runtime re-spawns a task (in the replaying thread's team)
         * that runs the body and completes the command.
         *
         * The command is recorded on the KMP prototype: launcher.kmp is the task's
         * ahead-of-time libomp routine `.omp_task_entry.(gtid, kmp_task_t*)`, run
         * directly for non-JIT replay. We ALSO fill `args` (the recorded kargs)
         * and the serialized IR (attached from the task format during graph build)
         * so prog-fuse/jit can compile the forwarded void(void**) kernel: the
         * `jit` pass then flips the prototype to VARIADIC. Keeping both coexisting
         * is why `args` lives outside the launcher union.
         *
         * For an unpacked task the recorded args are the per-value &value
         * slots (n_kargs of them; see task_args_t), so prog-fuse can deduplicate/
         * align the captured scalars across bodies and fuse their LOOPS. For a
         * packed task the single slot is the kmp_task_t* (args[0] == tt), and
         * fusion stays at the PROGRAM level. */
        kmp_task_t * ktask = ktask_from_task(task);
        assert(ktask);
        assert(ktask->routine);

        /* Fill kargs now, with the captures live (the body ran just above, so
         * `shareds`/privates are populated). Deferred from task alloc because a
         * SHARED_PTR slot loads a pointer out of `shareds`, which is empty at alloc
         * time. Runs once per recorded task; the slots feed prog-fuse/jit. */
        task_args_t * args = (task_args_t *) TASK_ARGS(task);
        if (args->scatter)
            ((kmp_task_scatter_t) args->scatter)(ktask, args->kargs);

        task_command_record_t * cmdrec = rec->commands.put();
        constexpr cgir::command_type_t ctype = cgir::COMMAND_TYPE_PROG;
        constexpr command_flag_t flags = COMMAND_FLAG_SYNCHRONOUS | COMMAND_FLAG_SERIALIZED;
        cmdrec->state = task->state.value;
        new (&cmdrec->command) command_t(ctype, flags);
        cmdrec->command.prog.launch_mode        = cgir::CGIR_COMMAND_PROG_LAUNCH_MODE_TASK_SPAWN;
        cmdrec->command.prog.prototype          = cgir::CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_KMP;
        cmdrec->command.prog.launcher.kmp.fn    = (int (*)(int, void *)) ktask->routine;   // .omp_task_entry.(gtid, tt)
        cmdrec->command.prog.launcher.kmp.task  = (void *) ktask;
        cmdrec->command.prog.args               = kargs_from_task(task);   // void** (&value slots, or [tt]) for the JIT path
        cmdrec->command.prog.n_args             = kargs_nslots(task);
        cmdrec->command.prog._args_owned        = false;  // owned by the task's data environment
    }
}

// called from a device thread
static inline void
body_omp_task_target(
    runtime_t * runtime,
    device_t * device,
    task_t * task
) {
    body_omp_task_run(task);
}

// Get (or lazily create) the task format for a task construct, copying the
// per-target functions from the shared 'omp-task' template and attaching the
// construct's LLVM-IR.
//
// The format is keyed by the embedded IR pointer, NOT by the source location
// (loc_ref): without -g/source info clang emits a single dummy ident
// (";unknown;unknown;0;0;;") shared by every task construct in the TU, so keying
// on loc_ref would collapse distinct task bodies onto the first format and, under
// JIT, run that one body for all of them. The IR is emitted once per construct
// (a distinct .omp_task_ir.<fn> global), so its pointer uniquely identifies the
// body. We fall back to loc_ref only when there is no IR (e.g. target
// directives, which do not use the host JIT path).
static inline task_format_id_t
get_or_create_loc_format(
    xkomp_t * xkomp,
    ident_t * loc_ref,
    const kmp_task_jit_desc_t * jit_desc,  // JIT source descriptor (compile-time global) or NULL
    void * scatter                          // != NULL => unpacked (individual params); NULL => packed (args[0]==tt)
) {
    // the forwarded task body IR + its resolution tables + ABI, or {NULL,0,...}
    // when no IR was forwarded (e.g. -fopenmp-task-jit-abi=none)
    void * ir      = jit_desc ? (void *) jit_desc->ir : NULL;
    size_t ir_size = jit_desc ? jit_desc->ir_size      : 0;

    // no per-construct key at all: fall back to the shared template format
    if (loc_ref == NULL && (ir == NULL || ir_size == 0))
        return xkomp->formats.kmp.host;

    // prefer the IR pointer as the per-construct key (see comment above)
    void * key = (ir != NULL && ir_size > 0) ? ir : (void *) loc_ref;

    SPINLOCK_LOCK(xkomp->formats.kmp.per_loc_lock);

    task_format_id_t fmtid;
    auto it = xkomp->formats.kmp.per_loc.find(key);
    if (it != xkomp->formats.kmp.per_loc.end())
    {
        fmtid = it->second;
    }
    else
    {
        task_format_t format;
        memset(&format, 0, sizeof(task_format_t));

        task_format_t * tmpl = xkomp->runtime.task_format_get(xkomp->formats.kmp.host);
        memcpy(format.f, tmpl->f, sizeof(format.f));
        format.suggest = tmpl->suggest;

        // label from the source location string ("file;func;line;line")
        if (loc_ref && loc_ref->psource)
            snprintf(format.label, sizeof(format.label), "%s", loc_ref->psource);
        else
            snprintf(format.label, sizeof(format.label), "omp-task");

        // the IR (and its externalized-global address / parameter tables) are
        // compile-time globals, so `_owned`/`_externs_owned`/`_params_owned` are false
        if (ir != NULL && ir_size > 0)
        {
            cgir_command_prog_source_t * src = &format.source[XKRT_TASK_FORMAT_TARGET_HOST];
            src->type                          = CGIR_COMMAND_PROG_SOURCE_TYPE_LLVMIR;
            src->content.llvmir.raw            = ir;
            src->content.llvmir.size           = ir_size;
            src->content.llvmir._owned         = false;
            // entry name (compile-time global): the JIT resolves the closure entry
            // by it rather than guessing (host closures may hold several defs)
            src->content.llvmir.symbol         = jit_desc->entry_name;
            src->content.llvmir.externs        = (const cgir_command_prog_extern_t *) jit_desc->externs;
            src->content.llvmir.externs_count  = jit_desc->externs_count;
            src->content.llvmir._externs_owned = false;
            // Parameter descriptors + entry prototype: an unpacked task body exposes
            // one value parameter per capture (the fusion unit); a packed body has
            // a single void** args block (args[0] == kmp_task_t*).
            // -fopenmp-task-jit-abi=packed (jit_desc->proto==2) requests a packed-buffer
            // fused program; otherwise an unpacked (individual-param) body has a
            // scatter, and a packed-args-block (args[0]==tt) proxy has none.
            src->content.llvmir.proto          =
                (jit_desc->proto == (int) CGIR_COMMAND_PROG_SOURCE_PROTO_PACKED_BUFFER)
                    ? CGIR_COMMAND_PROG_SOURCE_PROTO_PACKED_BUFFER
                : scatter ? CGIR_COMMAND_PROG_SOURCE_PROTO_UNPACKED_PARAMS
                          : CGIR_COMMAND_PROG_SOURCE_PROTO_VOID_PTRPTR;
            src->content.llvmir.params         = (const cgir_command_prog_param_t *) jit_desc->params;
            src->content.llvmir.param_count    = jit_desc->params_count;
            src->content.llvmir._params_owned  = false;
        }

        fmtid = xkomp->runtime.task_format_create(&format);
        xkomp->formats.kmp.per_loc[key] = fmtid;
    }

    SPINLOCK_UNLOCK(xkomp->formats.kmp.per_loc_lock);
    return fmtid;
}

inline static kmp_task *
task_alloc(
    ident_t * loc_ref,
    kmp_int32 gtid,
    kmp_tasking_flags_t flags,
    size_t sizeof_kmp_task_t,
    size_t sizeof_shareds,
    kmp_routine_entry_t task_entry,
    kmp_int32 ndeps,
    kmp_int32 nacs,
    kmp_int32 device_id,
    const kmp_task_jit_desc_t * jit_desc,  // JIT source descriptor (compile-time global) or NULL
    size_t n_args,          // number of &value slots the routine consumes
    void * scatter          // unpacked scatter (kmp_task_scatter_t) or NULL (packed)
) {
    if (device_id == -1)
        device_id = omp_get_default_device();

    // there is a '1' offset between omp device id and xkaapi device id
    static_assert(XKRT_HOST_DEVICE_UNIQUE_ID == 0);
    device_unique_id_t device_unique_id = omp_device_id_to_xkomp(device_id);

    const kmp_int32 naccesses = ndeps + nacs;
    assert(naccesses <= XKRT_TASK_MAX_ACCESSES);

    thread_t * thread = thread_t::get_tls();
    assert(thread);

    xkomp_t * xkomp = xkomp_get();
    assert(xkomp);

    // compute flags
    task_flag_bitfield_t xkflags  = TASK_FLAG_ZERO;

    // if tasks has accesses, set flag
    if (naccesses)
        xkflags |= TASK_FLAG_ACCESSES;

    // if task is not running on the host, set device/detachable flags, since libomptarget may submit commands
    if (device_unique_id != XKRT_HOST_DEVICE_UNIQUE_ID)
    {
        xkflags |= TASK_FLAG_DEVICE;        // execute on a device thread
        xkflags |= TASK_FLAG_DETACHABLE;    // a device task may submit commands
    }

    // undeferred task (OpenMP `if(0)`): the compiler set this bit at creation.
    // The task is still committed/scheduled normally (any thread may run it); the
    // encountering thread just suspends until it completes (see task_wait(task)
    // in __kmpc_omp_task / __kmpc_omp_task_with_deps_v2).
    if (flags.undeferred)
        xkflags |= TASK_FLAG_UNDEFERABLE;

    // if recording flag is set on parent, record that task.
    // TODO: if the `replayable(false)` is set, do not record

    // resolve the per-source-location task format (carries the LLVM-IR + the
    // externalized-global address table)
    const task_format_id_t fmtid = get_or_create_loc_format(xkomp, loc_ref, jit_desc, scatter);

    // Layout of the args region (see task_args_t):
    //   [task_args_t] [kmp_task_t + shareds (rounded)] [n_kargs void* slots]
    // The kargs slots live at the end so task_args_t stays a fixed size (the
    // kmp_task_t offset must not depend on n_args; see ktask_from_task). We
    // reserve at least one slot so `kargs` never points past the allocation.
    const size_t nslots = n_args ? n_args : 1;
    const size_t kmp_block = round_up_to_val(sizeof_kmp_task_t + sizeof_shareds, sizeof(void *));
    const size_t args_size = sizeof(task_args_t) + kmp_block + nslots * sizeof(void *);
    task_t * task = xkomp->runtime.task_new(fmtid, xkflags, NULL, args_size, naccesses);
    assert(task);

    // init acs infos
    if (naccesses)
    {
        task_acs_info_t * acs = TASK_ACS_INFO(task);

        // if spawning a device task, set spawning_thread so that
        // successor tasks are pushed to it
        thread_t * spawning_thread = (device_unique_id == XKRT_HOST_DEVICE_UNIQUE_ID) ? thread : NULL;

        new (acs) task_acs_info_t(spawning_thread, naccesses);
    }

    // init dev/det infos
    if (device_unique_id != XKRT_HOST_DEVICE_UNIQUE_ID)
    {
        assert(xkflags & TASK_FLAG_DEVICE);
        task_dev_info_t * dev = TASK_DEV_INFO(task);
        new (dev) task_dev_info_t(device_unique_id, XKRT_UNSPECIFIED_TASK_ACCESS);

        assert(xkflags & TASK_FLAG_DETACHABLE);
        task_det_info_t * det = TASK_DET_INFO(task);
        new (det) task_det_info_t();
    }

    // init args
    task_args_t * args = (task_args_t *) TASK_ARGS(task);
    assert(args);
    args->task = task;

    // init kmp task info
    kmp_task_t * ktask = (kmp_task_t *) (args + 1);
    assert(ktask);

    ktask->shareds = (void *) (sizeof_shareds <= 0 ? 0 : ((uintptr_t) ktask) + sizeof_kmp_task_t);
    ktask->routine = task_entry;
    ktask->part_id = 0;

    // the kargs slots sit just past the kmp_task_t + shareds block
    args->kargs   = (void **) (((char *) ktask) + kmp_block);
    args->n_kargs = n_args;
    args->scatter = scatter;

    if (scatter)
    {
        /* Unpacked form: the compiler-emitted scatter fills kargs[k] for the k-th
         * capture. Deferred to record time (body_omp_task), NOT run here: a
         * SHARED_PTR slot loads a pointer from `shareds`, which the caller copies
         * in only after this alloc returns -- reading it now yields garbage. By
         * record time the captures are live and the slots are stable for replay. */
    }
    else
    {
        /* Packed form: the single slot is the task's kmp_task_t* (args[0] ==
         * tt); the routine reads it back to recover shareds/privates. */
        assert(n_args == 1);
        args->kargs[0] = (void *) ktask;
    }

    # ifndef NDEBUG
    snprintf(task->label, sizeof(task->label), "omp-task");
    # endif /* NDEBUG */

    return ktask;
}

extern "C"
kmp_task_t *
__kmpc_omp_task_alloc_with_deps(
    ident_t * loc_ref,
    kmp_int32 gtid,
    kmp_tasking_flags_t flags,
    size_t sizeof_kmp_task_t,
    size_t sizeof_shareds,
    kmp_routine_entry_t task_entry,
    kmp_int32 ndeps,
    kmp_int32 nacs,
    const kmp_task_jit_desc_t * jit_desc,  // JIT source descriptor (or NULL)
    size_t n_args,
    void * scatter
) {
    const kmp_int32 device_id = omp_get_initial_device();
    return task_alloc(loc_ref, gtid, flags, sizeof_kmp_task_t, sizeof_shareds, task_entry, ndeps, nacs, device_id, jit_desc, n_args, scatter);
}

extern "C"
kmp_task_t *
__kmpc_omp_target_task_alloc_with_deps(
    ident_t * loc_ref,
    kmp_int32 gtid,
    kmp_tasking_flags_t flags,
    size_t sizeof_kmp_task_t,
    size_t sizeof_shareds,
    kmp_routine_entry_t task_entry,
    kmp_int64 device_id,
    kmp_int32 ndeps,
    kmp_int32 nacs,
    const kmp_task_jit_desc_t * jit_desc,  // JIT source descriptor (or NULL)
    size_t n_args,
    void * scatter
) {
    // target task is untied defined in the specification
    # define TASK_UNTIED    0
    # define TASK_TIED      1
    flags.tiedness = TASK_UNTIED;

    return task_alloc(loc_ref, gtid, flags, sizeof_kmp_task_t, sizeof_shareds, task_entry, ndeps, nacs, device_id, jit_desc, n_args, scatter);
}

// Stock-LLVM ABI entry point (no dependence count at alloc time): reserve a
// FIXED number of access slots and pad them with inert accesses, keeping the
// kmp_task_t at a fixed offset.  Real dependences (if any, up to
// XKOMP_FIXED_ACCESSES) overwrite the leading slots in __kmpc_omp_task_with_deps.
extern "C"
kmp_task_t *
__kmpc_omp_target_task_alloc(
    ident_t * loc_ref,
    kmp_int32 gtid,
    kmp_tasking_flags_t flags,
    size_t sizeof_kmp_task_t,
    size_t sizeof_shareds,
    kmp_routine_entry_t task_entry,
    kmp_int64 device_id
) {
    LOGGER_WARN("Stock LLVM/clang ABI: at most %d dependences/accesses per task "
                "are supported; use the patched LLVM/clang for XKOMP for unlimited "
                "dependences and best performance.", (int) XKOMP_FIXED_ACCESSES);
    flags.tiedness = TASK_UNTIED;   // target tasks are untied per the spec
    constexpr kmp_int32 ndeps = XKOMP_FIXED_ACCESSES;
    constexpr kmp_int32 nacs  = 0;
    constexpr kmp_task_jit_desc_t * jit_desc = NULL;   // stock ABI forwards no IR
    constexpr size_t n_args = 1;
    constexpr void * scatter = NULL;
    kmp_task_t * ktask = task_alloc(loc_ref, gtid, flags, sizeof_kmp_task_t, sizeof_shareds, task_entry, ndeps, nacs, device_id, jit_desc, n_args, scatter);
    task_t * task = task_from_ktask(ktask);
    task_pad_accesses(task, TASK_ACCESSES(task), 0, XKOMP_FIXED_ACCESSES);
    return ktask;
}

// Stock-LLVM ABI entry point: see __kmpc_omp_target_task_alloc above.
extern "C"
kmp_task_t *
__kmpc_omp_task_alloc(
    ident_t * loc_ref,
    kmp_int32 gtid,
    kmp_tasking_flags_t flags,
    size_t sizeof_kmp_task_t,
    size_t sizeof_shareds,
    kmp_routine_entry_t task_entry
) {
    LOGGER_WARN("Stock LLVM/clang ABI: at most %d dependences/accesses per task "
                "are supported; use the patched LLVM/clang for XKOMP for unlimited "
                "dependences and best performance.", (int) XKOMP_FIXED_ACCESSES);
    constexpr kmp_int32 ndeps = XKOMP_FIXED_ACCESSES;
    constexpr kmp_int32 nacs  = 0;
    constexpr kmp_task_jit_desc_t * jit_desc = NULL;   // stock ABI forwards no IR
    constexpr size_t n_args = 1;
    constexpr void * scatter = NULL;
    const kmp_int32 device_id = omp_get_initial_device();
    kmp_task_t * ktask = task_alloc(loc_ref, gtid, flags, sizeof_kmp_task_t, sizeof_shareds, task_entry, ndeps, nacs, device_id, jit_desc, n_args, scatter);
    task_t * task = task_from_ktask(ktask);
    task_pad_accesses(task, TASK_ACCESSES(task), 0, XKOMP_FIXED_ACCESSES);
    return ktask;
}

extern "C"
kmp_int32
__kmpc_omp_task(
    ident_t * loc_ref,
    kmp_int32 gtid,
    kmp_task_t * ktask
) {
    task_t * task = task_from_ktask(ktask);

    xkomp_t * xkomp = xkomp_get();
    assert(xkomp);

    // TODO: im not sure why we need all that mess, see MPC and LLVM impl to
    // figure out what's going on with untied tasks
    /* LLVM calls both
     *  '__kmpc_omp_task' and '__kmpc_omp_task_with_deps'
     * if a task is untied and has dependences.
     * In such case, the task was already commit in the previous call to
     * '__kmpc_omp_task_with_deps'
     */
    if (task->state.value == TASK_STATE_EXECUTING)
    {
        body_omp_task(NULL, NULL, task);
        return 1;
    }

    // stock-LLVM no-dep task: its reserved access slots were already padded with
    // inert accesses at allocation time, so it can be committed as-is.
    xkomp->runtime.task_commit(task);

    // undeferred task (`if(0)`): suspend the encountering thread until it
    // completes. The task went through the normal scheduling path above, so any
    // thread of the team may execute it meanwhile.
    if (task->flags & TASK_FLAG_UNDEFERABLE)
        xkomp->runtime.task_wait(task);

    return 0;
}

// NOTE: OpenMP `if(0)` (undeferred tasks) are NOT lowered through
// __kmpc_omp_task_begin_if0 / __kmpc_omp_task_complete_if0 anymore. The patched
// clang instead sets an `undeferred` bit in kmp_tasking_flags at creation and
// submits the task on the normal path; the runtime suspends the encountering
// thread after committing it (see task_alloc + the TASK_FLAG_UNDEFERABLE checks
// in __kmpc_omp_task / __kmpc_omp_task_with_deps_v2).

extern "C"
void
__kmp_omp_task_acs_to_xkomp_accesses(
    task_t * task,
    access_t * accesses,
    kmp_access_info_t * acs_list,
    kmp_int32 nacs
) {
    for (kmp_int32 i = 0; i < nacs ; ++i)
    {
        const void             * ptr = (const void *) acs_list[i].base_addr;
        const size_t               n = (size_t) acs_list[i].len;
        constexpr size_t sizeof_type = 1;
        if (ptr)
        {
            access_mode_t           mode        = (access_mode_t) 0;
            access_concurrency_t    concurrency = ACCESS_CONCURRENCY_SEQUENTIAL;
            access_scope_t          scope       = ACCESS_SCOPE_NONUNIFIED;

            if (acs_list[i].flags.storage)
            {
                acs_list[i].flags.write = 1;
                acs_list[i].flags.noncoherent = 1;
            }

            if (acs_list[i].flags.read)
                mode = (access_mode_t) (mode | ACCESS_MODE_R);

            if (acs_list[i].flags.write || acs_list[i].flags.concurrentwrite)
            {
                mode = (access_mode_t) (mode | ACCESS_MODE_W);
                if (acs_list[i].flags.concurrentwrite)
                    concurrency = ACCESS_CONCURRENCY_CONCURRENT;
            }

            if (acs_list[i].flags.noncoherent)
                mode = (access_mode_t) (mode | ACCESS_MODE_V);

            if (acs_list[i].flags.nostorage)
                LOGGER_FATAL("XKRT does not support the no-storage modifier yet");

            new (accesses + i) access_t(task, ptr, n, sizeof_type, mode, concurrency, scope);
        }
    }
}

extern "C"
void
__kmp_omp_task_deps_to_xkomp_accesses(
    task_t * task,
    access_t * accesses,
    kmp_depend_info_t * dep_list,
    kmp_int32 ndeps
) {
    for (kmp_int32 i = 0; i < ndeps ; ++i)
    {
        const void * ptr = (const void *) dep_list[i].base_addr;
        if (ptr)
        {
            access_mode_t           mode        = ACCESS_MODE_V;
            access_concurrency_t    concurrency = ACCESS_CONCURRENCY_SEQUENTIAL;
            access_scope_t          scope       = ACCESS_SCOPE_NONUNIFIED;

            if (dep_list[i].flags.in)
            {
                mode = (access_mode_t) (mode | ACCESS_MODE_R);
                concurrency = ACCESS_CONCURRENCY_SEQUENTIAL;
            }
            if (dep_list[i].flags.out)
            {
                mode = (access_mode_t) (mode | ACCESS_MODE_W);
                concurrency = ACCESS_CONCURRENCY_SEQUENTIAL;
            }
            if (dep_list[i].flags.mtx)
            {
                mode = (access_mode_t) (mode | ACCESS_MODE_W);
                concurrency = ACCESS_CONCURRENCY_COMMUTATIVE;
            }
            if (dep_list[i].flags.set)
            {
                mode = (access_mode_t) (mode | ACCESS_MODE_W);
                concurrency = ACCESS_CONCURRENCY_CONCURRENT;
            }
            if (dep_list[i].flags.all)
            {
                LOGGER_FATAL("Not implemented");
            }

            // new (accesses + access_idx++) access_t(task, ptr, 1, 1, mode, concurrency, scope);
            new (accesses + i) access_t(task, ptr, mode, concurrency, scope);
        }
    }
}

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param ktask task thunk allocated by __kmp_omp_task_alloc() for the ''new
task''
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

@return Returns either TASK_CURRENT_NOT_QUEUED if the current task was not
suspended and queued, or TASK_CURRENT_QUEUED if it was suspended and queued

Schedule a non-thread-switchable task with dependences for execution
*/
extern "C"
kmp_int32
__kmpc_omp_task_with_deps_v2(
    ident_t * loc_ref,
    kmp_int32 gtid,
    kmp_task_t * ktask,
    kmp_int32 ndeps,
    kmp_depend_info_t * dep_list,
    kmp_int32 ndeps_noalias,
    kmp_depend_info_t * noalias_dep_list,
    kmp_int32 nacs,
    kmp_access_info_t * acs_list
) {
    xkomp_t * xkomp = xkomp_get();
    assert(xkomp);

    // total accesses = access-clause (XKOMP ext) + aliased deps + non-aliased deps.
    // Stock LLVM splits dependences into an aliasing and a non-aliasing list;
    // both must be honoured (XKOMP previously ignored the non-aliasing one).
    const kmp_int32 naccesses = nacs + ndeps + ndeps_noalias;

    task_t * task = task_from_ktask(ktask);

    if (naccesses)
    {
        task_acs_info_t * acs = TASK_ACS_INFO(task);

        // The task was sized at allocation time: exactly (ndeps+nacs) for the
        // patched ABI, or XKOMP_FIXED_ACCESSES reserved (and padded) slots for the
        // stock ABI.  More accesses than were reserved cannot fit in the inline
        // array, so reject loudly.
        if (naccesses > acs->ac)
            LOGGER_FATAL("OpenMP task needs %d dependences/accesses but the stock "
                         "LLVM/clang ABI only supports %d per task; use the patched "
                         "LLVM/clang for XKOMP.",
                         (int) naccesses, (int) acs->ac);

        access_t * accesses = TASK_ACCESSES(task);

        /* access clause (XKOMP extension) */
        __kmp_omp_task_acs_to_xkomp_accesses(task, accesses, acs_list, nacs);

        /* depend clause: aliasing deps, then non-aliasing deps */
        __kmp_omp_task_deps_to_xkomp_accesses(task, accesses + nacs, dep_list, ndeps);
        __kmp_omp_task_deps_to_xkomp_accesses(task, accesses + nacs + ndeps, noalias_dep_list, ndeps_noalias);

        // resolve only the real accesses; the trailing reserved slots keep the
        // inert padding installed at allocation time.
        xkomp->runtime.task_accesses_resolve(accesses, naccesses);
    }

    xkomp->runtime.task_commit(task);

    // undeferred task (`if(0)` with a depend clause): suspend the encountering
    // thread until it completes. It carries its own dependences and was committed
    // on the normal path, so it is ordered after its predecessors and any thread
    // may execute it -- no separate empty task / taskwait_deps is needed.
    if (task->flags & TASK_FLAG_UNDEFERABLE)
        xkomp->runtime.task_wait(task);

    # define TASK_CURRENT_NOT_QUEUED    0
    # define TASK_CURRENT_QUEUED        1
    return TASK_CURRENT_QUEUED;
}



extern "C"
kmp_int32
__kmpc_omp_task_with_deps(
    ident_t * loc_ref,
    kmp_int32 gtid,
    kmp_task_t * ktask,
    kmp_int32 ndeps,
    kmp_depend_info_t * dep_list,
    kmp_int32 ndeps_noalias,
    kmp_depend_info_t * noalias_dep_list
) {
    return __kmpc_omp_task_with_deps_v2(loc_ref, gtid, ktask, ndeps, dep_list, ndeps_noalias, noalias_dep_list, 0, NULL);
}

extern "C"
kmp_int32
__kmpc_omp_taskwait(
    ident_t * loc_ref,
    kmp_int32 gtid
) {
    xkomp_t * xkomp = xkomp_get();
    xkomp->runtime.task_wait();
    return 0;
}

extern "C"
void
__kmpc_omp_wait_deps(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 ndeps,
        kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
        kmp_depend_info_t *noalias_dep_list) {
}

// Wait until a set of dependences is satisfied, WITHOUT running a user task body
// (emitted by the compiler for `#pragma omp taskwait depend(...)`).
//
// We realize it as an EMPTY undeferred task carrying exactly those dependences:
// committed on the normal path, it becomes ready only once its predecessors (the
// tasks those deps refer to) have completed, and we then block on that specific
// task -- so this waits for precisely those dependences (work-stealing
// meanwhile, no over-synchronization).
static void
xkomp_taskwait_deps(
    kmp_int32 ndeps,          kmp_depend_info_t * dep_list,
    kmp_int32 ndeps_noalias,  kmp_depend_info_t * noalias_dep_list,
    kmp_int32 nacs,           kmp_access_info_t * acs_list
) {
    const kmp_int32 naccesses = ndeps + ndeps_noalias + nacs;
    if (naccesses == 0)
        return;

    xkomp_t * xkomp = xkomp_get();
    assert(xkomp);

    task_t * task = xkomp->runtime.task_instanciate<TASK_FLAG_ACCESSES>(
        XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID,
        (task_access_counter_t) naccesses,
        // fill the empty task's accesses from the depend/access clauses
        [=](task_t * t, access_t * accesses) {
            __kmp_omp_task_acs_to_xkomp_accesses (t, accesses,                    acs_list,         nacs);
            __kmp_omp_task_deps_to_xkomp_accesses(t, accesses + nacs,             dep_list,         ndeps);
            __kmp_omp_task_deps_to_xkomp_accesses(t, accesses + nacs + ndeps,     noalias_dep_list, ndeps_noalias);
        },
        // not moldable
        nullptr,
        // empty body: nothing to do, it exists only to carry the dependences
        [](runtime_t *, device_t *, task_t *) { }
    );

    xkomp->runtime.task_commit(task);
    xkomp->runtime.task_wait(task);
}

extern "C"
void
__kmpc_omp_taskwait_deps_51(ident_t * loc_ref, kmp_int32 gtid,
        kmp_int32 ndeps, kmp_depend_info_t * dep_list,
        kmp_int32 ndeps_noalias, kmp_depend_info_t * noalias_dep_list,
        kmp_int32 has_no_wait) {
    (void) loc_ref; (void) gtid; (void) has_no_wait;
    xkomp_taskwait_deps(ndeps, dep_list, ndeps_noalias, noalias_dep_list, 0, NULL);
}

extern "C"
void
__kmpc_omp_taskwait_deps_51_v2(ident_t * loc_ref, kmp_int32 gtid,
        kmp_int32 ndeps, kmp_depend_info_t * dep_list,
        kmp_int32 ndeps_noalias, kmp_depend_info_t * noalias_dep_list,
        kmp_int32 has_no_wait,
        kmp_int32 nacs, kmp_access_info_t * acs_list) {
    (void) loc_ref; (void) gtid; (void) has_no_wait;
    xkomp_taskwait_deps(ndeps, dep_list, ndeps_noalias, noalias_dep_list, nacs, acs_list);
}

void
xkomp_task_register_formats_kmp_task(xkomp_t * xkomp)
{
    task_format_t format;
    memset(&format, 0, sizeof(task_format_t));
    for (int i = 0 ; i < XKRT_TASK_FORMAT_TARGET_MAX ; ++i)
        format.f[i] = (task_format_func_t) body_omp_task_target;
    format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) body_omp_task;
    snprintf(format.label, sizeof(format.label), "omp-task");
    xkomp->formats.kmp.host = xkomp->runtime.task_format_create(&format);
}

extern "C"
kmp_int32
__kmpc_omp_taskyield(ident_t *loc_ref, kmp_int32 gtid, int end_part)
{
    static bool warned = false;
    if (!warned) { LOGGER_WARN("taskyield is currently a noop");  warned = true; }
    // TODO: implement taskyield in XKRT, and tiedness
    return 0;
}

// Open a taskgroup region. XKRT binds every task created from now on (and all
// their descendants, across workers) to this group, and counts them; see
// runtime_t::taskgroup_begin.
extern "C"
void
__kmpc_taskgroup(ident_t *loc, int gtid)
{
    (void) loc; (void) gtid;
    xkomp_t * xkomp = xkomp_get();
    xkomp->runtime.taskgroup_begin();
}

// Close a taskgroup region: deep-sync on the group (child tasks AND all their
// descendants), not a plain taskwait. Decrement happens at true task completion
// in XKRT, so device/detachable tasks are waited for correctly too.
extern "C"
void
__kmpc_end_taskgroup(ident_t *loc, int gtid)
{
    (void) loc; (void) gtid;
    xkomp_t * xkomp = xkomp_get();
    xkomp->runtime.taskgroup_end();
}
