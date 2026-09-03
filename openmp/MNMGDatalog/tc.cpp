/*
 * tc.cpp - Datalog Transitive Closure (TC) as OpenMP target tasks / taskgraph.
 *
 * This is an OpenMP port of MNMGDatalog-reference/tc_benchmark (CUDA). It computes
 * the semi-naive TC fixpoint
 *
 *     path(a, c) :- path(a, b), edge(b, c).
 *
 * over fixed pre-allocated device buffers, with all per-iteration SIZES resident
 * in device memory (d_frontier_size / the device copy of new_count). The two
 * size-driven kernels are grid-stride loops over a FIXED host-constant worker
 * count (ctx.n_workers) that read those sizes from device memory in their body,
 * so the per-iteration kernel sequence is byte-identical every round -- which is
 * exactly what lets one recorded task graph be replayed. Using a size as the
 * OpenMP loop bound would not work: clang evaluates a target loop's trip count on
 * the host (see the comment above the fixpoint kernels).
 *
 * One source, several backends, chosen at compile time by the shared toggles in
 * ../tasking.h (see ../common.mk). The two CUDA "versions" map to toggles:
 *
 *     USE_TASKGRAPH == 0  ->  "baseline"  (CUDA v1_baseline): the fixpoint runs
 *                             as a host while-loop that spawns the kernels every
 *                             iteration (plain omp target tasks).
 *     USE_TASKGRAPH == 1  ->  "cudagraph" (CUDA v2_cudagraph): the loop-invariant
 *                             per-iteration kernel sequence is recorded once with
 *                             TASKGRAPH_BEGIN/END and replayed each iteration; the
 *                             host still reads new_count back to test convergence.
 *
 * USE_TARGET selects GPU offload (1) vs host CPU tasks (0); USE_SYNC selects a
 * blocking (synchronous) schedule. v3_conditional (on-GPU conditional loop) is
 * intentionally NOT ported: CGIR/XKOMP has no conditional-node support yet.
 *
 * Metrics: the measured unit is one fixpoint ROUND, reported as round 0 (record) /
 * round 1 (1st replay) / rounds 2..N-1 (avg, stddev) like the Krylov drivers'
 * iterations, plus the MNMGDatalog paper's end-to-end total and per-phase
 * breakdown (file IO, H2D, setup, compute, D2H). A few untimed, ungraphed warm-up
 * rounds precede round 0. See the "Timing model and report" section.
 *
 * Memory model: the large buffers are device-only (omp_target_alloc) and every
 * target construct reaches them via is_device_ptr(...) -- the direct analog of
 * the reference's cudaMalloc, with no host mirror of the (possibly multi-GB)
 * result set. The ONE exception is new_count, the scalar the host must read every
 * iteration: it is pinned host memory (shared ../alloc.h host_alloc) mapped with
 * map(alloc:)/map(present:) like the Krylov scalars, refreshed by an in-graph
 * async D2H (k_writeback). The remaining scalars are only read once, after the
 * fixpoint, via omp_target_memcpy. On the CPU backend every pointer is plain
 * host memory and the kernels become host tasks.
 *
 * Atomics: the open-addressing hash set / edge table are built with a
 * compare-and-swap (#pragma omp atomic compare capture, OpenMP 5.1) and the
 * append counters with fetch-add (#pragma omp atomic capture) -- one portable
 * code path for host and device. NOTE: on-device `omp atomic compare capture`
 * codegen must be confirmed on your XKOMP/clang + NVPTX toolchain.
 */
#include "tasking.h"
#include "alloc.h"

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <sys/stat.h>

typedef unsigned long long u64;

/* Empty-slot marker: 0xFF bytes -> every int becomes -1, every u64 becomes this. */
#define TC_EMPTY64 0xFFFFFFFFFFFFFFFFULL

/* Number of grid-stride workers of the two size-driven fixpoint kernels.
 *
 * An OpenMP loop bound is NOT a device-only expression: for a combined
 * `target teams distribute parallel for`, clang evaluates the trip count on the
 * HOST, inside the target task, to fill the LoopTripCount of __tgt_target_kernel
 * (CGStmtOpenMP.cpp SizeEmitter -> CGOpenMPRuntime::emitTargetNumIterationsCall,
 * reached from emitTargetCall). The bound must therefore be a loop-invariant
 * HOST scalar; the device-resident size is read inside the body by a grid-stride
 * loop instead -- the direct analog of the reference's fixed <<<32*numSM, 512>>>
 * launch geometry with `int n = *frontier_size;` read in the kernel.
 *
 * Default ~= 32*132*512 (an H100 at the reference's occupancy); override at build
 * time with -DTC_WORKERS_DEFAULT=<n> or at run time with TC_WORKERS=<n>. It is
 * resolved once in tc_setup, so it stays constant across taskgraph replays. */
#ifndef TC_WORKERS_DEFAULT
# define TC_WORKERS_DEFAULT (1L << 21)
#endif

/* Edge slot for the open-addressing edge table (key = source, value = dest). */
struct Entity { int key; int value; };

/* Version string (matches the reference CSV names). */
#if USE_SYNC
# define TC_VERSION "synchronous"
#elif USE_TASKGRAPH
# define TC_VERSION "cudagraph"
#else
# define TC_VERSION "baseline"
#endif

/* ------------------------------------------------------------------------- */
/* Device-callable helpers (hashing + atomics). declare target only on GPU.   */
/* ------------------------------------------------------------------------- */
#if USE_TARGET
# pragma omp declare target
#endif

/* Murmur3 finalizer (mirrors get_position in the codebase). */
static inline int tc_get_position(int key, int cap)
{
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key & (cap - 1);
}

/* splitmix64 finalizer -> slot in [0, cap). */
static inline u64 tc_hash64(u64 k, long cap)
{
    k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
    k ^= k >> 27; k *= 0x94d049bb133111ebULL;
    k ^= k >> 31;
    return k & (u64)(cap - 1);
}

static inline u64 tc_pack(int a, int b)
{
    return (((u64)(unsigned)a) << 32) | (unsigned)b;
}

/* CAS: atomically set *addr = desired iff *addr == expected; return old value. */
static inline u64 tc_cas_u64(u64 *addr, u64 expected, u64 desired)
{
    u64 old;
    #pragma omp atomic compare capture
    { old = *addr; if (*addr == expected) *addr = desired; }
    return old;
}

static inline int tc_cas_i32(int *addr, int expected, int desired)
{
    int old;
    #pragma omp atomic compare capture
    { old = *addr; if (*addr == expected) *addr = desired; }
    return old;
}

/* Fetch-add: return the old value, then *addr += v (the append/index pattern). */
static inline int tc_fetch_add_i32(int *addr, int v)
{
    int old;
    #pragma omp atomic capture
    { old = *addr; *addr += v; }
    return old;
}

static inline u64 tc_fetch_add_u64(u64 *addr, u64 v)
{
    u64 old;
    #pragma omp atomic capture
    { old = *addr; *addr += v; }
    return old;
}

/* Insert key into the open-addressing result set. Returns true iff newly
 * inserted (won an empty slot) -- i.e. this is a genuinely new fact. Bounded by
 * capacity: if the set is full we set *overflow and return false (no hang). */
static inline bool tc_set_insert(u64 *set, long cap, u64 key, int *overflow)
{
    u64 mask = (u64)(cap - 1);
    u64 pos  = tc_hash64(key, cap);
    for (long probes = 0; probes < cap; probes++) {
        u64 old = tc_cas_u64(&set[pos], TC_EMPTY64, key);
        if (old == TC_EMPTY64) return true;   /* won the slot  -> new fact  */
        if (old == key)        return false;  /* already present -> duplicate */
        pos = (pos + 1) & mask;
    }
    *overflow = 1;                             /* set is full (benign store) */
    return false;
}

/* Per-element bodies, shared by the GPU (target) and CPU (host task) loops. */

/* Build one edge into the edge table: claim the first empty slot (duplicate
 * edges land in separate slots; only produces duplicate candidates the result
 * set dedups). */
static inline void tc_build_one(int i, const int *edges, Entity *table, int cap)
{
    int key = edges[i * 2], value = edges[i * 2 + 1];
    int pos = tc_get_position(key, cap);
    while (true) {
        int existing = tc_cas_i32(&table[pos].key, -1, key);
        if (existing == -1) { table[pos].value = value; break; }
        pos = (pos + 1) & (cap - 1);
    }
}

/* Seed the fixpoint with the base facts: path(a,b) :- edge(a,b), deduped. */
static inline void tc_init_base_one(int i, const int *edges, u64 *set, long rcap,
                                    u64 *frontier, int fcap, int *fsize,
                                    u64 *rcount, int *overflow)
{
    int a = edges[i * 2], b = edges[i * 2 + 1];
    u64 p = tc_pack(a, b);
    if (tc_set_insert(set, rcap, p, overflow)) {
        int w = tc_fetch_add_i32(fsize, 1);
        if (w < fcap) frontier[w] = p; else *overflow = 1;
        tc_fetch_add_u64(rcount, 1ULL);
    }
}

/* Expand one frontier fact path(a,b): for every edge(b,c), try path(a,c). */
static inline void tc_expand_one(int i, const Entity *edge_table, int edge_cap,
                                 const u64 *frontier, u64 *set, long rcap,
                                 u64 *new_frontier, int nfcap, int *new_count,
                                 u64 *rcount, int *overflow)
{
    u64 f = frontier[i];
    int a = (int)(f >> 32);
    int b = (int)(f & 0xffffffffULL);
    int pos = tc_get_position(b, edge_cap);
    while (true) {
        int k = edge_table[pos].key;
        if (k == b) {
            int c = edge_table[pos].value;
            u64 np = tc_pack(a, c);
            if (tc_set_insert(set, rcap, np, overflow)) {
                int w = tc_fetch_add_i32(new_count, 1);
                if (w < nfcap) new_frontier[w] = np; else *overflow = 1;
                tc_fetch_add_u64(rcount, 1ULL);
            }
        } else if (k == -1) {
            break;
        }
        pos = (pos + 1) & (edge_cap - 1);
    }
}

/* Stream-compact one result-set slot into the dense output array. */
static inline void tc_compact_one(long i, const u64 *set, u64 *out, u64 *out_count)
{
    u64 s = set[i];
    if (s != TC_EMPTY64) {
        u64 w = tc_fetch_add_u64(out_count, 1ULL);
        out[w] = s;
    }
}

#if USE_TARGET
# pragma omp end declare target
#endif

/* ------------------------------------------------------------------------- */
/* Host <-> device memory abstraction.                                        */
/*   GPU (USE_TARGET==1): omp_target_alloc + omp_target_memcpy (device-only). */
/*   CPU (USE_TARGET==0): plain malloc / memcpy (host memory).                */
/* ------------------------------------------------------------------------- */
#if USE_TARGET
static int g_dev  = 0;
static int g_host = 0;
static inline void *dalloc(size_t b)            { return omp_target_alloc(b, g_dev); }
static inline void  dfree(void *p)              { if (p) omp_target_free(p, g_dev); }
static inline void  to_dev(void *d, const void *s, size_t b)   { omp_target_memcpy(d, (void *)s, b, 0, 0, g_dev, g_host); }
static inline void  from_dev(void *d, const void *s, size_t b) { omp_target_memcpy(d, (void *)s, b, 0, 0, g_host, g_dev); }
#else
static inline void *dalloc(size_t b)            { return malloc(b); }
static inline void  dfree(void *p)              { free(p); }
static inline void  to_dev(void *d, const void *s, size_t b)   { memcpy(d, s, b); }
static inline void  from_dev(void *d, const void *s, size_t b) { memcpy(d, s, b); }
#endif

/* Byte-wise fill of a device (or host) buffer -- mirrors cudaMemset. */
static void dmemset(void *p, int byte, size_t bytes)
{
#if USE_TARGET
    char *c = (char *)p;
    #pragma omp target teams distribute parallel for is_device_ptr(c)
    for (long long i = 0; i < (long long)bytes; i++) c[i] = (char)byte;
#else
    memset(p, byte, bytes);
#endif
}

static inline int tc_read_i32(int *dev) { int h;  from_dev(&h, dev, sizeof(int)); return h; }
static inline u64 tc_read_u64(u64 *dev) { u64 h;  from_dev(&h, dev, sizeof(u64)); return h; }

/* Abort on a failed allocation instead of letting the NULL reach a kernel. On the
 * GPU backend this also catches an OpenMP runtime whose omp_target_alloc is a
 * stub: every buffer here is device-only and dereferenced by device code, so a
 * silent NULL would only surface much later as an unattributable fault. */
#if USE_TARGET
# define TC_ALLOC_HINT "omp_target_alloc failed or is not implemented by the OpenMP runtime in use"
#else
# define TC_ALLOC_HINT "out of host memory"
#endif
static void *tc_dcheck(void *p, const char *what, size_t bytes)
{
    if (!p) {
        fprintf(stderr, "ERROR: allocation of %s (%zu bytes) returned NULL.\n"
                        "       %s.\n", what, bytes, TC_ALLOC_HINT);
        exit(2);
    }
    return p;
}

/* ------------------------------------------------------------------------- */
/* Context.                                                                   */
/* ------------------------------------------------------------------------- */
struct TCContext {
    int  n_edges   = 0;
    int  input_rows = 0;
    int *d_edges   = nullptr;

    Entity *d_edge_table = nullptr;
    int  edge_cap  = 0;

    u64 *d_result_set = nullptr;
    long result_cap = 0;

    u64 *d_frontier     = nullptr;
    u64 *d_new_frontier = nullptr;
    int  frontier_cap   = 0;

    /* Fixed grid-stride worker count of k_expand / k_promote (see
     * TC_WORKERS_DEFAULT). Resolved once in tc_setup and never changed, so the
     * recorded kernel launch is identical on every replay. 1 on the CPU backend,
     * where the grid-stride nest collapses to the plain loop. */
    int  n_workers      = 1;

    int *d_frontier_size = nullptr;
    /* new_count is the ONE scalar the host reads every iteration (the fixpoint
     * convergence test), so unlike the other buffers it is pinned HOST memory
     * (host_alloc) mapped onto the device with map(alloc:)/map(present:), as the
     * Krylov solvers do for their scalars. k_writeback then refreshes the host
     * copy with an in-graph async D2H, so no blocking omp_target_memcpy is
     * needed inside the timed loop. */
    int *new_count       = nullptr;
    u64 *d_result_count  = nullptr;
    int *d_overflow      = nullptr;

    double t_fileio = 0.0, t_h2d = 0.0, t_setup = 0.0, peak_mem_mb = 0.0;
};

/* ------------------------------------------------------------------------- */
/* Setup / finalize kernels (blocking; NOT part of the recorded task graph).  */
/* ------------------------------------------------------------------------- */
static void build_edges(TCContext &ctx)
{
    int n = ctx.n_edges, cap = ctx.edge_cap;
    int *edges = ctx.d_edges; Entity *table = ctx.d_edge_table;
#if USE_TARGET
    #pragma omp target teams distribute parallel for is_device_ptr(edges, table)
    for (int i = 0; i < n; i++) tc_build_one(i, edges, table, cap);
#else
    #pragma omp parallel for
    for (int i = 0; i < n; i++) tc_build_one(i, edges, table, cap);
#endif
}

static void init_base(TCContext &ctx)
{
    int n = ctx.n_edges, fcap = ctx.frontier_cap; long rcap = ctx.result_cap;
    int *edges = ctx.d_edges; u64 *set = ctx.d_result_set;
    u64 *fr = ctx.d_frontier; int *fs = ctx.d_frontier_size;
    u64 *rc = ctx.d_result_count; int *ov = ctx.d_overflow;
#if USE_TARGET
    #pragma omp target teams distribute parallel for is_device_ptr(edges, set, fr, fs, rc, ov)
    for (int i = 0; i < n; i++) tc_init_base_one(i, edges, set, rcap, fr, fcap, fs, rc, ov);
#else
    #pragma omp parallel for
    for (int i = 0; i < n; i++) tc_init_base_one(i, edges, set, rcap, fr, fcap, fs, rc, ov);
#endif
}

static void compact(TCContext &ctx, u64 *out, u64 *out_count)
{
    u64 *set = ctx.d_result_set; long cap = ctx.result_cap;
#if USE_TARGET
    #pragma omp target teams distribute parallel for is_device_ptr(set, out, out_count)
    for (long i = 0; i < cap; i++) tc_compact_one(i, set, out, out_count);
#else
    #pragma omp parallel for
    for (long i = 0; i < cap; i++) tc_compact_one(i, set, out, out_count);
#endif
}

/* ------------------------------------------------------------------------- */
/* Fixpoint kernels (the recorded/replayed per-iteration task sequence).      */
/*                                                                            */
/* Each is emitted as the SAME combined construct the Krylov solvers use --   */
/* OMP_TILE (GPU: one `omp target teams distribute parallel for`; CPU: one    */
/* `omp task`) for the data-parallel loops, and OMP_TARGET_TASK for the two   */
/* single-statement device ops. A combined construct is recorded as ONE kernel */
/* command: pragma_omp_taskgraph() runs the region body only on the first     */
/* (record) pass and re-submits those commands on replay.                     */
/*                                                                            */
/* The per-iteration sizes therefore CANNOT be OpenMP loop bounds. An OpenMP  */
/* loop bound is not a device-only expression: for a combined                 */
/* `target teams distribute parallel for` clang evaluates the trip count on   */
/* the HOST, inside the target task, to fill the LoopTripCount argument of    */
/* __tgt_target_kernel (clang: CGStmtOpenMP.cpp SizeEmitter ->                */
/* CGOpenMPRuntime::emitTargetNumIterationsCall, called from emitTargetCall). */
/* Writing `for (i = 0; i < fs[0]; ...)` over an is_device_ptr buffer makes   */
/* the host load a device address -> SIGSEGV; and even for a host-resident    */
/* scalar the host would bake a stale value into the recorded launch.         */
/*                                                                            */
/* So both size-driven kernels are GRID-STRIDE loops: the OpenMP loop bound is */
/* the loop-invariant host scalar ctx.n_workers (fixed once in tc_setup, hence */
/* an identical launch on every replay) and the size is read from device       */
/* memory INSIDE the body, once per worker. This is exactly the reference's    */
/* fixed <<<32*numSM, 512>>> geometry with `int n = *frontier_size;` read in   */
/* the kernel (MNMGDatalog-reference/tc_benchmark/common/tc_core.cuh,          */
/* tc_expand / tc_promote), so each replay sees the current frontier size.     */
/* On the CPU backend n_workers == 1 and the nest collapses to the plain loop. */
/* ------------------------------------------------------------------------- */
static void k_reset(TCContext &ctx)
{
    int *nc = ctx.new_count;
#if USE_TARGET
    OMP_TARGET_TASK(DEPEND(out, nc[0]) MAP(present: nc[0:1]))
    { nc[0] = 0; }
#else
    OMP_TASK(DEFAULT_NONE firstprivate(nc) DEPEND(out, nc[0]))
    { nc[0] = 0; }
#endif
}

static void k_expand(TCContext &ctx)
{
    Entity *et = ctx.d_edge_table; int ec = ctx.edge_cap;
    u64 *fr = ctx.d_frontier; int *fs = ctx.d_frontier_size;
    u64 *rs = ctx.d_result_set; long rc = ctx.result_cap;
    u64 *nf = ctx.d_new_frontier; int nfc = ctx.frontier_cap;
    int *ncnt = ctx.new_count; u64 *rcnt = ctx.d_result_count; int *ov = ctx.d_overflow;
    int nw = ctx.n_workers;
    /* GPU: is_device_ptr (mp slot) carries the device-only buffers and
     * map(present:) the pinned-host new_count; CPU: default(none) firstprivate
     * (fp slot) captures the pointers/scalars. The OpenMP bound is the host
     * constant nw; the frontier size fs[0] is read on the DEVICE by every worker,
     * so replay uses the current frontier size. */
    OMP_TILE(DEPEND(in, fs[0], fr[0]) DEPEND(inout, rs[0], ncnt[0], rcnt[0], ov[0]) DEPEND(out, nf[0]),
             is_device_ptr(et, fr, fs, rs, nf, rcnt, ov) MAP(present: ncnt[0:1]),
             DEFAULT_NONE firstprivate(et, ec, fr, fs, rs, rc, nf, nfc, ncnt, rcnt, ov, nw))
    for (int t = 0; t < nw; t++) {
        const int n = fs[0];
        for (int i = t; i < n; i += nw)
            tc_expand_one(i, et, ec, fr, rs, rc, nf, nfc, ncnt, rcnt, ov);
    }
}

static void k_promote(TCContext &ctx)
{
    u64 *fr = ctx.d_frontier; u64 *nf = ctx.d_new_frontier; int *nc = ctx.new_count;
    int nw = ctx.n_workers;
    /* Same grid-stride shape as k_expand: nc[0] is the DEVICE copy of new_count
     * (map(present:)), read inside the body. Reading it as the OpenMP bound would
     * take the host copy, which still holds the PREVIOUS round's count. */
    OMP_TILE(DEPEND(in, nc[0], nf[0]) DEPEND(out, fr[0]),
             is_device_ptr(fr, nf) MAP(present: nc[0:1]),
             DEFAULT_NONE firstprivate(fr, nf, nc, nw))
    for (int t = 0; t < nw; t++) {
        const int n = nc[0];
        for (int i = t; i < n; i += nw) fr[i] = nf[i];
    }
}

static void k_set_sizes(TCContext &ctx)
{
    int *fs = ctx.d_frontier_size; int *nc = ctx.new_count;
#if USE_TARGET
    OMP_TARGET_TASK(DEPEND(in, nc[0]) DEPEND(out, fs[0]) is_device_ptr(fs) MAP(present: nc[0:1]))
    { fs[0] = nc[0]; }
#else
    OMP_TASK(DEFAULT_NONE firstprivate(fs, nc) DEPEND(in, nc[0]) DEPEND(out, fs[0]))
    { fs[0] = nc[0]; }
#endif
}

/* Refresh the HOST copy of new_count so the fixpoint loop can test convergence.
 * This is an async D2H recorded INSIDE the taskgraph (depend-ordered after the
 * kernels that update it), i.e. one more replayed command -- the same shape as
 * the Krylov residual read-back (cg.cpp) and xkomp's taskgraph_dot_target test.
 * On the host backend it vanishes: new_count already IS the host memory. */
static void k_writeback(TCContext &ctx)
{
    int *nc = ctx.new_count;
    OMP_TARGET_UPDATE(from(nc[0:1]) NOWAIT DEPEND(in, nc[0]))
}

/* The loop-invariant per-round kernel sequence -- the body that is recorded once
 * and replayed. Byte-identical every round: same kernels, same buffers, same
 * launch geometry; only the device-resident sizes read inside the kernels move. */
static inline void tc_round(TCContext &ctx)
{
    k_reset(ctx);
    k_expand(ctx);
    k_promote(ctx);
    k_set_sizes(ctx);
    k_writeback(ctx);
}

/* Per-round wall times. The number of rounds is data-dependent (the fixpoint runs
 * until no new fact is produced), so the vector grows as the solve proceeds. */
typedef struct { double *v; int n, cap; } TCTimes;

static void tc_times_push(TCTimes *t, double x)
{
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 128;
        t->v   = (double *)realloc(t->v, (size_t)t->cap * sizeof(double));
        if (!t->v) { fprintf(stderr, "ERROR: out of memory for round timings.\n"); exit(2); }
    }
    t->v[t->n++] = x;
}

/* Untimed warm-up: run the per-round kernel sequence a few times WITHOUT the
 * taskgraph wrapper. This pays OpenMP team creation, device bring-up (context,
 * module load, kernel JIT) and first touch of the buffers up front, but does NOT
 * consume the recording pass -- XKOMP records a task only when its parent carries
 * TASK_FLAG_GRAPH_RECORDING, which only xkomp_taskgraph_begin sets. Round 0 of
 * the measured fixpoint is therefore still the round that records the graph. */
static void tc_warmup(TCContext &ctx, int nrounds)
{
    for (int i = 0; i < nrounds; i++)
    {
        tc_round(ctx);
        #pragma omp taskwait
    }
}

/* The fixpoint solve; returns the number of rounds and fills `times` with the
 * per-round wall time. MUST be called from inside a single region (see the
 * enclosing `omp parallel/single` in main): the round body is recorded on round 0
 * and REPLAYED on every later round -- the direct analog of CUDA v2_cudagraph's
 * build-once / replay. XKOMP records on the first taskgraph entry (rc == 1),
 * builds and optimizes the command graph on the second (rc == 2) and replays
 * afterwards, so round 0 measures the record and round 1 the graph build + first
 * replay -- exactly why they are reported separately from the steady state.
 *
 * Convergence is tested on the pinned-host new_count, refreshed by k_writeback's
 * in-graph async D2H -- the same host round-trip the CUDA v2_cudagraph pays,
 * minus the blocking copy.
 *
 * The taskwait is required: it is what makes new_count[0] complete before the
 * host reads it (and what bounds each round's time). Under USE_TASKGRAPH the
 * region is already effectively blocking (xkomp_taskgraph_end does an implicit
 * taskwait while recording, and replay is synchronous), so it costs nothing
 * there; but with USE_TASKGRAPH=0 the TASKGRAPH_BEGIN/END macros vanish and the
 * nowait tasks would still be in flight, so without it the loop would read a
 * stale count and stop early. */
static int tc_run_fixpoint(TCContext &ctx, TCTimes *times)
{
    int rounds = 0;
    int *nc = ctx.new_count;
    nc[0] = 1;                      /* prime: enter the loop (host copy only) */
    for (rounds = 0; nc[0] > 0; ++rounds)
    {
        const double r0 = omp_get_wtime();

        TASKGRAPH_BEGIN
        {
            tc_round(ctx);
        }
        TASKGRAPH_END

        #pragma omp taskwait

        tc_times_push(times, omp_get_wtime() - r0);
    }
    return rounds;
}

/* ------------------------------------------------------------------------- */
/* Host helpers: binary IO, sizing, stats.                                    */
/* ------------------------------------------------------------------------- */
static double tc_now()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int *tc_read_bin(const char *path, int *n_edges_out)
{
    struct stat st{};
    if (stat(path, &st) != 0) { fprintf(stderr, "Cannot stat input file %s\n", path); exit(EXIT_FAILURE); }
    long n = st.st_size / (long)(sizeof(int) * 2);
    int *data = (int *)malloc((size_t)n * 2 * sizeof(int));
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(EXIT_FAILURE); }
    size_t got = fread(data, sizeof(int), (size_t)n * 2, fp);
    fclose(fp);
    if (got != (size_t)(n * 2)) { fprintf(stderr, "Short read on %s\n", path); exit(EXIT_FAILURE); }
    *n_edges_out = (int)n;
    return data;
}

static long tc_next_pow2(long v) { long p = 1; while (p < v) p <<= 1; return p; }

static void tc_mean_std(const double *v, int n, double *mean, double *sd)
{
    if (n <= 0) { *mean = 0.0; *sd = 0.0; return; }
    double m = 0.0; for (int i = 0; i < n; i++) m += v[i]; m /= n;
    double s = 0.0; for (int i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m);
    *mean = m; *sd = (n > 1) ? std::sqrt(s / (n - 1)) : 0.0;
}

/* ------------------------------------------------------------------------- */
/* Setup / reset / teardown.                                                  */
/* ------------------------------------------------------------------------- */
static void tc_setup(TCContext &ctx, const char *input_file, long capacity_mult,
                     long frontier_slots)
{
#if USE_TARGET
    g_dev  = omp_get_default_device();
    g_host = omp_get_initial_device();
    #pragma omp target        /* untimed device/context warm-up */
    { }
#endif

    double t0 = tc_now();
    int *edges_host = tc_read_bin(input_file, &ctx.n_edges);
    ctx.input_rows = ctx.n_edges;
    ctx.t_fileio = tc_now() - t0;

    t0 = tc_now();
    size_t nb = (size_t)ctx.n_edges * 2 * sizeof(int);
    ctx.d_edges = (int *)tc_dcheck(dalloc(nb), "edges", nb);
    to_dev(ctx.d_edges, edges_host, nb);
    ctx.t_h2d = tc_now() - t0;
    free(edges_host);

    t0 = tc_now();
    ctx.edge_cap = (int)tc_next_pow2((long)std::ceil(ctx.n_edges / 0.6));
    if (ctx.edge_cap < 2) ctx.edge_cap = 2;
    nb = (size_t)ctx.edge_cap * sizeof(Entity);
    ctx.d_edge_table = (Entity *)tc_dcheck(dalloc(nb), "edge table", nb);
    dmemset(ctx.d_edge_table, 0xFF, nb);
    build_edges(ctx);

    long est = (long)ctx.n_edges * capacity_mult;
    if (est < 4096) est = 4096;
    ctx.result_cap = tc_next_pow2(est);

    long fcap = (frontier_slots > 0) ? tc_next_pow2(frontier_slots) : (1L << 28);
    if (fcap > ctx.result_cap) fcap = ctx.result_cap;
    ctx.frontier_cap = (int)fcap;

    /* Fixed grid-stride worker count of k_expand / k_promote. Resolved ONCE, here,
     * so the recorded task graph replays with an identical launch (see
     * TC_WORKERS_DEFAULT). Never more workers than the frontier can ever hold. */
#if USE_TARGET
    long nworkers = TC_WORKERS_DEFAULT;
    const char *wenv = getenv("TC_WORKERS");
    if (wenv && wenv[0]) nworkers = atol(wenv);
    if (nworkers < 1) nworkers = 1;
    if (nworkers > ctx.frontier_cap) nworkers = ctx.frontier_cap;
    ctx.n_workers = (int)nworkers;
#else
    ctx.n_workers = 1;              /* host task: the nest collapses to one loop */
#endif

    nb = (size_t)ctx.result_cap * sizeof(u64);
    ctx.d_result_set    = (u64 *)tc_dcheck(dalloc(nb), "result set", nb);
    dmemset(ctx.d_result_set, 0xFF, nb);
    nb = (size_t)ctx.frontier_cap * sizeof(u64);
    ctx.d_frontier      = (u64 *)tc_dcheck(dalloc(nb), "frontier", nb);
    ctx.d_new_frontier  = (u64 *)tc_dcheck(dalloc(nb), "new frontier", nb);
    ctx.d_frontier_size = (int *)tc_dcheck(dalloc(sizeof(int)),  "frontier size", sizeof(int));
    ctx.d_result_count  = (u64 *)tc_dcheck(dalloc(sizeof(u64)),  "result count", sizeof(u64));
    ctx.d_overflow      = (int *)tc_dcheck(dalloc(sizeof(int)),  "overflow flag", sizeof(int));

    /* new_count: pinned host memory (shared ../alloc.c) with a device copy
     * created here, so the kernels reach it with map(present:) and k_writeback
     * can refresh the host side with an in-graph async D2H. */
    ctx.new_count = (int *)tc_dcheck(host_alloc(sizeof(int)), "new_count", sizeof(int));
    int *new_count = ctx.new_count;
    new_count[0] = 0;
    OMP_TARGET_ENTER_DATA(MAP(alloc: new_count[0:1]))
    ctx.t_setup = tc_now() - t0;

    ctx.peak_mem_mb = (double)((size_t)ctx.n_edges * 2 * sizeof(int)
                             + (size_t)ctx.edge_cap * sizeof(Entity)
                             + (size_t)ctx.result_cap * sizeof(u64)
                             + 2 * (size_t)ctx.frontier_cap * sizeof(u64))
                    / (1024.0 * 1024.0);
}

/* Re-seed the fixpoint state: once before the warm-up rounds, once before the
 * measured fixpoint (the warm-up rounds leave real facts behind).
 * Buffer addresses stay stable, so a recorded task graph stays valid. */
static void tc_reset_state(TCContext &ctx)
{
    int z = 0; u64 z64 = 0;
    dmemset(ctx.d_result_set, 0xFF, ctx.result_cap * sizeof(u64));
    to_dev(ctx.d_frontier_size, &z, sizeof(int));
    to_dev(ctx.d_result_count,  &z64, sizeof(u64));
    to_dev(ctx.d_overflow,      &z, sizeof(int));
    /* new_count needs no reset here: k_reset zeroes the device copy at the top
     * of every fixpoint round, and tc_run_fixpoint primes the host copy. */
    init_base(ctx);
}

static void tc_check_overflow(TCContext &ctx)
{
    if (tc_read_i32(ctx.d_overflow)) {
        fprintf(stderr,
            "ERROR: result set / frontier overflow (capacity too small).\n"
            "       Increase capacity_mult (arg 2). Current result_cap=%ld slots.\n",
            ctx.result_cap);
        exit(2);
    }
}

static void tc_teardown(TCContext &ctx)
{
    dfree(ctx.d_edges);       dfree(ctx.d_edge_table);   dfree(ctx.d_result_set);
    dfree(ctx.d_frontier);    dfree(ctx.d_new_frontier);
    dfree(ctx.d_frontier_size);
    dfree(ctx.d_result_count);  dfree(ctx.d_overflow);

    int *new_count = ctx.new_count;
    if (new_count) {
        OMP_TARGET_EXIT_DATA(MAP(release: new_count[0:1]))
        host_free(new_count);
        ctx.new_count = nullptr;
    }
}

/* ------------------------------------------------------------------------- */
/* Result output (disk write only; the D2H is timed separately).              */
/* ------------------------------------------------------------------------- */
static void tc_write_output(const u64 *host, long long n, const char *input_file)
{
    char path[4096];
    snprintf(path, sizeof(path), "%s_%s_tc.bin", input_file, TC_VERSION);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open output file %s\n", path); return; }
    for (long long i = 0; i < n; i++) {
        u64 s = host[i];
        int pair[2] = { (int)(s >> 32), (int)(s & 0xffffffffULL) };
        fwrite(pair, sizeof(int), 2, f);
    }
    fclose(f);
    printf("# wrote %lld tuples to %s\n", n, path);
}

static void tc_dump(const u64 *host, long long n, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open dump file %s\n", path); return; }
    for (long long i = 0; i < n; i++)
        fprintf(f, "%d %d\n", (int)(host[i] >> 32), (int)(host[i] & 0xffffffffULL));
    fclose(f);
}

/* 15-column machine row (same schema as the reference tc_benchmark). */
static void tc_print_csv(const char *csv, int input, int iters, u64 tc, double total,
                         double fileio, double h2d, double setup, double build,
                         double compute, double compute_min, double d2h,
                         double peak, int repeats, const char *data)
{
    FILE *cf = fopen(csv, "w");
    if (!cf) return;
    fprintf(cf, "%s,%d,%d,%llu,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.2lf,%d,%s\n",
            TC_VERSION, input, iters, tc, total, fileio, h2d, setup, build,
            compute, compute_min, d2h, peak, repeats, data);
    fclose(cf);
}

/* ------------------------------------------------------------------------- */
/* Timing model and report.                                                   */
/*                                                                            */
/* The measured unit is one fixpoint ROUND, reported the way the Krylov drivers */
/* report iterations (krylov/common/driver.cpp):                               */
/*                                                                            */
/*   round 0            the round that RECORDS the task graph (XKOMP rc == 1)  */
/*   round 1            first replay; also where the command graph is built    */
/*                      and optimized (XKOMP rc == 2)                          */
/*   rounds 2..N-1      steady state -> avg / stddev                           */
/*                                                                            */
/* CAVEAT: unlike a Krylov iteration, TC rounds do very different amounts of   */
/* work -- the frontier grows for the first rounds and then collapses. The     */
/* steady-state stddev therefore mostly reflects that frontier-size profile,   */
/* NOT run-to-run jitter. The record/replay comparison (round 0 and round 1    */
/* against the steady mean) is what the split is for.                          */
/*                                                                            */
/* End-to-end "total time" is the MNMGDatalog paper's metric and per-phase     */
/* breakdown (Table "End-to-end total time (ms)" / Fig. "TC per-phase total    */
/* time breakdown"): file IO + H2D + setup + compute + D2H, where compute is   */
/* the whole measured fixpoint plus the one-shot result compaction. Unlike the  */
/* CUDA reference -- which captures and instantiates the graph in a separate,  */
/* separately-timed phase -- XKOMP records and builds INSIDE the loop, so there */
/* is no separate build phase: that cost sits in compute, visible as rounds 0  */
/* and 1. Every number printed is measured; none is extrapolated.              */
/* ------------------------------------------------------------------------- */
typedef struct {
    double round0_s;        /* round 0: records the task graph                */
    double round1_s;        /* round 1: graph build + first replay            */
    double steady_s;        /* mean of the steady-state rounds                */
    double steady_sd_s;     /* sample stddev of the steady-state rounds       */
    int    steady_from;     /* index of the first steady-state round          */
    int    steady_n;        /* number of steady-state rounds                  */
} TCTimings;

/* Split the per-round times into record / first-replay / steady state. With
 * fewer than 3 rounds the steady-state window degrades gracefully (2 rounds ->
 * round 1 alone; 1 round -> round 0 alone) so tiny inputs still report a number. */
static TCTimings tc_timings(const double *t, int n)
{
    TCTimings s{};
    if (n <= 0) return s;

    s.round0_s    = t[0];
    s.round1_s    = (n >= 2) ? t[1] : 0.0;
    s.steady_from = (n >= 3) ? 2 : (n >= 2 ? 1 : 0);
    s.steady_n    = n - s.steady_from;
    tc_mean_std(t + s.steady_from, s.steady_n, &s.steady_s, &s.steady_sd_s);
    return s;
}

/* ------------------------------------------------------------------------- */
/* main. Usage: ./tc.x <data.bin> [capacity_mult] [frontier_slots]            */
/*   Env: TC_WARMUP=<n> untimed warm-up rounds before round 0 (default 3);     */
/*        TC_WORKERS=<n> grid-stride worker count (GPU);                       */
/*        TC_WRITE=1 writes <input>_<version>_tc.bin; TC_DUMP=<f> text dump;   */
/*        TC_CSV=<f> writes the 15-column machine row.                         */
/* ------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    const char *input_file = (argc >= 2) ? argv[1] : "MNMGDatalog-reference/data/data_10.bin";
    long capacity_mult  = (argc >= 3) ? atol(argv[2]) : 64;
    long frontier_slots = (argc >= 4) ? atol(argv[3]) : 0;

    int warmups = 3;
    const char *wu = getenv("TC_WARMUP");
    if (wu && wu[0]) warmups = atoi(wu);
    if (warmups < 0) warmups = 0;

    TCContext ctx;
    tc_setup(ctx, input_file, capacity_mult, frontier_slots);

    /* One enclosing parallel/single spans the warm-up AND the measured fixpoint.
     * The warm-up rounds run the same kernels with the taskgraph wrapper DISABLED,
     * so they absorb team creation / first-launch / module-load / JIT /
     * first-touch cost without consuming the record pass; round 0 of the measured
     * fixpoint is then the round that records the graph, and every later round is
     * a replay -- like CUDA v2_cudagraph's build-once / replay. */
    int rounds = 0;
    double fixpoint_s = 0.0;
    TCTimes times{};

    #pragma omp parallel
    #pragma omp single
    {
        tc_reset_state(ctx);
        tc_warmup(ctx, warmups);

        tc_reset_state(ctx);
        double f0 = omp_get_wtime();
        rounds = tc_run_fixpoint(ctx, &times);
        fixpoint_s = omp_get_wtime() - f0;
    }
    tc_check_overflow(ctx);
    u64 tc = tc_read_u64(ctx.d_result_count);

    /* Free the frontier buffers to make room for the compact result buffer. */
    dfree(ctx.d_frontier);     ctx.d_frontier = nullptr;
    dfree(ctx.d_new_frontier); ctx.d_new_frontier = nullptr;

    /* Materialize: stream-compact the sparse set into a dense device array, then
     * copy exactly TC tuples to the host. Compaction is result-producing GPU work
     * (folded into compute); d2h times only the device->host copy. */
    double c0 = tc_now();
    size_t cb = (size_t)(tc ? tc : 1) * sizeof(u64);
    u64 *d_compact = (u64 *)tc_dcheck(dalloc(cb), "compact output", cb);
    u64 *d_cnt = (u64 *)tc_dcheck(dalloc(sizeof(u64)), "compact counter", sizeof(u64));
    u64 z64 = 0; to_dev(d_cnt, &z64, sizeof(u64));
    compact(ctx, d_compact, d_cnt);
    double compact_s = tc_now() - c0;

    u64 *host = (u64 *)malloc((size_t)(tc ? tc : 1) * sizeof(u64));
    double t0 = tc_now();
    from_dev(host, d_compact, (size_t)tc * sizeof(u64));
    double d2h = tc_now() - t0;
    dfree(d_compact); dfree(d_cnt);

    double fileio = ctx.t_fileio;
    if (getenv("TC_WRITE")) {
        double tw = tc_now();
        tc_write_output(host, (long long)tc, input_file);
        fileio += tc_now() - tw;
    }

    /* Per-phase end-to-end accounting, as in the MNMGDatalog paper. compute is
     * the whole measured fixpoint plus the one-shot result compaction; the graph
     * record/build cost lives inside it (rounds 0 and 1), because XKOMP records
     * and builds inside the loop rather than in a separate phase. */
    const TCTimings st      = tc_timings(times.v, times.n);
    const double compute_s  = fixpoint_s + compact_s;
    const double total_s    = fileio + ctx.t_h2d + ctx.t_setup + compute_s + d2h;

    /* ---- Banner + statistics (same shape as the Krylov drivers) ---- */
    printf("MNMGDatalog TC (transitive closure)\n");
    printf("  %-11s: %s\n", "backend", USE_TARGET ? "GPU (omp target, device-resident buffers)"
                                                  : "CPU (omp task, host memory)");
    printf("  %-11s: %s\n", "exec mode", USE_SYNC ? "synchronous (blocking, no tasks)"
                                                  : "asynchronous (tasks)");
    printf("  %-11s: %s\n", "taskgraph", (USE_TASKGRAPH && !USE_SYNC) ? "on (record once, replay)" : "off");
    printf("  %-11s: %s\n", "version", TC_VERSION);
    printf("  %-11s: %s\n", "input", input_file);
    printf("  %-11s: %d edges  ->  TC = %llu tuples in %d rounds\n",
           "size", ctx.input_rows, tc, rounds);
#if USE_TARGET
    printf("  %-11s: %d grid-stride workers\n", "geometry", ctx.n_workers);
#endif
    printf("  %-11s: %d untimed round%s (ungraphed)\n",
           "warm-up", warmups, warmups == 1 ? "" : "s");
    printf("  %-11s: %.2f MB\n", "peak memory", ctx.peak_mem_mb);

    printf("Statistics\n");
    printf("  %-27s : %10.3f ms\n", "total time (end-to-end)", total_s * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  file IO",      fileio      * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  H2D transfer", ctx.t_h2d   * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  setup",        ctx.t_setup * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  compute",      compute_s   * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  D2H transfer", d2h         * 1000.0);

    {
        char lbl[64];
        const int graphed = (USE_TASKGRAPH && !USE_SYNC);
        snprintf(lbl, sizeof lbl, "round 0%s", graphed ? " (record)" : "");
        printf("  %-27s : %10.3f ms\n", lbl, st.round0_s * 1000.0);
        if (times.n >= 2) {
            snprintf(lbl, sizeof lbl, "round 1%s", graphed ? " (1st replay)" : "");
            printf("  %-27s : %10.3f ms\n", lbl, st.round1_s * 1000.0);
        }
        if (times.n >= 3) {
            snprintf(lbl, sizeof lbl, "rounds %d..%d (avg)", st.steady_from, times.n - 1);
            printf("  %-27s : %10.3f ms   (%d rounds)\n", lbl, st.steady_s * 1000.0, st.steady_n);
            snprintf(lbl, sizeof lbl, "rounds %d..%d (stddev)", st.steady_from, times.n - 1);
            printf("  %-27s : %10.3f ms\n", lbl, st.steady_sd_s * 1000.0);
        }
    }
    fflush(stdout);

    /* 15-column reference row. There is no separate graph-build phase (build =
     * 0, folded into compute) and no repeats, so compute_min == compute. */
    const char *csv = getenv("TC_CSV");
    if (csv && csv[0])
        tc_print_csv(csv, ctx.input_rows, rounds, tc, total_s, fileio, ctx.t_h2d,
                     ctx.t_setup, 0.0, compute_s, compute_s, d2h,
                     ctx.peak_mem_mb, 1, input_file);

    const char *dump = getenv("TC_DUMP");
    if (dump && dump[0]) tc_dump(host, (long long)tc, dump);

    free(host); free(times.v);
    tc_teardown(ctx);
    return 0;
}
