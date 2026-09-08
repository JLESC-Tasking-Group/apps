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

/* Memory-model A/B toggle, see the "Host <-> device memory abstraction" block.
 * 0 (default): device-only buffers via omp_target_alloc + is_device_ptr.
 * 1          : pinned host buffers + map(present:), as krylov/lulesh do. */
#ifndef TC_MAPPED_MEM
# define TC_MAPPED_MEM 0
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
/*                                                                            */
/* Every one of them is marked always_inline, and that is load-bearing, not    */
/* cosmetic: on this toolchain a `target teams distribute parallel for` whose   */
/* body makes real device function CALLS faults with                          */
/* CUDA_ERROR_ILLEGAL_ADDRESS once ~10^5 threads are concurrently inside the   */
/* call chain. -O3 inlines these anyway and is unaffected, but a plain         */
/* `static inline` leaves them as calls at -O0, which is exactly how the bug   */
/* was hit for weeks (only k_expand / build_edges / init_base -- the kernels    */
/* with call chains -- ever faulted; k_promote and the fills, whose bodies are  */
/* inline, never did). Forcing the inline here keeps -O0 -g GPU builds usable.  */
/* Not a stack-size problem: LIBOMPTARGET_STACK_SIZE up to 256 KiB changes      */
/* nothing. See REPRODUCER.md and ../common.mk's OPT warning.                   */
/* ------------------------------------------------------------------------- */
#define TC_DEVFN static inline __attribute__((always_inline))

#if USE_TARGET
# pragma omp begin declare target
#endif

/* Murmur3 finalizer (mirrors get_position in the codebase). */
TC_DEVFN int tc_get_position(int key, int cap)
{
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key & (cap - 1);
}

/* splitmix64 finalizer -> slot in [0, cap). */
TC_DEVFN u64 tc_hash64(u64 k, long cap)
{
    k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
    k ^= k >> 27; k *= 0x94d049bb133111ebULL;
    k ^= k >> 31;
    return k & (u64)(cap - 1);
}

TC_DEVFN u64 tc_pack(int a, int b)
{
    return (((u64)(unsigned)a) << 32) | (unsigned)b;
}

/* CAS: atomically set *addr = desired iff *addr == expected; return old value. */
TC_DEVFN u64 tc_cas_u64(u64 *addr, u64 expected, u64 desired)
{
    u64 old;
    #pragma omp atomic compare capture
    { old = *addr; if (*addr == expected) *addr = desired; }
    return old;
}

TC_DEVFN int tc_cas_i32(int *addr, int expected, int desired)
{
    int old;
    #pragma omp atomic compare capture
    { old = *addr; if (*addr == expected) *addr = desired; }
    return old;
}

/* Fetch-add: return the old value, then *addr += v (the append/index pattern). */
TC_DEVFN int tc_fetch_add_i32(int *addr, int v)
{
    int old;
    #pragma omp atomic capture
    { old = *addr; *addr += v; }
    return old;
}

TC_DEVFN u64 tc_fetch_add_u64(u64 *addr, u64 v)
{
    u64 old;
    #pragma omp atomic capture
    { old = *addr; *addr += v; }
    return old;
}

/* Insert key into the open-addressing result set. Returns true iff newly
 * inserted (won an empty slot) -- i.e. this is a genuinely new fact. Bounded by
 * capacity: if the set is full we set *overflow and return false (no hang). */
TC_DEVFN bool tc_set_insert(u64 *set, long cap, u64 key, int *overflow)
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
 * set dedups). Bounded by capacity exactly like tc_set_insert: a table that is
 * full -- or that was never initialised to the -1 empty marker -- sets *overflow
 * instead of spinning forever, so the failure is reported rather than hung. */
TC_DEVFN void tc_build_one(int i, const int *edges, Entity *table, int cap,
                                int *overflow)
{
    int key = edges[i * 2], value = edges[i * 2 + 1];
    int pos = tc_get_position(key, cap);
    for (int probes = 0; probes < cap; probes++) {
        int existing = tc_cas_i32(&table[pos].key, -1, key);
        if (existing == -1) { table[pos].value = value; return; }
        pos = (pos + 1) & (cap - 1);
    }
    *overflow = 1;                             /* table full (benign store) */
}

/* Seed the fixpoint with the base facts: path(a,b) :- edge(a,b), deduped. */
TC_DEVFN void tc_init_base_one(int i, const int *edges, u64 *set, long rcap,
                                    u64 *frontier, int fcap, int *fsize,
                                    u64 *rcount, int *overflow)
{
    int a = edges[i * 2], b = edges[i * 2 + 1];
    u64 p = tc_pack(a, b);
    if (tc_set_insert(set, rcap, p, overflow)) {
        /* The counter is bumped for every new fact, so it can run past fcap (and,
         * in the extreme, wrap negative) once the frontier is full: the unsigned
         * compare covers both, and the append is skipped. */
        int w = tc_fetch_add_i32(fsize, 1);
        if ((unsigned)w < (unsigned)fcap) frontier[w] = p; else *overflow = 1;
        tc_fetch_add_u64(rcount, 1ULL);
    }
}

/* Expand one frontier fact path(a,b): for every edge(b,c), try path(a,c). */
TC_DEVFN void tc_expand_one(int i, const Entity *edge_table, int edge_cap,
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
                /* See tc_init_base_one: new_count counts every new fact, so it
                 * overruns nfcap (and may wrap negative) once the frontier fills.
                 * The unsigned compare keeps the append in bounds; k_promote and
                 * k_set_sizes clamp the *readers* of new_count to nfcap. */
                int w = tc_fetch_add_i32(new_count, 1);
                if ((unsigned)w < (unsigned)nfcap) new_frontier[w] = np; else *overflow = 1;
                tc_fetch_add_u64(rcount, 1ULL);
            }
        } else if (k == -1) {
            break;
        }
        pos = (pos + 1) & (edge_cap - 1);
    }
}

/* Stream-compact one result-set slot into the dense output array. */
TC_DEVFN void tc_compact_one(long i, const u64 *set, u64 *out, u64 *out_count)
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
/*                                                                            */
/*   CPU (USE_TARGET==0):                 plain malloc / memcpy.              */
/*   GPU, TC_MAPPED_MEM==0 (default):     omp_target_alloc + omp_target_memcpy */
/*                                        (device-only, reached with           */
/*                                        is_device_ptr) -- the direct analog  */
/*                                        of the reference's cudaMalloc, with  */
/*                                        no host mirror of a multi-GB set.    */
/*   GPU, TC_MAPPED_MEM==1:               pinned host buffers + map(present:), */
/*                                        i.e. the model krylov / lulesh /     */
/*                                        HPCCG use. Strictly a DIAGNOSTIC     */
/*                                        A/B: it doubles the footprint (a     */
/*                                        host mirror of every buffer) and is  */
/*                                        only practical on small inputs. It   */
/*                                        exists because tc.cpp is the only    */
/*                                        app here using the is_device_ptr     */
/*                                        model, and the only one that faults. */
/* ------------------------------------------------------------------------- */
#if USE_TARGET
static int g_dev  = 0;
static int g_host = 0;
#endif

#if USE_TARGET && TC_MAPPED_MEM

/* Every buffer is host memory with a device copy created at allocation, so the
 * kernels reach it with map(present:) exactly like the Krylov solvers. The
 * host side is kept as the master copy; to_dev / from_dev are target updates. */
static inline void *dalloc(size_t b)
{
    char *p = (char *)host_alloc(b);
    if (p) {
        #pragma omp target enter data map(alloc: p[0:b])
    }
    return p;
}
static inline void dfree(void *vp)
{
    char *p = (char *)vp;
    if (p) {
        #pragma omp target exit data map(release: p[0:1])
        host_free(p);
    }
}
static inline void to_dev(void *d, const void *s, size_t b)
{
    char *p = (char *)d;
    memcpy(p, s, b);
    #pragma omp target update to(p[0:b])
}
static inline void from_dev(void *d, const void *s, size_t b)
{
    char *p = (char *)s;
    #pragma omp target update from(p[0:b])
    memcpy(d, p, b);
}

#elif USE_TARGET

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

/* ----------------------------------------------------------------------------
 * Per-kernel device-memory clauses. The ONLY place the two memory models differ
 * in the kernels: TC_MAPPED_MEM==0 passes the device-only buffers by value with
 * is_device_ptr, TC_MAPPED_MEM==1 asserts their mapped presence instead. The two
 * spellings need different syntax (bare pointers vs array sections), so they
 * cannot be folded into one macro -- hence one macro per kernel, all here.
 * new_count and overflow are pinned-host + mapped in BOTH models.
 * ------------------------------------------------------------------------- */
#if USE_TARGET && TC_MAPPED_MEM
# define TC_MEM_FILL_TABLE  map(present: table[0:cap])
# define TC_MEM_FILL_SET    map(present: set[0:cap])
# define TC_MEM_BUILD       map(present: edges[0:2*n], table[0:cap], ov[0:1])
# define TC_MEM_INIT_BASE   map(present: edges[0:2*n], set[0:rcap], fr[0:fcap], \
                                         fs[0:1], rc[0:1], ov[0:1])
# define TC_MEM_COMPACT     map(present: set[0:cap], out[0:1], out_count[0:1])
# define TC_MEM_EXPAND      MAP(present: et[0:ec], fr[0:nfc], fs[0:1], rs[0:rc], \
                                         nf[0:nfc], rcnt[0:1], ncnt[0:1], ov[0:1])
# define TC_MEM_PROMOTE     MAP(present: fr[0:nfc], nf[0:nfc], nc[0:1])
# define TC_MEM_SET_SIZES   map(present: fs[0:1], nc[0:1])
#elif USE_TARGET
# define TC_MEM_FILL_TABLE  is_device_ptr(table)
# define TC_MEM_FILL_SET    is_device_ptr(set)
# define TC_MEM_BUILD       is_device_ptr(edges, table) map(present: ov[0:1])
# define TC_MEM_INIT_BASE   is_device_ptr(edges, set, fr, fs, rc) map(present: ov[0:1])
# define TC_MEM_COMPACT     is_device_ptr(set, out, out_count)
# define TC_MEM_EXPAND      is_device_ptr(et, fr, fs, rs, nf, rcnt) \
                            MAP(present: ncnt[0:1], ov[0:1])
# define TC_MEM_PROMOTE     is_device_ptr(fr, nf) MAP(present: nc[0:1])
# define TC_MEM_SET_SIZES   is_device_ptr(fs) map(present: nc[0:1])
#else
# define TC_MEM_EXPAND
# define TC_MEM_PROMOTE
#endif

/* ------------------------------------------------------------------------- */
/* TC_TRACE=1: one flushed stderr line per kernel dispatch, printed BEFORE the  */
/* launch, naming the kernel and the sizes it was given.                       */
/*                                                                             */
/* Under USE_SYNC=1 every launch is blocking, so after an abort the LAST line   */
/* printed names the kernel that faulted -- which the CUDA error alone does not */
/* tell you (xkrt reports the stream sync, not the kernel). Under the           */
/* asynchronous schedules the lines mark task CREATION, not execution, so they  */
/* bound the failure but do not pinpoint it; debug with USE_SYNC=1.             */
/*                                                                             */
/* Off by default: one cached getenv and a predictable branch per dispatch.     */
/* ------------------------------------------------------------------------- */
static int g_trace = -1;

static inline bool tc_tracing(void)
{
    if (g_trace < 0) {
        const char *v = getenv("TC_TRACE");
        g_trace = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return g_trace != 0;
}

#define TC_TRACE_K(fmt, ...)                                                   \
    do {                                                                       \
        if (tc_tracing()) {                                                    \
            fprintf(stderr, "# TC_TRACE " fmt "\n" __VA_OPT__(,) __VA_ARGS__); \
            fflush(stderr);                                                    \
        }                                                                      \
    } while (0)

/* Typed fills of the two open-addressing tables with their empty marker -- the
 * analog of the reference's cudaMemset(..., 0xFF, ...), but one store per SLOT
 * instead of one per byte: 8x fewer iterations and coalesced. (The byte-wise
 * form cost 268 M single-byte stores just to clear the result set of a 410 K-edge
 * graph, and tens of GB of them on the paper's billion-pair graphs.) The bit
 * patterns are identical: -1 == 0xFFFFFFFF, TC_EMPTY64 == 0xFFFF...FF. */
static void fill_edge_table(Entity *table, int cap)
{
    TC_TRACE_K("-> fill_edge_table  cap=%d", cap);
#if USE_TARGET
    #pragma omp target teams distribute parallel for TC_MEM_FILL_TABLE
    for (int i = 0; i < cap; i++) { table[i].key = -1; table[i].value = -1; }
#else
    #pragma omp parallel for
    for (int i = 0; i < cap; i++) { table[i].key = -1; table[i].value = -1; }
#endif
}

static void fill_result_set(u64 *set, long cap)
{
    TC_TRACE_K("-> fill_result_set  cap=%ld", cap);
#if USE_TARGET
    #pragma omp target teams distribute parallel for TC_MEM_FILL_SET
    for (long i = 0; i < cap; i++) set[i] = TC_EMPTY64;
#else
    #pragma omp parallel for
    for (long i = 0; i < cap; i++) set[i] = TC_EMPTY64;
#endif
}

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
    /* new_count and overflow are the TWO scalars the host reads every instance
     * (the fixpoint convergence test and the capacity check), so unlike the other
     * buffers they are pinned HOST memory (host_alloc) mapped onto the device with
     * map(alloc:)/map(present:), as the Krylov solvers do for their scalars.
     * k_writeback refreshes both with a single in-graph async D2H, so no blocking
     * omp_target_memcpy is needed inside the timed loop. */
    int *new_count       = nullptr;
    int *overflow        = nullptr;
    u64 *d_result_count  = nullptr;

    /* Kept for the overflow diagnostic: it names the knob to raise. */
    long capacity_mult   = 0;

    /* Monotonic round counter, TC_TRACE output only (warm-up rounds included). */
    int  trace_round     = 0;

    /* ---- Capacity provenance (see tc_resolve_capacity) --------------------
     * result_cap / frontier_cap cannot be derived from the input: the TC/edge
     * ratio spans 21x (OL.cedge) to 5977x (p2p-Gnutella31). They are therefore
     * DISCOVERED by an untimed, ungraphed probe solve, then cached. `max_node`
     * bounds the probe's provisional table; sized_tc / sized_peak are what the
     * probe found. */
    int    max_node      = 0;    /* largest node id in the input               */
    u64    sized_tc      = 0;    /* closure size the probe discovered          */
    long   sized_peak    = 0;    /* peak per-round new_count the probe saw     */
    enum { CAP_OVERRIDE, CAP_CACHED, CAP_DISCOVERED } cap_source = CAP_OVERRIDE;
    double t_sizing      = 0.0;  /* wall time of the probe (0 if not run)      */

    double t_fileio = 0.0, t_h2d = 0.0, t_setup = 0.0, peak_mem_mb = 0.0;
};

/* The overflow flag is a benign store from every insert path: the edge table
 * (build_edges), the result set and the frontiers (init_base / k_expand). It is
 * pinned host memory mapped onto the device, and k_writeback refreshes it once
 * per instance, so an undersized run is caught within milliseconds instead of
 * grinding through a saturated open-addressing table (every insert into a full
 * table probes result_cap times before giving up).
 *
 * `instance` is the fixpoint instance that tripped it, or -1 during setup. */
static void tc_check_overflow(TCContext &ctx, const char *what, int instance)
{
    if (!ctx.overflow || !ctx.overflow[0]) return;

    fprintf(stderr, "\nERROR: %s overflow", what);
    if (instance >= 0) fprintf(stderr, " at instance %d", instance);
    fprintf(stderr, ".\n");

    if (ctx.result_cap > 0)
        fprintf(stderr,
            "       %d edges, capacity_mult=%ld  ->  result_cap=%ld slots (%.2f GB),\n"
            "       frontier_cap=%d slots (%.2f GB each).\n"
            "       The transitive closure does not fit: the result set must hold\n"
            "       >= ~2x TC. Re-run with a larger capacity_mult (arg 2); see the\n"
            "       capacity table in README.md for the per-dataset values.\n",
            ctx.input_rows, ctx.capacity_mult, ctx.result_cap,
            (double)ctx.result_cap * (double)sizeof(u64) / (1024.0 * 1024.0 * 1024.0),
            ctx.frontier_cap,
            (double)ctx.frontier_cap * (double)sizeof(u64) / (1024.0 * 1024.0 * 1024.0));
    else
        fprintf(stderr,
            "       %d edges, edge_cap=%d slots. The edge table could not absorb the\n"
            "       input: it was either full or never initialised to the -1 marker.\n",
            ctx.input_rows, ctx.edge_cap);
    exit(2);
}

/* ------------------------------------------------------------------------- */
/* Setup / finalize kernels (blocking; NOT part of the recorded task graph).  */
/* ------------------------------------------------------------------------- */
static void build_edges(TCContext &ctx)
{
    int n = ctx.n_edges, cap = ctx.edge_cap;
    int *edges = ctx.d_edges; Entity *table = ctx.d_edge_table; int *ov = ctx.overflow;
    TC_TRACE_K("-> build_edges      n=%d edge_cap=%d", n, cap);
#if USE_TARGET
    #pragma omp target teams distribute parallel for TC_MEM_BUILD
    for (int i = 0; i < n; i++) tc_build_one(i, edges, table, cap, ov);
#else
    #pragma omp parallel for
    for (int i = 0; i < n; i++) tc_build_one(i, edges, table, cap, ov);
#endif
}

static void init_base(TCContext &ctx)
{
    int n = ctx.n_edges, fcap = ctx.frontier_cap; long rcap = ctx.result_cap;
    int *edges = ctx.d_edges; u64 *set = ctx.d_result_set;
    u64 *fr = ctx.d_frontier; int *fs = ctx.d_frontier_size;
    u64 *rc = ctx.d_result_count; int *ov = ctx.overflow;
    TC_TRACE_K("-> init_base        n=%d rcap=%ld fcap=%d", n, rcap, fcap);
#if USE_TARGET
    #pragma omp target teams distribute parallel for TC_MEM_INIT_BASE
    for (int i = 0; i < n; i++) tc_init_base_one(i, edges, set, rcap, fr, fcap, fs, rc, ov);
#else
    #pragma omp parallel for
    for (int i = 0; i < n; i++) tc_init_base_one(i, edges, set, rcap, fr, fcap, fs, rc, ov);
#endif
}

static void compact(TCContext &ctx, u64 *out, u64 *out_count)
{
    u64 *set = ctx.d_result_set; long cap = ctx.result_cap;
    TC_TRACE_K("-> compact          rcap=%ld", cap);
#if USE_TARGET
    #pragma omp target teams distribute parallel for TC_MEM_COMPACT
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
    TC_TRACE_K("   -> k_reset");
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
    int *ncnt = ctx.new_count; u64 *rcnt = ctx.d_result_count; int *ov = ctx.overflow;
    int nw = ctx.n_workers;
    TC_TRACE_K("   -> k_expand     nw=%d edge_cap=%d rcap=%ld nfcap=%d", nw, ec, rc, nfc);
    /* GPU: TC_MEM_EXPAND is the memory-model clause (is_device_ptr for the
     * device-only buffers plus map(present:) for the two pinned-host scalars, or
     * all-map(present:) under TC_MAPPED_MEM); CPU: default(none) firstprivate
     * captures the pointers/scalars. The OpenMP bound is the host constant nw;
     * the frontier size fs[0] is read on the DEVICE by every worker, so replay
     * uses the current frontier size. */
    OMP_TILE(DEPEND(in, fs[0], fr[0]) DEPEND(inout, rs[0], ncnt[0], rcnt[0], ov[0]) DEPEND(out, nf[0]),
             TC_MEM_EXPAND,
             DEFAULT_NONE firstprivate(et, ec, fr, fs, rs, rc, nf, nfc, ncnt, rcnt, ov, nw))
    for (int t = 0; t < nw; t++) {
        const int n = fs[0];
        for (int i = t; i < n; i += nw)
            tc_expand_one(i, et, ec, fr, rs, rc, nf, nfc, ncnt, rcnt, ov);
    }
}

/* new_count counts every new fact of the round, so it can exceed frontier_cap
 * when the result set / frontier is undersized for the graph (see the capacity
 * table in the README). Only the first frontier_cap facts were actually stored,
 * so BOTH readers of new_count clamp to frontier_cap -- otherwise they would walk
 * off the end of the frontier buffers, which is an illegal access on the device,
 * not merely a wrong answer. The overflow flag is already set in that case and
 * the fixpoint aborts on the next convergence test. */
static void k_promote(TCContext &ctx)
{
    u64 *fr = ctx.d_frontier; u64 *nf = ctx.d_new_frontier; int *nc = ctx.new_count;
    int nw = ctx.n_workers; int nfc = ctx.frontier_cap;
    TC_TRACE_K("   -> k_promote    nw=%d nfcap=%d", nw, nfc);
    /* Same grid-stride shape as k_expand: nc[0] is the DEVICE copy of new_count
     * (map(present:)), read inside the body. Reading it as the OpenMP bound would
     * take the host copy, which still holds the PREVIOUS round's count. */
    OMP_TILE(DEPEND(in, nc[0], nf[0]) DEPEND(out, fr[0]),
             TC_MEM_PROMOTE,
             DEFAULT_NONE firstprivate(fr, nf, nc, nw, nfc))
    for (int t = 0; t < nw; t++) {
        const int n = nc[0] < nfc ? nc[0] : nfc;
        for (int i = t; i < n; i += nw) fr[i] = nf[i];
    }
}

static void k_set_sizes(TCContext &ctx)
{
    int *fs = ctx.d_frontier_size; int *nc = ctx.new_count; int nfc = ctx.frontier_cap;
    TC_TRACE_K("   -> k_set_sizes  nfcap=%d", nfc);
#if USE_TARGET
    OMP_TARGET_TASK(DEPEND(in, nc[0]) DEPEND(out, fs[0]) TC_MEM_SET_SIZES)
    { fs[0] = nc[0] < nfc ? nc[0] : nfc; }
#else
    OMP_TASK(DEFAULT_NONE firstprivate(fs, nc, nfc) DEPEND(in, nc[0]) DEPEND(out, fs[0]))
    { fs[0] = nc[0] < nfc ? nc[0] : nfc; }
#endif
}

/* Refresh the HOST copies of new_count (convergence test) and overflow (capacity
 * check) so the fixpoint loop can act on both. This is ONE async D2H recorded
 * INSIDE the taskgraph (depend-ordered after the kernels that update them), i.e.
 * one more replayed command -- the same shape as the Krylov residual read-back
 * (cg.cpp) and xkomp's taskgraph_dot_target test. On the host backend it
 * vanishes: both already ARE the host memory. */
static void k_writeback(TCContext &ctx)
{
    int *nc = ctx.new_count; int *ov = ctx.overflow;
    TC_TRACE_K("   -> k_writeback");
    (void) nc; (void) ov;      /* the directive vanishes on the host backend */
    OMP_TARGET_UPDATE(from(nc[0:1], ov[0:1]) NOWAIT DEPEND(in, nc[0], ov[0]))
}

/* The loop-invariant per-round kernel sequence -- the body that is recorded once
 * and replayed. Byte-identical every round: same kernels, same buffers, same
 * launch geometry; only the device-resident sizes read inside the kernels move. */
static inline void tc_round(TCContext &ctx)
{
    /* new_count is the HOST copy, i.e. the previous round's result: under
     * TC_TRACE it shows the frontier growing round by round, which is what
     * distinguishes a size-driven failure from a work-driven one. */
    TC_TRACE_K("== round %d  (prev nc=%d)", ctx.trace_round++, ctx.new_count[0]);
    k_reset(ctx);
    k_expand(ctx);
    k_promote(ctx);
    k_set_sizes(ctx);
    k_writeback(ctx);
}

/* Per-instance wall times, in seconds PER ROUND (the epilogue divides by the
 * unroll). The number of rounds is data-dependent (the fixpoint runs until no new
 * fact is produced), so the vector grows as the solve proceeds. */
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
        /* Catch an undersized capacity here rather than letting the warm-up churn
         * through a saturated hash set for minutes before the measured run. */
        tc_check_overflow(ctx, "result set / frontier (warm-up)", i);
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
 * there; but with USE_TASKGRAPH=0 the taskgraph macros vanish and the nowait
 * tasks would still be in flight, so without it the loop would read a stale
 * count and stop early.
 *
 * `unroll` (-u) folds that many rounds into one recorded instance, so they
 * overlap inside the graph rather than being separated by its implicit
 * taskgroup. The convergence test is then only evaluated every `unroll` rounds,
 * so the fixpoint OVERSHOOTS by up to unroll-1 rounds. That is semantically
 * harmless -- a converged round has an empty frontier, so expand/promote copy
 * nothing -- but it is wasted work and it inflates the reported round count,
 * which is why the count returned here is the number of rounds actually
 * executed. */
static int tc_run_fixpoint(TCContext &ctx, TCTimes *times, int unroll)
{
    int *nc = ctx.new_count;
    nc[0] = 1;                      /* prime: enter the loop (host copy only) */

    double r0 = omp_get_wtime();

    const size_t rounds = TASKGRAPH_LOOP(unroll,
        [&] (size_t done) { (void) done; return nc[0] > 0; },
        [&] (size_t inst, size_t done)
    {
        (void) done;
        TASKWAIT
        const double now = omp_get_wtime();
        tc_times_push(times, (now - r0) / (double) unroll);
        r0 = now;
        /* k_writeback brought the flag back with new_count in the same D2H, so
         * an undersized run aborts here -- before it can produce wrong results
         * or spend minutes probing a full table. */
        tc_check_overflow(ctx, "result set / frontier", (int) inst);
    })
    {
        tc_round(ctx);
    }
    TASKGRAPH_LOOP_END

    return (int) rounds;
}

/* ------------------------------------------------------------------------- */
/* Host helpers: binary IO, sizing, stats.                                    */
/* ------------------------------------------------------------------------- */
static double tc_now()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* Also reports the largest node id: tc_resolve_capacity uses (max_node+1)^2 as
 * the trivial upper bound on the closure size when sizing the probe's table. One
 * extra pass over data that is already hot in cache. */
static int *tc_read_bin(const char *path, int *n_edges_out, int *max_node_out)
{
    struct stat st{};
    if (stat(path, &st) != 0) { fprintf(stderr, "Cannot stat input file %s\n", path); exit(EXIT_FAILURE); }
    long n = st.st_size / (long)(sizeof(int) * 2);
    int *data = (int *)malloc((size_t)n * 2 * sizeof(int));
    if (!data) { fprintf(stderr, "Out of host memory reading %s\n", path); exit(EXIT_FAILURE); }
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(EXIT_FAILURE); }
    size_t got = fread(data, sizeof(int), (size_t)n * 2, fp);
    fclose(fp);
    if (got != (size_t)(n * 2)) { fprintf(stderr, "Short read on %s\n", path); exit(EXIT_FAILURE); }

    int hi = 0;
    for (long i = 0; i < n * 2; i++) if (data[i] > hi) hi = data[i];

    *n_edges_out  = (int)n;
    *max_node_out = hi;
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

/* TC_VERIFY=1: read the edge table back and count the occupied slots. Off by
 * default. Run once after fill_edge_table (expect 0 occupied -- every slot must
 * carry the -1 empty marker) and once after build_edges (expect exactly n_edges,
 * since duplicate edges each claim their own slot). This separates "the fill
 * kernel did not cover the whole table" from "the CAS insert is wrong" from
 * "the pointers/capacity are wrong" without a debugger. */
static void tc_verify_edge_table(TCContext &ctx, const char *stage, long expected)
{
    const char *v = getenv("TC_VERIFY");
    if (!(v && v[0] && v[0] != '0')) return;

    const size_t nb = (size_t)ctx.edge_cap * sizeof(Entity);
    Entity *h = (Entity *)malloc(nb);
    if (!h) { fprintf(stderr, "# TC_VERIFY [%s]: out of host memory, skipped\n", stage); return; }
    from_dev(h, ctx.d_edge_table, nb);

    long occupied = 0, first = -1;
    for (int i = 0; i < ctx.edge_cap; i++)
        if (h[i].key != -1) { occupied++; if (first < 0) first = i; }

    fprintf(stderr, "# TC_VERIFY [%-11s]: edge_cap=%d occupied=%ld expected=%ld  %s"
                    "  (first occupied slot %ld)\n",
            stage, ctx.edge_cap, occupied, expected,
            occupied == expected ? "OK" : "*** MISMATCH ***", first);
    free(h);
}

/* ----------------------------------------------------------------------------
 * Phase 1 of setup: everything that does NOT depend on the closure size --
 * the input, the edge table, and the two pinned-host scalars. Runs outside the
 * parallel region. The result set and frontiers are sized separately, because
 * their size cannot be derived from the input (see tc_resolve_capacity).
 * ------------------------------------------------------------------------- */
static void tc_setup_input(TCContext &ctx, const char *input_file)
{
#if USE_TARGET
    g_dev  = omp_get_default_device();
    g_host = omp_get_initial_device();
    #pragma omp target        /* untimed device/context warm-up */
    { }
#endif

    double t0 = tc_now();
    int *edges_host = tc_read_bin(input_file, &ctx.n_edges, &ctx.max_node);
    ctx.input_rows = ctx.n_edges;
    ctx.t_fileio = tc_now() - t0;

    t0 = tc_now();
    size_t nb = (size_t)ctx.n_edges * 2 * sizeof(int);
    ctx.d_edges = (int *)tc_dcheck(dalloc(nb), "edges", nb);
    to_dev(ctx.d_edges, edges_host, nb);
    ctx.t_h2d = tc_now() - t0;
    free(edges_host);

    t0 = tc_now();
    /* The overflow flag is pinned host memory with a device copy, like new_count:
     * k_writeback then brings both back in one async D2H so the fixpoint can test
     * capacity every instance. Allocated up front because build_edges (below)
     * already needs it to report a full / uninitialised edge table. */
    ctx.overflow = (int *)tc_dcheck(host_alloc(sizeof(int)), "overflow flag", sizeof(int));
    int *ov = ctx.overflow;
    ov[0] = 0;
    OMP_TARGET_ENTER_DATA(MAP(to: ov[0:1]))

    /* new_count: pinned host memory (shared ../alloc.c) with a device copy
     * created here, so the kernels reach it with map(present:) and k_writeback
     * can refresh the host side with an in-graph async D2H. */
    ctx.new_count = (int *)tc_dcheck(host_alloc(sizeof(int)), "new_count", sizeof(int));
    int *new_count = ctx.new_count;
    new_count[0] = 0;
    OMP_TARGET_ENTER_DATA(MAP(alloc: new_count[0:1]))

    ctx.edge_cap = (int)tc_next_pow2((long)std::ceil(ctx.n_edges / 0.6));
    if (ctx.edge_cap < 2) ctx.edge_cap = 2;
    nb = (size_t)ctx.edge_cap * sizeof(Entity);
    ctx.d_edge_table = (Entity *)tc_dcheck(dalloc(nb), "edge table", nb);
    fill_edge_table(ctx.d_edge_table, ctx.edge_cap);
    tc_verify_edge_table(ctx, "after fill", 0);
    build_edges(ctx);
    OMP_TARGET_UPDATE(from(ov[0:1]))        /* build_edges is synchronous */
    tc_check_overflow(ctx, "edge table", -1);
    tc_verify_edge_table(ctx, "after build", ctx.n_edges);

    ctx.t_setup = tc_now() - t0;
}

/* ----------------------------------------------------------------------------
 * Phase 2: the size-dependent buffers. Split out because the discovery probe
 * allocates them at a provisional size, runs, and frees them again before the
 * real ones are allocated. `try_only` returns false instead of aborting when an
 * allocation fails, which is how the probe finds the largest table the device
 * will give it.
 * ------------------------------------------------------------------------- */
static bool tc_alloc_fixpoint(TCContext &ctx, long rcap, long fcap, bool try_only)
{
    ctx.result_cap   = rcap;
    ctx.frontier_cap = (int)fcap;

    /* Fixed grid-stride worker count of k_expand / k_promote. Resolved here, so
     * the recorded task graph replays with an identical launch (see
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

    const size_t nb_set = (size_t)rcap * sizeof(u64);
    const size_t nb_fr  = (size_t)fcap * sizeof(u64);

    ctx.d_result_set    = (u64 *)dalloc(nb_set);
    ctx.d_frontier      = (u64 *)dalloc(nb_fr);
    ctx.d_new_frontier  = (u64 *)dalloc(nb_fr);
    ctx.d_frontier_size = (int *)dalloc(sizeof(int));
    ctx.d_result_count  = (u64 *)dalloc(sizeof(u64));

    const bool ok = ctx.d_result_set && ctx.d_frontier && ctx.d_new_frontier
                 && ctx.d_frontier_size && ctx.d_result_count;
    if (!ok) {
        if (try_only) {
            dfree(ctx.d_result_set);    ctx.d_result_set    = nullptr;
            dfree(ctx.d_frontier);      ctx.d_frontier      = nullptr;
            dfree(ctx.d_new_frontier);  ctx.d_new_frontier  = nullptr;
            dfree(ctx.d_frontier_size); ctx.d_frontier_size = nullptr;
            dfree(ctx.d_result_count);  ctx.d_result_count  = nullptr;
            return false;
        }
        tc_dcheck(ctx.d_result_set,   "result set",    nb_set);
        tc_dcheck(ctx.d_frontier,     "frontier",      nb_fr);
        tc_dcheck(ctx.d_new_frontier, "new frontier",  nb_fr);
        tc_dcheck(ctx.d_frontier_size,"frontier size", sizeof(int));
        tc_dcheck(ctx.d_result_count, "result count",  sizeof(u64));
    }

    ctx.peak_mem_mb = (double)((size_t)ctx.n_edges * 2 * sizeof(int)
                             + (size_t)ctx.edge_cap * sizeof(Entity)
                             + nb_set + 2 * nb_fr)
                    / (1024.0 * 1024.0);
    return true;
}

static void tc_free_fixpoint(TCContext &ctx)
{
    dfree(ctx.d_result_set);    ctx.d_result_set    = nullptr;
    dfree(ctx.d_frontier);      ctx.d_frontier      = nullptr;
    dfree(ctx.d_new_frontier);  ctx.d_new_frontier  = nullptr;
    dfree(ctx.d_frontier_size); ctx.d_frontier_size = nullptr;
    dfree(ctx.d_result_count);  ctx.d_result_count  = nullptr;
}

/* Re-seed the fixpoint state: once before the warm-up rounds, once before the
 * measured fixpoint (the warm-up rounds leave real facts behind).
 * Buffer addresses stay stable, so a recorded task graph stays valid. */
static void tc_reset_state(TCContext &ctx)
{
    int z = 0; u64 z64 = 0; int *ov = ctx.overflow;
    fill_result_set(ctx.d_result_set, ctx.result_cap);
    to_dev(ctx.d_frontier_size, &z, sizeof(int));
    to_dev(ctx.d_result_count,  &z64, sizeof(u64));
    ov[0] = 0;
    OMP_TARGET_UPDATE(to(ov[0:1]))
    /* new_count needs no reset here: k_reset zeroes the device copy at the top
     * of every fixpoint round, and tc_run_fixpoint primes the host copy. */
    init_base(ctx);
}

/* ----------------------------------------------------------------------------
 * Capacity: discovered, not configured.
 *
 * The result set is a fixed pre-allocated open-addressing table -- that is what
 * makes the per-round kernel sequence loop-invariant and therefore replayable,
 * and it is the whole point of the fused design (the original MNMGDatalog engine
 * needs no capacity because it cudaMallocs and re-sorts the entire relation
 * every round; see MNMGDatalog-reference/tc.cu, which is exactly the cost this
 * design removes).
 *
 * A fixed table needs a size up front, and that size CANNOT be derived from the
 * input: the TC/edge ratio spans 21x (OL.cedge) to 5977x (p2p-Gnutella31). The
 * CUDA benchmark this is ported from resolves that with a hand-maintained
 * per-dataset table (tc_benchmark/tests/benchmark.sh) and a mandatory
 * capacity_mult argument. We instead DISCOVER it:
 *
 *   1. explicit capacity_mult (arg 2)  -> reference-compatible, no cache
 *   2. cache hit                       -> sizes recomputed from a stored TC
 *   3. otherwise                       -> probe solve, then cache
 *
 * The probe runs the real fixpoint once, untimed and UNGRAPHED, on a provisional
 * table, and reports the exact closure size and the peak per-round new_count.
 * It must run before the measured solve because the taskgraph records buffer
 * pointers -- growing the table mid-solve would invalidate a recorded graph, and
 * xkomp has no graph-reset (XKOMP_TASKGRAPH_FLAG_RESET is not supported).
 * ------------------------------------------------------------------------- */

/* Load factor bound: result_cap >= TC_LOAD_HEADROOM * TC, i.e. <= 50% full by
 * default, which is what the reference targets ("must be >= ~2x TC"). */
#ifndef TC_LOAD_HEADROOM
# define TC_LOAD_HEADROOM 2
#endif

/* First table the probe tries, before the trial allocation shrinks it to what
 * the device will actually give. 2^32 slots = 34 GiB covers every dataset in the
 * README capacity table in a single attempt; smaller devices just shrink. */
#ifndef TC_PROBE_MAX_SLOTS
# define TC_PROBE_MAX_SLOTS (1L << 32)
#endif

static const char *tc_cache_path(void)
{
    const char *p = getenv("TC_CACHE");
    return (p && p[0]) ? p : "tc_capacity.cache";
}

static const char *tc_basename(const char *path)
{
    const char *b = strrchr(path, '/');
    return b ? b + 1 : path;
}

/* One line per dataset: "<basename> <edges> <tc> <peak_frontier>". Only the
 * measured quantities are stored, never the derived capacities, so the
 * derivation below can change without invalidating existing entries. */
static bool tc_cache_lookup(const char *input_file, int n_edges, u64 *tc, long *peak)
{
    FILE *f = fopen(tc_cache_path(), "r");
    if (!f) return false;

    const char *want = tc_basename(input_file);
    char line[512];
    bool hit = false;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        char name[256]; long e = 0, pk = 0; unsigned long long t = 0;
        if (sscanf(line, "%255s %ld %llu %ld", name, &e, &t, &pk) != 4) continue;
        if (strcmp(name, want) != 0 || e != (long)n_edges) continue;
        *tc = (u64)t; *peak = pk; hit = true; break;
    }
    fclose(f);
    return hit;
}

/* Best-effort: the cache is an optimisation, so a write failure is silent. */
static void tc_cache_store(const char *input_file, int n_edges, u64 tc, long peak)
{
    const char *path = tc_cache_path();
    const char *want = tc_basename(input_file);

    /* Rewrite, dropping any stale entry for this dataset. */
    char (*keep)[512] = nullptr; int nkeep = 0, cap = 0;
    FILE *f = fopen(path, "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            char name[256];
            if (line[0] != '#' && sscanf(line, "%255s", name) == 1 && !strcmp(name, want))
                continue;
            if (nkeep == cap) {
                cap = cap ? cap * 2 : 32;
                keep = (char (*)[512])realloc(keep, (size_t)cap * 512);
                if (!keep) { fclose(f); return; }
            }
            snprintf(keep[nkeep++], 512, "%s", line);
        }
        fclose(f);
    }

    f = fopen(path, "w");
    if (!f) { free(keep); return; }
    fprintf(f, "# tc.x discovered capacities -- safe to delete, it is only a cache.\n"
               "# <dataset> <edges> <tc> <peak_frontier>\n");
    for (int i = 0; i < nkeep; i++) fputs(keep[i], f);
    fprintf(f, "%s %d %llu %ld\n", want, n_edges, tc, peak);
    fclose(f);
    free(keep);
}

/* The probe: run the fixpoint to convergence, ungraphed and untimed, recording
 * the closure size and the largest per-round new_count. Returns false if the
 * provisional table overflowed, in which case the caller retries bigger.
 *
 * Must be called from inside the enclosing single region: it creates tasks. It
 * deliberately calls tc_round directly rather than tc_run_fixpoint, so that (a)
 * no taskgraph is recorded and (b) every round's new_count is observed -- an
 * unrolled taskgraph loop only samples the count every `unroll` rounds and would
 * under-report the peak. */
static bool tc_size_probe(TCContext &ctx, u64 *tc_out, long *peak_out)
{
    tc_reset_state(ctx);

    int *nc = ctx.new_count; int *ov = ctx.overflow;
    long peak = 0;
    nc[0] = 1;
    while (nc[0] > 0) {
        tc_round(ctx);
        #pragma omp taskwait
        if (ov[0]) return false;
        if (nc[0] > peak) peak = nc[0];
    }
    *tc_out  = tc_read_u64(ctx.d_result_count);
    *peak_out = peak;
    return true;
}

static long tc_cap_from_tc(u64 tc)
{
    long want = (long)(tc * (u64)TC_LOAD_HEADROOM);
    if (want < 4096) want = 4096;
    return tc_next_pow2(want);
}

static long tc_cap_from_peak(long peak, long result_cap)
{
    long want = peak * TC_LOAD_HEADROOM;
    if (want < 4096) want = 4096;
    long fcap = tc_next_pow2(want);
    if (fcap > result_cap) fcap = result_cap;
    return fcap;
}

/* Resolve result_cap / frontier_cap and leave the FINAL buffers allocated.
 * Must be called from inside the enclosing single region (the probe runs tasks). */
static void tc_resolve_capacity(TCContext &ctx, const char *input_file,
                                long capacity_mult, long frontier_slots)
{
    /* --- 1. explicit override: reproduce the reference's sizing exactly ----- */
    if (capacity_mult > 0) {
        ctx.capacity_mult = capacity_mult;
        ctx.cap_source    = TCContext::CAP_OVERRIDE;
        long est = (long)ctx.n_edges * capacity_mult;
        if (est < 4096) est = 4096;
        long rcap = tc_next_pow2(est);
        long fcap = (frontier_slots > 0) ? tc_next_pow2(frontier_slots) : (1L << 28);
        if (fcap > rcap) fcap = rcap;
        tc_alloc_fixpoint(ctx, rcap, fcap, false);
        return;
    }

    /* --- 2. cache ---------------------------------------------------------- */
    u64 tc = 0; long peak = 0;
    if (tc_cache_lookup(input_file, ctx.n_edges, &tc, &peak)) {
        ctx.cap_source = TCContext::CAP_CACHED;
        ctx.sized_tc = tc; ctx.sized_peak = peak;
        long rcap = tc_cap_from_tc(tc);
        long fcap = (frontier_slots > 0) ? tc_next_pow2(frontier_slots)
                                         : tc_cap_from_peak(peak, rcap);
        if (fcap > rcap) fcap = rcap;
        tc_alloc_fixpoint(ctx, rcap, fcap, false);
        return;
    }

    /* --- 3. discovery ------------------------------------------------------ */
    const double s0 = tc_now();

    /* Two bounds on the probe's provisional table: (max_node+1)^2 is the trivial
     * upper bound on the closure size, and TC_PROBE_MAX_SLOTS keeps the first
     * attempt to something a GPU plausibly has (34 GiB by default, which covers
     * every dataset in the README's capacity table in ONE attempt). The trial
     * allocation below shrinks further if the device says no. */
    const long nodes = (long)ctx.max_node + 1;
    long vbound = (nodes < 3037000499L) ? tc_next_pow2(nodes * nodes) : TC_PROBE_MAX_SLOTS;
    long floor_ = tc_next_pow2((long)ctx.n_edges * 64);
    if (floor_ < 4096) floor_ = 4096;

    long prov = vbound < TC_PROBE_MAX_SLOTS ? vbound : TC_PROBE_MAX_SLOTS;
    if (prov < floor_) prov = floor_;

    fprintf(stderr,
        "# sizing pass: closure size unknown for %s -- probing (untimed, cached\n"
        "#              in %s for later runs; pass capacity_mult to skip).\n",
        tc_basename(input_file), tc_cache_path());
    fflush(stderr);

    /* Largest table already known to be TOO SMALL. If the trial allocation ever
     * has to shrink back to it, the closure simply does not fit on this device --
     * without this the loop could oscillate (grow x4, shrink /2) forever. */
    long too_small = 0;

    for (;;) {
        long pf = prov < (1L << 28) ? prov : (1L << 28);
        while (prov >= floor_ && !tc_alloc_fixpoint(ctx, prov, pf, /*try_only=*/true)) {
            prov >>= 1;
            pf = prov < (1L << 28) ? prov : (1L << 28);
        }
        if (prov < floor_ || prov <= too_small) {
            fprintf(stderr,
                "ERROR: cannot size %s -- the device cannot hold a table large\n"
                "       enough for its transitive closure (largest that fits: %ld\n"
                "       slots, known too small: %ld). Use a larger device, or pass\n"
                "       an explicit capacity_mult to bypass the sizing pass.\n",
                tc_basename(input_file), prov, too_small);
            exit(2);
        }

        if (tc_size_probe(ctx, &tc, &peak))     /* tc_reset_state fills the table */
            break;

        /* Overflowed: the closure is bigger than this table. A failed probe is
         * cheap -- it aborts as soon as the table fills. Grow and retry. */
        tc_free_fixpoint(ctx);
        too_small = prov;
        prov <<= 2;
    }
    tc_free_fixpoint(ctx);

    ctx.cap_source = TCContext::CAP_DISCOVERED;
    ctx.sized_tc = tc; ctx.sized_peak = peak;

    long rcap = tc_cap_from_tc(tc);
    long fcap = (frontier_slots > 0) ? tc_next_pow2(frontier_slots)
                                     : tc_cap_from_peak(peak, rcap);
    if (fcap > rcap) fcap = rcap;
    tc_alloc_fixpoint(ctx, rcap, fcap, false);

    ctx.t_sizing = tc_now() - s0;
    tc_cache_store(input_file, ctx.n_edges, tc, peak);

    fprintf(stderr, "# sizing pass: TC=%llu, peak frontier=%ld  ->  result_cap=%ld, "
                    "frontier_cap=%d  (%.3f s)\n",
            tc, peak, ctx.result_cap, ctx.frontier_cap, ctx.t_sizing);
    fflush(stderr);
}

static void tc_teardown(TCContext &ctx)
{
    dfree(ctx.d_edges);       dfree(ctx.d_edge_table);   dfree(ctx.d_result_set);
    dfree(ctx.d_frontier);    dfree(ctx.d_new_frontier);
    dfree(ctx.d_frontier_size);
    dfree(ctx.d_result_count);

    int *new_count = ctx.new_count;
    if (new_count) {
        OMP_TARGET_EXIT_DATA(MAP(release: new_count[0:1]))
        host_free(new_count);
        ctx.new_count = nullptr;
    }

    int *ov = ctx.overflow;
    if (ov) {
        OMP_TARGET_EXIT_DATA(MAP(release: ov[0:1]))
        host_free(ov);
        ctx.overflow = nullptr;
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
/* The reported unit is one taskgraph INSTANCE -- a group of `unroll` fixpoint  */
/* rounds (-u), one round when unrolling is off -- and the value is always ms   */
/* per ROUND, so it stays comparable across -u. The split is the Krylov drivers'*/
/* (krylov/common/driver.cpp), and it is the runtime's own phase structure:     */
/*                                                                            */
/*   instance 0         RECORDS the task graph (XKOMP rc == 1)                 */
/*   instance 1         first replay; also where the command graph is built    */
/*                      and optimized (XKOMP rc == 2)                          */
/*   instances 2..N-1   steady state -> avg / stddev                           */
/*                                                                            */
/* CAVEAT: unlike a Krylov iteration, TC rounds do very different amounts of   */
/* work -- the frontier grows for the first rounds and then collapses. The     */
/* steady-state stddev therefore mostly reflects that frontier-size profile,   */
/* NOT run-to-run jitter. The record/replay comparison (instance 0 and instance */
/* 1 against the steady mean) is what the split is for.                        */
/*                                                                            */
/* CAVEAT (-u > 1): convergence is only tested between instances, so the LAST  */
/* instance may hold up to unroll-1 converged rounds. Those do no work (an     */
/* empty frontier copies nothing), which makes that instance cheap and pulls   */
/* the steady-state mean down -- the more so the fewer instances there are.    */
/*                                                                            */
/* End-to-end "total time" is the MNMGDatalog paper's metric and per-phase     */
/* breakdown (Table "End-to-end total time (ms)" / Fig. "TC per-phase total    */
/* time breakdown"): file IO + H2D + setup + compute + D2H, where compute is   */
/* the whole measured fixpoint plus the one-shot result compaction. Unlike the  */
/* CUDA reference -- which captures and instantiates the graph in a separate,  */
/* separately-timed phase -- XKOMP records and builds INSIDE the loop, so there */
/* is no separate build phase: that cost sits in compute, visible as instances */
/* 0 and 1. Every number printed is measured; none is extrapolated.            */
/* ------------------------------------------------------------------------- */
typedef struct {
    double round0_s;        /* instance 0: records the task graph             */
    double round1_s;        /* instance 1: graph build + first replay         */
    double steady_s;        /* mean of the steady-state instances             */
    double steady_sd_s;     /* sample stddev of the steady-state instances    */
    int    steady_from;     /* index of the first steady-state instance       */
    int    steady_n;        /* number of steady-state instances               */
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
/* main. Usage: ./tc.x <data.bin> [capacity_mult] [frontier_slots] [-u <n>]    */
/*   -u <n> : rounds per taskgraph instance (default 1). The taskgraph carries  */
/*            an implicit taskgroup, so consecutive instances cannot overlap;   */
/*            unrolling recovers that overlap inside the graph. The convergence  */
/*            test then only runs every n rounds, so the fixpoint may overshoot. */
/*   Env: TC_WARMUP=<n> untimed warm-up rounds before round 0 (default 3);     */
/*        TC_UNROLL=<n> same as -u;                                            */
/*        TC_WORKERS=<n> grid-stride worker count (GPU);                       */
/*        TC_WRITE=1 writes <input>_<version>_tc.bin; TC_DUMP=<f> text dump;   */
/*        TC_CSV=<f> writes the 15-column machine row.                         */
/* ------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    /* positional args, skipping any flags */
    const char *pos[3] = { NULL, NULL, NULL };
    int npos = 0, unroll = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-u") && i + 1 < argc) unroll = atoi(argv[++i]);
        else if (npos < 3)                          pos[npos++] = argv[i];
    }

    const char *input_file = pos[0] ? pos[0] : "MNMGDatalog-reference/data/data_10.bin";
    /* 0 = auto: discover the closure size and right-size the tables. A positive
     * value reproduces the CUDA reference's next_pow2(edges*mult) sizing. */
    long capacity_mult  = pos[1] ? atol(pos[1]) : 0;
    long frontier_slots = pos[2] ? atol(pos[2]) : 0;

    const char *ur = getenv("TC_UNROLL");
    if (ur && ur[0]) unroll = atoi(ur);
    if (unroll < 1) unroll = 1;

    int warmups = 3;
    const char *wu = getenv("TC_WARMUP");
    if (wu && wu[0]) warmups = atoi(wu);
    if (warmups < 0) warmups = 0;

    TCContext ctx;
    tc_setup_input(ctx, input_file);

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
        /* Sizes the result set / frontiers. May run an untimed, ungraphed probe
         * solve; it must happen HERE, before anything is recorded, because the
         * taskgraph captures buffer pointers. */
        tc_resolve_capacity(ctx, input_file, capacity_mult, frontier_slots);

        tc_reset_state(ctx);
        tc_warmup(ctx, warmups);

        tc_reset_state(ctx);
        double f0 = omp_get_wtime();
        rounds = tc_run_fixpoint(ctx, &times, unroll);
        fixpoint_s = omp_get_wtime() - f0;
    }
    tc_check_overflow(ctx, "result set / frontier", -1);
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
    if (unroll > 1)
        printf("  %-11s: %d rounds per taskgraph instance (%d instances; the "
               "fixpoint may overshoot by up to %d rounds)\n",
               "unroll", unroll, rounds / unroll, unroll - 1);
#if USE_TARGET
    printf("  %-11s: %d grid-stride workers\n", "geometry", ctx.n_workers);
#endif
    printf("  %-11s: %d untimed round%s (ungraphed)\n",
           "warm-up", warmups, warmups == 1 ? "" : "s");
    printf("  %-11s: result_cap=%ld slots (%.2f GiB), frontier_cap=%d (%.2f GiB each)\n",
           "capacity", ctx.result_cap,
           (double)ctx.result_cap * (double)sizeof(u64) / (1024.0 * 1024.0 * 1024.0),
           ctx.frontier_cap,
           (double)ctx.frontier_cap * (double)sizeof(u64) / (1024.0 * 1024.0 * 1024.0));
    if (ctx.cap_source == TCContext::CAP_OVERRIDE)
        printf("  %-11s  from capacity_mult=%ld (reference sizing)\n", "", ctx.capacity_mult);
    else
        printf("  %-11s  %s: TC=%llu, peak frontier=%ld\n", "",
               ctx.cap_source == TCContext::CAP_CACHED ? "cached" : "discovered",
               ctx.sized_tc, ctx.sized_peak);
    printf("  %-11s: %.2f MB\n", "peak memory", ctx.peak_mem_mb);

    printf("Statistics\n");
    printf("  %-27s : %10.3f ms\n", "total time (end-to-end)", total_s * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  file IO",      fileio      * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  H2D transfer", ctx.t_h2d   * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  setup",        ctx.t_setup * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  compute",      compute_s   * 1000.0);
    printf("  %-27s : %10.3f ms\n", "  D2H transfer", d2h         * 1000.0);
    if (ctx.t_sizing > 0.0)
        printf("  %-27s : %10.3f ms   (once per dataset, cached; NOT in the total)\n",
               "sizing pass", ctx.t_sizing * 1000.0);

    {
        char lbl[64];
        const int graphed = (USE_TASKGRAPH && !USE_SYNC);
        /* Entries are per taskgraph INSTANCE -- a group of `unroll` rounds (-u),
         * one round when unrolling is off -- and always in ms per ROUND, so they
         * stay comparable across -u. Instance 0 records the graph, instance 1
         * builds and optimizes the command graph and runs the first replay, and
         * instances 2.. are the steady state. */
        snprintf(lbl, sizeof lbl, "instance 0%s", graphed ? " (record)" : "");
        printf("  %-27s : %10.3f ms\n", lbl, st.round0_s * 1000.0);
        if (times.n >= 2) {
            snprintf(lbl, sizeof lbl, "instance 1%s", graphed ? " (1st replay)" : "");
            printf("  %-27s : %10.3f ms\n", lbl, st.round1_s * 1000.0);
        }
        if (times.n >= 3) {
            snprintf(lbl, sizeof lbl, "instances %d..%d (avg)", st.steady_from, times.n - 1);
            printf("  %-27s : %10.3f ms   (%d instances)\n", lbl, st.steady_s * 1000.0, st.steady_n);
            snprintf(lbl, sizeof lbl, "instances %d..%d (stddev)", st.steady_from, times.n - 1);
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
