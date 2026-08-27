/*
 * tc.cpp - Datalog Transitive Closure (TC) as OpenMP target tasks / taskgraph.
 *
 * This is an OpenMP port of MNMGDatalog-reference/tc_benchmark (CUDA). It computes
 * the semi-naive TC fixpoint
 *
 *     path(a, c) :- path(a, b), edge(b, c).
 *
 * over fixed pre-allocated device buffers, with all per-iteration SIZES resident
 * in device memory (d_frontier_size / d_new_count). The kernels read those sizes
 * on the device at launch, so the per-iteration kernel sequence is byte-identical
 * every round -- which is exactly what lets one recorded task graph be replayed.
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
 *                             host still copies new_count back to test convergence.
 *
 * USE_TARGET selects GPU offload (1) vs host CPU tasks (0); USE_SYNC selects a
 * blocking (synchronous) schedule. v3_conditional (on-GPU conditional loop) is
 * intentionally NOT ported: CGIR/XKOMP has no conditional-node support yet.
 *
 * Memory model: the large buffers are device-only (omp_target_alloc) and every
 * target construct reaches them via is_device_ptr(...) -- the direct analog of
 * the reference's cudaMalloc, with no host mirror of the (possibly multi-GB)
 * result set. Small scalars are read back with omp_target_memcpy. On the CPU
 * backend the same pointers are plain malloc and the kernels become host tasks.
 *
 * Atomics: the open-addressing hash set / edge table are built with a
 * compare-and-swap (#pragma omp atomic compare capture, OpenMP 5.1) and the
 * append counters with fetch-add (#pragma omp atomic capture) -- one portable
 * code path for host and device. NOTE: on-device `omp atomic compare capture`
 * codegen must be confirmed on your XKOMP/clang + NVPTX toolchain.
 */
#include "tasking.h"

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

    int *d_frontier_size = nullptr;
    int *d_new_count     = nullptr;
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
/* single-statement device ops. This is what replays correctly:               */
/* pragma_omp_taskgraph() runs the region body only on the first (record)     */
/* pass; on replay it re-submits the recorded *kernel commands*. A combined   */
/* construct is recorded as one such kernel command, and the loop bound       */
/* `fs[0]`/`nc[0]` is an is_device_ptr dereference evaluated ON THE DEVICE at  */
/* every launch -- the recorded command captures only the (stable) buffer     */
/* pointer, never the size value -- so each replay reads the current frontier  */
/* size, exactly like the CUDA reference's device-resident *d_frontier_size.   */
/* (A bare `omp target` + a separate `omp teams distribute parallel for` is    */
/* NOT this shape and does not replay; see xkomp's taskgraph target tests.)   */
/* ------------------------------------------------------------------------- */
static void k_reset(TCContext &ctx)
{
    int *nc = ctx.d_new_count;
#if USE_TARGET
    OMP_TARGET_TASK(DEPEND(out, nc[0]) is_device_ptr(nc))
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
    int *ncnt = ctx.d_new_count; u64 *rcnt = ctx.d_result_count; int *ov = ctx.d_overflow;
    /* GPU: is_device_ptr (mp slot) carries the device buffers; CPU: default(none)
     * firstprivate (fp slot) captures the pointers/scalars. Bound fs[0] is read on
     * the device each launch, so replay uses the current frontier size. */
    OMP_TILE(DEPEND(in, fs[0], fr[0]) DEPEND(inout, rs[0], ncnt[0], rcnt[0], ov[0]) DEPEND(out, nf[0]),
             is_device_ptr(et, fr, fs, rs, nf, ncnt, rcnt, ov),
             DEFAULT_NONE firstprivate(et, ec, fr, fs, rs, rc, nf, nfc, ncnt, rcnt, ov))
    for (int i = 0; i < fs[0]; i++)
        tc_expand_one(i, et, ec, fr, rs, rc, nf, nfc, ncnt, rcnt, ov);
}

static void k_promote(TCContext &ctx)
{
    u64 *fr = ctx.d_frontier; u64 *nf = ctx.d_new_frontier; int *nc = ctx.d_new_count;
    OMP_TILE(DEPEND(in, nc[0], nf[0]) DEPEND(out, fr[0]),
             is_device_ptr(fr, nf, nc),
             DEFAULT_NONE firstprivate(fr, nf, nc))
    for (int i = 0; i < nc[0]; i++) fr[i] = nf[i];
}

static void k_set_sizes(TCContext &ctx)
{
    int *fs = ctx.d_frontier_size; int *nc = ctx.d_new_count;
#if USE_TARGET
    OMP_TARGET_TASK(DEPEND(in, nc[0]) DEPEND(out, fs[0]) is_device_ptr(fs, nc))
    { fs[0] = nc[0]; }
#else
    OMP_TASK(DEFAULT_NONE firstprivate(fs, nc) DEPEND(in, nc[0]) DEPEND(out, fs[0]))
    { fs[0] = nc[0]; }
#endif
}

/* One full fixpoint solve; returns the number of rounds. MUST be called from
 * inside a single region (see the enclosing `omp parallel/single` in main): the
 * warm-up and every timed repeat share that one region so the per-iteration body
 * (reset -> expand -> promote -> set_sizes) is recorded once (first round of the
 * warm-up) and REPLAYED for every subsequent round of every repeat -- the direct
 * analog of CUDA v2_cudagraph's build-once / replay. After each round the host
 * copies new_count back (a taskwait, then a device->host read) to test
 * convergence -- the same host round-trip the CUDA v2_cudagraph pays. */
static int tc_run_fixpoint(TCContext &ctx)
{
    int iterations = 0;
    while (true) {
        TASKGRAPH_BEGIN
        {
            k_reset(ctx);
            k_expand(ctx);
            k_promote(ctx);
            k_set_sizes(ctx);
        }
        TASKGRAPH_END

        iterations++;
        if (tc_read_i32(ctx.d_new_count) == 0) break;
    }
    return iterations;
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

static double tc_median(double *v, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; }
    if (n == 0) return 0.0;
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
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
    ctx.d_edges = (int *)dalloc((size_t)ctx.n_edges * 2 * sizeof(int));
    to_dev(ctx.d_edges, edges_host, (size_t)ctx.n_edges * 2 * sizeof(int));
    ctx.t_h2d = tc_now() - t0;
    free(edges_host);

    t0 = tc_now();
    ctx.edge_cap = (int)tc_next_pow2((long)std::ceil(ctx.n_edges / 0.6));
    if (ctx.edge_cap < 2) ctx.edge_cap = 2;
    ctx.d_edge_table = (Entity *)dalloc((size_t)ctx.edge_cap * sizeof(Entity));
    dmemset(ctx.d_edge_table, 0xFF, (size_t)ctx.edge_cap * sizeof(Entity));
    build_edges(ctx);

    long est = (long)ctx.n_edges * capacity_mult;
    if (est < 4096) est = 4096;
    ctx.result_cap = tc_next_pow2(est);

    long fcap = (frontier_slots > 0) ? tc_next_pow2(frontier_slots) : (1L << 28);
    if (fcap > ctx.result_cap) fcap = ctx.result_cap;
    ctx.frontier_cap = (int)fcap;

    ctx.d_result_set    = (u64 *)dalloc(ctx.result_cap * sizeof(u64));
    dmemset(ctx.d_result_set, 0xFF, ctx.result_cap * sizeof(u64));
    ctx.d_frontier      = (u64 *)dalloc((size_t)ctx.frontier_cap * sizeof(u64));
    ctx.d_new_frontier  = (u64 *)dalloc((size_t)ctx.frontier_cap * sizeof(u64));
    ctx.d_frontier_size = (int *)dalloc(sizeof(int));
    ctx.d_new_count     = (int *)dalloc(sizeof(int));
    ctx.d_result_count  = (u64 *)dalloc(sizeof(u64));
    ctx.d_overflow      = (int *)dalloc(sizeof(int));
    ctx.t_setup = tc_now() - t0;

    ctx.peak_mem_mb = (double)((size_t)ctx.n_edges * 2 * sizeof(int)
                             + (size_t)ctx.edge_cap * sizeof(Entity)
                             + (size_t)ctx.result_cap * sizeof(u64)
                             + 2 * (size_t)ctx.frontier_cap * sizeof(u64))
                    / (1024.0 * 1024.0);
}

/* Re-seed the fixpoint state (called before every warm-up and timed solve).
 * Buffer addresses stay stable, so a recorded task graph stays valid. */
static void tc_reset_state(TCContext &ctx)
{
    int z = 0; u64 z64 = 0;
    dmemset(ctx.d_result_set, 0xFF, ctx.result_cap * sizeof(u64));
    to_dev(ctx.d_frontier_size, &z, sizeof(int));
    to_dev(ctx.d_new_count,     &z, sizeof(int));
    to_dev(ctx.d_result_count,  &z64, sizeof(u64));
    to_dev(ctx.d_overflow,      &z, sizeof(int));
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
    dfree(ctx.d_frontier_size); dfree(ctx.d_new_count);
    dfree(ctx.d_result_count);  dfree(ctx.d_overflow);
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
/* main. Usage: ./tc.x <data.bin> [capacity_mult] [repeats] [frontier_slots]  */
/*   Env: TC_WRITE=1 writes <input>_<version>_tc.bin; TC_DUMP=<f> text dump;   */
/*        TC_CSV=<f> writes the 15-column machine row.                         */
/* ------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    const char *input_file = (argc >= 2) ? argv[1] : "../MNMGDatalog-reference/data/data_10.bin";
    long capacity_mult  = (argc >= 3) ? atol(argv[2]) : 64;
    int  repeats        = (argc >= 4) ? atoi(argv[3]) : 1;
    long frontier_slots = (argc >= 5) ? atol(argv[4]) : 0;
    if (repeats < 1) repeats = 1;

    TCContext ctx;
    tc_setup(ctx, input_file, capacity_mult, frontier_slots);

    /* One enclosing parallel/single spans the warm-up AND all timed repeats, so
     * the per-iteration task graph is recorded once (first warm-up round) and
     * replayed for every later round of every repeat -- like CUDA v2_cudagraph's
     * build-once / replay. The warm-up (untimed) absorbs first-launch / JIT /
     * graph-record cost; each repeat is then a pure replay-driven solve. */
    int iterations = 0;
    double warm = 0.0;
    double *times = (double *)malloc(repeats * sizeof(double));
    double min_t = 1e300;

    #pragma omp parallel
    #pragma omp single
    {
        tc_reset_state(ctx);
        double w0 = omp_get_wtime();
        iterations = tc_run_fixpoint(ctx);
        warm = omp_get_wtime() - w0;

        for (int r = 0; r < repeats; r++) {
            tc_reset_state(ctx);
            double s0 = omp_get_wtime();
            iterations = tc_run_fixpoint(ctx);
            times[r] = omp_get_wtime() - s0;
            if (times[r] < min_t) min_t = times[r];
        }
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
    u64 *d_compact = (u64 *)dalloc((size_t)(tc ? tc : 1) * sizeof(u64));
    u64 *d_cnt = (u64 *)dalloc(sizeof(u64)); u64 z64 = 0; to_dev(d_cnt, &z64, sizeof(u64));
    compact(ctx, d_compact, d_cnt);
    double compact_s = tc_now() - c0;

    u64 *host = (u64 *)malloc((size_t)(tc ? tc : 1) * sizeof(u64));
    double t0 = tc_now();
    from_dev(host, d_compact, (size_t)tc * sizeof(u64));
    double d2h = tc_now() - t0;
    dfree(d_compact); dfree(d_cnt);

    double avg, sd; tc_mean_std(times, repeats, &avg, &sd);
    double med = tc_median(times, repeats) + compact_s;
    min_t += compact_s;
    double compute_ms = avg * 1000.0;

    double fileio = ctx.t_fileio;
    if (getenv("TC_WRITE")) {
        double tw = tc_now();
        tc_write_output(host, (long long)tc, input_file);
        fileio += tc_now() - tw;
    }

    /* Human / harness-parseable stdout. */
    printf("# MNMGDatalog TC (OpenMP)  version=%s  input=%s  edges=%d  iterations=%d  TC=%llu\n",
           TC_VERSION, input_file, ctx.input_rows, iterations, tc);
    printf("solve 0 (warmup)      : %.3f ms\n", warm * 1000.0);
    printf("timed solves (avg)    : %.3f ms\n", compute_ms);
    printf("timed solves (stddev) : %.3f ms\n", sd * 1000.0);
    printf("total solve time      : %.6f s\n", med);
    fflush(stdout);

    const char *csv = getenv("TC_CSV");
    if (csv && csv[0]) {
        double total = fileio + ctx.t_h2d + ctx.t_setup + 0.0 + med + d2h;
        tc_print_csv(csv, ctx.input_rows, iterations, tc, total, fileio, ctx.t_h2d,
                     ctx.t_setup, 0.0, med, min_t, d2h, ctx.peak_mem_mb, repeats, input_file);
    }

    const char *dump = getenv("TC_DUMP");
    if (dump && dump[0]) tc_dump(host, (long long)tc, dump);

    free(host); free(times);
    tc_teardown(ctx);
    return 0;
}
