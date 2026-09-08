/*
 * tc_repro.cpp - minimal standalone reproducer for the MNMGDatalog TC device
 * fault (`cuStreamSynchronize failed with an illegal memory access`, and at -O3
 * `Invalid access of peer GPU memory over nvlink`).
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * tc.cpp is the only application in apps/openmp that uses these three OpenMP
 * constructs; krylov, lulesh and HPCCG use none of them:
 *
 *   1. omp_target_alloc device buffers reached with is_device_ptr (never map'd)
 *   2. #pragma omp atomic compare capture   on the device (open-addressing CAS)
 *   3. #pragma omp atomic capture           on the device (append counter)
 *
 * and it is the only one that faults. This file isolates exactly those three, in
 * the same grid-stride shape as tc.cpp's k_expand, with every dimension on the
 * command line and each construct switchable, so the failure can be bisected
 * without any of tc.cpp's surrounding machinery (no tasking.h, no alloc.h, no
 * xkomp headers, no taskgraph, no tasks, no dependences -- plain blocking
 * `omp target` regions only).
 *
 * If this reproduces, it is a self-contained bug report for the toolchain. If it
 * does not, the fault is in tc.cpp's own logic and the difference between the two
 * is the next thing to look at.
 *
 * WHAT IT MIRRORS
 * ---------------
 * k_expand's nest, including the loop bound read from DEVICE memory:
 *
 *     for (t = 0; t < workers; t++) {          // host-constant OpenMP bound
 *         const int n = dn[0];                 // device-resident size
 *         for (i = t; i < n; i += workers)     // grid-stride
 *             insert(hash(i)) into an open-addressing set, append if new;
 *     }
 *
 * BUILD
 *   make repro USE_TARGET=1
 * or by hand:
 *   clang++ -std=c++20 -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
 *           --offload-arch=sm_90 -fopenmp-offload-mandatory \
 *           tc_repro.cpp -o tc_repro.x
 *
 * RUN
 *   ./tc_repro.x [keys] [slots] [workers] [mode] [rounds]
 *     keys     number of insert attempts per round        (default 1000000)
 *     slots    open-addressing table slots, power of two  (default 4*keys, po2)
 *     workers  OpenMP loop bound = grid-stride worker count (default 2097152,
 *              i.e. tc.cpp's TC_WORKERS_DEFAULT)
 *     mode     bitmask, default 3:
 *                bit 0 (1) = atomic compare capture for the table insert
 *                bit 1 (2) = atomic capture for the append counter
 *                0         = neither: plain stores, pure memory traffic
 *     rounds   repeat count, mimicking the fixpoint       (default 8)
 *
 * SWEEPS THAT MATTER (the two knobs the real app is sensitive to)
 *   for w in 4096 65536 1048576 2097152; do ./tc_repro.x 1000000 4194304 $w; done
 *   for k in 10000 100000 1000000 10000000; do ./tc_repro.x $k; done
 *   for m in 0 1 2 3; do ./tc_repro.x 1000000 4194304 2097152 $m; done
 *
 * A -DREPRO_MAPPED build swaps the memory model to malloc + map(present:), i.e.
 * what krylov/lulesh do, keeping the kernels otherwise identical.
 */
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef unsigned long long u64;

#define REPRO_EMPTY 0xFFFFFFFFFFFFFFFFULL

/* ------------------------------------------------------------------------- */
/* Device-callable helpers -- the same shapes as tc.cpp's.                     */
/* ------------------------------------------------------------------------- */
#pragma omp declare target

static inline u64 repro_hash(u64 k, long cap)
{
    k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
    k ^= k >> 27; k *= 0x94d049bb133111ebULL;
    k ^= k >> 31;
    return k & (u64)(cap - 1);
}

/* #pragma omp atomic compare capture (OpenMP 5.1) -- tc.cpp's tc_cas_u64. */
static inline u64 repro_cas(u64 *addr, u64 expected, u64 desired)
{
    u64 old;
    #pragma omp atomic compare capture
    { old = *addr; if (*addr == expected) *addr = desired; }
    return old;
}

/* #pragma omp atomic capture -- tc.cpp's tc_fetch_add_i32. */
static inline int repro_fetch_add(int *addr, int v)
{
    int old;
    #pragma omp atomic capture
    { old = *addr; *addr += v; }
    return old;
}

/* Bounded open-addressing insert; returns true iff the key was newly inserted.
 * Mirrors tc_set_insert, including the probe bound and the overflow flag. */
static inline bool repro_insert(u64 *set, long cap, u64 key, int *flag, int mode)
{
    const u64 mask = (u64)(cap - 1);
    u64 pos = repro_hash(key, cap);

    if (!(mode & 1)) {              /* no atomics: plain racy store */
        set[pos] = key;
        return true;
    }
    for (long probes = 0; probes < cap; probes++) {
        u64 old = repro_cas(&set[pos], REPRO_EMPTY, key);
        if (old == REPRO_EMPTY) return true;
        if (old == key)        return false;
        pos = (pos + 1) & mask;
    }
    *flag = 1;                      /* table full (benign store) */
    return false;
}

/* One "fact": derive a key from the index, insert it, append it if new.
 * The append is exactly tc_expand_one's, unsigned-compared so an overrunning
 * counter cannot index out of bounds. */
static inline void repro_one(int i, u64 *set, long cap, u64 *out, int ocap,
                             int *cnt, int *flag, int mode)
{
    /* |1 keeps the key away from the REPRO_EMPTY marker. */
    u64 key = ((u64)(unsigned)i * 0x9e3779b97f4a7c15ULL) | 1ULL;

    if (!repro_insert(set, cap, key, flag, mode))
        return;

    if (mode & 2) {
        int w = repro_fetch_add(cnt, 1);
        if ((unsigned)w < (unsigned)ocap) out[w] = key; else *flag = 1;
    } else {
        if ((unsigned)i < (unsigned)ocap) out[i] = key;
    }
}

#pragma omp end declare target

/* ----------------------------------------------------------------------------
 * The memory-model clause, and the two directives that carry it.
 *
 * A `#pragma` line is NOT macro-expanded, so the clause has to go through
 * _Pragma with the usual two-level indirection (the outer macro expands its
 * arguments, the inner one stringizes them) -- the same trick tasking.h uses.
 * The clause names variables that exist in main(); macros are textual, so that
 * is fine and it keeps the two models side by side in one place.
 * ------------------------------------------------------------------------- */
#define RP_PRAGMA(...)  _Pragma(#__VA_ARGS__)
#define RP_XPRAGMA(...) RP_PRAGMA(__VA_ARGS__)

#ifdef REPRO_MAPPED
/* krylov / lulesh model: host buffers with a device copy, asserted present. */
# define RP_MEM map(present: set[0:slots], out[0:keys], cnt[0:1], flg[0:1], dn[0:1])
#else
/* tc.cpp model: device-only buffers passed by value, never mapped. */
# define RP_MEM is_device_ptr(set, out, cnt, flg, dn)
#endif

#define RP_TEAMS_LOOP RP_XPRAGMA(omp target teams distribute parallel for RP_MEM)
#define RP_TARGET     RP_XPRAGMA(omp target RP_MEM)

/* ------------------------------------------------------------------------- */
/* Host side.                                                                 */
/* ------------------------------------------------------------------------- */
static long next_pow2(long v) { long p = 1; while (p < v) p <<= 1; return p; }

static void *dcheck(void *p, const char *what, size_t bytes)
{
    if (!p) {
        fprintf(stderr, "ERROR: allocation of %s (%zu bytes) failed\n", what, bytes);
        exit(2);
    }
    return p;
}

int main(int argc, char **argv)
{
    const long keys    = (argc >= 2) ? atol(argv[1]) : 1000000;
    const long slots   = (argc >= 3) ? next_pow2(atol(argv[2])) : next_pow2(keys * 4);
    const int  workers = (argc >= 4) ? atoi(argv[3]) : (1 << 21);
    const int  mode    = (argc >= 5) ? atoi(argv[4]) : 3;
    const int  rounds  = (argc >= 6) ? atoi(argv[5]) : 8;

    if (keys <= 0 || slots <= 0 || workers <= 0 || rounds <= 0) {
        fprintf(stderr, "usage: %s [keys] [slots] [workers] [mode] [rounds]\n", argv[0]);
        return 1;
    }
    if (keys > 0x7fffffffL) {          /* the append counter and dn[0] are int */
        fprintf(stderr, "ERROR: keys must fit in an int\n");
        return 1;
    }

    const int dev  = omp_get_default_device();
    const int host = omp_get_initial_device();
    (void) dev; (void) host;      /* unused by the REPRO_MAPPED build */

    printf("tc_repro: keys=%ld slots=%ld workers=%d mode=%d rounds=%d\n"
           "          atomics: insert=%s append=%s   memory=%s\n"
           "          table=%.1f MiB  out=%.1f MiB\n",
           keys, slots, workers, mode, rounds,
           (mode & 1) ? "atomic compare capture" : "plain store",
           (mode & 2) ? "atomic capture"         : "indexed store",
#ifdef REPRO_MAPPED
           "malloc + map(present:)",
#else
           "omp_target_alloc + is_device_ptr",
#endif
           (double)slots * 8.0 / (1024.0 * 1024.0),
           (double)keys  * 8.0 / (1024.0 * 1024.0));
    fflush(stdout);

    const int  ocap  = (int)keys;
    const long nb_set = slots * (long)sizeof(u64);
    const long nb_out = keys  * (long)sizeof(u64);

#ifdef REPRO_MAPPED
    /* krylov / lulesh memory model: host allocations mapped onto the device. */
    u64 *set = (u64 *)dcheck(malloc((size_t)nb_set), "set", (size_t)nb_set);
    u64 *out = (u64 *)dcheck(malloc((size_t)nb_out), "out", (size_t)nb_out);
    int *cnt = (int *)dcheck(malloc(sizeof(int)),    "cnt", sizeof(int));
    int *flg = (int *)dcheck(malloc(sizeof(int)),    "flg", sizeof(int));
    int *dn  = (int *)dcheck(malloc(sizeof(int)),    "dn",  sizeof(int));
    dn[0] = (int)keys; cnt[0] = 0; flg[0] = 0;
    #pragma omp target enter data map(alloc: set[0:slots], out[0:keys]) \
                                  map(to: cnt[0:1], flg[0:1], dn[0:1])
#else
    /* tc.cpp memory model: device-only buffers, never mapped. */
    u64 *set = (u64 *)dcheck(omp_target_alloc((size_t)nb_set, dev), "set", (size_t)nb_set);
    u64 *out = (u64 *)dcheck(omp_target_alloc((size_t)nb_out, dev), "out", (size_t)nb_out);
    int *cnt = (int *)dcheck(omp_target_alloc(sizeof(int), dev),    "cnt", sizeof(int));
    int *flg = (int *)dcheck(omp_target_alloc(sizeof(int), dev),    "flg", sizeof(int));
    int *dn  = (int *)dcheck(omp_target_alloc(sizeof(int), dev),    "dn",  sizeof(int));
    { int z = 0, k = (int)keys;
      omp_target_memcpy(cnt, &z, sizeof(int), 0, 0, dev, host);
      omp_target_memcpy(flg, &z, sizeof(int), 0, 0, dev, host);
      omp_target_memcpy(dn,  &k, sizeof(int), 0, 0, dev, host); }
#endif

    for (int r = 0; r < rounds; r++)
    {
        printf("round %d: fill ", r); fflush(stdout);

        /* Clear the table -- tc.cpp's fill_result_set. */
        RP_TEAMS_LOOP
        for (long i = 0; i < slots; i++) set[i] = REPRO_EMPTY;

        printf("reset "); fflush(stdout);

        RP_TARGET
        { cnt[0] = 0; }

        printf("expand "); fflush(stdout);

        /* k_expand's nest: host-constant OpenMP bound, device-resident size. */
        RP_TEAMS_LOOP
        for (int t = 0; t < workers; t++) {
            const int n = dn[0];
            for (int i = t; i < n; i += workers)
                repro_one(i, set, slots, out, ocap, cnt, flg, mode);
        }

        printf("readback "); fflush(stdout);

        int h_cnt = 0, h_flg = 0;
#ifdef REPRO_MAPPED
        #pragma omp target update from(cnt[0:1], flg[0:1])
        h_cnt = cnt[0]; h_flg = flg[0];
#else
        omp_target_memcpy(&h_cnt, cnt, sizeof(int), 0, 0, host, dev);
        omp_target_memcpy(&h_flg, flg, sizeof(int), 0, 0, host, dev);
#endif
        printf("-> inserted=%d overflow=%d %s\n", h_cnt, h_flg,
               (mode & 2) ? ((h_cnt == (int)keys) ? "OK" : "*** WRONG COUNT ***") : "");
        fflush(stdout);
    }

    printf("tc_repro: completed %d rounds without a fault\n", rounds);

#ifdef REPRO_MAPPED
    #pragma omp target exit data map(release: set[0:slots], out[0:keys], \
                                              cnt[0:1], flg[0:1], dn[0:1])
    free(set); free(out); free(cnt); free(flg); free(dn);
#else
    omp_target_free(set, dev); omp_target_free(out, dev);
    omp_target_free(cnt, dev); omp_target_free(flg, dev); omp_target_free(dn, dev);
#endif
    return 0;
}
