/*
 * jacobi-2d - PolyBench 5-point Jacobi stencil, ported to the shared tasking.h
 * backends (CPU tasks / GPU target / synchronous / taskgraph). One timestep is
 * two ping-pong sweeps (A->B then B->A), each a fully-parallel stencil tiled
 * into -nb row blocks. The per-timestep task body is recorded/replayed by the
 * taskgraph.
 *
 * Dependency convention (portable across OpenMP runtimes, incl. the ones that
 * only match depend sections by their *start address*): every tile of block b
 * keys its I/O on the block-aligned address X[b*RBS*n]. A tile reading the
 * 5-point stencil also touches the last/first row of the neighbour blocks b-1
 * and b+1, so it names those blocks' aligned addresses as extra `in` deps. The
 * writer of a block and any reader naming that block therefore collide exactly,
 * which is what the runtime needs to build the edge. This mirrors the LULESH
 * port (single-element block-aligned keys + explicit neighbour deps).
 *
 * Flat row-major A[i*n + j].
 */
#include "pb_common.h"

#ifndef PB_N
# define PB_N 1300
#endif
#ifndef PB_TSTEPS
# define PB_TSTEPS 500
#endif

static void init_array(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i * n + j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
            B[i * n + j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
        }
}

int main(int argc, char **argv)
{
    pb_opts opts;
    pb_parse_args(argc, argv, &opts);

    const int n      = PB_N;
    const int tsteps = (opts.iters > 0) ? opts.iters : PB_TSTEPS;
    const size_t npts = (size_t)n * n;

    DATA_TYPE *A = pb_alloc(npts);
    DATA_TYPE *B = pb_alloc(npts);
    init_array(n, A, B);

    /* Interior rows are [1, n-1). Block them into RBS-row tiles aligned to the
     * absolute row index, so block b owns rows [b*RBS, (b+1)*RBS) clipped to
     * [1, n-1). The aligned key for block b is X[b*RBS*n]. */
    const int RBS   = pb_block_size(n, opts.nb);
    const int nblk  = (n + RBS - 1) / RBS;

#if USE_SYNC
    const char *backend = "synchronous (blocking)";
#elif USE_TARGET
    const char *backend = "GPU target tasks";
#else
    const char *backend = "CPU tasks";
#endif

    double *iter_times = (double *)malloc(sizeof(double) * (tsteps > 0 ? tsteps : 1));
    double prev_ts, t0, tf;

    OMP_TARGET_ENTER_DATA(MAP(to: A[0:npts], B[0:npts]))

#pragma omp parallel
{
#pragma omp single
{
    t0 = omp_get_wtime();
    prev_ts = t0;
    for (int t = 0; t < tsteps; t++) {
        TASKGRAPH_BEGIN
        {
            /* sweep 1: B = 0.2*(A 5-point) */
            for (int b = 0; b < nblk; b++) {
                const int rb = b * RBS < 1 ? 1 : b * RBS;
                const int re = (b + 1) * RBS > n - 1 ? n - 1 : (b + 1) * RBS;
                if (rb >= re) continue;
                const int bm = (b > 0) ? (b - 1) : b;          /* neighbour blocks */
                const int bp = (b < nblk - 1) ? (b + 1) : b;
                OMP_TILE(DEPEND(in, A[b * RBS * n], A[bm * RBS * n], A[bp * RBS * n])
                         DEPEND(out, B[b * RBS * n]),
                         MAP(present: A[0:npts], B[0:npts]) num_teams((re - rb)) thread_limit(256),
                         firstprivate(A, B, rb, re, n))
                for (int i = rb; i < re; i++)
                    for (int j = 1; j < n - 1; j++)
                        B[i * n + j] = SCALAR_VAL(0.2) *
                            (A[i * n + j] + A[i * n + (j - 1)] + A[i * n + (j + 1)]
                           + A[(i + 1) * n + j] + A[(i - 1) * n + j]);
            }
            /* sweep 2: A = 0.2*(B 5-point) */
            for (int b = 0; b < nblk; b++) {
                const int rb = b * RBS < 1 ? 1 : b * RBS;
                const int re = (b + 1) * RBS > n - 1 ? n - 1 : (b + 1) * RBS;
                if (rb >= re) continue;
                const int bm = (b > 0) ? (b - 1) : b;
                const int bp = (b < nblk - 1) ? (b + 1) : b;
                OMP_TILE(DEPEND(in, B[b * RBS * n], B[bm * RBS * n], B[bp * RBS * n])
                         DEPEND(out, A[b * RBS * n]),
                         MAP(present: A[0:npts], B[0:npts]) num_teams((re - rb)) thread_limit(256),
                         firstprivate(A, B, rb, re, n))
                for (int i = rb; i < re; i++)
                    for (int j = 1; j < n - 1; j++)
                        A[i * n + j] = SCALAR_VAL(0.2) *
                            (B[i * n + j] + B[i * n + (j - 1)] + B[i * n + (j + 1)]
                           + B[(i + 1) * n + j] + B[(i - 1) * n + j]);
            }
        }
        TASKGRAPH_END

        /* timing host task: depend on every A block so it fires after sweep 2. */
        OMP_HOST_TASK(firstprivate(t, iter_times, n, RBS, nblk) shared(prev_ts)
                      DEPEND(in, A[0]))
        {
            const double now = omp_get_wtime();
            iter_times[t] = (now - prev_ts) * 1000.0;
            prev_ts = now;
        }
    }
#pragma omp taskwait
    tf = omp_get_wtime();
} /* single */
} /* parallel */

    OMP_TARGET_UPDATE(from(A[0:npts]))
    OMP_TARGET_EXIT_DATA(MAP(release: A[0:npts], B[0:npts]))

    const double lup = (double)(n - 2) * (n - 2) * 2.0;   /* two sweeps */
    pb_report("jacobi-2d", USE_TASKGRAPH, backend, opts.nb,
              iter_times, tsteps, tf - t0, lup);
    if (opts.verify)
        printf("checksum             = %.10e\n", pb_checksum(A, npts));

    free(iter_times);
    pb_free(A);
    pb_free(B);
    return 0;
}
