/*
 * jacobi-1d - PolyBench 3-point Jacobi stencil on the shared tasking.h backends.
 * One timestep is two ping-pong sweeps (A->B, B->A); each sweep is tiled into
 * -nb blocks over i.
 *
 * Dependency convention (see jacobi-2d.c): block b keys its I/O on the aligned
 * address X[b*BS]; a tile reading the 3-point stencil also names its neighbour
 * blocks b-1/b+1 as `in` deps so the producer/consumer addresses collide. This
 * is portable across runtimes that match depend sections only by start address.
 */
#include "pb_common.h"

#ifndef PB_N
# define PB_N 2000
#endif
#ifndef PB_TSTEPS
# define PB_TSTEPS 500
#endif

static void init_array(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    for (int i = 0; i < n; i++) {
        A[i] = ((DATA_TYPE)i + 2) / n;
        B[i] = ((DATA_TYPE)i + 3) / n;
    }
}

int main(int argc, char **argv)
{
    pb_opts opts;
    pb_parse_args(argc, argv, &opts);

    const int n      = PB_N;
    const int tsteps = (opts.iters > 0) ? opts.iters : PB_TSTEPS;
    const size_t npts = (size_t)n;

    DATA_TYPE *A = pb_alloc(npts);
    DATA_TYPE *B = pb_alloc(npts);
    init_array(n, A, B);

    const int BS   = pb_block_size(n, opts.nb);
    const int nblk = (n + BS - 1) / BS;

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
            for (int b = 0; b < nblk; b++) {
                const int rb = b * BS < 1 ? 1 : b * BS;
                const int re = (b + 1) * BS > n - 1 ? n - 1 : (b + 1) * BS;
                if (rb >= re) continue;
                const int bm = (b > 0) ? (b - 1) : b;
                const int bp = (b < nblk - 1) ? (b + 1) : b;
                OMP_TILE(DEPEND(in, A[b * BS], A[bm * BS], A[bp * BS]) DEPEND(out, B[b * BS]),
                         MAP(present: A[0:npts], B[0:npts]) num_teams((re - rb)) thread_limit(256),
                         firstprivate(A, B, rb, re))
                for (int i = rb; i < re; i++)
                    B[i] = SCALAR_VAL(0.33333) * (A[i - 1] + A[i] + A[i + 1]);
            }
            for (int b = 0; b < nblk; b++) {
                const int rb = b * BS < 1 ? 1 : b * BS;
                const int re = (b + 1) * BS > n - 1 ? n - 1 : (b + 1) * BS;
                if (rb >= re) continue;
                const int bm = (b > 0) ? (b - 1) : b;
                const int bp = (b < nblk - 1) ? (b + 1) : b;
                OMP_TILE(DEPEND(in, B[b * BS], B[bm * BS], B[bp * BS]) DEPEND(out, A[b * BS]),
                         MAP(present: A[0:npts], B[0:npts]) num_teams((re - rb)) thread_limit(256),
                         firstprivate(A, B, rb, re))
                for (int i = rb; i < re; i++)
                    A[i] = SCALAR_VAL(0.33333) * (B[i - 1] + B[i] + B[i + 1]);
            }
        }
        TASKGRAPH_END

        OMP_HOST_TASK(firstprivate(t, iter_times) shared(prev_ts) DEPEND(in, A[0]))
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

    const double lup = (double)(n - 2) * 2.0;
    pb_report("jacobi-1d", USE_TASKGRAPH, backend, opts.nb,
              iter_times, tsteps, tf - t0, lup);
    if (opts.verify)
        printf("checksum             = %.10e\n", pb_checksum(A, npts));

    free(iter_times);
    pb_free(A);
    pb_free(B);
    return 0;
}
