/*
 * heat-3d - PolyBench 7-point 3D Jacobi stencil on the shared tasking.h backends.
 * One timestep is two ping-pong sweeps (A->B, B->A); each sweep is tiled into
 * -nb blocks over the outer index i (planes of size n*n).
 *
 * Dependency convention (see jacobi-2d.c): block b keys its I/O on the aligned
 * address X[b*RBS*plane]; the 7-point stencil reads neighbour planes i-1/i+1, so
 * a tile names its neighbour blocks b-1/b+1 as `in` deps too. Flat row-major
 * A[(i*n + j)*n + k]; the PolyBench heat-3d spatial N is small (3D), default 120.
 */
#include "pb_common.h"

#ifndef PB_N
# define PB_N 120
#endif
#ifndef PB_TSTEPS
# define PB_TSTEPS 500
#endif

static void init_array(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                A[(i * n + j) * n + k] = B[(i * n + j) * n + k] =
                    (DATA_TYPE)(i + j + (n - k)) * 10 / n;
}

int main(int argc, char **argv)
{
    pb_opts opts;
    pb_parse_args(argc, argv, &opts);

    const int n      = PB_N;
    const int tsteps = pb_tsteps(&opts, PB_TSTEPS);
    const int unroll = opts.unroll;
    const int ninst  = tsteps / unroll;   /* taskgraph instances */
    const size_t npts = (size_t)n * n * n;
    const int plane = n * n;

    DATA_TYPE *A = pb_alloc(npts);
    DATA_TYPE *B = pb_alloc(npts);
    init_array(n, A, B);

    const int RBS  = pb_block_size(n, opts.nb);
    const int nblk = (n + RBS - 1) / RBS;

    /* ms per timestep, one entry per taskgraph instance (see tg_report.h) */
    double *inst_ms = (double *)malloc(sizeof(double) * ninst);
    double prev_ts, t0, tf;

    OMP_TARGET_ENTER_DATA(MAP(to: A[0:npts], B[0:npts]))

    #pragma omp parallel
    {
        #pragma omp single
        {
            t0 = omp_get_wtime();
            prev_ts = t0;
            /* `unroll` timesteps per taskgraph instance (-u). The body is one timestep and
             * is loop-invariant: it takes no timestep index, so recording it once and
             * replaying it is exact. */
            TASKGRAPH_LOOP(unroll,
                           [&] (size_t done) { return done < (size_t) tsteps; },
                           [&] (size_t inst, size_t done)
            {
                (void) done;
                /* Per-instance timing, OUTSIDE the recorded region, in ms per timestep.
                 * Inline where the instance is provably drained (any taskgraph build);
                 * elsewhere a host task ordered after the last sweep of every block, which
                 * keeps the no-taskgraph configuration pipelined. */
                EPILOGUE_TASK(firstprivate(inst, unroll, inst_ms) shared(prev_ts)
                              DEPEND_MULTI(in, (b=0:nblk), A[b * RBS * plane]))
                {
                    const double now = omp_get_wtime();
                    inst_ms[inst] = (now - prev_ts) * 1000.0 / (double) unroll;
                    prev_ts = now;
                }
            })
            {
                for (int b = 0; b < nblk; b++) {
                    const int rb = b * RBS < 1 ? 1 : b * RBS;
                    const int re = (b + 1) * RBS > n - 1 ? n - 1 : (b + 1) * RBS;
                    if (rb >= re) continue;
                    const int bm = (b > 0) ? (b - 1) : b;
                    const int bp = (b < nblk - 1) ? (b + 1) : b;
                    OMP_TILE(DEPEND(in, A[b * RBS * plane], A[bm * RBS * plane], A[bp * RBS * plane])
                             DEPEND(out, B[b * RBS * plane]),
                             MAP(present: A[0:npts], B[0:npts]) num_teams((re - rb)) thread_limit(256),
                             firstprivate(A, B, rb, re, n, plane))
                    for (int i = rb; i < re; i++)
                        for (int j = 1; j < n - 1; j++)
                            for (int k = 1; k < n - 1; k++) {
                                const size_t c = (size_t)(i * n + j) * n + k;
                                B[c] = SCALAR_VAL(0.125) * (A[c + plane] - SCALAR_VAL(2.0) * A[c] + A[c - plane])
                                     + SCALAR_VAL(0.125) * (A[c + n]     - SCALAR_VAL(2.0) * A[c] + A[c - n])
                                     + SCALAR_VAL(0.125) * (A[c + 1]     - SCALAR_VAL(2.0) * A[c] + A[c - 1])
                                     + A[c];
                            }
                }
                for (int b = 0; b < nblk; b++) {
                    const int rb = b * RBS < 1 ? 1 : b * RBS;
                    const int re = (b + 1) * RBS > n - 1 ? n - 1 : (b + 1) * RBS;
                    if (rb >= re) continue;
                    const int bm = (b > 0) ? (b - 1) : b;
                    const int bp = (b < nblk - 1) ? (b + 1) : b;
                    OMP_TILE(DEPEND(in, B[b * RBS * plane], B[bm * RBS * plane], B[bp * RBS * plane])
                             DEPEND(out, A[b * RBS * plane]),
                             MAP(present: A[0:npts], B[0:npts]) num_teams((re - rb)) thread_limit(256),
                             firstprivate(A, B, rb, re, n, plane))
                    for (int i = rb; i < re; i++)
                        for (int j = 1; j < n - 1; j++)
                            for (int k = 1; k < n - 1; k++) {
                                const size_t c = (size_t)(i * n + j) * n + k;
                                A[c] = SCALAR_VAL(0.125) * (B[c + plane] - SCALAR_VAL(2.0) * B[c] + B[c - plane])
                                     + SCALAR_VAL(0.125) * (B[c + n]     - SCALAR_VAL(2.0) * B[c] + B[c - n])
                                     + SCALAR_VAL(0.125) * (B[c + 1]     - SCALAR_VAL(2.0) * B[c] + B[c - 1])
                                     + B[c];
                            }
                }
            }
            TASKGRAPH_LOOP_END
            #pragma omp taskwait
            tf = omp_get_wtime();
        }
    }

    OMP_TARGET_UPDATE(from(A[0:npts]))
    OMP_TARGET_EXIT_DATA(MAP(release: A[0:npts], B[0:npts]))

    const double lup = (double)(n - 2) * (n - 2) * (n - 2) * 2.0;
    pb_report("heat-3d", opts.nb, inst_ms, ninst, unroll, tf - t0, lup,
              pb_checksum(A, npts));
    if (opts.verify)
        printf("checksum A           = %.10e\n", pb_checksum(A, npts));

    free(inst_ms);
    pb_free(A);
    pb_free(B);
    return 0;
}
