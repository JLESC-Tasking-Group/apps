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
    const int tsteps = pb_tsteps(&opts, PB_TSTEPS);
    const int unroll = opts.unroll;
    const int ninst  = tsteps / unroll;   /* taskgraph instances */
    const size_t npts = (size_t)n;

    DATA_TYPE *A = pb_alloc(npts);
    DATA_TYPE *B = pb_alloc(npts);
    init_array(n, A, B);

    const int BS   = pb_block_size(n, opts.nb);
    const int nblk = (n + BS - 1) / BS;

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
                              DEPEND_MULTI(in, (b=0:nblk), A[b * BS]))
                {
                    const double now = omp_get_wtime();
                    inst_ms[inst] = (now - prev_ts) * 1000.0 / (double) unroll;
                    prev_ts = now;
                }
            })
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
            TASKGRAPH_LOOP_END
            #pragma omp taskwait
            tf = omp_get_wtime();
        }
    }

    OMP_TARGET_UPDATE(from(A[0:npts]))
    OMP_TARGET_EXIT_DATA(MAP(release: A[0:npts], B[0:npts]))

    const double lup = (double)(n - 2) * 2.0;
    pb_report("jacobi-1d", opts.nb, inst_ms, ninst, unroll, tf - t0, lup,
              pb_checksum(A, npts));
    if (opts.verify)
        printf("checksum A           = %.10e\n", pb_checksum(A, npts));

    free(inst_ms);
    pb_free(A);
    pb_free(B);
    return 0;
}
