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
    const int tsteps = pb_tsteps(&opts, PB_TSTEPS);
    const int unroll = opts.unroll;
    const int ninst  = tsteps / unroll;   /* taskgraph instances */
    const size_t npts = (size_t)n * n;

    DATA_TYPE *A = pb_alloc(npts);
    DATA_TYPE *B = pb_alloc(npts);
    init_array(n, A, B);

    /* Interior rows are [1, n-1). Block them into RBS-row tiles aligned to the
     * absolute row index, so block b owns rows [b*RBS, (b+1)*RBS) clipped to
     * [1, n-1). The aligned key for block b is X[b*RBS*n]. */
    const int RBS   = pb_block_size(n, opts.nb);
    const int nblk  = (n + RBS - 1) / RBS;


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
                              DEPEND_MULTI(in, (b=0:nblk), A[b * RBS * n]))
                {
                    const double now = omp_get_wtime();
                    inst_ms[inst] = (now - prev_ts) * 1000.0 / (double) unroll;
                    prev_ts = now;
                }
            })
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
            TASKGRAPH_LOOP_END
            #pragma omp taskwait
            tf = omp_get_wtime();
        }
    }

    OMP_TARGET_UPDATE(from(A[0:npts]))
    OMP_TARGET_EXIT_DATA(MAP(release: A[0:npts], B[0:npts]))

    const double lup = (double)(n - 2) * (n - 2) * 2.0;   /* two sweeps */
    pb_report("jacobi-2d", opts.nb, inst_ms, ninst, unroll, tf - t0, lup,
              pb_checksum(A, npts));
    if (opts.verify)
        printf("checksum A           = %.10e\n", pb_checksum(A, npts));

    free(inst_ms);
    pb_free(A);
    pb_free(B);
    return 0;
}
