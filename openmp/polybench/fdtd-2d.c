/*
 * fdtd-2d - PolyBench 2D FDTD (Yee) stencil on the shared tasking.h backends.
 * One timestep is four ordered sub-loops over the ex/ey/hz fields:
 *   (0) ey[0][*]   = _fict_[t]                 (source row, block 0)
 *   (1) ey[i][j]  -= 0.5*(hz[i][j]-hz[i-1][j]) (reads hz row i-1,i ; i=1..nx-1)
 *   (2) ex[i][j]  -= 0.5*(hz[i][j]-hz[i][j-1]) (reads hz row i      ; i=0..nx-1)
 *   (3) hz[i][j]  -= 0.7*(ex[i][j+1]-ex[i][j] + ey[i+1][j]-ey[i][j])
 *                                              (reads ex,ey rows i,i+1; i=0..nx-2)
 *
 * Dependency convention (see jacobi-2d.c): each spatial loop is tiled into -nb
 * row blocks aligned on X[b*RBS*ny]; stencil neighbour rows are covered by naming
 * the adjacent blocks (b-1 for the hz->ey read, b+1 for the ex/ey->hz read). The
 * cross-sub-loop ordering within a step and the hz->(next step) dependence are
 * automatic because the per-block keys collide between sub-loops.
 *
 * The source row reads _fict_[t], so the recorded body reads the mapped _fict_
 * array indexed by t; the graph shape is loop-invariant.
 */
#include "pb_common.h"

#ifndef PB_NX
# define PB_NX 1000
#endif
#ifndef PB_NY
# define PB_NY 1200
#endif
#ifndef PB_TMAX
# define PB_TMAX 500
#endif

static void init_array(int nx, int ny, DATA_TYPE *ex, DATA_TYPE *ey,
                       DATA_TYPE *hz, DATA_TYPE *fict, int tmax)
{
    for (int i = 0; i < tmax; i++) fict[i] = (DATA_TYPE)i;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            ex[i * ny + j] = ((DATA_TYPE)i * (j + 1)) / nx;
            ey[i * ny + j] = ((DATA_TYPE)i * (j + 2)) / ny;
            hz[i * ny + j] = ((DATA_TYPE)i * (j + 3)) / nx;
        }
}

int main(int argc, char **argv)
{
    pb_opts opts;
    pb_parse_args(argc, argv, &opts);

    const int nx   = PB_NX;
    const int ny   = PB_NY;
    const int tmax = pb_tsteps(&opts, PB_TMAX);
    const int unroll = opts.unroll;
    const int ninst  = tmax / unroll;   /* taskgraph instances */
    const size_t npts = (size_t)nx * ny;

    DATA_TYPE *ex   = pb_alloc(npts);
    DATA_TYPE *ey   = pb_alloc(npts);
    DATA_TYPE *hz   = pb_alloc(npts);
    DATA_TYPE *fict = pb_alloc(tmax > 0 ? tmax : 1);
    init_array(nx, ny, ex, ey, hz, fict, tmax);

    /* The source term is fict[t], i.e. it varies per timestep. A recorded graph
     * replays its captured values verbatim, so t cannot be a captured scalar: it
     * lives in this 1-element buffer instead, read by the source task and bumped
     * by a task of its own, both inside the graph (cf. LULESH's `cycle`). */
    int *tstep = (int *)malloc(sizeof(int));
    tstep[0] = 0;

    const int RBS  = pb_block_size(nx, opts.nb);
    const int nblk = (nx + RBS - 1) / RBS;

    /* ms per timestep, one entry per taskgraph instance (see tg_report.h) */
    double *inst_ms = (double *)malloc(sizeof(double) * ninst);
    double prev_ts, t0, tf;

    OMP_TARGET_ENTER_DATA(MAP(to: ex[0:npts], ey[0:npts], hz[0:npts], fict[0:(size_t)tmax], tstep[0:1]))

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
                   [&] (size_t done) { return done < (size_t) tmax; },
                   [&] (size_t inst, size_t done)
    {
        (void) done;
        /* Per-instance timing, OUTSIDE the recorded region, in ms per timestep.
         * Inline where the instance is provably drained (any taskgraph build);
         * elsewhere a host task ordered after the last sweep of every block, which
         * keeps the no-taskgraph configuration pipelined. */
        EPILOGUE_TASK(firstprivate(inst, unroll, inst_ms) shared(prev_ts)
                      DEPEND_MULTI(in, (b=0:nblk), hz[b * RBS * ny]))
        {
            const double now = omp_get_wtime();
            inst_ms[inst] = (now - prev_ts) * 1000.0 / (double) unroll;
            prev_ts = now;
        }
    })
    {
        /* (0) source row ey[0][*] = fict[t] (part of block 0, keyed ey[0]),
         *     t read from the device-side counter, then t += 1 */
        OMP_TILE(DEPEND(in, tstep[0]) DEPEND(inout, ey[0]),
                 MAP(present: ey[0:npts], fict[0:(size_t)tmax], tstep[0:1]) num_teams(1) thread_limit(256),
                 firstprivate(ey, fict, tstep, ny))
        for (int j = 0; j < ny; j++)
            ey[0 * ny + j] = fict[tstep[0]];
        OMP_TARGET_TASK(DEPEND(inout, tstep[0]) MAP(present: tstep[0:1]))
        {
            tstep[0] += 1;
        }

        /* (1) ey[i][j] -= 0.5*(hz[i][j]-hz[i-1][j]) ; reads hz rows i-1,i
         *     -> depend on hz block b and the previous block b-1. */
        for (int b = 0; b < nblk; b++) {
            const int rb = b * RBS < 1 ? 1 : b * RBS;
            const int re = (b + 1) * RBS > nx ? nx : (b + 1) * RBS;
            if (rb >= re) continue;
            const int bm = (b > 0) ? (b - 1) : b;
            OMP_TILE(DEPEND(in, hz[b * RBS * ny], hz[bm * RBS * ny]) DEPEND(inout, ey[b * RBS * ny]),
                     MAP(present: ey[0:npts], hz[0:npts]) num_teams((re - rb)) thread_limit(256),
                     firstprivate(ey, hz, rb, re, ny))
            for (int i = rb; i < re; i++)
                for (int j = 0; j < ny; j++)
                    ey[i * ny + j] -= SCALAR_VAL(0.5) * (hz[i * ny + j] - hz[(i - 1) * ny + j]);
        }

        /* (2) ex[i][j] -= 0.5*(hz[i][j]-hz[i][j-1]) ; reads hz row i only */
        for (int b = 0; b < nblk; b++) {
            const int rb = b * RBS;
            const int re = (b + 1) * RBS > nx ? nx : (b + 1) * RBS;
            if (rb >= re) continue;
            OMP_TILE(DEPEND(in, hz[b * RBS * ny]) DEPEND(inout, ex[b * RBS * ny]),
                     MAP(present: ex[0:npts], hz[0:npts]) num_teams((re - rb)) thread_limit(256),
                     firstprivate(ex, hz, rb, re, ny))
            for (int i = rb; i < re; i++)
                for (int j = 1; j < ny; j++)
                    ex[i * ny + j] -= SCALAR_VAL(0.5) * (hz[i * ny + j] - hz[i * ny + (j - 1)]);
        }

        /* (3) hz[i][j] -= 0.7*(ex[i][j+1]-ex[i][j] + ey[i+1][j]-ey[i][j])
         *     reads ex,ey rows i,i+1 -> depend on blocks b and b+1. i=0..nx-2 */
        for (int b = 0; b < nblk; b++) {
            const int rb = b * RBS;
            const int re = (b + 1) * RBS > nx - 1 ? nx - 1 : (b + 1) * RBS;
            if (rb >= re) continue;
            const int bp = (b < nblk - 1) ? (b + 1) : b;
            OMP_TILE(DEPEND(in, ex[b * RBS * ny], ex[bp * RBS * ny], ey[b * RBS * ny], ey[bp * RBS * ny])
                     DEPEND(inout, hz[b * RBS * ny]),
                     MAP(present: ex[0:npts], ey[0:npts], hz[0:npts]) num_teams((re - rb)) thread_limit(256),
                     firstprivate(ex, ey, hz, rb, re, ny))
            for (int i = rb; i < re; i++)
                for (int j = 0; j < ny - 1; j++)
                    hz[i * ny + j] -= SCALAR_VAL(0.7) *
                        (ex[i * ny + (j + 1)] - ex[i * ny + j]
                       + ey[(i + 1) * ny + j] - ey[i * ny + j]);
        }
    }
    TASKGRAPH_LOOP_END
#pragma omp taskwait
    tf = omp_get_wtime();
} /* single */
} /* parallel */

    OMP_TARGET_UPDATE(from(ex[0:npts], ey[0:npts], hz[0:npts]))
    OMP_TARGET_EXIT_DATA(MAP(release: ex[0:npts], ey[0:npts], hz[0:npts], fict[0:(size_t)tmax], tstep[0:1]))

    const double lup = 3.0 * (double)nx * ny;   /* three field sub-loops */
    pb_report("fdtd-2d", opts.nb, inst_ms, ninst, unroll, tf - t0, lup,
              pb_checksum(ex, npts) + pb_checksum(ey, npts) + pb_checksum(hz, npts));
    if (opts.verify) {
        printf("checksum ex          = %.10e\n", pb_checksum(ex, npts));
        printf("checksum ey          = %.10e\n", pb_checksum(ey, npts));
        printf("checksum hz          = %.10e\n", pb_checksum(hz, npts));
    }

    free(inst_ms);
    pb_free(ex); pb_free(ey); pb_free(hz); pb_free(fict);
    free(tstep);
    return 0;
}
