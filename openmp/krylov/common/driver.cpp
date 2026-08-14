/*
 * driver.cpp - the single command-line driver shared by every Krylov solver.
 *
 * It parses the command line, builds the requested test system, prints the
 * banner, runs the solver (via `krylov_descriptor.solve`), verifies the result
 * against the known all-ones exact solution, and reports the timing statistics.
 * Each solver only provides its algorithm plus a `krylov_descriptor`.
 */
#include "driver.h"
#include "spmat.h"
#include "kalloc.h"
#include "tasking.h" /* USE_TARGET / USE_TASKGRAPH for the banner */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GRID_N
# define GRID_N 64
#endif

/* ==========================================================================
 * Statistics.
 * ========================================================================== */
void krylov_stats_init(KrylovStats *s, int niter)
{
    s->niter   = niter;
    s->total_s = 0.0;
    s->iter_ms = (double *) calloc((size_t)(niter > 0 ? niter : 1), sizeof(double));
}

void krylov_stats_free(KrylovStats *s)
{
    free(s->iter_ms);
    s->iter_ms = NULL;
}

void krylov_stats_report(const KrylovStats *s, const char *unit, int taskgraph)
{
    char lbl[64];

    printf("Statistics\n");
    printf("  %-27s : %10.4f s\n", "total solve time", s->total_s);

    if (s->niter >= 1) {
        snprintf(lbl, sizeof lbl, "%s 0%s", unit, taskgraph ? " (record)" : "");
        printf("  %-27s : %10.3f ms\n", lbl, s->iter_ms[0]);
    }
    if (s->niter >= 2) {
        snprintf(lbl, sizeof lbl, "%s 1%s", unit, taskgraph ? " (1st replay)" : "");
        printf("  %-27s : %10.3f ms\n", lbl, s->iter_ms[1]);
    }
    if (s->niter >= 3) {
        /* Steady-state iterations 2..niter-1 (iteration 0 = record and iteration
         * 1 = first replay are excluded as outliers). Report mean and the
         * (sample) standard deviation of the per-iteration time. */
        const int cnt = s->niter - 2;
        double sum = 0.0;
        for (int i = 2; i < s->niter; i++) sum += s->iter_ms[i];
        const double mean = sum / cnt;

        double var = 0.0;
        for (int i = 2; i < s->niter; i++) {
            const double dv = s->iter_ms[i] - mean;
            var += dv * dv;
        }
        const double stddev = (cnt > 1) ? sqrt(var / (cnt - 1)) : 0.0;

        snprintf(lbl, sizeof lbl, "%ss 2..%d (avg)", unit, s->niter - 1);
        printf("  %-27s : %10.3f ms   (%d %ss)\n", lbl, mean, cnt, unit);
        snprintf(lbl, sizeof lbl, "%ss 2..%d (stddev)", unit, s->niter - 1);
        printf("  %-27s : %10.3f ms\n", lbl, stddev);
    }
}

/* ==========================================================================
 * Command line.
 * ========================================================================== */
static void usage(const char *prog, const KrylovDescriptor *d)
{
    const char *iw = d->restarted ? "RESTARTS" : "ITER";
    fprintf(stderr, "Usage: %s [-n N] [-i %s] [-t T1] [-s T2] [-M FILE]", prog, iw);
    if (d->opt_mask & OPT_MEM)     fprintf(stderr, " [-m MEM]");
    if (d->opt_mask & OPT_STENCIL) fprintf(stderr, " [-S {7|27}]");
    if (d->opt_mask & OPT_SHIFT)   fprintf(stderr, " [-g SIGMA]");
    if (d->opt_mask & OPT_CONV)    fprintf(stderr, " [-c CONV]");
    fprintf(stderr, " [-p]\n");

    fprintf(stderr,
            "  -M FILE    import the matrix from a Matrix Market .mtx file\n"
            "             (SuiteSparse); overrides the built-in generator and -n\n"
            "  -n N       cubic grid: solve an N*N*N system   (default %d)\n",
            GRID_N);
    if (d->restarted)
        fprintf(stderr,
            "  -i RESTARTS number of restart cycles            (default %d)\n",
            d->default_iters);
    else
        fprintf(stderr,
            "  -i ITER    fixed number of %-8s iterations (default %d)\n",
            d->name, d->default_iters);
    fprintf(stderr,
            "  -t T1      number of tasks per vector op        (default: omp threads)\n"
            "  -s T2      number of SpMV sub-tasks per block   (default: omp threads)\n");
    if (d->opt_mask & OPT_MEM)
        fprintf(stderr,
            "  -m MEM     Krylov subspace size (restart)      (default %d)\n",
            d->default_m);
    if (d->opt_mask & OPT_STENCIL)
        fprintf(stderr,
            "  -S STENCIL 7- or 27-point 3-D stencil           (default 27)\n");
    if (d->opt_mask & OPT_SHIFT)
        fprintf(stderr,
            "  -g SIGMA   shift the diagonal by -SIGMA (indefinite test) (default 0)\n");
    if (d->opt_mask & OPT_CONV)
        fprintf(stderr,
            "  -c CONV    convection strength (0 => symmetric) (default 1.00)\n");
    fprintf(stderr,
            "  -p         print the residual at each %s\n",
            d->restarted ? "restart" : "iteration");
}

int main(int argc, char **argv)
{
    const KrylovDescriptor *d = &krylov_descriptor;

    KrylovParams prm;
    prm.N         = GRID_N;
    prm.iters     = d->default_iters;
    prm.m         = d->default_m;
    prm.T1        = 0;
    prm.T2        = 0;
    prm.stencil   = SPMAT_STENCIL_27PT;
    prm.conv      = 1.0;
    prm.sigma     = 0.0;
    prm.print_dbg = 0;
    prm.mtx       = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-n") && i + 1 < argc) prm.N     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) prm.iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) prm.T1    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) prm.T2    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-M") && i + 1 < argc) prm.mtx   = argv[++i];
        else if ((d->opt_mask & OPT_MEM)     && !strcmp(argv[i], "-m") && i + 1 < argc) prm.m       = atoi(argv[++i]);
        else if ((d->opt_mask & OPT_STENCIL) && !strcmp(argv[i], "-S") && i + 1 < argc) prm.stencil = atoi(argv[++i]);
        else if ((d->opt_mask & OPT_SHIFT)   && !strcmp(argv[i], "-g") && i + 1 < argc) prm.sigma   = atof(argv[++i]);
        else if ((d->opt_mask & OPT_CONV)    && !strcmp(argv[i], "-c") && i + 1 < argc) prm.conv    = atof(argv[++i]);
        else if (!strcmp(argv[i], "-p"))                 prm.print_dbg = 1;
        else if (!strcmp(argv[i], "-h")) { usage(argv[0], d); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); usage(argv[0], d); return 1; }
    }
    if (prm.T1 <= 0) prm.T1 = omp_get_max_threads();
    if (prm.T2 <= 0) prm.T2 = omp_get_max_threads();
    if (prm.m  <= 0) prm.m  = d->default_m;
    if ((d->opt_mask & OPT_STENCIL) &&
        prm.stencil != SPMAT_STENCIL_7PT && prm.stencil != SPMAT_STENCIL_27PT)
        prm.stencil = SPMAT_STENCIL_27PT;

    /* Build the test system (exact solution is all-ones for every generator and
     * for the importer, which sets b = A*1). A '-M <file>' import overrides the
     * solver's built-in generator (and the -n/-S/-c/-g knobs). */
    SpMatrix A;
    real_t *b = NULL, *xexact = NULL;
    if (prm.mtx) {
        spmat_load_matrixmarket(&A, prm.mtx, &b, &xexact);
    } else {
        switch (d->problem) {
            case KR_CONVDIFF:
                spmat_generate_convdiff(&A, prm.N, prm.N, prm.N, (real_t) prm.conv, &b, &xexact);
                break;
            case KR_STENCIL_SHIFT:
                spmat_generate_stencil(&A, prm.N, prm.N, prm.N, prm.stencil, &b, &xexact);
                if (prm.sigma != 0.0) {
                    spmat_shift_diagonal(&A, (real_t) prm.sigma);
                    for (idx_t i = 0; i < A.n; i++) b[i] -= (real_t) prm.sigma; /* keep xexact = 1 */
                }
                break;
            case KR_SPD_STENCIL:
            default:
                spmat_generate_stencil(&A, prm.N, prm.N, prm.N, prm.stencil, &b, &xexact);
                break;
        }
    }

    /* Banner. */
    if (d->restarted) printf("Krylov %s(%d) (Jacobi-preconditioned)\n", d->name, prm.m);
    else              printf("Krylov %s (Jacobi-preconditioned)\n", d->name);
    printf("  backend    : %s\n", USE_TARGET ? "GPU (omp target, pinned host mem)" : "CPU (omp task, malloc)");
    printf("  taskgraph  : %s\n", USE_TASKGRAPH ? "on" : "off");
    if (prm.mtx) {
        /* imported: the loader already printed the detailed matrix info block */
        printf("  matrix     : Matrix Market file %s\n", prm.mtx);
        printf("  size       : n = %d, nnz = %d\n", A.n, A.nnz);
    } else {
        switch (d->problem) {
            case KR_SPD_STENCIL:
                printf("  matrix     : %d-pt stencil (SPD)\n", prm.stencil);
                break;
            case KR_STENCIL_SHIFT:
                printf("  matrix     : %d-pt stencil, diagonal shift sigma = %.3f (%s)\n",
                       prm.stencil, prm.sigma, prm.sigma == 0.0 ? "SPD" : "indefinite");
                break;
            case KR_CONVDIFF:
                printf("  matrix     : convection-diffusion (conv = %.2f, %s)\n",
                       prm.conv, prm.conv == 0.0 ? "symmetric" : "nonsymmetric");
                break;
        }
        printf("  grid       : %d x %d x %d  (n = %d, nnz = %d)\n", prm.N, prm.N, prm.N, A.n, A.nnz);
    }
    if (d->restarted)
        printf("  restarts   : %d  x  m = %d  (%d Arnoldi steps)\n", prm.iters, prm.m, prm.iters * prm.m);
    else
        printf("  iterations : %d\n", prm.iters);
    printf("  tasks      : T1 = %d (vectors), T2 = %d (SpMV sub-blocks)\n", prm.T1, prm.T2);
    printf("  omp threads: %d\n", omp_get_max_threads());

    real_t *x = (real_t *) kr_alloc((size_t) A.n * sizeof(real_t));

    KrylovStats st;
    krylov_stats_init(&st, prm.iters);

    d->solve(&A, b, x, &prm, &st);

    /* Verification: true residual ||b - A x||_2 and error ||x - xexact||_2. */
    real_t *Ax = (real_t *) malloc((size_t) A.n * sizeof(real_t));
    spmat_spmv(&A, x, Ax);
    double res2 = 0.0, b2 = 0.0, err2 = 0.0, xe2 = 0.0;
    for (idx_t i = 0; i < A.n; i++) {
        const double ri = (double) b[i] - (double) Ax[i];
        const double ei = (double) x[i] - (double) xexact[i];
        res2 += ri * ri; b2 += (double) b[i] * (double) b[i];
        err2 += ei * ei; xe2 += (double) xexact[i] * (double) xexact[i];
    }
    printf("Results\n");
    printf("  relative residual     : %.6e   (||b-Ax|| / ||b||)\n", sqrt(res2 / b2));
    printf("  relative error        : %.6e   (||x-xexact|| / ||xexact||)\n", sqrt(err2 / xe2));

    krylov_stats_report(&st, d->restarted ? "restart" : "iteration", USE_TASKGRAPH);

    free(Ax); kr_free(x); free(b); free(xexact);
    spmat_free(&A);
    krylov_stats_free(&st);
    return 0;
}
