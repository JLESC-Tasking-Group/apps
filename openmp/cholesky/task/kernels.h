#ifndef KERNELS_H
# define KERNELS_H

// # include <cblas.h>

/* prototypes */
void dpotrf_ (const char* uplo, const int * n, double* a,
        const int * lda, int * info );

void dgemm_ (const char *transa, const char *transb, int *l, int *n, int *m, double *alpha,
        const void *a, int *lda, void *b, int *ldb, double *beta, void *c, int *ldc);

void dtrsm_ (char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha,
        double *a, int *lda, double *b, int *ldb);

void dsyrk_ (char *uplo, char *trans, int *n, int *k, double *alpha, double *a, int *lda,
        double *beta, double *c, int *ldc);

void dlarnv_(int * idist, int iseed[4], const int * n, double * tile);

/* http://www.netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga2f55f604a6003d03b5cd4a0adcfb74d6.html */
static void
potrf(double * const A, int ts, int ld)
{
    static int INFO;
    static const char L = 'L';
    dpotrf_(&L, &ts, A, &ld, &INFO);
}

/* http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_ga6a0a7704f4a747562c1bd9487e89795c.html */
static void
trsm(double * A, double * B, int ts, int ld)
{
    static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
    static double DONE = 1.0;
    dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, A, &ld, B, &ld );
}

/* http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html */
static void
gemm(double * A, double * B, double * C, int ts, int ld)
{
    static const char TR = 'T', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;
    dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, A, &ld, B, &ld, &DONE, C, &ld);
}

/* https://www.netlib.org/lapack/explore-html/db/def/group__complex__blas__level3_ga1b4f63daf04fdf3061bd25dfec0d3e84.html */
static void
syrk(double * A, double * C, int ts, int ld)
{
    static char LO = 'L', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;
    dsyrk_(&LO, &NT, &ts, &ts, &DMONE, A, &ld, &DONE, C, &ld);
}

static void
initialize_tile(const int n, double * tile)
{
	int ISEED[4] = {0, 0, 0, 1};
	int intONE = 1;
    dlarnv_(&intONE, &ISEED[0], &n, tile);
}

# endif /* KERNELS_H */
