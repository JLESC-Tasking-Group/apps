//////////////////////////////////////////////////////////////////////////////////////////
// sw4lite - tasking.h port (self-contained proxy)
//
// A single-source proxy for the LLNL sw4lite seismic-wave mini-app, built on the
// shared apps/openmp tasking.h backends (CPU tasks / GPU target / synchronous /
// taskgraph). It reproduces sw4lite's numerical core:
//
//   * rhs4sg : the 4th-order 3D elastic-wave stencil L(u) = div(stress), the
//              dominant kernel, lifted verbatim (interior centered stencil) from
//              src/rhs4sg_rev.C (component-last "corder" layout);
//   * predictor / corrector / DpDm : the leap-frog predictor-corrector time
//              integrator, from src/ew-cfromfort.C (predfort/corrfort/dpdmtfort);
//   * the per-step sequence  Force -> rhs4sg -> predictor -> BC -> Force ->
//              DpDm -> rhs4sg -> corrector -> BC -> cycle(Um,U,Up).
//
// The full sw4lite needs MPI + BLAS/LAPACK + supergrid/curvilinear machinery; the
// proxy drops those (single node, uniform material, unit stretch, a Cartesian box
// with Dirichlet-zero boundaries and an initial Gaussian displacement pulse that
// radiates and damps). This keeps the exact hot-loop arithmetic and the
// predictor/BC/corrector/BC dependency structure that the tasking port exercises,
// while being self-contained and verifiable.
//
// Decomposition: every kernel is tiled into -nb blocks of z-planes. The stencil
// reaches +/-2 in k, so an rhs block names its neighbour blocks (b-2..b+2) as
// extra `in` state deps (block-aligned keys, as in the LULESH / PolyBench ports);
// predictor/corrector/DpDm are element-wise and tile cleanly. One timestep body is
// wrapped in TASKGRAPH_BEGIN/END for record/replay.
//
// Verification: component-last layout, so a tasks run must reproduce the
// synchronous run bit-for-bit; -v prints the final displacement checksum and the
// L2 norm of U (a decaying standing wave).
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "tasking.h"

#ifndef float_sw4
# define float_sw4 double
#endif

#if USE_TARGET
# ifdef ALLOC_CUDA
#  include <cuda_runtime.h>
# elif defined(ALLOC_HIP)
#  define __HIP_PLATFORM_AMD__
#  include <hip/hip_runtime.h>
# endif
#endif

static inline float_sw4 *sw4_alloc(size_t n)
{
    float_sw4 *p = NULL;
#if USE_TARGET && defined(ALLOC_CUDA)
    cudaMallocHost((void **)&p, sizeof(float_sw4) * n);
#elif USE_TARGET && defined(ALLOC_HIP)
    hipHostMalloc((void **)&p, sizeof(float_sw4) * n, hipHostMallocDefault);
#else
    p = (float_sw4 *)malloc(sizeof(float_sw4) * n);
#endif
    return p;
}
static inline void sw4_free(float_sw4 *p)
{
    if (!p) return;
#if USE_TARGET && defined(ALLOC_CUDA)
    cudaFreeHost(p);
#elif USE_TARGET && defined(ALLOC_HIP)
    hipHostFree(p);
#else
    free(p);
#endif
}

// ---- problem size (compile-time, overridable via -DSW4_N / -DSW4_NZ) ----
#ifndef SW4_N
# define SW4_N 64      // interior cells per horizontal dimension
#endif
#ifndef SW4_NZ
# define SW4_NZ 64     // interior cells in z
#endif

constexpr int hs = 2;  // ghost layers (4th-order stencil reach)

// Grid dimensions incl. ghosts. Index ranges [0, N+2hs) in each dim.
int ni, nj, nk;        // padded extents (N + 2hs)
long npts;             // ni*nj*nk
int N  = SW4_N;
int NZ = SW4_NZ;

// Uniform material
float_sw4 muval, laval, rhoval, h, dt;
float_sw4 *mu, *la, *rho;      // scalar fields (npts)
float_sw4 *U, *Um, *Up, *Lu;   // displacement (3*npts, component-last) + RHS
float_sw4 *strx, *stry, *strz; // unit stretch (=1)

int NB  = 32;  // tasks per loop
int KBS = 1;   // z-planes per tile

// component-last indexing: u[c*npts + i + ni*j + nij*k]
static inline long IDX(int i, int j, int k) { return (long)i + (long)ni * j + (long)ni * nj * k; }

//////////////////////////////////////////////////////////////////////////////////////////
// rhs4sg: interior 4th-order elastic stencil (from rhs4sg_rev.C). Tiled over k.
// With unit stretch (strx=stry=strz=1). lu = (1/h^2) * L(u). Computes interior
// k in [hs, nk-hs), j in [hs, nj-hs), i in [hs, ni-hs).
//////////////////////////////////////////////////////////////////////////////////////////
void rhs4sg_block(const float_sw4 *src, int kb, int ke)
{
    const float_sw4 i6 = 1.0 / 6, i144 = 1.0 / 144, tf = 0.75;
    const float_sw4 cof = 1.0 / (h * h);
    const long NP = npts;
#define MU(i,j,k)  mu[IDX(i,j,k)]
#define LA(i,j,k)  la[IDX(i,j,k)]
#define U1(i,j,k)  src[0*NP + IDX(i,j,k)]
#define U2(i,j,k)  src[1*NP + IDX(i,j,k)]
#define U3(i,j,k)  src[2*NP + IDX(i,j,k)]
    for (int k = kb; k < ke; k++)
        for (int j = hs; j < nj - hs; j++)
            for (int i = hs; i < ni - hs; i++) {
                float_sw4 mux1 = MU(i-1,j,k) - tf*(MU(i,j,k)+MU(i-2,j,k));
                float_sw4 mux2 = MU(i-2,j,k)+MU(i+1,j,k)+3*(MU(i,j,k)+MU(i-1,j,k));
                float_sw4 mux3 = MU(i-1,j,k)+MU(i+2,j,k)+3*(MU(i+1,j,k)+MU(i,j,k));
                float_sw4 mux4 = MU(i+1,j,k)-tf*(MU(i,j,k)+MU(i+2,j,k));
                float_sw4 muy1 = MU(i,j-1,k)-tf*(MU(i,j,k)+MU(i,j-2,k));
                float_sw4 muy2 = MU(i,j-2,k)+MU(i,j+1,k)+3*(MU(i,j,k)+MU(i,j-1,k));
                float_sw4 muy3 = MU(i,j-1,k)+MU(i,j+2,k)+3*(MU(i,j+1,k)+MU(i,j,k));
                float_sw4 muy4 = MU(i,j+1,k)-tf*(MU(i,j,k)+MU(i,j+2,k));
                float_sw4 muz1 = MU(i,j,k-1)-tf*(MU(i,j,k)+MU(i,j,k-2));
                float_sw4 muz2 = MU(i,j,k-2)+MU(i,j,k+1)+3*(MU(i,j,k)+MU(i,j,k-1));
                float_sw4 muz3 = MU(i,j,k-1)+MU(i,j,k+2)+3*(MU(i,j,k+1)+MU(i,j,k));
                float_sw4 muz4 = MU(i,j,k+1)-tf*(MU(i,j,k)+MU(i,j,k+2));

                float_sw4 r1 = i6*(
                    (2*mux1+LA(i-1,j,k)-tf*(LA(i,j,k)+LA(i-2,j,k)))*(U1(i-2,j,k)-U1(i,j,k))+
                    (2*mux2+LA(i-2,j,k)+LA(i+1,j,k)+3*(LA(i,j,k)+LA(i-1,j,k)))*(U1(i-1,j,k)-U1(i,j,k))+
                    (2*mux3+LA(i-1,j,k)+LA(i+2,j,k)+3*(LA(i+1,j,k)+LA(i,j,k)))*(U1(i+1,j,k)-U1(i,j,k))+
                    (2*mux4+LA(i+1,j,k)-tf*(LA(i,j,k)+LA(i+2,j,k)))*(U1(i+2,j,k)-U1(i,j,k))+
                    muy1*(U1(i,j-2,k)-U1(i,j,k))+muy2*(U1(i,j-1,k)-U1(i,j,k))+
                    muy3*(U1(i,j+1,k)-U1(i,j,k))+muy4*(U1(i,j+2,k)-U1(i,j,k))+
                    muz1*(U1(i,j,k-2)-U1(i,j,k))+muz2*(U1(i,j,k-1)-U1(i,j,k))+
                    muz3*(U1(i,j,k+1)-U1(i,j,k))+muz4*(U1(i,j,k+2)-U1(i,j,k)) );

                float_sw4 r2 = i6*(
                    mux1*(U2(i-2,j,k)-U2(i,j,k))+mux2*(U2(i-1,j,k)-U2(i,j,k))+
                    mux3*(U2(i+1,j,k)-U2(i,j,k))+mux4*(U2(i+2,j,k)-U2(i,j,k))+
                    (2*muy1+LA(i,j-1,k)-tf*(LA(i,j,k)+LA(i,j-2,k)))*(U2(i,j-2,k)-U2(i,j,k))+
                    (2*muy2+LA(i,j-2,k)+LA(i,j+1,k)+3*(LA(i,j,k)+LA(i,j-1,k)))*(U2(i,j-1,k)-U2(i,j,k))+
                    (2*muy3+LA(i,j-1,k)+LA(i,j+2,k)+3*(LA(i,j+1,k)+LA(i,j,k)))*(U2(i,j+1,k)-U2(i,j,k))+
                    (2*muy4+LA(i,j+1,k)-tf*(LA(i,j,k)+LA(i,j+2,k)))*(U2(i,j+2,k)-U2(i,j,k))+
                    muz1*(U2(i,j,k-2)-U2(i,j,k))+muz2*(U2(i,j,k-1)-U2(i,j,k))+
                    muz3*(U2(i,j,k+1)-U2(i,j,k))+muz4*(U2(i,j,k+2)-U2(i,j,k)) );

                float_sw4 r3 = i6*(
                    mux1*(U3(i-2,j,k)-U3(i,j,k))+mux2*(U3(i-1,j,k)-U3(i,j,k))+
                    mux3*(U3(i+1,j,k)-U3(i,j,k))+mux4*(U3(i+2,j,k)-U3(i,j,k))+
                    muy1*(U3(i,j-2,k)-U3(i,j,k))+muy2*(U3(i,j-1,k)-U3(i,j,k))+
                    muy3*(U3(i,j+1,k)-U3(i,j,k))+muy4*(U3(i,j+2,k)-U3(i,j,k))+
                    (2*muz1+LA(i,j,k-1)-tf*(LA(i,j,k)+LA(i,j,k-2)))*(U3(i,j,k-2)-U3(i,j,k))+
                    (2*muz2+LA(i,j,k-2)+LA(i,j,k+1)+3*(LA(i,j,k)+LA(i,j,k-1)))*(U3(i,j,k-1)-U3(i,j,k))+
                    (2*muz3+LA(i,j,k-1)+LA(i,j,k+2)+3*(LA(i,j,k+1)+LA(i,j,k)))*(U3(i,j,k+1)-U3(i,j,k))+
                    (2*muz4+LA(i,j,k+1)-tf*(LA(i,j,k)+LA(i,j,k+2)))*(U3(i,j,k+2)-U3(i,j,k)) );

                // mixed derivatives (la*v_y)_x etc., 116 ops each (from rhs4sg_rev.C)
                r1 += i144*( LA(i-2,j,k)*(U2(i-2,j-2,k)-U2(i-2,j+2,k)+8*(-U2(i-2,j-1,k)+U2(i-2,j+1,k)))
                        -8*( LA(i-1,j,k)*(U2(i-1,j-2,k)-U2(i-1,j+2,k)+8*(-U2(i-1,j-1,k)+U2(i-1,j+1,k))) )
                        +8*( LA(i+1,j,k)*(U2(i+1,j-2,k)-U2(i+1,j+2,k)+8*(-U2(i+1,j-1,k)+U2(i+1,j+1,k))) )
                        -  ( LA(i+2,j,k)*(U2(i+2,j-2,k)-U2(i+2,j+2,k)+8*(-U2(i+2,j-1,k)+U2(i+2,j+1,k))) ) )
                    + i144*( LA(i-2,j,k)*(U3(i-2,j,k-2)-U3(i-2,j,k+2)+8*(-U3(i-2,j,k-1)+U3(i-2,j,k+1)))
                        -8*( LA(i-1,j,k)*(U3(i-1,j,k-2)-U3(i-1,j,k+2)+8*(-U3(i-1,j,k-1)+U3(i-1,j,k+1))) )
                        +8*( LA(i+1,j,k)*(U3(i+1,j,k-2)-U3(i+1,j,k+2)+8*(-U3(i+1,j,k-1)+U3(i+1,j,k+1))) )
                        -  ( LA(i+2,j,k)*(U3(i+2,j,k-2)-U3(i+2,j,k+2)+8*(-U3(i+2,j,k-1)+U3(i+2,j,k+1))) ) )
                    + i144*( MU(i,j-2,k)*(U2(i-2,j-2,k)-U2(i+2,j-2,k)+8*(-U2(i-1,j-2,k)+U2(i+1,j-2,k)))
                        -8*( MU(i,j-1,k)*(U2(i-2,j-1,k)-U2(i+2,j-1,k)+8*(-U2(i-1,j-1,k)+U2(i+1,j-1,k))) )
                        +8*( MU(i,j+1,k)*(U2(i-2,j+1,k)-U2(i+2,j+1,k)+8*(-U2(i-1,j+1,k)+U2(i+1,j+1,k))) )
                        -  ( MU(i,j+2,k)*(U2(i-2,j+2,k)-U2(i+2,j+2,k)+8*(-U2(i-1,j+2,k)+U2(i+1,j+2,k))) ) )
                    + i144*( MU(i,j,k-2)*(U3(i-2,j,k-2)-U3(i+2,j,k-2)+8*(-U3(i-1,j,k-2)+U3(i+1,j,k-2)))
                        -8*( MU(i,j,k-1)*(U3(i-2,j,k-1)-U3(i+2,j,k-1)+8*(-U3(i-1,j,k-1)+U3(i+1,j,k-1))) )
                        +8*( MU(i,j,k+1)*(U3(i-2,j,k+1)-U3(i+2,j,k+1)+8*(-U3(i-1,j,k+1)+U3(i+1,j,k+1))) )
                        -  ( MU(i,j,k+2)*(U3(i-2,j,k+2)-U3(i+2,j,k+2)+8*(-U3(i-1,j,k+2)+U3(i+1,j,k+2))) ) );

                r2 += i144*( MU(i-2,j,k)*(U1(i-2,j-2,k)-U1(i-2,j+2,k)+8*(-U1(i-2,j-1,k)+U1(i-2,j+1,k)))
                        -8*( MU(i-1,j,k)*(U1(i-1,j-2,k)-U1(i-1,j+2,k)+8*(-U1(i-1,j-1,k)+U1(i-1,j+1,k))) )
                        +8*( MU(i+1,j,k)*(U1(i+1,j-2,k)-U1(i+1,j+2,k)+8*(-U1(i+1,j-1,k)+U1(i+1,j+1,k))) )
                        -  ( MU(i+2,j,k)*(U1(i+2,j-2,k)-U1(i+2,j+2,k)+8*(-U1(i+2,j-1,k)+U1(i+2,j+1,k))) ) )
                    + i144*( LA(i,j-2,k)*(U1(i-2,j-2,k)-U1(i+2,j-2,k)+8*(-U1(i-1,j-2,k)+U1(i+1,j-2,k)))
                        -8*( LA(i,j-1,k)*(U1(i-2,j-1,k)-U1(i+2,j-1,k)+8*(-U1(i-1,j-1,k)+U1(i+1,j-1,k))) )
                        +8*( LA(i,j+1,k)*(U1(i-2,j+1,k)-U1(i+2,j+1,k)+8*(-U1(i-1,j+1,k)+U1(i+1,j+1,k))) )
                        -  ( LA(i,j+2,k)*(U1(i-2,j+2,k)-U1(i+2,j+2,k)+8*(-U1(i-1,j+2,k)+U1(i+1,j+2,k))) ) )
                    + i144*( LA(i,j-2,k)*(U3(i,j-2,k-2)-U3(i,j-2,k+2)+8*(-U3(i,j-2,k-1)+U3(i,j-2,k+1)))
                        -8*( LA(i,j-1,k)*(U3(i,j-1,k-2)-U3(i,j-1,k+2)+8*(-U3(i,j-1,k-1)+U3(i,j-1,k+1))) )
                        +8*( LA(i,j+1,k)*(U3(i,j+1,k-2)-U3(i,j+1,k+2)+8*(-U3(i,j+1,k-1)+U3(i,j+1,k+1))) )
                        -  ( LA(i,j+2,k)*(U3(i,j+2,k-2)-U3(i,j+2,k+2)+8*(-U3(i,j+2,k-1)+U3(i,j+2,k+1))) ) )
                    + i144*( MU(i,j,k-2)*(U3(i,j-2,k-2)-U3(i,j+2,k-2)+8*(-U3(i,j-1,k-2)+U3(i,j+1,k-2)))
                        -8*( MU(i,j,k-1)*(U3(i,j-2,k-1)-U3(i,j+2,k-1)+8*(-U3(i,j-1,k-1)+U3(i,j+1,k-1))) )
                        +8*( MU(i,j,k+1)*(U3(i,j-2,k+1)-U3(i,j+2,k+1)+8*(-U3(i,j-1,k+1)+U3(i,j+1,k+1))) )
                        -  ( MU(i,j,k+2)*(U3(i,j-2,k+2)-U3(i,j+2,k+2)+8*(-U3(i,j-1,k+2)+U3(i,j+1,k+2))) ) );

                r3 += i144*( MU(i-2,j,k)*(U1(i-2,j,k-2)-U1(i-2,j,k+2)+8*(-U1(i-2,j,k-1)+U1(i-2,j,k+1)))
                        -8*( MU(i-1,j,k)*(U1(i-1,j,k-2)-U1(i-1,j,k+2)+8*(-U1(i-1,j,k-1)+U1(i-1,j,k+1))) )
                        +8*( MU(i+1,j,k)*(U1(i+1,j,k-2)-U1(i+1,j,k+2)+8*(-U1(i+1,j,k-1)+U1(i+1,j,k+1))) )
                        -  ( MU(i+2,j,k)*(U1(i+2,j,k-2)-U1(i+2,j,k+2)+8*(-U1(i+2,j,k-1)+U1(i+2,j,k+1))) ) )
                    + i144*( MU(i,j-2,k)*(U2(i,j-2,k-2)-U2(i,j-2,k+2)+8*(-U2(i,j-2,k-1)+U2(i,j-2,k+1)))
                        -8*( MU(i,j-1,k)*(U2(i,j-1,k-2)-U2(i,j-1,k+2)+8*(-U2(i,j-1,k-1)+U2(i,j-1,k+1))) )
                        +8*( MU(i,j+1,k)*(U2(i,j+1,k-2)-U2(i,j+1,k+2)+8*(-U2(i,j+1,k-1)+U2(i,j+1,k+1))) )
                        -  ( MU(i,j+2,k)*(U2(i,j+2,k-2)-U2(i,j+2,k+2)+8*(-U2(i,j+2,k-1)+U2(i,j+2,k+1))) ) )
                    + i144*( LA(i,j,k-2)*(U1(i-2,j,k-2)-U1(i+2,j,k-2)+8*(-U1(i-1,j,k-2)+U1(i+1,j,k-2)))
                        -8*( LA(i,j,k-1)*(U1(i-2,j,k-1)-U1(i+2,j,k-1)+8*(-U1(i-1,j,k-1)+U1(i+1,j,k-1))) )
                        +8*( LA(i,j,k+1)*(U1(i-2,j,k+1)-U1(i+2,j,k+1)+8*(-U1(i-1,j,k+1)+U1(i+1,j,k+1))) )
                        -  ( LA(i,j,k+2)*(U1(i-2,j,k+2)-U1(i+2,j,k+2)+8*(-U1(i-1,j,k+2)+U1(i+1,j,k+2))) ) )
                    + i144*( LA(i,j,k-2)*(U2(i,j-2,k-2)-U2(i,j+2,k-2)+8*(-U2(i,j-1,k-2)+U2(i,j+1,k-2)))
                        -8*( LA(i,j,k-1)*(U2(i,j-2,k-1)-U2(i,j+2,k-1)+8*(-U2(i,j-1,k-1)+U2(i,j+1,k-1))) )
                        +8*( LA(i,j,k+1)*(U2(i,j-2,k+1)-U2(i,j+2,k+1)+8*(-U2(i,j-1,k+1)+U2(i,j+1,k+1))) )
                        -  ( LA(i,j,k+2)*(U2(i,j-2,k+2)-U2(i,j+2,k+2)+8*(-U2(i,j-1,k+2)+U2(i,j+1,k+2))) ) );

                Lu[0*NP + IDX(i,j,k)] = cof * r1;
                Lu[1*NP + IDX(i,j,k)] = cof * r2;
                Lu[2*NP + IDX(i,j,k)] = cof * r3;
            }
#undef MU
#undef LA
#undef U1
#undef U2
#undef U3
}

//////////////////////////////////////////////////////////////////////////////////////////
// kernels, tiled over k. Dependencies keyed on block-aligned plane addresses.
//////////////////////////////////////////////////////////////////////////////////////////
// Lu = L(src). src is captured by firstprivate so deferred tasks read the right
// buffer (the global-pointer-swap trick does NOT work with deferred tasks).
void evalRHS_src(float_sw4 *src)
{
    const long nij = (long)ni * nj;
    const int nblk = (nk + KBS - 1) / KBS;
    // interior k in [hs, nk-hs): rhs block [kb,ke) reads src planes [kb-2, ke+1]
    // -> name neighbour blocks b-2..b+2. Writes Lu (whole-buffer token Lu[0]).
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS < hs ? hs : b * KBS;
        const int ke = (b + 1) * KBS > nk - hs ? nk - hs : (b + 1) * KBS;
        if (kb >= ke) continue;
        const int bm2 = (b - 2 >= 0) ? (b - 2) : 0;
        const int bm1 = (b - 1 >= 0) ? (b - 1) : 0;
        const int bp1 = (b + 1 < nblk) ? (b + 1) : b;
        const int bp2 = (b + 2 < nblk) ? (b + 2) : bp1;
        OMP_TILE(DEPEND(in, src[bm2 * KBS * nij], src[bm1 * KBS * nij], src[b * KBS * nij],
                            src[bp1 * KBS * nij], src[bp2 * KBS * nij])
                 DEPEND(out, Lu[0]),
                 MAP(present: src[0:(size_t)3*npts], Lu[0:(size_t)3*npts],
                              mu[0:(size_t)npts], la[0:(size_t)npts])
                 num_teams((ke - kb)) thread_limit(256),
                 firstprivate(src, kb, ke))
        rhs4sg_block(src, kb, ke);
    }
}
void evalRHS() { evalRHS_src(U); }

// predictor: Up = 2U - Um + (dt^2/rho)*(Lu + F), with F=0. Elementwise, tiled over k.
void evalPredictor(float_sw4 dt2)
{
    const long nij = (long)ni * nj;
    const long NP = npts;
    const int nblk = (nk + KBS - 1) / KBS;
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nk ? nk : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(in, U[b * KBS * nij], Um[b * KBS * nij], Lu[0])
                 DEPEND(out, Up[b * KBS * nij]),
                 MAP(present: U[0:(size_t)3*npts], Um[0:(size_t)3*npts],
                              Up[0:(size_t)3*npts], Lu[0:(size_t)3*npts], rho[0:(size_t)npts])
                 num_teams((ke - kb)) thread_limit(256),
                 firstprivate(kb, ke, dt2, NP))
        for (int k = kb; k < ke; k++)
            for (int j = 0; j < nj; j++)
                for (int i = 0; i < ni; i++) {
                    const long p = IDX(i, j, k);
                    const float_sw4 dt2orh = dt2 / rho[p];
                    for (int c = 0; c < 3; c++)
                        Up[c * NP + p] = 2 * U[c * NP + p] - Um[c * NP + p] + dt2orh * Lu[c * NP + p];
                }
    }
}

// corrector: Up += (dt^4/12/rho)*(Lu + F), F=0. Elementwise, tiled over k.
void evalCorrector(float_sw4 dt4)
{
    const long nij = (long)ni * nj;
    const long NP = npts;
    const float_sw4 dt4i12 = dt4 / 12;
    const int nblk = (nk + KBS - 1) / KBS;
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nk ? nk : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(in, Lu[0]) DEPEND(inout, Up[b * KBS * nij]),
                 MAP(present: Up[0:(size_t)3*npts], Lu[0:(size_t)3*npts], rho[0:(size_t)npts])
                 num_teams((ke - kb)) thread_limit(256),
                 firstprivate(kb, ke, dt4i12, NP))
        for (int k = kb; k < ke; k++)
            for (int j = 0; j < nj; j++)
                for (int i = 0; i < ni; i++) {
                    const long p = IDX(i, j, k);
                    const float_sw4 dt4i12orh = dt4i12 / rho[p];
                    for (int c = 0; c < 3; c++)
                        Up[c * NP + p] += dt4i12orh * Lu[c * NP + p];
                }
    }
}

// DpDm: compute acceleration-like U2 = (Up - 2U + Um)/dt^2 into Lu's slot? In sw4lite
// evalDpDmInTime writes into Uacc which then feeds evalRHS. Here we reuse U as the
// forcing operand: Uacc = (Up - 2U + Um)/dt^2, stored in a dedicated buffer.
float_sw4 *Uacc;
void evalDpDm(float_sw4 dt2i)
{
    const long nij = (long)ni * nj;
    const long NP = npts;
    const int nblk = (nk + KBS - 1) / KBS;
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nk ? nk : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(in, Up[b * KBS * nij], U[b * KBS * nij], Um[b * KBS * nij])
                 DEPEND(out, Uacc[b * KBS * nij]),
                 MAP(present: Up[0:(size_t)3*npts], U[0:(size_t)3*npts],
                              Um[0:(size_t)3*npts], Uacc[0:(size_t)3*npts])
                 num_teams((ke - kb)) thread_limit(256),
                 firstprivate(kb, ke, dt2i, NP))
        for (int k = kb; k < ke; k++)
            for (int j = 0; j < nj; j++)
                for (int i = 0; i < ni; i++) {
                    const long p = IDX(i, j, k);
                    for (int c = 0; c < 3; c++)
                        Uacc[c * NP + p] = dt2i * (Up[c * NP + p] - 2 * U[c * NP + p] + Um[c * NP + p]);
                }
    }
}

// Dirichlet-zero boundary: zero the ghost + boundary planes of Up. One task per
// k-block (keyed inout on the block) so it orders after predictor/corrector.
void enforceBC()
{
    const long nij = (long)ni * nj;
    const long NP = npts;
    const int nblk = (nk + KBS - 1) / KBS;
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nk ? nk : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(inout, Up[b * KBS * nij]),
                 MAP(present: Up[0:(size_t)3*npts]) num_teams((ke - kb)) thread_limit(256),
                 firstprivate(kb, ke, NP))
        for (int k = kb; k < ke; k++)
            for (int j = 0; j < nj; j++)
                for (int i = 0; i < ni; i++) {
                    const bool bnd = (i < hs || i >= ni - hs || j < hs || j >= nj - hs ||
                                      k < hs || k >= nk - hs);
                    if (bnd) {
                        const long p = IDX(i, j, k);
                        for (int c = 0; c < 3; c++) Up[c * NP + p] = 0.0;
                    }
                }
    }
}

// cycle Um <- U, U <- Up  (host swap of pointers would break the mapped buffers, so
// we copy, tiled over k). WAW/RAW keyed per block.
void cycleArrays()
{
    const long nij = (long)ni * nj;
    const long NP = npts;
    const int nblk = (nk + KBS - 1) / KBS;
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nk ? nk : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(in, Up[b * KBS * nij]) DEPEND(out, Um[b * KBS * nij], U[b * KBS * nij]),
                 MAP(present: Up[0:(size_t)3*npts], U[0:(size_t)3*npts], Um[0:(size_t)3*npts])
                 num_teams((ke - kb)) thread_limit(256),
                 firstprivate(kb, ke, NP))
        for (int k = kb; k < ke; k++)
            for (int j = 0; j < nj; j++)
                for (int i = 0; i < ni; i++) {
                    const long p = IDX(i, j, k);
                    for (int c = 0; c < 3; c++) {
                        Um[c * NP + p] = U[c * NP + p];
                        U[c * NP + p] = Up[c * NP + p];
                    }
                }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// one timestep: predictor-corrector (Force=0 throughout).
//////////////////////////////////////////////////////////////////////////////////////////
void timestep()
{
    const float_sw4 dt2 = dt * dt;
    const float_sw4 dt4 = dt2 * dt2;
    const float_sw4 dt2i = 1.0 / dt2;
    evalRHS();               // Lu = L(U)
    evalPredictor(dt2);      // Up = 2U - Um + dt^2/rho * Lu
    enforceBC();             // Up boundary = 0
    evalDpDm(dt2i);          // Uacc = (Up - 2U + Um)/dt^2
    evalRHS_src(Uacc);        // Lu = L(Uacc)
    evalCorrector(dt4);      // Up += dt^4/12/rho * Lu
    enforceBC();             // Up boundary = 0
    cycleArrays();           // Um<-U, U<-Up
}

//////////////////////////////////////////////////////////////////////////////////////////
// setup
//////////////////////////////////////////////////////////////////////////////////////////
void init()
{
    ni = N + 2 * hs;
    nj = N + 2 * hs;
    nk = NZ + 2 * hs;
    npts = (long)ni * nj * nk;

    muval = 1.0; laval = 1.0; rhoval = 1.0; h = 1.0;
    // CFL-stable dt for the 4th-order elastic operator with cp=sqrt((2mu+la)/rho).
    const float_sw4 cp = sqrt((2 * muval + laval) / rhoval);
    dt = 0.3 * h / cp;

    mu  = sw4_alloc(npts);
    la  = sw4_alloc(npts);
    rho = sw4_alloc(npts);
    U   = sw4_alloc(3 * npts);
    Um  = sw4_alloc(3 * npts);
    Up  = sw4_alloc(3 * npts);
    Lu  = sw4_alloc(3 * npts);
    Uacc = sw4_alloc(3 * npts);

    for (long p = 0; p < npts; p++) { mu[p] = muval; la[p] = laval; rho[p] = rhoval; }
    for (long p = 0; p < 3 * npts; p++) { U[p] = Um[p] = Up[p] = Lu[p] = Uacc[p] = 0.0; }

    // Initial Gaussian displacement pulse (x-component) centred in the box.
    const double x0 = ni / 2.0, y0 = nj / 2.0, z0 = nk / 2.0;
    const double sig = N / 8.0;
    for (int k = hs; k < nk - hs; k++)
        for (int j = hs; j < nj - hs; j++)
            for (int i = hs; i < ni - hs; i++) {
                const double r2 = (i - x0) * (i - x0) + (j - y0) * (j - y0) + (k - z0) * (k - z0);
                const double g = exp(-r2 / (2 * sig * sig));
                const long p = IDX(i, j, k);
                U[0 * npts + p]  = g;
                Um[0 * npts + p] = g;   // start at rest (Um = U)
            }
}

void finalize()
{
    sw4_free(mu); sw4_free(la); sw4_free(rho);
    sw4_free(U); sw4_free(Um); sw4_free(Up); sw4_free(Lu); sw4_free(Uacc);
}

double l2norm()
{
    double s = 0.0;
    for (long p = 0; p < 3 * npts; p++) s += (double)U[p] * (double)U[p];
    return sqrt(s);
}
double checksum()
{
    double s = 0.0;
    for (long p = 0; p < 3 * npts; p++) s += (double)U[p];
    return s;
}

//////////////////////////////////////////////////////////////////////////////////////////
// main
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int steps = 100, verify = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-nb") && a + 1 < argc) NB = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-i") && a + 1 < argc) steps = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-v")) verify = 1;
        else if (!strcmp(argv[a], "-h")) { printf("usage: %s [-i steps] [-nb tasks-per-loop] [-v]\n", argv[0]); return 0; }
    }

    init();
    if (NB < 1) NB = 1;
    KBS = nk / NB + (nk % NB != 0);
    if (KBS < 1) KBS = 1;

#if USE_SYNC
    const char *backend = "synchronous (blocking)";
#elif USE_TARGET
    const char *backend = "GPU target tasks";
#else
    const char *backend = "CPU tasks";
#endif
    printf("grid: %d x %d x %d  (interior %d x %d x %d)\n", ni, nj, nk, N, N, NZ);
    printf("dt: %g\n", dt);
    printf("backend=%s taskgraph=%d\n", backend, USE_TASKGRAPH);
    printf("Num tasks per loop: %d\n", NB);

    const long nij = (long)ni * nj;
    double *iter_times = (double *)malloc(sizeof(double) * (steps > 0 ? steps : 1));
    double prev_ts, t0, tf;

    OMP_TARGET_ENTER_DATA(MAP(to: U[0:(size_t)3*npts], Um[0:(size_t)3*npts],
                                  mu[0:(size_t)npts], la[0:(size_t)npts], rho[0:(size_t)npts])
                          MAP(alloc: Up[0:(size_t)3*npts], Lu[0:(size_t)3*npts], Uacc[0:(size_t)3*npts]))

#pragma omp parallel
{
#pragma omp single
{
    t0 = omp_get_wtime();
    prev_ts = t0;
    for (int s = 0; s < steps; s++) {
        TASKGRAPH_BEGIN
        {
            timestep();
        }
        TASKGRAPH_END

        OMP_HOST_TASK(firstprivate(s, iter_times, nij) shared(prev_ts)
                      DEPEND(in, U[(nk / 2) * nij]))
        {
            const double now = omp_get_wtime();
            iter_times[s] = (now - prev_ts) * 1000.0;
            prev_ts = now;
        }
    }
#pragma omp taskwait
    tf = omp_get_wtime();
} /* single */
} /* parallel */

    OMP_TARGET_UPDATE(from(U[0:(size_t)3*npts]))
    OMP_TARGET_EXIT_DATA(MAP(release: U[0:(size_t)3*npts], Um[0:(size_t)3*npts],
                                      Up[0:(size_t)3*npts], Lu[0:(size_t)3*npts], Uacc[0:(size_t)3*npts],
                                      mu[0:(size_t)npts], la[0:(size_t)npts], rho[0:(size_t)npts]))

    printf("Elapsed time         = %10.6f\n", tf - t0);
    if (steps > 0) {
        const int lo  = (steps > 2) ? 2 : (steps > 1 ? 1 : 0);
        const int cnt = steps - lo;
        double mean = 0.0;
        for (int r = lo; r < steps; r++) mean += iter_times[r];
        if (cnt > 0) mean /= cnt;
        double var = 0.0;
        for (int r = lo; r < steps; r++) { double d = iter_times[r] - mean; var += d * d; }
        const double sd = (cnt > 1) ? sqrt(var / (cnt - 1)) : 0.0;
        printf("iteration 0          : %10.3f ms  (record)\n", iter_times[0]);
        printf("iteration 1          : %10.3f ms  (1st replay)\n", (steps > 1) ? iter_times[1] : iter_times[0]);
        printf("iterations 2.. (avg)    : %10.3f ms\n", mean);
        printf("iterations 2.. (stddev) : %10.3f ms\n", sd);
        const double gpupd = (double)N * N * NZ;   // interior grid-point updates / step
        const double fom = (mean > 0.0) ? gpupd / (mean / 1000.0) : 0.0;
        printf("FOM                  = %10.8g (gridpt-updates/s)\n", fom);
    }
    printf("L2 norm of U         = %.10e\n", l2norm());
    if (verify) printf("checksum             = %.10e\n", checksum());

    free(iter_times);
    finalize();
    return 0;
}
