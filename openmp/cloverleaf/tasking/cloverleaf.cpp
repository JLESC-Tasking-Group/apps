// CloverLeaf - tasking.h port
//
// 2D compressible Euler (Lagrangian step + Eulerian remap) on one chunk, from the
// UoB-HPC C++ CloverLeaf vendored alongside (kernels: src/omp/*.cpp, sequence:
// driver/{hydro,timestep,PdV,advection}.cpp). Runs the clover_bm*.in problem (a
// dense, energetic rectangle in a 10x10 box) on an -s x -s mesh.
//
// Every kernel loop is tiled into whole mesh rows, one OMP_TILE per tile. A tile
// writes only its own rows and reads at most 2 rows away, so with >= 2 rows per
// tile its dependences are block keys K(A,b) on blocks b-1..b+1; halo rows belong
// to the first / last block.
//
// What varies between timesteps lives in data, not in the recorded graph:
//   dt   : per-row CFL minima + one finalize task into the device scalars dtv[];
//          kernels read dtv[DT] when they run;
//   sweep: the advection order alternates x-first / y-first, so one iteration of
//          the TASKGRAPH_LOOP is a PAIR of timesteps (-i counts pairs).
//
// answer: the final kinetic energy of the field summary.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <omp.h>

#include "tasking.h"
#include "tg_report.h"

// driver/definitions.h, read_input.cpp defaults, and the clover_bm deck
static constexpr double g_small = 1.0e-16, g_big = 1.0e+21;
static constexpr double dtinit = 0.04, dtrise = 1.5, dtmax = 0.04, dtmin = 0.0000001;
static constexpr double dtc_safe = 0.7, dtu_safe = 0.5, dtv_safe = 0.5, dtdiv_safe = 0.7;
enum { XDIR = 1, YDIR = 2 };
enum { DT, DTOLD, TIME, BAD, NDTV };   // dtv[] slots

// Mesh: cells (i,j) in [2,xmax+2) x [2,ymax+2), 2 halo layers per side, vertex
// arrays one wider. Every 2D array is S x (ymax+5), row-major in j.
static int xmax, ymax, S, L1;   // L1: length of the 1D (per-row / per-column) arrays
static size_t N;
static int KBS, nblk;           // rows per tile, tiles per loop

static double *density0, *density1, *energy0, *energy1, *pressure, *viscosity, *soundspeed;
static double *xvel0, *xvel1, *yvel0, *yvel1;
static double *vol_flux_x, *vol_flux_y, *mass_flux_x, *mass_flux_y;
static double *volume, *xarea, *yarea;
static double *pre_vol, *ener_flux;                                        // advec_cell
static double *post_vol, *node_flux, *node_mass_post, *node_mass_pre, *mom_flux;  // advec_mom
static double *vertexx, *vertexy, *vertexdx, *vertexdy, *cellx, *celly, *celldx, *celldy;
static double *dtrow;   // per-row CFL minimum
static double *dtv;     // dt, dtold, time, bad-dt flag

#define F(a, i, j) (a)[(size_t)(j) * s + (i)]
// dependency key of block q of a 2D (K) / per-row 1D (KR) array
#define K(a, q)  (a)[(size_t)(2 + (q) * KBS) * S]
#define KR(a, q) (a)[2 + (q) * KBS]
// read rows of the blocks around b (bm..bp-1, set by FOR_TILES)
#define RD(a) DEPEND_MULTI(in, (q = bm:bp), K(a, q))

// Rows [jlo,jhi) of tile b for a loop over rows [lo,hi): block b's rows, the
// first / last block also taking the halo rows below / above.
static inline bool tile_rows(int b, int lo, int hi, int &jlo, int &jhi, int &bm, int &bp)
{
    jlo = (b == 0) ? lo : std::max(lo, 2 + b * KBS);
    jhi = (b == nblk - 1) ? hi : std::min(hi, 2 + (b + 1) * KBS);
    bm = std::max(b - 1, 0);
    bp = std::min(b + 1, nblk - 1) + 1;
    return jlo < jhi;
}
#define FOR_TILES(lo, hi) \
    for (int b = 0, jlo = 0, jhi = 0, bm = 0, bp = 0; b < nblk; b++) \
        if (tile_rows(b, (lo), (hi), jlo, jhi, bm, bp))

#define LOCALS const int s = S, xm = xmax, ym = ymax; (void) ym;

//////////////////////////////////////////////////////////////////////////////
// Halo exchange (update_halo_kernel, every face external): reflect `d` layers.
// Per field: bottom / top source-row offsets, the extra vertex column/row of
// the copied range, and the sign (normal velocities and fluxes flip).
//////////////////////////////////////////////////////////////////////////////
struct Halo { int bo, tdo, tso, xe, sy, lo, rdo, rso, ye, sx; };
static constexpr Halo H_CELL = {0, 0, 0, 0,  1, 0, 0, 0, 0,  1};
static constexpr Halo H_XVEL = {1, 1, 0, 1,  1, 1, 1, 0, 1, -1};
static constexpr Halo H_YVEL = {1, 1, 0, 1, -1, 1, 1, 0, 1,  1};
static constexpr Halo H_FX   = {1, 0, 1, 1,  1, 1, 1, 0, 0, -1};   // vol/mass_flux_x
static constexpr Halo H_FY   = {1, 1, 0, 0, -1, 1, 0, 1, 1,  1};   // vol/mass_flux_y

static void halo(double *f, const Halo &h, int d)
{
    LOCALS
    const int bo = h.bo, tdo = h.tdo, tso = h.tso, xe = h.xe, lo = h.lo, rdo = h.rdo, rso = h.rso, ye = h.ye;
    const double sy = h.sy, sx = h.sx;
    const int last = nblk - 1, b1 = std::min(1, last), bl = std::max(last - 1, 0);

    // bottom / top before left / right: the side copies cover the halo rows
    OMP_TILE(DEPEND(inout, K(f, 0), K(f, b1)), MAP(present: f[0:N]),
             firstprivate(f, s, xm, d, bo, xe, sy))
    for (int i = 2 - d; i < xm + xe + 2 + d; i++)
        for (int k = 0; k < d; k++) F(f, i, 1 - k) = sy * F(f, i, 2 + bo + k);

    OMP_TILE(DEPEND(inout, K(f, last), K(f, bl)), MAP(present: f[0:N]),
             firstprivate(f, s, xm, ym, d, tdo, tso, xe, sy))
    for (int i = 2 - d; i < xm + xe + 2 + d; i++)
        for (int k = 0; k < d; k++) F(f, i, ym + 2 + tdo + k) = sy * F(f, i, ym + 1 - tso - k);

    FOR_TILES(2 - d, ym + ye + 2 + d) {
        OMP_TILE(DEPEND(inout, K(f, b)), MAP(present: f[0:N]),
                 firstprivate(f, s, xm, d, lo, rdo, rso, sx, jlo, jhi))
        for (int k = jlo; k < jhi; k++)
            for (int jj = 0; jj < d; jj++) {
                F(f, 1 - jj, k)            = sx * F(f, 2 + lo + jj, k);
                F(f, xm + 2 + rdo + jj, k) = sx * F(f, xm + 1 - rso - jj, k);
            }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Lagrangian step kernels
//////////////////////////////////////////////////////////////////////////////
static void ideal_gas(double *d, double *e)
{
    LOCALS
    double *p = pressure, *ss = soundspeed;
    FOR_TILES(2, ym + 2) {
        OMP_TILE(DEPEND(in, K(d, b), K(e, b)) DEPEND(inout, K(p, b), K(ss, b)),
                 MAP(present: d[0:N], e[0:N], p[0:N], ss[0:N]) collapse(2),
                 firstprivate(d, e, p, ss, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++)
            for (int i = 2; i < xm + 2; i++) {
                const double v = 1.0 / F(d, i, j);
                F(p, i, j) = (1.4 - 1.0) * F(d, i, j) * F(e, i, j);
                const double pressurebyenergy = (1.4 - 1.0) * F(d, i, j);
                const double pressurebyvolume = -F(d, i, j) * F(p, i, j);
                const double sound_speed_squared = v * v * (F(p, i, j) * pressurebyenergy - pressurebyvolume);
                F(ss, i, j) = sqrt(sound_speed_squared);
            }
    }
}

static void viscosity_kernel()
{
    LOCALS
    double *d0 = density0, *p = pressure, *q0 = viscosity, *u = xvel0, *v = yvel0, *cdx = celldx, *cdy = celldy;
    FOR_TILES(2, ym + 2) {
        OMP_TILE(RD(u) RD(v) RD(p) DEPEND(in, K(d0, b)) DEPEND(inout, K(q0, b)),
                 MAP(present: d0[0:N], p[0:N], q0[0:N], u[0:N], v[0:N], cdx[0:L1], cdy[0:L1]) collapse(2),
                 firstprivate(d0, p, q0, u, v, cdx, cdy, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++)
            for (int i = 2; i < xm + 2; i++) {
                const double ugrad = (F(u, i + 1, j) + F(u, i + 1, j + 1)) - (F(u, i, j) + F(u, i, j + 1));
                const double vgrad = (F(v, i, j + 1) + F(v, i + 1, j + 1)) - (F(v, i, j) + F(v, i + 1, j));
                const double div = (cdx[i] * (ugrad) + cdy[j] * (vgrad));
                const double strain2 = 0.5 * (F(u, i, j + 1) + F(u, i + 1, j + 1) - F(u, i, j) - F(u, i + 1, j)) / cdy[j] +
                                       0.5 * (F(v, i + 1, j) + F(v, i + 1, j + 1) - F(v, i, j) - F(v, i, j + 1)) / cdx[i];
                double pgradx = (F(p, i + 1, j) - F(p, i - 1, j)) / (cdx[i] + cdx[i + 1]);
                double pgrady = (F(p, i, j + 1) - F(p, i, j - 1)) / (cdy[j] + cdy[j + 1]);
                const double pgradx2 = pgradx * pgradx;
                const double pgrady2 = pgrady * pgrady;
                const double limiter = ((0.5 * (ugrad) / cdx[i]) * pgradx2 + (0.5 * (vgrad) / cdy[j]) * pgrady2 +
                                        strain2 * pgradx * pgrady) / fmax(pgradx2 + pgrady2, g_small);
                if ((limiter > 0.0) || (div >= 0.0)) {
                    F(q0, i, j) = 0.0;
                } else {
                    const double dirx = (pgradx < 0.0) ? -1.0 : 1.0;
                    pgradx = dirx * fmax(g_small, fabs(pgradx));
                    const double diry = (pgradx < 0.0) ? -1.0 : 1.0;   // sic (upstream)
                    pgrady = diry * fmax(g_small, fabs(pgrady));
                    const double pgrad = sqrt(pgradx * pgradx + pgrady * pgrady);
                    const double xgrad = fabs(cdx[i] * pgrad / pgradx);
                    const double ygrad = fabs(cdy[j] * pgrad / pgrady);
                    const double grad = fmin(xgrad, ygrad);
                    const double grad2 = grad * grad;
                    F(q0, i, j) = 2.0 * F(d0, i, j) * grad2 * limiter * limiter;
                }
            }
    }
}

// CFL timestep: per-row minima, then one task reduces them into dtv[].
static void calc_dt()
{
    LOCALS
    double *d0 = density0, *q0 = viscosity, *ss = soundspeed, *u = xvel0, *v = yvel0, *vol = volume,
           *xa = xarea, *ya = yarea, *cdx = celldx, *cdy = celldy, *dr = dtrow, *dt_ = dtv;
    FOR_TILES(2, ym + 2) {
        OMP_TILE(RD(u) RD(v) DEPEND(in, K(d0, b), K(q0, b), K(ss, b)) DEPEND(inout, KR(dr, b)),
                 MAP(present: d0[0:N], q0[0:N], ss[0:N], u[0:N], v[0:N], vol[0:N], xa[0:N], ya[0:N],
                              cdx[0:L1], cdy[0:L1], dr[0:L1]),
                 firstprivate(d0, q0, ss, u, v, vol, xa, ya, cdx, cdy, dr, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++) {
            double dt_min_val = g_big;
            for (int i = 2; i < xm + 2; i++) {
                const double dsx = cdx[i], dsy = cdy[j];
                double cc = F(ss, i, j) * F(ss, i, j);
                cc = cc + 2.0 * F(q0, i, j) / F(d0, i, j);
                cc = fmax(sqrt(cc), g_small);
                const double dtct = dtc_safe * fmin(dsx, dsy) / cc;
                double div = 0.0;
                double dv1 = (F(u, i, j) + F(u, i, j + 1)) * F(xa, i, j);
                double dv2 = (F(u, i + 1, j) + F(u, i + 1, j + 1)) * F(xa, i + 1, j);
                div = div + dv2 - dv1;
                const double dtut = dtu_safe * 2.0 * F(vol, i, j) / fmax(fmax(fabs(dv1), fabs(dv2)), g_small * F(vol, i, j));
                dv1 = (F(v, i, j) + F(v, i + 1, j)) * F(ya, i, j);
                dv2 = (F(v, i, j + 1) + F(v, i + 1, j + 1)) * F(ya, i, j + 1);
                div = div + dv2 - dv1;
                const double dtvt = dtv_safe * 2.0 * F(vol, i, j) / fmax(fmax(fabs(dv1), fabs(dv2)), g_small * F(vol, i, j));
                div = div / (2.0 * F(vol, i, j));
                const double dtdivt = (div < -g_small) ? dtdiv_safe * (-1.0 / div) : g_big;
                const double mins = fmin(dtct, fmin(dtut, fmin(dtvt, fmin(dtdivt, g_big))));
                dt_min_val = fmin(mins, dt_min_val);
            }
            dr[j] = dt_min_val;
        }
    }

    // timestep.cpp: dt = min(min(cfl, dtold*dtrise), dtmax); dtold = dt
    const int nb = nblk, kbs = KBS;
    OMP_TILE(DEPEND_MULTI(in, (q = 0:nb), dr[2 + q * kbs]) DEPEND(inout, dt_[0]),
             MAP(present: dr[0:L1], dt_[0:NDTV]),
             firstprivate(dr, dt_, ym))
    for (int t = 0; t < 1; t++) {
        double dt = g_big;
        for (int j = 2; j < ym + 2; j++)
            if (dr[j] <= dt) dt = dr[j];
        const double rise = dt_[DTOLD] * dtrise;
        dt = (rise < dt) ? rise : dt;
        dt = (dtmax < dt) ? dtmax : dt;
        if (!(dt >= dtmin) || !(dt_[TIME] + dt > dt_[TIME])) dt_[BAD] = 1.0;   // also catches NaN
        dt_[DT] = dt;
        dt_[DTOLD] = dt;
        dt_[TIME] += dt;
    }
}

// PdV: predictor (half step, *0) or corrector (full step, average of *0 and *1)
static void PdV(bool predict)
{
    LOCALS
    double *xa = xarea, *ya = yarea, *vol = volume, *d0 = density0, *d1 = density1, *e0 = energy0, *e1 = energy1,
           *p = pressure, *q0 = viscosity, *u0 = xvel0, *v0 = yvel0, *dt_ = dtv;
    double *u1 = predict ? xvel0 : xvel1, *v1 = predict ? yvel0 : yvel1;
    const double half = predict ? 0.5 : 1.0;
    FOR_TILES(2, ym + 2) {
        OMP_TILE(RD(u0) RD(v0) RD(u1) RD(v1) DEPEND(in, K(d0, b), K(e0, b), K(p, b), K(q0, b), dt_[0])
                 DEPEND(inout, K(d1, b), K(e1, b)),
                 MAP(present: xa[0:N], ya[0:N], vol[0:N], d0[0:N], d1[0:N], e0[0:N], e1[0:N], p[0:N], q0[0:N],
                              u0[0:N], v0[0:N], u1[0:N], v1[0:N], dt_[0:NDTV]) collapse(2),
                 firstprivate(xa, ya, vol, d0, d1, e0, e1, p, q0, u0, v0, u1, v1, dt_, half, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++)
            for (int i = 2; i < xm + 2; i++) {
                const double dt = dt_[DT];
                const double left_flux = (F(xa, i, j) * (F(u0, i, j) + F(u0, i, j + 1) + F(u1, i, j) + F(u1, i, j + 1))) * 0.25 * dt * half;
                const double right_flux = (F(xa, i + 1, j) * (F(u0, i + 1, j) + F(u0, i + 1, j + 1) + F(u1, i + 1, j) + F(u1, i + 1, j + 1))) * 0.25 * dt * half;
                const double bottom_flux = (F(ya, i, j) * (F(v0, i, j) + F(v0, i + 1, j) + F(v1, i, j) + F(v1, i + 1, j))) * 0.25 * dt * half;
                const double top_flux = (F(ya, i, j + 1) * (F(v0, i, j + 1) + F(v0, i + 1, j + 1) + F(v1, i, j + 1) + F(v1, i + 1, j + 1))) * 0.25 * dt * half;
                const double total_flux = right_flux - left_flux + top_flux - bottom_flux;
                const double volume_change_s = F(vol, i, j) / (F(vol, i, j) + total_flux);
                const double recip_volume = 1.0 / F(vol, i, j);
                const double energy_change = (F(p, i, j) / F(d0, i, j) + F(q0, i, j) / F(d0, i, j)) * total_flux * recip_volume;
                F(e1, i, j) = F(e0, i, j) - energy_change;
                F(d1, i, j) = F(d0, i, j) * volume_change_s;
            }
    }
}

static void revert()
{
    LOCALS
    double *d0 = density0, *d1 = density1, *e0 = energy0, *e1 = energy1;
    FOR_TILES(2, ym + 2) {
        OMP_TILE(DEPEND(in, K(d0, b), K(e0, b)) DEPEND(inout, K(d1, b), K(e1, b)),
                 MAP(present: d0[0:N], d1[0:N], e0[0:N], e1[0:N]) collapse(2),
                 firstprivate(d0, d1, e0, e1, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++)
            for (int i = 2; i < xm + 2; i++) {
                F(d1, i, j) = F(d0, i, j);
                F(e1, i, j) = F(e0, i, j);
            }
    }
}

static void accelerate()
{
    LOCALS
    double *xa = xarea, *ya = yarea, *vol = volume, *d0 = density0, *p = pressure, *q0 = viscosity,
           *u0 = xvel0, *v0 = yvel0, *u1 = xvel1, *v1 = yvel1, *dt_ = dtv;
    FOR_TILES(2, ym + 3) {
        OMP_TILE(RD(d0) RD(p) RD(q0) DEPEND(in, K(u0, b), K(v0, b), dt_[0]) DEPEND(inout, K(u1, b), K(v1, b)),
                 MAP(present: xa[0:N], ya[0:N], vol[0:N], d0[0:N], p[0:N], q0[0:N], u0[0:N], v0[0:N], u1[0:N], v1[0:N],
                              dt_[0:NDTV]) collapse(2),
                 firstprivate(xa, ya, vol, d0, p, q0, u0, v0, u1, v1, dt_, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++)
            for (int i = 2; i < xm + 3; i++) {
                const double halfdt = 0.5 * dt_[DT];
                const double stepbymass_s = halfdt / ((F(d0, i - 1, j - 1) * F(vol, i - 1, j - 1) + F(d0, i - 1, j) * F(vol, i - 1, j) +
                                                       F(d0, i, j) * F(vol, i, j) + F(d0, i, j - 1) * F(vol, i, j - 1)) * 0.25);
                F(u1, i, j) = F(u0, i, j) - stepbymass_s * (F(xa, i, j) * (F(p, i, j) - F(p, i - 1, j)) +
                                                            F(xa, i, j - 1) * (F(p, i, j - 1) - F(p, i - 1, j - 1)));
                F(v1, i, j) = F(v0, i, j) - stepbymass_s * (F(ya, i, j) * (F(p, i, j) - F(p, i, j - 1)) +
                                                            F(ya, i - 1, j) * (F(p, i - 1, j) - F(p, i - 1, j - 1)));
                F(u1, i, j) = F(u1, i, j) - stepbymass_s * (F(xa, i, j) * (F(q0, i, j) - F(q0, i - 1, j)) +
                                                            F(xa, i, j - 1) * (F(q0, i, j - 1) - F(q0, i - 1, j - 1)));
                F(v1, i, j) = F(v1, i, j) - stepbymass_s * (F(ya, i, j) * (F(q0, i, j) - F(q0, i, j - 1)) +
                                                            F(ya, i - 1, j) * (F(q0, i - 1, j) - F(q0, i - 1, j - 1)));
            }
    }
}

static void flux_calc()
{
    LOCALS
    double *xa = xarea, *ya = yarea, *u0 = xvel0, *v0 = yvel0, *u1 = xvel1, *v1 = yvel1,
           *vfx = vol_flux_x, *vfy = vol_flux_y, *dt_ = dtv;
    FOR_TILES(2, ym + 3) {
        OMP_TILE(RD(u0) RD(u1) DEPEND(in, K(v0, b), K(v1, b), dt_[0]) DEPEND(inout, K(vfx, b), K(vfy, b)),
                 MAP(present: xa[0:N], ya[0:N], u0[0:N], v0[0:N], u1[0:N], v1[0:N], vfx[0:N], vfy[0:N],
                              dt_[0:NDTV]) collapse(2),
                 firstprivate(xa, ya, u0, v0, u1, v1, vfx, vfy, dt_, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++)
            for (int i = 2; i < xm + 3; i++) {
                const double dt = dt_[DT];
                F(vfx, i, j) = 0.25 * dt * F(xa, i, j) * (F(u0, i, j) + F(u0, i, j + 1) + F(u1, i, j) + F(u1, i, j + 1));
                F(vfy, i, j) = 0.25 * dt * F(ya, i, j) * (F(v0, i, j) + F(v0, i + 1, j) + F(v1, i, j) + F(v1, i + 1, j));
            }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Advection (remap)
//////////////////////////////////////////////////////////////////////////////
// van Leer-limited donor-cell flux of density then energy across face `c`
// (upstream advec_cell; `st` is the array stride along the sweep direction)
#define ADVEC_CELL_FLUX(vf, mf, vd, pos, lim_hi, st)                                            \
    {                                                                                           \
        const size_t c = (size_t) j * s + i;                                                    \
        int upwind, donor, downwind, dif;                                                       \
        if (vf[c] > 0.0) { upwind = pos - 2; donor = pos - 1; downwind = pos; dif = donor; }    \
        else { upwind = std::min(pos + 1, lim_hi); donor = pos; downwind = pos - 1; dif = upwind; } \
        const size_t cu = c + (size_t) (upwind - pos) * st, cd = c + (size_t) (donor - pos) * st, \
                     cw = c + (size_t) (downwind - pos) * st;                                   \
        const double sigmat = fabs(vf[c]) / pre[cd];                                            \
        const double sigma3 = (1.0 + sigmat) * (vd[pos] / vd[dif]);                             \
        const double sigma4 = 2.0 - sigmat;                                                     \
        const double sigmav = sigmat;                                                           \
        double diffuw = d1[cd] - d1[cu], diffdw = d1[cw] - d1[cd];                              \
        double wind = (diffdw <= 0.0) ? -1.0 : 1.0, limiter;                                    \
        if (diffuw * diffdw > 0.0)                                                              \
            limiter = (1.0 - sigmav) * wind * fmin(fmin(fabs(diffuw), fabs(diffdw)),            \
                                                   one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))); \
        else limiter = 0.0;                                                                     \
        mf[c] = vf[c] * (d1[cd] + limiter);                                                     \
        const double sigmam = fabs(mf[c]) / (d1[cd] * pre[cd]);                                 \
        diffuw = e1[cd] - e1[cu]; diffdw = e1[cw] - e1[cd];                                     \
        wind = (diffdw <= 0.0) ? -1.0 : 1.0;                                                    \
        if (diffuw * diffdw > 0.0)                                                              \
            limiter = (1.0 - sigmam) * wind * fmin(fmin(fabs(diffuw), fabs(diffdw)),            \
                                                   one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))); \
        else limiter = 0.0;                                                                     \
        ef[c] = mf[c] * (e1[cd] + limiter);                                                     \
    }

static void advec_cell(int dir, int sweep)
{
    LOCALS
    double *vol = volume, *vfx = vol_flux_x, *vfy = vol_flux_y, *mfx = mass_flux_x, *mfy = mass_flux_y,
           *pre = pre_vol, *ef = ener_flux, *d1 = density1, *e1 = energy1, *vdx = vertexdx, *vdy = vertexdy;
    const double one_by_six = 1.0 / 6.0;

    // pre-sweep volume (post_vol is computed upstream but never read)
    FOR_TILES(0, ym + 4) {
        OMP_TILE(RD(vfy) DEPEND(in, K(vfx, b)) DEPEND(inout, K(pre, b)),
                 MAP(present: vol[0:N], vfx[0:N], vfy[0:N], pre[0:N]) collapse(2),
                 firstprivate(vol, vfx, vfy, pre, dir, sweep, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++)
            for (int i = 0; i < xm + 4; i++) {
                if (dir == XDIR)
                    F(pre, i, j) = (sweep == 1) ? F(vol, i, j) + (F(vfx, i + 1, j) - F(vfx, i, j) + F(vfy, i, j + 1) - F(vfy, i, j))
                                                : F(vol, i, j) + F(vfx, i + 1, j) - F(vfx, i, j);
                else
                    F(pre, i, j) = (sweep == 1) ? F(vol, i, j) + (F(vfy, i, j + 1) - F(vfy, i, j) + F(vfx, i + 1, j) - F(vfx, i, j))
                                                : F(vol, i, j) + F(vfy, i, j + 1) - F(vfy, i, j);
            }
    }

    if (dir == XDIR) {
        FOR_TILES(2, ym + 2) {
            OMP_TILE(DEPEND(in, K(vfx, b), K(pre, b), K(d1, b), K(e1, b)) DEPEND(inout, K(mfx, b), K(ef, b)),
                     MAP(present: vfx[0:N], pre[0:N], d1[0:N], e1[0:N], mfx[0:N], ef[0:N], vdx[0:L1]) collapse(2),
                     firstprivate(vfx, pre, d1, e1, mfx, ef, vdx, one_by_six, s, xm, jlo, jhi))
            for (int j = jlo; j < jhi; j++)
                for (int i = 2; i < xm + 4; i++)
                    ADVEC_CELL_FLUX(vfx, mfx, vdx, i, xm + 3, 1)
        }
        FOR_TILES(2, ym + 2) {
            OMP_TILE(DEPEND(in, K(vfx, b), K(pre, b), K(mfx, b), K(ef, b)) DEPEND(inout, K(d1, b), K(e1, b)),
                     MAP(present: vfx[0:N], pre[0:N], d1[0:N], e1[0:N], mfx[0:N], ef[0:N]) collapse(2),
                     firstprivate(vfx, pre, d1, e1, mfx, ef, s, xm, jlo, jhi))
            for (int j = jlo; j < jhi; j++)
                for (int i = 2; i < xm + 2; i++) {
                    const double pre_mass_s = F(d1, i, j) * F(pre, i, j);
                    const double post_mass_s = pre_mass_s + F(mfx, i, j) - F(mfx, i + 1, j);
                    const double post_ener_s = (F(e1, i, j) * pre_mass_s + F(ef, i, j) - F(ef, i + 1, j)) / post_mass_s;
                    const double advec_vol_s = F(pre, i, j) + F(vfx, i, j) - F(vfx, i + 1, j);
                    F(d1, i, j) = post_mass_s / advec_vol_s;
                    F(e1, i, j) = post_ener_s;
                }
        }
    } else {
        FOR_TILES(2, ym + 4) {
            OMP_TILE(RD(pre) RD(d1) RD(e1) DEPEND(in, K(vfy, b)) DEPEND(inout, K(mfy, b), K(ef, b)),
                     MAP(present: vfy[0:N], pre[0:N], d1[0:N], e1[0:N], mfy[0:N], ef[0:N], vdy[0:L1]) collapse(2),
                     firstprivate(vfy, pre, d1, e1, mfy, ef, vdy, one_by_six, s, xm, ym, jlo, jhi))
            for (int j = jlo; j < jhi; j++)
                for (int i = 2; i < xm + 2; i++)
                    ADVEC_CELL_FLUX(vfy, mfy, vdy, j, ym + 3, s)
        }
        FOR_TILES(2, ym + 2) {
            OMP_TILE(RD(mfy) RD(ef) RD(vfy) DEPEND(in, K(pre, b)) DEPEND(inout, K(d1, b), K(e1, b)),
                     MAP(present: vfy[0:N], pre[0:N], d1[0:N], e1[0:N], mfy[0:N], ef[0:N]) collapse(2),
                     firstprivate(vfy, pre, d1, e1, mfy, ef, s, xm, jlo, jhi))
            for (int j = jlo; j < jhi; j++)
                for (int i = 2; i < xm + 2; i++) {
                    const double pre_mass_s = F(d1, i, j) * F(pre, i, j);
                    const double post_mass_s = pre_mass_s + F(mfy, i, j) - F(mfy, i, j + 1);
                    const double post_ener_s = (F(e1, i, j) * pre_mass_s + F(ef, i, j) - F(ef, i, j + 1)) / post_mass_s;
                    const double advec_vol_s = F(pre, i, j) + F(vfy, i, j) - F(vfy, i, j + 1);
                    F(d1, i, j) = post_mass_s / advec_vol_s;
                    F(e1, i, j) = post_ener_s;
                }
        }
    }
}

// Momentum flux of one velocity component through node `c` along the sweep.
#define ADVEC_MOM_FLUX(pos, width, cd_, st)                                                      \
    {                                                                                            \
        const size_t c = (size_t) j * s + i;                                                     \
        int upwind, donor, downwind, dif;                                                        \
        if (F(nf, i, j) < 0.0) { upwind = pos + 2; donor = pos + 1; downwind = pos; dif = donor; } \
        else { upwind = pos - 1; donor = pos; downwind = pos + 1; dif = upwind; }                \
        const size_t cu = c + (size_t) (upwind - pos) * st, cd = c + (size_t) (donor - pos) * st, \
                     cw = c + (size_t) (downwind - pos) * st;                                    \
        const double sigma = fabs(nf[c]) / (npre[cd]);                                           \
        const double vdiffuw = vel[cd] - vel[cu];                                                \
        const double vdiffdw = vel[cw] - vel[cd];                                                \
        double limiter = 0.0;                                                                    \
        if (vdiffuw * vdiffdw > 0.0) {                                                           \
            const double auw = fabs(vdiffuw), adw = fabs(vdiffdw);                               \
            const double wind = (vdiffdw <= 0.0) ? -1.0 : 1.0;                                   \
            limiter = wind * fmin(fmin(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / cd_[dif]) / 6.0, auw), adw); \
        }                                                                                        \
        const double advec_vel_s = vel[cd] + (1.0 - sigma) * limiter;                            \
        mf[c] = advec_vel_s * nf[c];                                                             \
    }

// advec_mom for one velocity; `first` (the x velocity, swept first) also builds
// the node fluxes / masses that the second one reuses, as upstream does.
static void advec_mom(double *vel, bool first, int dir, int sweep)
{
    LOCALS
    double *vol = volume, *vfx = vol_flux_x, *vfy = vol_flux_y, *mfx = mass_flux_x, *mfy = mass_flux_y,
           *d1 = density1, *pv = post_vol, *nf = node_flux, *npost = node_mass_post, *npre = node_mass_pre,
           *mf = mom_flux, *cdx = celldx, *cdy = celldy;

    if (first) {
        // post-sweep volume (pre_vol is computed upstream but never read)
        FOR_TILES(0, ym + 4) {
            OMP_TILE(RD(vfy) DEPEND(in, K(vfx, b)) DEPEND(inout, K(pv, b)),
                     MAP(present: vol[0:N], vfx[0:N], vfy[0:N], pv[0:N]) collapse(2),
                     firstprivate(vol, vfx, vfy, pv, dir, sweep, s, xm, jlo, jhi))
            for (int j = jlo; j < jhi; j++)
                for (int i = 0; i < xm + 4; i++) {
                    if (sweep == 2)        F(pv, i, j) = F(vol, i, j);
                    else if (dir == XDIR)  F(pv, i, j) = F(vol, i, j) + F(vfy, i, j + 1) - F(vfy, i, j);
                    else                   F(pv, i, j) = F(vol, i, j) + F(vfx, i + 1, j) - F(vfx, i, j);
                }
        }
        if (dir == XDIR) {
            FOR_TILES(2, ym + 3) {
                OMP_TILE(RD(mfx) DEPEND(inout, K(nf, b)),
                         MAP(present: mfx[0:N], nf[0:N]) collapse(2),
                         firstprivate(mfx, nf, s, xm, jlo, jhi))
                for (int j = jlo; j < jhi; j++)
                    for (int i = 0; i < xm + 4; i++)
                        F(nf, i, j) = 0.25 * (F(mfx, i, j - 1) + F(mfx, i, j) + F(mfx, i + 1, j - 1) + F(mfx, i + 1, j));
            }
            FOR_TILES(2, ym + 3) {
                OMP_TILE(RD(d1) RD(pv) DEPEND(in, K(nf, b)) DEPEND(inout, K(npost, b), K(npre, b)),
                         MAP(present: d1[0:N], pv[0:N], nf[0:N], npost[0:N], npre[0:N]) collapse(2),
                         firstprivate(d1, pv, nf, npost, npre, s, xm, jlo, jhi))
                for (int j = jlo; j < jhi; j++)
                    for (int i = 1; i < xm + 4; i++) {
                        F(npost, i, j) = 0.25 * (F(d1, i, j - 1) * F(pv, i, j - 1) + F(d1, i, j) * F(pv, i, j) +
                                                 F(d1, i - 1, j - 1) * F(pv, i - 1, j - 1) + F(d1, i - 1, j) * F(pv, i - 1, j));
                        F(npre, i, j) = F(npost, i, j) - F(nf, i - 1, j) + F(nf, i, j);
                    }
            }
        } else {
            FOR_TILES(0, ym + 4) {
                OMP_TILE(RD(mfy) DEPEND(inout, K(nf, b)),
                         MAP(present: mfy[0:N], nf[0:N]) collapse(2),
                         firstprivate(mfy, nf, s, xm, jlo, jhi))
                for (int j = jlo; j < jhi; j++)
                    for (int i = 2; i < xm + 3; i++)
                        F(nf, i, j) = 0.25 * (F(mfy, i - 1, j) + F(mfy, i, j) + F(mfy, i - 1, j + 1) + F(mfy, i, j + 1));
            }
            FOR_TILES(1, ym + 4) {
                OMP_TILE(RD(d1) RD(pv) RD(nf) DEPEND(inout, K(npost, b), K(npre, b)),
                         MAP(present: d1[0:N], pv[0:N], nf[0:N], npost[0:N], npre[0:N]) collapse(2),
                         firstprivate(d1, pv, nf, npost, npre, s, xm, jlo, jhi))
                for (int j = jlo; j < jhi; j++)
                    for (int i = 2; i < xm + 3; i++) {
                        F(npost, i, j) = 0.25 * (F(d1, i, j - 1) * F(pv, i, j - 1) + F(d1, i, j) * F(pv, i, j) +
                                                 F(d1, i - 1, j - 1) * F(pv, i - 1, j - 1) + F(d1, i - 1, j) * F(pv, i - 1, j));
                        F(npre, i, j) = F(npost, i, j) - F(nf, i, j - 1) + F(nf, i, j);
                    }
            }
        }
    }

    if (dir == XDIR) {
        FOR_TILES(2, ym + 3) {
            OMP_TILE(DEPEND(in, K(nf, b), K(npre, b), K(vel, b)) DEPEND(inout, K(mf, b)),
                     MAP(present: nf[0:N], npre[0:N], vel[0:N], mf[0:N], cdx[0:L1]) collapse(2),
                     firstprivate(nf, npre, vel, mf, cdx, s, xm, jlo, jhi))
            for (int j = jlo; j < jhi; j++)
                for (int i = 1; i < xm + 3; i++)
                    ADVEC_MOM_FLUX(i, cdx[i], cdx, 1)
        }
        FOR_TILES(2, ym + 3) {
            OMP_TILE(DEPEND(in, K(npre, b), K(npost, b), K(mf, b)) DEPEND(inout, K(vel, b)),
                     MAP(present: npre[0:N], npost[0:N], mf[0:N], vel[0:N]) collapse(2),
                     firstprivate(npre, npost, mf, vel, s, xm, jlo, jhi))
            for (int j = jlo; j < jhi; j++)
                for (int i = 2; i < xm + 3; i++)
                    F(vel, i, j) = (F(vel, i, j) * F(npre, i, j) + F(mf, i - 1, j) - F(mf, i, j)) / F(npost, i, j);
        }
    } else {
        FOR_TILES(1, ym + 3) {
            OMP_TILE(RD(npre) RD(vel) DEPEND(in, K(nf, b)) DEPEND(inout, K(mf, b)),
                     MAP(present: nf[0:N], npre[0:N], vel[0:N], mf[0:N], cdy[0:L1]) collapse(2),
                     firstprivate(nf, npre, vel, mf, cdy, s, xm, jlo, jhi))
            for (int j = jlo; j < jhi; j++)
                for (int i = 2; i < xm + 3; i++)
                    ADVEC_MOM_FLUX(j, cdy[j], cdy, s)
        }
        FOR_TILES(2, ym + 3) {
            OMP_TILE(RD(mf) DEPEND(in, K(npre, b), K(npost, b)) DEPEND(inout, K(vel, b)),
                     MAP(present: npre[0:N], npost[0:N], mf[0:N], vel[0:N]) collapse(2),
                     firstprivate(npre, npost, mf, vel, s, xm, jlo, jhi))
            for (int j = jlo; j < jhi; j++)
                for (int i = 2; i < xm + 3; i++)
                    F(vel, i, j) = (F(vel, i, j) * F(npre, i, j) + F(mf, i, j - 1) - F(mf, i, j)) / F(npost, i, j);
        }
    }
}

static void advection(bool advect_x)
{
    const int dir1 = advect_x ? XDIR : YDIR, dir2 = advect_x ? YDIR : XDIR;
    halo(energy1, H_CELL, 2); halo(density1, H_CELL, 2);
    halo(vol_flux_x, H_FX, 2); halo(vol_flux_y, H_FY, 2);
    for (int sweep = 1; sweep <= 2; sweep++) {
        const int dir = (sweep == 1) ? dir1 : dir2;
        advec_cell(dir, sweep);
        halo(density1, H_CELL, 2); halo(energy1, H_CELL, 2);
        halo(xvel1, H_XVEL, 2); halo(yvel1, H_YVEL, 2);
        halo(mass_flux_x, H_FX, 2); halo(mass_flux_y, H_FY, 2);
        advec_mom(xvel1, true, dir, sweep);
        advec_mom(yvel1, false, dir, sweep);
    }
}

static void reset_field()
{
    LOCALS
    double *d0 = density0, *d1 = density1, *e0 = energy0, *e1 = energy1, *u0 = xvel0, *u1 = xvel1, *v0 = yvel0, *v1 = yvel1;
    FOR_TILES(2, ym + 2) {
        OMP_TILE(DEPEND(in, K(d1, b), K(e1, b)) DEPEND(inout, K(d0, b), K(e0, b)),
                 MAP(present: d0[0:N], d1[0:N], e0[0:N], e1[0:N]) collapse(2),
                 firstprivate(d0, d1, e0, e1, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++)
            for (int i = 2; i < xm + 2; i++) {
                F(d0, i, j) = F(d1, i, j);
                F(e0, i, j) = F(e1, i, j);
            }
    }
    FOR_TILES(2, ym + 3) {
        OMP_TILE(DEPEND(in, K(u1, b), K(v1, b)) DEPEND(inout, K(u0, b), K(v0, b)),
                 MAP(present: u0[0:N], u1[0:N], v0[0:N], v1[0:N]) collapse(2),
                 firstprivate(u0, u1, v0, v1, s, xm, jlo, jhi))
        for (int j = jlo; j < jhi; j++)
            for (int i = 2; i < xm + 3; i++) {
                F(u0, i, j) = F(u1, i, j);
                F(v0, i, j) = F(v1, i, j);
            }
    }
}

// One timestep: driver/timestep.cpp then driver/hydro.cpp's kernel chain.
static void timestep(bool advect_x)
{
    ideal_gas(density0, energy0);
    halo(pressure, H_CELL, 1); halo(energy0, H_CELL, 1); halo(density0, H_CELL, 1);
    halo(xvel0, H_XVEL, 1); halo(yvel0, H_YVEL, 1);
    viscosity_kernel();
    halo(viscosity, H_CELL, 1);
    calc_dt();

    PdV(true);
    ideal_gas(density1, energy1);
    halo(pressure, H_CELL, 1);
    revert();
    accelerate();
    PdV(false);
    flux_calc();
    advection(advect_x);
    reset_field();
}

//////////////////////////////////////////////////////////////////////////////
// Setup (initialise_chunk + generate_chunk for the clover_bm states) and summary
//////////////////////////////////////////////////////////////////////////////
static double *alloc(size_t n) { return (double *) calloc(n, sizeof(double)); }

static void generate(int cells)
{
    const int s = S;
    const double dx = 10.0 / cells, dy = 10.0 / cells;   // domain [0,10]^2
    for (int i = 0; i < L1; i++) {
        vertexx[i] = dx * (i - 2); vertexdx[i] = dx; celldx[i] = dx;
        vertexy[i] = dy * (i - 2); vertexdy[i] = dy; celldy[i] = dy;
    }
    for (int i = 0; i < L1 - 1; i++) {
        cellx[i] = 0.5 * (vertexx[i] + vertexx[i + 1]);
        celly[i] = 0.5 * (vertexy[i] + vertexy[i + 1]);
    }
    for (int j = 0; j < ymax + 5; j++)
        for (int i = 0; i < S; i++) {
            F(volume, i, j) = dx * dy;
            F(xarea, i, j) = celldy[j];
            F(yarea, i, j) = celldx[i];
        }
    // state 1 everywhere, then state 2: the rectangle [0,5] x [0,2]
    for (int j = 0; j < ymax + 4; j++)
        for (int i = 0; i < xmax + 4; i++) {
            F(energy0, i, j) = 1.0;
            F(density0, i, j) = 0.2;
            if (vertexx[i + 1] >= 0.0 && vertexx[i] < 5.0 && vertexy[j + 1] >= 0.0 && vertexy[j] < 2.0) {
                F(energy0, i, j) = 2.5;
                F(density0, i, j) = 1.0;
            }
        }
}

struct Summary { double vol, mass, ie, ke, press; };

static Summary field_summary()
{
    const int s = S;
    Summary r = {0, 0, 0, 0, 0};
    for (int j = 2; j < ymax + 2; j++)
        for (int i = 2; i < xmax + 2; i++) {
            double vsqrd = 0.0;
            for (int kv = j; kv <= j + 1; ++kv)
                for (int jv = i; jv <= i + 1; ++jv)
                    vsqrd += 0.25 * (F(xvel0, jv, kv) * F(xvel0, jv, kv) + F(yvel0, jv, kv) * F(yvel0, jv, kv));
            const double cell_vol = F(volume, i, j);
            const double cell_mass = cell_vol * F(density0, i, j);
            r.vol += cell_vol;
            r.mass += cell_mass;
            r.ie += cell_mass * F(energy0, i, j);
            r.ke += cell_mass * 0.5 * vsqrd;
            r.press += cell_vol * (1.4 - 1.0) * F(density0, i, j) * F(energy0, i, j);
        }
    return r;
}

int main(int argc, char **argv)
{
    int cells = 512, NB = 32;
    long iters = 50, unroll = 1;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-s") && a + 1 < argc) cells = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-i") && a + 1 < argc) iters = atol(argv[++a]);
        else if (!strcmp(argv[a], "-nb") && a + 1 < argc) NB = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-u") && a + 1 < argc) unroll = atol(argv[++a]);
        else if (!strcmp(argv[a], "-h")) {
            printf("usage: %s [-s cells-per-side] [-i timestep-pairs] [-nb tasks-per-loop] [-u unroll]\n", argv[0]);
            return 0;
        } else { fprintf(stderr, "%s: unknown option '%s' (-h for help)\n", argv[0], argv[a]); return 1; }
    }
    if (cells < 2) { fprintf(stderr, "-s must be >= 2\n"); return 1; }
    if (unroll < 1) unroll = 1;
    if (NB < 1) NB = 1;

    xmax = ymax = cells;
    S = xmax + 5;
    N = (size_t) S * (ymax + 5);
    L1 = std::max(xmax, ymax) + 5;
    KBS = std::max(2, (ymax + NB - 1) / NB);   // >= 2 rows: stencils reach 2 rows
    nblk = (ymax + KBS - 1) / KBS;

    iters = tg_round_iters(iters, unroll);
    if (iters <= 0) return 1;
    const long ninst = iters / unroll;

    double **fields2d[] = {&density0, &density1, &energy0, &energy1, &pressure, &viscosity, &soundspeed,
                           &xvel0, &xvel1, &yvel0, &yvel1, &vol_flux_x, &vol_flux_y, &mass_flux_x, &mass_flux_y,
                           &volume, &xarea, &yarea, &pre_vol, &ener_flux,
                           &post_vol, &node_flux, &node_mass_post, &node_mass_pre, &mom_flux};
    double **fields1d[] = {&vertexx, &vertexy, &vertexdx, &vertexdy, &cellx, &celly, &celldx, &celldy, &dtrow};
    for (double **f : fields2d) *f = alloc(N);
    for (double **f : fields1d) *f = alloc(L1);
    dtv = alloc(NDTV);
    dtv[DT] = dtinit; dtv[DTOLD] = dtinit;

    generate(cells);

    printf("CloverLeaf: %d x %d cells, %ld timesteps (%ld pairs)\n", xmax, ymax, 2 * iters, iters);
    printf("backend=%s taskgraph=%d\n", tg_backend_name(), USE_TASKGRAPH);
    printf("Num tasks per loop: %d\n", nblk);

#define ALL2D(m) m(density0[0:N], density1[0:N], energy0[0:N], energy1[0:N], pressure[0:N], viscosity[0:N],   \
                   soundspeed[0:N], xvel0[0:N], xvel1[0:N], yvel0[0:N], yvel1[0:N], vol_flux_x[0:N],              \
                   vol_flux_y[0:N], mass_flux_x[0:N], mass_flux_y[0:N], volume[0:N], xarea[0:N], yarea[0:N],      \
                   pre_vol[0:N], ener_flux[0:N], post_vol[0:N], node_flux[0:N], node_mass_post[0:N],              \
                   node_mass_pre[0:N], mom_flux[0:N], vertexdx[0:L1], vertexdy[0:L1], celldx[0:L1], celldy[0:L1], \
                   dtrow[0:L1], dtv[0:NDTV])
#define MAP_TO(...)      MAP(to: __VA_ARGS__)
#define MAP_RELEASE(...) MAP(release: __VA_ARGS__)
    OMP_TARGET_ENTER_DATA(ALL2D(MAP_TO))

    double *inst_ms = (double *) malloc(sizeof(double) * ninst);
    double prev_ts = 0.0, t0 = 0.0, tf = 0.0;
    const int nb = nblk;
    #pragma omp parallel
    {
        #pragma omp single
        {
            // start.cpp: equation of state, then every halo at depth 2
            ideal_gas(density0, energy0);
            halo(density0, H_CELL, 2); halo(energy0, H_CELL, 2); halo(pressure, H_CELL, 2);
            halo(viscosity, H_CELL, 2); halo(density1, H_CELL, 2); halo(energy1, H_CELL, 2);
            halo(xvel0, H_XVEL, 2); halo(yvel0, H_YVEL, 2); halo(xvel1, H_XVEL, 2); halo(yvel1, H_YVEL, 2);
            #pragma omp taskwait

            t0 = omp_get_wtime();
            prev_ts = t0;
            TASKGRAPH_LOOP(unroll,
                           [&] (size_t done) { return done < (size_t) iters; },
                           [&] (size_t inst, size_t done)
            {
                (void) done;
                // ms per iteration, after reset_field on every block
                EPILOGUE_TASK(firstprivate(inst, unroll, inst_ms) shared(prev_ts)
                              DEPEND_MULTI(in, (q = 0:nb), K(density0, q))
                              DEPEND_MULTI(in, (q = 0:nb), K(xvel0, q)))
                {
                    const double now = omp_get_wtime();
                    inst_ms[inst] = (now - prev_ts) * 1000.0 / (double) unroll;
                    prev_ts = now;
                }
            })
            {
                timestep(true);    // x sweep first
                timestep(false);   // y sweep first
            }
            TASKGRAPH_LOOP_END
            #pragma omp taskwait
            tf = omp_get_wtime();
        }
    }

    OMP_TARGET_UPDATE(from(density0[0:N], energy0[0:N], xvel0[0:N], yvel0[0:N], dtv[0:NDTV]))
    OMP_TARGET_EXIT_DATA(ALL2D(MAP_RELEASE))

    const Summary r = field_summary();
    printf(" %10s %20s %20s %20s %20s %20s\n", "time", "volume", "mass", "internal energy", "kinetic energy", "pressure");
    printf(" %10.6f %20.12e %20.12e %20.12e %20.12e %20.12e\n", dtv[TIME], r.vol, r.mass, r.ie, r.ke, r.press);
    printf("last dt              : %.12e\n", dtv[DT]);
    const double mean = tg_report_instances(inst_ms, ninst, unroll);
    printf("Elapsed time (s)     : %10.6f\n", tf - t0);
    const double cell_updates = 2.0 * xmax * ymax;   // per iteration (2 timesteps)
    printf("FOM (cell-updates/s) : %10.8g\n", (mean > 0.0) ? cell_updates / (mean / 1000.0) : 0.0);
    printf("answer (kinetic energy): %.15e\n", r.ke);

    const int bad = dtv[BAD] != 0.0;
    if (bad) fprintf(stderr, "error: invalid or too small timestep (< %g) during the run\n", dtmin);

    free(inst_ms);
    for (double **f : fields2d) free(*f);
    for (double **f : fields1d) free(*f);
    free(dtv);
    return bad;
}
