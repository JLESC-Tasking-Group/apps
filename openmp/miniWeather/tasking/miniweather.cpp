//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather - tasking.h port
//
// Dry, stratified, compressible, non-hydrostatic 2D (x-z) fluid solver. Derived
// from the ORNL miniWeather serial reference (c/miniWeather_serial.cpp), stripped
// of MPI / parallel-netcdf (single node, periodic halo, no file output) and
// re-expressed on the shared apps/openmp tasking.h backends:
//
//   USE_TARGET=0 : each spatial loop is tiled into -nb CPU tasks over the z-rows
//   USE_TARGET=1 : each tiled loop is one offloaded parallel-for
//   USE_SYNC=1   : synchronous blocking baseline (one kernel at a time)
//   USE_TASKGRAPH: one perform_timestep() body recorded/replayed
//
// The solver advances with a 3-stage low-storage RK and Strang (dimensional)
// splitting that alternates x-first / z-first every step. Because the alternation
// makes consecutive timestep bodies differ, we record TWO taskgraphs (one per
// parity) and replay the matching one -- the toggle itself is kept out of the
// recorded region.
//
// Decomposition: every spatial loop is tiled into blocks of KBS z-rows. Task
// dependences are keyed on block-aligned addresses X[b*KBS*<stride>] and, where a
// z-stencil reaches into neighbouring blocks (+/-hs rows), the adjacent blocks are
// named as extra `in` deps so producer/consumer keys collide. This is the same
// convention the LULESH / PolyBench ports use and is portable across runtimes
// that match depend sections only by their start address.
//
// Verification: the native mass / total-energy conservation check (d_mass, d_te).
// The reference test (NX=100 NZ=50 SIM_TIME=400) requires |d_mass|<1e-13 and
// d_te<0 with |d_te|<4.5e-5.
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <omp.h>

#include "tasking.h"

constexpr double pi        = 3.14159265358979323846264338327;
constexpr double grav      = 9.8;
constexpr double cp        = 1004.;
constexpr double cv        = 717.;
constexpr double rd        = 287.;
constexpr double p0        = 1.e5;
constexpr double C0        = 27.5629410929725921310572974482;
constexpr double gamm      = 1.40027894002789400278940027894;
constexpr double xlen      = 2.e4;
constexpr double zlen      = 1.e4;
constexpr double hv_beta   = 0.05;
constexpr double cfl       = 1.50;
constexpr double max_speed = 450;
constexpr int    hs        = 2;
constexpr int    sten_size = 4;

constexpr int NUM_VARS = 4;
constexpr int ID_DENS  = 0;
constexpr int ID_UMOM  = 1;
constexpr int ID_WMOM  = 2;
constexpr int ID_RHOT  = 3;
constexpr int DIR_X = 1;
constexpr int DIR_Z = 2;
constexpr int DATA_SPEC_COLLISION       = 1;
constexpr int DATA_SPEC_THERMAL         = 2;
constexpr int DATA_SPEC_GRAVITY_WAVES   = 3;
constexpr int DATA_SPEC_DENSITY_CURRENT = 5;
constexpr int DATA_SPEC_INJECTION       = 6;

constexpr int nqpoints = 3;
constexpr double qpoints [] = { 0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0 };
constexpr double qweights[] = { 0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0 };

#ifndef _NX
# define _NX 100
#endif
#ifndef _NZ
# define _NZ 50
#endif
#ifndef _SIM_TIME
# define _SIM_TIME 400
#endif
#ifndef _OUT_FREQ
# define _OUT_FREQ (-1)
#endif
#ifndef _DATA_SPEC
# define _DATA_SPEC DATA_SPEC_THERMAL
#endif

int    constexpr nx_glob       = _NX;
int    constexpr nz_glob       = _NZ;
double constexpr sim_time      = _SIM_TIME;
double constexpr output_freq   = _OUT_FREQ;
int    constexpr data_spec_int = _DATA_SPEC;
double constexpr dx            = xlen / nx_glob;
double constexpr dz            = zlen / nz_glob;

// --- static (loop-invariant) state ---
double dt;
int    nx, nz;
int    i_beg, k_beg;
int    mainproc = 1;
double *hy_dens_cell;
double *hy_dens_theta_cell;
double *hy_dens_int;
double *hy_dens_theta_int;
double *hy_pressure_int;

// --- dynamic state ---
double etime;
double *state;
double *state_tmp;
double *flux;
double *tend;
int    direction_switch = 1;
double mass0, te0, mass, te;

// --- tiling knob ---
int NB  = 32;    // tasks per loop (-nb)
int KBS = 1;     // z-rows per tile (derived from NB and nz)

double dmin(double a, double b) { return a < b ? a : b; }

// strides for block-aligned dependency keys
static inline int st_row() { return nx + 2 * hs; }              // state row stride (with x halo)
static inline int fl_row() { return nx + 1; }                   // flux row stride
static inline int td_row() { return nx; }                       // tend row stride

void init();
void finalize();
void injection      (double x,double z,double&r,double&u,double&w,double&t,double&hr,double&ht);
void density_current(double x,double z,double&r,double&u,double&w,double&t,double&hr,double&ht);
void gravity_waves  (double x,double z,double&r,double&u,double&w,double&t,double&hr,double&ht);
void thermal        (double x,double z,double&r,double&u,double&w,double&t,double&hr,double&ht);
void collision      (double x,double z,double&r,double&u,double&w,double&t,double&hr,double&ht);
void hydro_const_theta (double z,double&r,double&t);
void hydro_const_bvfreq(double z,double bv_freq0,double&r,double&t);
double sample_ellipse_cosine(double x,double z,double amp,double x0,double z0,double xrad,double zrad);
void perform_timestep  (double dt);
void semi_discrete_step(double*state_init,double*state_forcing,double*state_out,double dt,int dir);
void compute_tendencies_x(double*state,double dt);
void compute_tendencies_z(double*state,double dt);
void set_halo_values_x (double*state);
void set_halo_values_z (double*state);
void reductions        (double&mass,double&te);


//////////////////////////////////////////////////////////////////////////////////////////
// semi-discrete RK stage: state_out = state_init + dt * rhs(state_forcing)
//////////////////////////////////////////////////////////////////////////////////////////
void semi_discrete_step(double *state_init, double *state_forcing, double *state_out,
                        double dt, int dir)
{
    const int SR = st_row(), TR = td_row();
    const int nblk = (nz + KBS - 1) / KBS;

    if (dir == DIR_X) {
        set_halo_values_x(state_forcing);
        compute_tendencies_x(state_forcing, dt);
    } else {
        set_halo_values_z(state_forcing);
        compute_tendencies_z(state_forcing, dt);
    }

    // Apply tendencies: state_out[ll,k,i] = state_init[ll,k,i] + dt*tend[ll,k,i]
    // Tiled over z-rows; each block reads its own rows of state_init and the whole
    // tend scratch (keyed on tend[0] -- see note in compute_tendencies_*), writes
    // its own rows of state_out.
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nz ? nz : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(in, state_init[(kb + hs) * SR], tend[0])
                 DEPEND(out, state_out[(kb + hs) * SR]),
                 MAP(present: state_init[0:(size_t)(nz+2*hs)*(nx+2*hs)*NUM_VARS],
                              state_out[0:(size_t)(nz+2*hs)*(nx+2*hs)*NUM_VARS],
                              tend[0:(size_t)nz*nx*NUM_VARS])
                 num_teams((ke - kb)) thread_limit(256),
                 firstprivate(state_init, state_out, tend, kb, ke, dt))
        for (int ll = 0; ll < NUM_VARS; ll++)
            for (int k = kb; k < ke; k++)
                for (int i = 0; i < nx; i++) {
                    const int inds = ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + hs;
                    const int indt = ll * nz * nx + k * nx + i;
                    state_out[inds] = state_init[inds] + dt * tend[indt];
                }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
// x-direction tendencies: flux (nz x nx+1) then tend (nz x nx). Rows independent in k.
//////////////////////////////////////////////////////////////////////////////////////////
void compute_tendencies_x(double *state, double dt)
{
    const double hv_coef = -hv_beta * dx / (16 * dt);
    const int SR = st_row(), FR = fl_row(), TR = td_row();
    const int nblk = (nz + KBS - 1) / KBS;
    const size_t nstate = (size_t)(nz + 2 * hs) * (nx + 2 * hs) * NUM_VARS;
    const size_t nfl = (size_t)(nz + 1) * (nx + 1) * NUM_VARS;
    const size_t ntd = (size_t)nz * nx * NUM_VARS;

    // flux/tend are per-stage scratch reused by every RK stage. To keep the WAR/WAW
    // chains across the ~24 kernels/step provably correct without fragile per-block
    // flux-interface alignment, they are keyed on a single whole-buffer token
    // (flux[0] / tend[0]); this serializes flux and tend *production* per stage (as
    // the openmp45 reference does via one asyncid token) while the parallelism-rich
    // state update stays tiled. `state` reads keep precise per-block keys so the
    // cross-timestep and cross-stage state chains stay fine-grained.
    //
    // flux: each k-row reads state rows [k] only (x-stencil in i), independent in k.
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nz ? nz : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(in, state[(kb + hs) * SR]) DEPEND(out, flux[0]),
                 MAP(present: state[0:nstate], flux[0:nfl],
                              hy_dens_cell[0:(size_t)(nz+2*hs)], hy_dens_theta_cell[0:(size_t)(nz+2*hs)])
                 num_teams((ke - kb)) thread_limit(256),
                 firstprivate(state, flux, kb, ke, hv_coef))
        for (int k = kb; k < ke; k++)
            for (int i = 0; i < nx + 1; i++) {
                double stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];
                for (int ll = 0; ll < NUM_VARS; ll++) {
                    for (int s = 0; s < sten_size; s++) {
                        const int inds = ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + s;
                        stencil[s] = state[inds];
                    }
                    vals[ll]    = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
                    d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
                }
                const double r = vals[ID_DENS] + hy_dens_cell[k + hs];
                const double u = vals[ID_UMOM] / r;
                const double w = vals[ID_WMOM] / r;
                const double t = (vals[ID_RHOT] + hy_dens_theta_cell[k + hs]) / r;
                const double p = C0 * pow(r * t, gamm);
                flux[ID_DENS * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * u       - hv_coef * d3_vals[ID_DENS];
                flux[ID_UMOM * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * u * u + p - hv_coef * d3_vals[ID_UMOM];
                flux[ID_WMOM * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * u * w   - hv_coef * d3_vals[ID_WMOM];
                flux[ID_RHOT * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * u * t   - hv_coef * d3_vals[ID_RHOT];
            }
    }

    // tend: reads flux row [k] only, independent in k
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nz ? nz : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(in, flux[0]) DEPEND(out, tend[0]),
                 MAP(present: flux[0:nfl], tend[0:ntd]) num_teams((ke - kb)) thread_limit(256),
                 firstprivate(flux, tend, kb, ke))
        for (int ll = 0; ll < NUM_VARS; ll++)
            for (int k = kb; k < ke; k++)
                for (int i = 0; i < nx; i++) {
                    const int indt  = ll * nz * nx + k * nx + i;
                    const int indf1 = ll * (nz + 1) * (nx + 1) + k * (nx + 1) + i;
                    const int indf2 = ll * (nz + 1) * (nx + 1) + k * (nx + 1) + i + 1;
                    tend[indt] = -(flux[indf2] - flux[indf1]) / dx;
                }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
// z-direction tendencies: flux (nz+1 x nx) then tend (nz x nx). z-stencil couples k rows.
//////////////////////////////////////////////////////////////////////////////////////////
void compute_tendencies_z(double *state, double dt)
{
    const double hv_coef = -hv_beta * dz / (16 * dt);
    const int SR = st_row(), FR = fl_row(), TR = td_row();
    const int nblk  = (nz + KBS - 1) / KBS;
    const int nblkf = (nz + 1 + KBS - 1) / KBS;   /* blocks covering nz+1 z-interfaces */
    const size_t nstate = (size_t)(nz + 2 * hs) * (nx + 2 * hs) * NUM_VARS;
    const size_t nfl = (size_t)(nz + 1) * (nx + 1) * NUM_VARS;
    const size_t ntd = (size_t)nz * nx * NUM_VARS;

    // flux at interfaces k=0..nz (nz+1 of them): reads state rows k..k+3 (z-stencil,
    // array rows [kb, ke+2]). A flux block names its own block b plus neighbours
    // b+1..b+3 (so the +3 stencil reach is covered even for KBS=1) AND the z-halo
    // boundary keys (state[0] for low rows 0,1; state[(nz+hs)*SR] for the top rows)
    // that set_halo_values_z writes, so the halo->flux edge is explicit.
    for (int b = 0; b < nblkf; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nz + 1 ? nz + 1 : (b + 1) * KBS;
        if (kb >= ke) continue;
        const int bp1 = (b + 1 < nblkf) ? (b + 1) : b;
        const int bp2 = (b + 2 < nblkf) ? (b + 2) : bp1;
        const int bp3 = (b + 3 < nblkf) ? (b + 3) : bp2;
        OMP_TILE(DEPEND(in, state[(kb + hs) * SR], state[(bp1 * KBS + hs) * SR],
                            state[(bp2 * KBS + hs) * SR], state[(bp3 * KBS + hs) * SR],
                            state[0], state[(nz + hs) * SR])
                 DEPEND(out, flux[0]),
                 MAP(present: state[0:nstate], flux[0:nfl],
                              hy_dens_int[0:(size_t)(nz+1)], hy_dens_theta_int[0:(size_t)(nz+1)],
                              hy_pressure_int[0:(size_t)(nz+1)])
                 num_teams((ke - kb)) thread_limit(256),
                 firstprivate(state, flux, kb, ke, hv_coef))
        for (int k = kb; k < ke; k++)
            for (int i = 0; i < nx; i++) {
                double stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];
                for (int ll = 0; ll < NUM_VARS; ll++) {
                    for (int s = 0; s < sten_size; s++) {
                        const int inds = ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + s) * (nx + 2 * hs) + i + hs;
                        stencil[s] = state[inds];
                    }
                    vals[ll]    = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
                    d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
                }
                double r = vals[ID_DENS] + hy_dens_int[k];
                double u = vals[ID_UMOM] / r;
                double w = vals[ID_WMOM] / r;
                double t = (vals[ID_RHOT] + hy_dens_theta_int[k]) / r;
                double p = C0 * pow(r * t, gamm) - hy_pressure_int[k];
                if (k == 0 || k == nz) { w = 0; d3_vals[ID_DENS] = 0; }
                flux[ID_DENS * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * w       - hv_coef * d3_vals[ID_DENS];
                flux[ID_UMOM * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * w * u   - hv_coef * d3_vals[ID_UMOM];
                flux[ID_WMOM * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * w * w + p - hv_coef * d3_vals[ID_WMOM];
                flux[ID_RHOT * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * w * t   - hv_coef * d3_vals[ID_RHOT];
            }
    }

    // tend: reads flux rows k, k+1 (-> block b and b+1) and state row k (gravity term)
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nz ? nz : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(in, flux[0], state[(kb + hs) * SR])
                 DEPEND(out, tend[0]),
                 MAP(present: flux[0:nfl], tend[0:ntd], state[0:nstate]) num_teams((ke - kb)) thread_limit(256),
                 firstprivate(flux, tend, state, kb, ke))
        for (int ll = 0; ll < NUM_VARS; ll++)
            for (int k = kb; k < ke; k++)
                for (int i = 0; i < nx; i++) {
                    const int indt  = ll * nz * nx + k * nx + i;
                    const int indf1 = ll * (nz + 1) * (nx + 1) + (k) * (nx + 1) + i;
                    const int indf2 = ll * (nz + 1) * (nx + 1) + (k + 1) * (nx + 1) + i;
                    double tv = -(flux[indf2] - flux[indf1]) / dz;
                    if (ll == ID_WMOM) {
                        const int inds = ID_DENS * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + hs;
                        tv -= state[inds] * grav;
                    }
                    tend[indt] = tv;
                }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
// halos (single rank, periodic in x, no-flux in z). Tiled over k for parity with kernels.
//////////////////////////////////////////////////////////////////////////////////////////
void set_halo_values_x(double *state)
{
    const int SR = st_row();
    const int nblk = (nz + KBS - 1) / KBS;
    const size_t nstate = (size_t)(nz + 2 * hs) * (nx + 2 * hs) * NUM_VARS;
    for (int b = 0; b < nblk; b++) {
        const int kb = b * KBS;
        const int ke = (b + 1) * KBS > nz ? nz : (b + 1) * KBS;
        if (kb >= ke) continue;
        OMP_TILE(DEPEND(inout, state[(kb + hs) * SR]),
                 MAP(present: state[0:nstate]) num_teams((ke - kb)) thread_limit(256),
                 firstprivate(state, kb, ke))
        for (int ll = 0; ll < NUM_VARS; ll++)
            for (int k = kb; k < ke; k++) {
                double *row = state + ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs);
                row[0]        = row[nx + hs - 2];
                row[1]        = row[nx + hs - 1];
                row[nx + hs]  = row[hs];
                row[nx + hs + 1] = row[hs + 1];
            }
    }
}

void set_halo_values_z(double *state)
{
    const int SR = st_row();
    const size_t nstate = (size_t)(nz + 2 * hs) * (nx + 2 * hs) * NUM_VARS;
    const int last_kb = ((nz - 1) / KBS) * KBS;   /* first row of the last interior block */
    // z halos touch rows 0,1 and nz+hs, nz+hs+1 -> one task covering the whole z
    // extent, keyed on state[0] (low rows) and state[(nz+hs)*SR] (top rows). These
    // keys are named by the z-flux tasks that read the boundary rows (see
    // compute_tendencies_z), making the halo->flux dependence explicit. It also
    // keys on block 0's interior row so the preceding stage's writer orders it.
    OMP_TILE(DEPEND(inout, state[0], state[(nz + hs) * SR], state[(0 + hs) * SR],
                           state[(last_kb + hs) * SR]),
             MAP(present: state[0:nstate], hy_dens_cell[0:(size_t)(nz+2*hs)]) num_teams(1) thread_limit(256),
             firstprivate(state))
    for (int ll = 0; ll < NUM_VARS; ll++)
        for (int i = 0; i < nx + 2 * hs; i++) {
            const int base = ll * (nz + 2 * hs) * (nx + 2 * hs);
            if (ll == ID_WMOM) {
                state[base + (0) * (nx + 2 * hs) + i] = 0.;
                state[base + (1) * (nx + 2 * hs) + i] = 0.;
                state[base + (nz + hs) * (nx + 2 * hs) + i] = 0.;
                state[base + (nz + hs + 1) * (nx + 2 * hs) + i] = 0.;
            } else if (ll == ID_UMOM) {
                state[base + (0) * (nx + 2 * hs) + i] = state[base + (hs) * (nx + 2 * hs) + i] / hy_dens_cell[hs] * hy_dens_cell[0];
                state[base + (1) * (nx + 2 * hs) + i] = state[base + (hs) * (nx + 2 * hs) + i] / hy_dens_cell[hs] * hy_dens_cell[1];
                state[base + (nz + hs) * (nx + 2 * hs) + i] = state[base + (nz + hs - 1) * (nx + 2 * hs) + i] / hy_dens_cell[nz + hs - 1] * hy_dens_cell[nz + hs];
                state[base + (nz + hs + 1) * (nx + 2 * hs) + i] = state[base + (nz + hs - 1) * (nx + 2 * hs) + i] / hy_dens_cell[nz + hs - 1] * hy_dens_cell[nz + hs + 1];
            } else {
                state[base + (0) * (nx + 2 * hs) + i] = state[base + (hs) * (nx + 2 * hs) + i];
                state[base + (1) * (nx + 2 * hs) + i] = state[base + (hs) * (nx + 2 * hs) + i];
                state[base + (nz + hs) * (nx + 2 * hs) + i] = state[base + (nz + hs - 1) * (nx + 2 * hs) + i];
                state[base + (nz + hs + 1) * (nx + 2 * hs) + i] = state[base + (nz + hs - 1) * (nx + 2 * hs) + i];
            }
        }
}


//////////////////////////////////////////////////////////////////////////////////////////
// one timestep: 3-stage RK x-direction then z-direction (or reversed by parity).
//////////////////////////////////////////////////////////////////////////////////////////
void perform_timestep(double dt)
{
    if (direction_switch) {
        semi_discrete_step(state, state,     state_tmp, dt / 3, DIR_X);
        semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X);
        semi_discrete_step(state, state_tmp, state,     dt / 1, DIR_X);
        semi_discrete_step(state, state,     state_tmp, dt / 3, DIR_Z);
        semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z);
        semi_discrete_step(state, state_tmp, state,     dt / 1, DIR_Z);
    } else {
        semi_discrete_step(state, state,     state_tmp, dt / 3, DIR_Z);
        semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z);
        semi_discrete_step(state, state_tmp, state,     dt / 1, DIR_Z);
        semi_discrete_step(state, state,     state_tmp, dt / 3, DIR_X);
        semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X);
        semi_discrete_step(state, state_tmp, state,     dt / 1, DIR_X);
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
// main
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    long iter_cap = -1;      // -i : cap #timesteps (for the harness); <0 = run to sim_time
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-nb") && a + 1 < argc) NB = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-i") && a + 1 < argc) iter_cap = atol(argv[++a]);
        else if (!strcmp(argv[a], "-h")) { printf("usage: %s [-i maxsteps] [-nb tasks-per-loop]\n", argv[0]); return 0; }
    }

    init();
    if (NB < 1) NB = 1;
    KBS = nz / NB + (nz % NB != 0);
    if (KBS < 1) KBS = 1;

#if USE_SYNC
    const char *backend = "synchronous (blocking)";
#elif USE_TARGET
    const char *backend = "GPU target tasks";
#else
    const char *backend = "CPU tasks";
#endif
    if (mainproc) {
        printf("nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
        printf("dt: %lf\n", dt);
        printf("backend=%s taskgraph=%d\n", backend, USE_TASKGRAPH);
        printf("Num tasks per loop: %d\n", NB);
    }

    reductions(mass0, te0);

    const size_t nstate = (size_t)(nz + 2 * hs) * (nx + 2 * hs) * NUM_VARS;
    const size_t nfl = (size_t)(nz + 1) * (nx + 1) * NUM_VARS;
    const size_t ntd = (size_t)nz * nx * NUM_VARS;
    const size_t nhc = (size_t)(nz + 2 * hs);
    const size_t nhi = (size_t)(nz + 1);

    OMP_TARGET_ENTER_DATA(MAP(to: state[0:nstate], state_tmp[0:nstate],
                                  hy_dens_cell[0:nhc], hy_dens_theta_cell[0:nhc],
                                  hy_dens_int[0:nhi], hy_dens_theta_int[0:nhi], hy_pressure_int[0:nhi])
                          MAP(alloc: flux[0:nfl], tend[0:ntd]))

    // estimate #timesteps for the timing array (fixed dt until the final partial step)
    long est_steps = (long)(sim_time / dt) + 2;
    if (iter_cap >= 0 && iter_cap < est_steps) est_steps = iter_cap + 1;
    double *iter_times = (double *)malloc(sizeof(double) * (est_steps > 0 ? est_steps : 1));
    long nsteps = 0;
    double prev_ts, t0, tf;

    auto wall1 = std::chrono::steady_clock::now();
#pragma omp parallel
{
#pragma omp single
{
    t0 = omp_get_wtime();
    prev_ts = t0;
    long step = 0;
    while (etime < sim_time) {
        if (etime + dt > sim_time) dt = sim_time - etime;
        const int parity = direction_switch;   // kept OUT of the recorded region

        // Two graphs: one per splitting parity. The body of a given parity is
        // loop-invariant, so it records once and replays thereafter.
        if (parity) {
            TASKGRAPH_BEGIN
            { perform_timestep(dt); }
            TASKGRAPH_END
        } else {
            TASKGRAPH_BEGIN
            { perform_timestep(dt); }
            TASKGRAPH_END
        }
        direction_switch = parity ? 0 : 1;

        OMP_HOST_TASK(firstprivate(step, iter_times) shared(prev_ts)
                      DEPEND(in, state[(0 + hs) * (nx + 2 * hs)]))
        {
            const double now = omp_get_wtime();
            iter_times[step] = (now - prev_ts) * 1000.0;
            prev_ts = now;
        }

        etime += dt;
        step++;
        if (iter_cap >= 0 && step >= iter_cap) break;
    }
    nsteps = step;
#pragma omp taskwait
    tf = omp_get_wtime();
} /* single */
} /* parallel */
    auto wall2 = std::chrono::steady_clock::now();
    const double cpu_time = std::chrono::duration<double>(wall2 - wall1).count();

    OMP_TARGET_UPDATE(from(state[0:nstate]))
    OMP_TARGET_EXIT_DATA(MAP(release: state[0:nstate], state_tmp[0:nstate],
                                      hy_dens_cell[0:nhc], hy_dens_theta_cell[0:nhc],
                                      hy_dens_int[0:nhi], hy_dens_theta_int[0:nhi],
                                      hy_pressure_int[0:nhi], flux[0:nfl], tend[0:ntd]))

    reductions(mass, te);
    if (mainproc) {
        printf("d_mass: %le\n", (mass - mass0) / mass0);
        printf("d_te:   %le\n", (te - te0) / te0);
        printf("CPU Time: %lf sec\n", cpu_time);
        printf("Elapsed time         = %10.6f\n", tf - t0);

        if (nsteps > 0) {
            const long lo  = (nsteps > 2) ? 2 : (nsteps > 1 ? 1 : 0);
            const long cnt = nsteps - lo;
            double m = 0.0;
            for (long r = lo; r < nsteps; r++) m += iter_times[r];
            if (cnt > 0) m /= cnt;
            double var = 0.0;
            for (long r = lo; r < nsteps; r++) { double d = iter_times[r] - m; var += d * d; }
            const double sd = (cnt > 1) ? sqrt(var / (cnt - 1)) : 0.0;
            printf("iteration 0          : %10.3f ms  (record)\n", iter_times[0]);
            printf("iteration 1          : %10.3f ms  (1st replay)\n", (nsteps > 1) ? iter_times[1] : iter_times[0]);
            printf("iterations 2.. (avg)    : %10.3f ms\n", m);
            printf("iterations 2.. (stddev) : %10.3f ms\n", sd);
            const double cell_updates = (double)nx * nz;   // cells advanced per step
            const double fom = (m > 0.0) ? cell_updates / (m / 1000.0) : 0.0;
            printf("FOM                  = %10.8g (cell-updates/s)\n", fom);
        }
    }

    free(iter_times);
    finalize();
    return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////
// init / finalize / data-spec helpers (host-side, from the serial reference)
//////////////////////////////////////////////////////////////////////////////////////////
void init()
{
    i_beg = 0; k_beg = 0;
    nx = nx_glob; nz = nz_glob;

    state              = (double *) malloc((nx + 2 * hs) * (nz + 2 * hs) * NUM_VARS * sizeof(double));
    state_tmp          = (double *) malloc((nx + 2 * hs) * (nz + 2 * hs) * NUM_VARS * sizeof(double));
    flux               = (double *) malloc((nx + 1) * (nz + 1) * NUM_VARS * sizeof(double));
    tend               = (double *) malloc(nx * nz * NUM_VARS * sizeof(double));
    hy_dens_cell       = (double *) malloc((nz + 2 * hs) * sizeof(double));
    hy_dens_theta_cell = (double *) malloc((nz + 2 * hs) * sizeof(double));
    hy_dens_int        = (double *) malloc((nz + 1) * sizeof(double));
    hy_dens_theta_int  = (double *) malloc((nz + 1) * sizeof(double));
    hy_pressure_int    = (double *) malloc((nz + 1) * sizeof(double));

    dt = dmin(dx, dz) / max_speed * cfl;
    etime = 0.;

    double x, z, r, u, w, t, hr, ht;
    for (int k = 0; k < nz + 2 * hs; k++) {
        for (int i = 0; i < nx + 2 * hs; i++) {
            for (int ll = 0; ll < NUM_VARS; ll++)
                state[ll * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i] = 0.;
            for (int kk = 0; kk < nqpoints; kk++)
                for (int ii = 0; ii < nqpoints; ii++) {
                    x = (i_beg + i - hs + 0.5) * dx + (qpoints[ii] - 0.5) * dx;
                    z = (k_beg + k - hs + 0.5) * dz + (qpoints[kk] - 0.5) * dz;
                    if (data_spec_int == DATA_SPEC_COLLISION)       collision(x, z, r, u, w, t, hr, ht);
                    if (data_spec_int == DATA_SPEC_THERMAL)         thermal(x, z, r, u, w, t, hr, ht);
                    if (data_spec_int == DATA_SPEC_GRAVITY_WAVES)   gravity_waves(x, z, r, u, w, t, hr, ht);
                    if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) density_current(x, z, r, u, w, t, hr, ht);
                    if (data_spec_int == DATA_SPEC_INJECTION)       injection(x, z, r, u, w, t, hr, ht);
                    int inds;
                    inds = ID_DENS * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
                    state[inds] += r * qweights[ii] * qweights[kk];
                    inds = ID_UMOM * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
                    state[inds] += (r + hr) * u * qweights[ii] * qweights[kk];
                    inds = ID_WMOM * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
                    state[inds] += (r + hr) * w * qweights[ii] * qweights[kk];
                    inds = ID_RHOT * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
                    state[inds] += ((r + hr) * (t + ht) - hr * ht) * qweights[ii] * qweights[kk];
                }
            for (int ll = 0; ll < NUM_VARS; ll++) {
                int inds = ll * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
                state_tmp[inds] = state[inds];
            }
        }
    }
    for (int k = 0; k < nz + 2 * hs; k++) {
        hy_dens_cell[k] = 0.; hy_dens_theta_cell[k] = 0.;
        for (int kk = 0; kk < nqpoints; kk++) {
            z = (k_beg + k - hs + 0.5) * dz;
            if (data_spec_int == DATA_SPEC_COLLISION)       collision(0., z, r, u, w, t, hr, ht);
            if (data_spec_int == DATA_SPEC_THERMAL)         thermal(0., z, r, u, w, t, hr, ht);
            if (data_spec_int == DATA_SPEC_GRAVITY_WAVES)   gravity_waves(0., z, r, u, w, t, hr, ht);
            if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) density_current(0., z, r, u, w, t, hr, ht);
            if (data_spec_int == DATA_SPEC_INJECTION)       injection(0., z, r, u, w, t, hr, ht);
            hy_dens_cell[k]       += hr * qweights[kk];
            hy_dens_theta_cell[k] += hr * ht * qweights[kk];
        }
    }
    for (int k = 0; k < nz + 1; k++) {
        z = (k_beg + k) * dz;
        if (data_spec_int == DATA_SPEC_COLLISION)       collision(0., z, r, u, w, t, hr, ht);
        if (data_spec_int == DATA_SPEC_THERMAL)         thermal(0., z, r, u, w, t, hr, ht);
        if (data_spec_int == DATA_SPEC_GRAVITY_WAVES)   gravity_waves(0., z, r, u, w, t, hr, ht);
        if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) density_current(0., z, r, u, w, t, hr, ht);
        if (data_spec_int == DATA_SPEC_INJECTION)       injection(0., z, r, u, w, t, hr, ht);
        hy_dens_int[k]       = hr;
        hy_dens_theta_int[k] = hr * ht;
        hy_pressure_int[k]   = C0 * pow(hr * ht, gamm);
    }
}

void finalize()
{
    free(state); free(state_tmp); free(flux); free(tend);
    free(hy_dens_cell); free(hy_dens_theta_cell);
    free(hy_dens_int); free(hy_dens_theta_int); free(hy_pressure_int);
}

void injection(double, double, double &r, double &u, double &w, double &t, double &hr, double &ht)
{ hydro_const_theta(0, hr, ht); r = 0; t = 0; u = 0; w = 0; }

void density_current(double x, double z, double &r, double &u, double &w, double &t, double &hr, double &ht)
{ hydro_const_theta(z, hr, ht); r = 0; t = 0; u = 0; w = 0; t += sample_ellipse_cosine(x, z, -20., xlen / 2, 5000., 4000., 2000.); }

void gravity_waves(double, double z, double &r, double &u, double &w, double &t, double &hr, double &ht)
{ hydro_const_bvfreq(z, 0.02, hr, ht); r = 0; t = 0; u = 15.; w = 0; }

void thermal(double x, double z, double &r, double &u, double &w, double &t, double &hr, double &ht)
{ hydro_const_theta(z, hr, ht); r = 0; t = 0; u = 0; w = 0; t += sample_ellipse_cosine(x, z, 3., xlen / 2, 2000., 2000., 2000.); }

void collision(double x, double z, double &r, double &u, double &w, double &t, double &hr, double &ht)
{ hydro_const_theta(z, hr, ht); r = 0; t = 0; u = 0; w = 0;
  t += sample_ellipse_cosine(x, z, 20., xlen / 2, 2000., 2000., 2000.);
  t += sample_ellipse_cosine(x, z, -20., xlen / 2, 8000., 2000., 2000.); }

void hydro_const_theta(double z, double &r, double &t)
{
    const double theta0 = 300., exner0 = 1.;
    double p, exner, rt;
    t = theta0;
    exner = exner0 - grav * z / (cp * theta0);
    p = p0 * pow(exner, (cp / rd));
    rt = pow((p / C0), (1. / gamm));
    r = rt / t;
}

void hydro_const_bvfreq(double z, double bv_freq0, double &r, double &t)
{
    const double theta0 = 300., exner0 = 1.;
    double p, exner, rt;
    t = theta0 * exp(bv_freq0 * bv_freq0 / grav * z);
    exner = exner0 - grav * grav / (cp * bv_freq0 * bv_freq0) * (t - theta0) / (t * theta0);
    p = p0 * pow(exner, (cp / rd));
    rt = pow((p / C0), (1. / gamm));
    r = rt / t;
}

double sample_ellipse_cosine(double x, double z, double amp, double x0, double z0, double xrad, double zrad)
{
    double dist = sqrt(((x - x0) / xrad) * ((x - x0) / xrad) + ((z - z0) / zrad) * ((z - z0) / zrad)) * pi / 2.;
    return (dist <= pi / 2.) ? amp * pow(cos(dist), 2.) : 0.;
}

void reductions(double &mass, double &te)
{
    mass = 0; te = 0;
    for (int k = 0; k < nz; k++)
        for (int i = 0; i < nx; i++) {
            int ind_r = ID_DENS * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + hs;
            int ind_u = ID_UMOM * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + hs;
            int ind_w = ID_WMOM * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + hs;
            int ind_t = ID_RHOT * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + hs;
            double r  = state[ind_r] + hy_dens_cell[hs + k];
            double u  = state[ind_u] / r;
            double w  = state[ind_w] / r;
            double th = (state[ind_t] + hy_dens_theta_cell[hs + k]) / r;
            double p  = C0 * pow(r * th, gamm);
            double temp = th / pow(p0 / p, rd / cp);
            double ke = r * (u * u + w * w);
            double ie = r * cv * temp;
            mass += r * dx * dz;
            te   += (ke + ie) * dx * dz;
        }
}
