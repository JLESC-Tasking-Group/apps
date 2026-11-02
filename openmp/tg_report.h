/*
 * tg_report.h - the timing summary shared by the apps/openmp ports that do not
 * carry their own (polybench, miniWeather, sw4lite, CloverLeaf).
 *
 * Same lines, in the same words, as LULESH / krylov / mnmg, so that
 * scripts/appspecs.py parses every app with one regex set (_parse_instances):
 *
 *   taskgraph unroll     :          4 iterations/instance (25 instances)
 *   instance 0 (record)  :      1.234 ms
 *   instance 1 (1st replay):    0.567 ms
 *   instances 2..24 (avg)    :  0.321 ms   (23 instances)
 *   instances 2..24 (stddev) :  0.012 ms
 *   Elapsed time (s)     :   0.123456
 *   FOM (<unit>)         : 1.2345678e+09
 *   answer (<what>)      : 1.234567890123e+03
 *
 * Entries are per taskgraph INSTANCE -- a group of `unroll` iterations -- and in
 * ms per ITERATION (the TASKGRAPH_LOOP epilogue divides by `unroll`), so they
 * stay comparable across -u. Instance 0 is the record, instance 1 the command-
 * graph build + first replay, instances 2.. the steady state.
 *
 * `answer` is a number computed from the final state that every configuration of
 * the same problem must reproduce (plot.py's report_answers() checks it).
 */
#ifndef OPENMP_TG_REPORT_H
#define OPENMP_TG_REPORT_H

#include <math.h>
#include <stdio.h>

#include "tasking.h"

/* Whether the build records a taskgraph, for the (record) / (1st replay) labels. */
#define TG_GRAPHED (USE_TASKGRAPH && !USE_SYNC)

/* The backend banner every app prints (parity with lulesh / krylov). */
static inline const char *tg_backend_name(void)
{
#if USE_SYNC
    return "synchronous (blocking)";
#elif USE_TARGET
    return "GPU target tasks";
#elif USE_OMPSS
    return "OmpSs-2 tasks";
#else
    return "CPU tasks";
#endif
}

/* The number of iterations actually run for a requested `iters` at `unroll`
 * iterations per instance: a whole number of instances, since a shorter last
 * instance would be a different graph and would not replay (same rule as
 * LULESH's -i / -u). Prints a note when it rounds; returns 0 if nothing runs. */
static inline long tg_round_iters(long iters, long unroll)
{
    if (unroll < 1) unroll = 1;
    const long r = (iters / unroll) * unroll;
    if (r != iters) {
        if (r == 0) printf("-u %ld exceeds -i %ld: nothing to run\n", unroll, iters);
        else        printf("-i %ld is not a multiple of -u %ld: running %ld iterations\n",
                           iters, unroll, r);
    }
    return r;
}

/* Timing summary. Returns the steady-state mean (ms per iteration), which the
 * caller turns into its FOM. */
static inline double tg_report_instances(const double *inst_ms, long ninst, long unroll)
{
    if (ninst <= 0) return 0.0;
    const int graphed = TG_GRAPHED;
    const long lo  = (ninst > 2) ? 2 : (ninst > 1 ? 1 : 0);   /* first steady instance */
    const long cnt = ninst - lo;
    double mean = 0.0;
    for (long r = lo; r < ninst; r++) mean += inst_ms[r];
    if (cnt > 0) mean /= (double) cnt;
    double var = 0.0;
    for (long r = lo; r < ninst; r++) { const double d = inst_ms[r] - mean; var += d * d; }
    const double sd = (cnt > 1) ? sqrt(var / (double) (cnt - 1)) : 0.0;

    if (unroll > 1)
        printf("taskgraph unroll     : %10ld iterations/instance (%ld instances)\n", unroll, ninst);
    printf("instance 0%-11s: %10.3f ms\n", graphed ? " (record)" : "", inst_ms[0]);
    if (ninst > 1)
        printf("instance 1%-11s: %10.3f ms\n", graphed ? " (1st replay)" : "", inst_ms[1]);
    /* the window actually averaged: it starts below 2 only when there are too
     * few instances for the record / first replay to be dropped */
    printf("instances %ld..%ld (avg)    : %10.3f ms   (%ld instances)\n", lo, ninst - 1, mean, cnt);
    printf("instances %ld..%ld (stddev) : %10.3f ms\n", lo, ninst - 1, sd);
    return mean;
}

#endif /* OPENMP_TG_REPORT_H */
