#!/bin/bash
# redo-krylov-lulesh-20261112.sh <machine>: redo the Krylov and LULESH cells of the
# 2026-11-12 06:33 sweep after two fixes:
#  - xkrt: XKRT_PROGRESS_THREAD_SPIN_TIME default 10 ms -> 0 (the spin slowed LULESH
#    +packing 1.5-2x on GH200/GB300); the stack is rebuilt before this runs.
#  - krylov: cg/cr/bicgstab freeze once converged (task_scalar_div_live, kernels.h):
#    past convergence their residual was NaN or ~1e+25..1e+92, run to run.
# xkomp + cache phases for krylov and lulesh, vanilla for krylov (vanilla LULESH did not
# change: no xkrt, same app), then the tables. Old rows are purged first (.bak kept).
set -u
PE=$(cd "$(dirname "$0")/.." && pwd); M=${1:?machine}; R=$PE/$M
K="krylov-cg krylov-cr krylov-bicgstab krylov-minres"
log() { echo "$(date -u '+%F %T') $*"; }
log "redo $M: $K lulesh"
for t in "" jit-disk-cold jit-disk-warm; do
    python3.12 $PE/scripts/purge-rows.py $R/xkomp --apps krylov,lulesh --tag "$t"
done
python3.12 $PE/scripts/purge-rows.py $R/vanilla --apps krylov
PHASES="snapshot xkomp cache" ONLY="$K lulesh" bash $PE/scripts/sweep.sh $M
PHASES="vanilla" ONLY="$K" bash $PE/scripts/sweep.sh $M
PHASES="table" bash $PE/scripts/sweep.sh $M
log "redo $M done"
