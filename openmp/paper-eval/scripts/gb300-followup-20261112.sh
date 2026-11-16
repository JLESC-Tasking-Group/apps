#!/bin/bash
# Follow-up of the 2026-11-12 GB300 sweep, run on graceblackwell01:
#  1. redo CloverLeaf 64 +prog-fuse in the main results: it completes, then aborts in
#     teardown (xkrt waits on a CUDA event after exit() shut CUDA down); the apps now
#     flush their report, so evaluate.py keeps it as ok_crashed_at_exit.
#  2. run-to-run variance: REPEAT=5 of synchronous + the taskgraph pipeline on the
#     problems whose speedup dropped >5% from one pass to the next, into gb300/variance.
# Same protocol as sweep.sh (W from gb300/warmup.env, 8*(W+50) iterations, grains).
set -u
PE=$(cd "$(dirname "$0")/.." && pwd); M=gb300; R=$PE/$M; W=$PE/work/$M/openmp
source $R/grains.env; declare -A WU; source $R/warmup.env
A=reduce-node,transitive-reduction; FULL=$A,critical-path,jit,prog-fuse,batch
OPTS="$A;$A,critical-path;$A,critical-path,jit;$A,critical-path,jit,prog-fuse;$FULL"
log() { echo "$(date -u '+%F %T') $*"; }
E() {  # outdir logname app variant size grain repeat [args]
    local out=$1 ln=$2 app=$3 var=$4 s=$5 g=$6 rep=$7; shift 7
    local w=${WU[$app:$var:$s]:-5} it; it=$((8 * (w + 50)))
    [ $app = mnmg ] && it=""
    log "  $app${var:+ $var} size=$s grain=$g W=$w repeat=$rep"
    python3.12 $W/scripts/evaluate.py --target gpu --apps $app ${var:+--variants $var} --unroll 8 \
        --sizes "$app=$s" ${it:+--iters "$app=$it"} ${g:+--grain "$app=$g"} --repeat $rep --timeout 300 \
        --env XKRT_OFFLOADER_CAPACITY=16384 --env TG_WARMUP=$w --outdir $out "$@" > $out/log.$ln.txt 2>&1
    echo "     rc=$? $(grep -E '^ok=' $out/log.$ln.txt | tail -1) $(grep -E '^fail by app' $out/log.$ln.txt | tail -1)"
}
log "followup $M"
bash $PE/scripts/snapshot.sh $M
source $PE/env/xkomp.sh $M || exit 1
log "1. cloverleaf 64 +prog-fuse (main results)"
python3.12 $PE/scripts/purge-rows.py $R/xkomp --apps cloverleaf:64 --configs "taskgraph:$A,critical-path,jit,prog-fuse"
E $R/xkomp cloverleaf.64.redo cloverleaf "" 64 ${CLG%%,*} 1 --opts "$A,critical-path,jit,prog-fuse" \
    --omit synchronous,no-taskgraph,taskgraph:none
log "2. variance, REPEAT=5 -> $R/variance"
V=$R/variance; mkdir -p $V
J2=(${J2G//,/ }); FD=(${FDG//,/ }); CL=(${CLG//,/ })
E $V cloverleaf.64   cloverleaf ""        64    ${CL[0]} 5 --opts "$OPTS" --omit no-taskgraph
E $V lulesh.16       lulesh     ""        16    1        5 --opts "$OPTS" --omit no-taskgraph
E $V mnmg.23874      mnmg       ""        23874 ""       5 --opts "$OPTS" --omit no-taskgraph
E $V fdtd-2d.256     polybench  fdtd-2d   256   ${FD[1]} 5 --opts "$OPTS" --omit no-taskgraph
E $V fdtd-2d.1024    polybench  fdtd-2d   1024  ${FD[2]} 5 --opts "$OPTS" --omit no-taskgraph
E $V jacobi-2d.64    polybench  jacobi-2d 64    ${J2[0]} 5 --opts "$OPTS" --omit no-taskgraph
E $V jacobi-2d.256   polybench  jacobi-2d 256   ${J2[1]} 5 --opts "$OPTS" --omit no-taskgraph
E $V jacobi-2d.1024  polybench  jacobi-2d 1024  ${J2[2]} 5 --opts "$OPTS" --omit no-taskgraph
log "followup $M done"
