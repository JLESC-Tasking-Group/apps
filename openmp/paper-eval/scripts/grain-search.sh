#!/bin/bash
# grain-search.sh <machine>: pick the tile count (-nb) per problem for the apps whose
# grain is not fixed by hand (cloverleaf, sw4lite, polybench, miniWeather). One
# taskgraph build per problem, then one short run per nb in $NBS with the full
# pipeline; the fastest steady-state mean wins. A timeout or failure counts as a bad
# grain (never retried with a longer limit). Krylov, LULESH and MNMG use the grains
# of the paper (sweep.sh).
# Output: <machine>/grains.csv (every run) and <machine>/grains.env (the picks).
set -u
PE=$(cd "$(dirname "$0")/.." && pwd); M=${1:?machine}
source $PE/env/xkomp.sh $M || { echo "env failed"; exit 1; }
W=$PE/work/$M/openmp; R=$PE/$M; mkdir -p $R
OUT=$R/grains.csv
NBS=${NBS:-"1 2 4 8 16 32 64"}; IT=${IT:-96}
OPT=reduce-node,transitive-reduction,jit,prog-fuse,batch
export OMP_PLACES=cores OMP_PROC_BIND=close XKRT_DRIVERS="host,2;cuda,1" OMP_NUM_THREADS=8
export XKRT_OFFLOADER_CAPACITY=16384 CGIR_JIT_CACHE_DIR=$PE/work/$M/jitcache-grains OMP_TASKGRAPH_OPT=$OPT
echo "app,variant,size,nb,avg_ms,stddev_ms,answer,status" > $OUT

g() { echo "$1" | grep -E "$2" | head -1 | sed -E 's/.*: *([-+0-9.eE]+).*/\1/'; }
run_nb() {  # app variant size dir exe args...
    local app=$1 var=$2 size=$3 dir=$4 exe=$5; shift 5
    for nb in $NBS; do
        out=$(cd $dir && timeout ${TMO:-120} taskset -c 0-7 $exe "$@" -nb $nb -u 8 2>&1); rc=$?
        avg=$(g "$out" 'instances [0-9]+\.\.[0-9]+ \(avg\)'); sd=$(g "$out" 'instances [0-9]+\.\.[0-9]+ \(stddev\)')
        ans=$(g "$out" '^answer'); st=ok; [ -z "$avg" ] && st=noavg; [ $rc -ne 0 ] && st=rc$rc
        echo "$app,$var,$size,$nb,$avg,$sd,$ans,$st" >> $OUT
    done
}
bld() {  # dir target defs...
    local d=$1 t=$2; shift 2
    (cd $W/$d && make clean > /dev/null 2>&1; make $t USE_SYNC=0 USE_TASKGRAPH=1 USE_TARGET=1 USE_OMPSS=0 "$@" > build.log 2>&1) \
        || { echo "!! build $d $t $*"; tail -5 $W/$d/build.log; return 1; }
}
echo "grain search $M $(date -u +%T)"
bld cloverleaf cloverleaf && for s in 64 256 1024; do run_nb cloverleaf "" $s $W/cloverleaf ./cloverleaf -s $s -i $IT; done
for s in 16 32 64; do bld sw4lite sw4lite "SW4_DEFS=-DSW4_N=$s -DSW4_NZ=$s" && run_nb sw4lite "" $s $W/sw4lite ./sw4lite -i $IT; done
for v in jacobi-2d heat-3d fdtd-2d jacobi-1d; do
    case $v in heat-3d) sizes="32 64 128";; jacobi-1d) sizes="1024 65536 1048576";; *) sizes="64 256 1024";; esac
    for s in $sizes; do
        bld polybench $v "PB_DEFS=-DPB_N=$s -DPB_NX=$s -DPB_NY=$s -DPB_TSTEPS=100 -DPB_TMAX=100" && run_nb polybench $v $s $W/polybench ./$v.x -i $IT
    done
done
for s in 50 200 800; do
    bld miniWeather miniweather "MW_DEFS=-D_NX=$s -D_NZ=$((s/2)) -D_SIM_TIME=400 -D_OUT_FREQ=-1 -D_DATA_SPEC=DATA_SPEC_THERMAL" || continue
    it=$IT; [ $s = 800 ] && it=24
    run_nb miniweather "" $s $W/miniWeather ./miniweather -i $it
done
python3 $PE/scripts/grain-pick.py $OUT > $R/grains.env
echo "grain search done $(date -u +%T)"; cat $R/grains.env
