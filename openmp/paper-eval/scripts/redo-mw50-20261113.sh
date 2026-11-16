#!/bin/bash
# redo-mw50-20261113.sh <machine>: miniWeather 50 only, every phase but snapshot and
# grains, after the SIM_TIME fix of scripts/appspecs.py (_mw_sim_time): 400 s held only
# 150 pairs at NX = 50, so synchronous (-u 1) ran 150 pairs and the taskgraph configs
# (-u 8) 144, i.e. different answers (rejected), and fewer instances than W + STEADY.
# Its rows are purged first (.bak kept). Generated from sweep.sh with apps() reduced
# to that problem, so sweep.sh is not edited while the GH200 sweep runs it.
# sweep.sh <machine>: the paper's performance sweep on one machine.
#
#   PHASES (default "snapshot xkomp cache vanilla table"; add "grains" to redo the search):
#   snapshot  copy the apps into work/<machine>/openmp (scripts/snapshot.sh)
#   grains    tile-count search for the apps without a hand-set grain (grain-search.sh);
#             without it, the grains of <machine>/grains.env are reused
#   xkomp     main sweep, XKOMP: synchronous, no-taskgraph, taskgraph:none, then the
#             cumulative pipeline +reduce +jit +prog-fuse +packing (critical-path is left
#             out: its effect is below the run-to-run noise, gb300/variance.md).
#             Untagged; no on-disk JIT cache, so every run compiles from scratch
#             ("cold"), and its pass times are the cold column of the cost table.
#   cache     full pipeline only, with an on-disk JIT cache: first populated
#             (CGIR_JIT_CACHE_MODE=w, tag jit-disk-cold), then read back
#             (CGIR_JIT_CACHE_MODE=r, tag jit-disk-warm: the "cached" column).
#   vanilla   vanilla clang/KMP: synchronous and no-taskgraph (KMP has no taskgraph)
#   table     table.py + plot.py: <machine>/table.md, <machine>/latex/, figures
#
# ONLY="<log names>" restricts every phase to some apps, XOMIT="<configs>" omits configs
# from the xkomp phase (see scripts/purge-rows.py to drop the rows being redone first).
# Unroll 8, one run per app/config/size (REPEAT=1): the statistics (mean, stddev) come
# from the steady state of that run, STEADY instances (default 50) after W warm-up
# instances, i.e. 8*(W+STEADY) iterations. W is per problem, from <machine>/warmup.env
# (scripts/warmup.py: ~1 s of GPU load after the first replay's JIT compile, so the GPU
# is back at its full clock), passed as TG_WARMUP; problems missing there use W = 5.
# Every app also runs a 1 s GPU warm-up before its first instance (TG_GPU_WARMUP_MS,
# tasking.h). MNMG's round count comes from its data (W = 5); miniWeather 800 runs
# MW800_ITERS = 104.
# 8 threads pinned to CPUs 0-7, XKRT_DRIVERS="host,2;cuda,1".
# Results: <machine>/{xkomp,vanilla}/{runs,cgstats,jitstats}.csv and per-app logs.
set -u
PE=$(cd "$(dirname "$0")/.." && pwd); M=${1:?machine}
PHASES=${PHASES:-"xkomp cache vanilla table"}
R=$PE/$M; W=$PE/work/$M/openmp; mkdir -p $R
REPEAT=${REPEAT:-1}
STEADY=${STEADY:-50}; MW800_ITERS=${MW800_ITERS:-104}
declare -A WU=()
A=reduce-node,transitive-reduction
FULL=$A,jit,prog-fuse,batch
OPTS="$A;$A,jit;$A,jit,prog-fuse;$FULL"
has() { [[ " $PHASES " == *" $1 "* ]]; }
log() { echo "$(date -u '+%F %T') $*"; }

ev() {  # outdir logname app sizes iters grain [evaluate.py args]
    local out=$1 ln=$2 app=$3 sz=$4 it=$5 gr=$6; shift 6
    # ONLY="<logname> ...": run just these entries of apps() (log names: mnmg krylov-<cg..>
    # lulesh cloverleaf sw4lite polybench-<variant> miniweather-a miniweather-b)
    [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $ln "* ]] && return 0
    [ -n "$it" ] && it=${ITERS_OVERRIDE:-$it}
    log "  $app${V:+ $V} sizes=$sz grain=$gr"
    python3.12 $W/scripts/evaluate.py --target gpu --apps $app --unroll 8 \
        --sizes "$app=$sz" ${it:+--iters "$app=$it"} ${gr:+--grain "$app=$gr"} \
        --repeat $REPEAT --timeout ${TMO:-300} --env XKRT_OFFLOADER_CAPACITY=16384 \
        --outdir $out "$@" > $out/log.$ln.txt 2>&1
    echo "     rc=$? $(grep -E '^ok=' $out/log.$ln.txt | tail -1) $(grep -E '^fail by app' $out/log.$ln.txt | tail -1)"
}

# evs outdir logname app variant sizes grains [evaluate.py args]: one evaluate.py call per
# size (and variant), each with its warm-up W = WU[app:variant:size] (TG_WARMUP) and
# 8*(W+STEADY) iterations; ITERS=<n> fixes the iterations instead. Logs log.<logname>.<size>.txt.
evs() {
    local out=$1 ln=$2 app=$3 var=$4 sz=$5 gr=$6; shift 6
    [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $ln "* ]] && return 0
    local -a S=(${sz//,/ }) G=(${gr//,/ }); local i s g w
    for i in "${!S[@]}"; do
        s=${S[$i]}; g=${G[$i]:-${G[0]:-}}; w=${WU[$app:$var:$s]:-5}
        V=$var ONLY= ev $out $ln.$s $app $s ${ITERS:-$((8 * (w + STEADY)))} "$g" \
            ${var:+--variants $var} --env TG_WARMUP=$w "$@"
    done
}

# every app and problem of the sweep, with its grain; "$@" goes to evaluate.py
apps() {
    local out=$1; shift
    local mw=(${MWG//,/ })
    evs $out miniweather-a   miniweather "" 50 ${mw[0]} "$@"
}

log "redo miniweather 50 $M: $PHASES"
grep -q _mw_sim_time $W/scripts/appspecs.py || { log "$W/scripts/appspecs.py has no SIM_TIME fix"; exit 1; }
for t in "" jit-disk-cold jit-disk-warm; do
    python3.12 $PE/scripts/purge-rows.py $R/xkomp --apps miniweather:50 --tag "$t"
done
python3.12 $PE/scripts/purge-rows.py $R/vanilla --apps miniweather:50
has snapshot && bash $PE/scripts/snapshot.sh $M
has grains && bash $PE/scripts/grain-search.sh $M > $R/grain-search.log 2>&1
source $R/grains.env || { echo "no $R/grains.env"; exit 1; }
cat $R/grains.env
if [ -f $R/warmup.env ]; then source $R/warmup.env; log "warm-up of ${#WU[@]} problems from $R/warmup.env"
else log "no $R/warmup.env: warm-up W = 5 everywhere (scripts/warmup.py writes it)"; fi

if has xkomp || has cache; then
    # The installed stack must be built from the sources it reports: the sweeps of
    # 2026-11-12 06:33 ran a cgir older than its source (prog-fuse 7-9x slower, a
    # JIT cache that missed). Refuse a source file newer than its installed library.
    ( source $PE/env/$M.conf
      for p in cgir:libcgir xkrt:libxkrt xkomp:libxkomp; do
          lib=$(ls -t $INSTALL/${p%%:*}/$TAG/$XT/lib*/${p#*:}.so* 2>/dev/null | head -1)
          new=$(find $REPO/${p%%:*}/src $REPO/${p%%:*}/include -type f -newer "$lib" 2>/dev/null | head -3)
          [ -n "$lib" ] && [ -z "$new" ] || { echo "stale ${p%%:*} install ($lib) -- newer: $new"; exit 1; }
      done ) || { log "rebuild the stack first"; exit 1; }
    ( source $PE/env/xkomp.sh $M || exit 1
      { echo "machine $M $(hostname)"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
        lscpu | grep 'Model name'; echo "xkcxx $(which xkcxx)"; clang --version | head -1
        for p in llvm-project cgir xkrt xkomp; do echo "$p $(git -C $REPO/$p rev-parse --short=12 HEAD) $(git -C $REPO/$p status --short -uno | wc -l) modified"; done
      } > $R/stack.txt
      mkdir -p $R/xkomp
      if has xkomp; then log "xkomp main sweep"; apps $R/xkomp --opts "$OPTS" ${XOMIT:+--omit $XOMIT}; fi
      if has cache; then
          # node-local disk: on NFS (/home, /vast) reading the cache made warm 2-3x slower
          C=/tmp/$USER-jitcache-$M; rm -rf $C
          log "jit cache: populate (cold)"
          # only its pass costs are used: a short run (one steady instance)
          # CGIR_NOALIAS=prove, as in the main sweep: a cache directory alone switches
          # CGIR to `trusted`, i.e. other code than the cold column measured
          ITERS_OVERRIDE=48 apps $R/xkomp --opts "$FULL" --omit synchronous,no-taskgraph,taskgraph:none \
               --env CGIR_NOALIAS=prove --env CGIR_JIT_CACHE_DIR=$C --env CGIR_JIT_CACHE_MODE=w --tag jit-disk-cold
          log "jit cache: read back (warm)"
          apps $R/xkomp --opts "$FULL" --omit synchronous,no-taskgraph,taskgraph:none \
               --env CGIR_NOALIAS=prove --env CGIR_JIT_CACHE_DIR=$C --env CGIR_JIT_CACHE_MODE=r --tag jit-disk-warm
      fi )
fi

if has vanilla; then
    ( source $PE/env/vanilla.sh $M || exit 1
      export VANILLA=1
      echo "vanilla $(which clang) $(clang --version | head -1)" >> $R/stack.txt
      mkdir -p $R/vanilla; log "vanilla sweep"
      apps $R/vanilla --opts skip --omit taskgraph:none )
fi

if has table; then
    python3.12 $PE/scripts/table.py $R > $R/table.md && log "table -> $R/table.md"
    # plot.py needs matplotlib: the user site-packages are built for aarch64, so run this
    # phase on a Grace node for every machine (PHASES=table bash sweep.sh a100).
    ( cd $W && PYTHONPATH=${PLOT_PYTHONPATH:-/vast/users/rpereira/python} python3.12 scripts/plot.py --outdir $R/xkomp --no-figures --pipeline "taskgraph:$FULL" \
          --latex-tables $R/latex --apps krylov,lulesh,mnmg > $R/plot.log 2>&1; log "latex tables rc=$? -> $R/latex" )
fi
# binaries kept for failed runs only
find $R -name .binstash -type d -prune -exec du -sh {} \; 2>/dev/null
log "sweep $M done"
