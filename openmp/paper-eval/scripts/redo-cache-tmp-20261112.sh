#!/bin/bash
# redo-cache-tmp-20261112.sh <machine>: redo the cache phase (populate + warm) of the
# 2026-11-12 ~20:00 sweep with the on-disk JIT cache in the node-local /tmp (sweep.sh
# now puts it there) instead of $PE/work/<machine>/jitcache-disk on NFS: the warm
# column measures reading the cache, and NFS made that 2-3x slower. The
# jit-disk-cold / jit-disk-warm rows are purged first (.bak kept), then the tables.
set -u
PE=$(cd "$(dirname "$0")/.." && pwd); M=${1:?machine}; R=$PE/$M
log() { echo "$(date -u '+%F %T') $*"; }
log "redo cache $M"
grep -q 'C=/tmp/' $PE/scripts/sweep.sh || { log "sweep.sh does not put the cache in /tmp"; exit 1; }
for t in jit-disk-cold jit-disk-warm; do
    python3.12 $PE/scripts/purge-rows.py $R/xkomp --apps mnmg,krylov,lulesh,cloverleaf,sw4lite,polybench,miniweather --tag "$t"
done
PHASES="cache" bash $PE/scripts/sweep.sh $M
PHASES="table" bash $PE/scripts/sweep.sh $M
log "redo cache $M done"
