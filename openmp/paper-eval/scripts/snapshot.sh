#!/bin/bash
# snapshot.sh <machine>: copy the apps (../.. = JLESC-apps/openmp, with its working
# tree changes) into work/<machine>/openmp, where the sweep builds them. evaluate.py
# builds in place, so each machine gets its own copy: the machines share the file
# system and would otherwise rebuild each other's binaries mid-sweep.
# Large inputs are symlinked, not copied.
set -eu
PE=$(cd "$(dirname "$0")/.." && pwd); SRC=$(cd $PE/.. && pwd)
D=$PE/work/${1:?machine}/openmp
mkdir -p $D
rsync -a --delete --exclude=/paper-eval --exclude=/results --exclude='/results-*' \
    --exclude=/krylov/matrices --exclude=/MNMGDatalog/MNMGDatalog-reference \
    --exclude='*.o' --exclude='*.x' --exclude='*.cobaltlog' --exclude='*.error' --exclude='*.output' \
    --exclude=__pycache__ --exclude=.binstash --exclude=build.log \
    $SRC/ $D/
ln -sfn $SRC/krylov/matrices $D/krylov/matrices
ln -sfn $(readlink -f $SRC/MNMGDatalog/MNMGDatalog-reference) $D/MNMGDatalog/MNMGDatalog-reference
(cd $SRC/.. && { git rev-parse HEAD; git status --short openmp; git diff openmp; }) > $D/../apps-snapshot.txt
echo "snapshot -> $D ($(head -1 $D/../apps-snapshot.txt | cut -c1-12) + working tree)"
