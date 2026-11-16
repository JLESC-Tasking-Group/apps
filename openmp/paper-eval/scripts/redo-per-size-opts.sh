#!/bin/bash
# redo-per-size-opts.sh <machine>: redo the taskgraph:<opt> rows of the apps rebuilt per
# size (sw4lite, polybench, miniWeather 50/200) in <machine>/xkomp, then the tables.
#
# Why: until 2026-11-11 16:10 UTC, evaluate.py stashed binaries only when --repeat > 1.
# All taskgraph:<opt> configs share one build per size, so with --repeat 1 the configs
# after taskgraph:none reused the in-tree binary -- the LAST size's, since every size's
# build cleans the directory -- and the smaller sizes ran the largest problem. Fixed in
# evaluate.py (always stash); this script replaces the rows of the sweeps run before it.
set -u
PE=$(cd "$(dirname "$0")/.." && pwd); M=${1:?machine}
A=reduce-node,transitive-reduction
CFGS="taskgraph:$A;taskgraph:$A,critical-path;taskgraph:$A,critical-path,jit;taskgraph:$A,critical-path,jit,prog-fuse;taskgraph:$A,critical-path,jit,prog-fuse,batch"
python3.12 $PE/scripts/purge-rows.py $PE/$M/xkomp --apps sw4lite,polybench,miniweather:50:200 --configs "$CFGS"
cp $PE/$M/stack.txt $PE/$M/stack.txt.keep   # the xkomp phase rewrites it (vanilla line)
ONLY="sw4lite polybench-jacobi-2d polybench-heat-3d polybench-fdtd-2d polybench-jacobi-1d miniweather-a" \
XOMIT=synchronous,no-taskgraph,taskgraph:none PHASES="xkomp table" bash $PE/scripts/sweep.sh $M
mv $PE/$M/stack.txt.keep $PE/$M/stack.txt
