# MNMGDatalog TC - OpenMP target-task port

An OpenMP port of `MNMGDatalog-reference/tc_benchmark` (CUDA). It computes the
Datalog-style semi-naive **Transitive Closure** fixpoint

```
path(a, c) :- path(a, b), edge(b, c).
```

on fixed pre-allocated device buffers, with all per-iteration **sizes resident in
device memory** (`d_frontier_size` / `d_new_count`). The kernels read those sizes
on the device, so the per-iteration kernel sequence is identical every round --
which is exactly what lets one recorded task graph be replayed.

## The two versions are compile-time toggles

Unlike the CUDA reference (separate `v1_baseline/` and `v2_cudagraph/` sources),
this is **one source** (`tc.cpp`) whose backend is chosen by the shared toggles in
`../tasking.h` / `../common.mk`. The two CUDA versions map to `USE_TASKGRAPH`:

| CUDA reference        | Build here                | Fixpoint driver                                   |
|-----------------------|---------------------------|---------------------------------------------------|
| `v1_baseline`         | `make USE_TASKGRAPH=0`    | host loop spawns the kernels every iteration      |
| `v2_cudagraph`        | `make USE_TASKGRAPH=1`    | per-iteration body recorded once, then replayed   |

`v3_conditional` (the on-GPU conditional WHILE loop) is **not** ported: CGIR/XKOMP
has no conditional-node support yet. The remaining toggles are orthogonal:

- `USE_TARGET=1` GPU offload (target tasks) vs `USE_TARGET=0` host CPU tasks.
- `USE_SYNC=1` synchronous blocking schedule (each kernel runs to completion).

The per-iteration body `reset -> expand -> promote -> set_sizes` is wrapped in
`TASKGRAPH_BEGIN/END` together with a `k_writeback` async D2H of `new_count`; the
host then tests convergence on that (pinned) host scalar after a `taskwait` -- the
same host round-trip the CUDA `v2_cudagraph` pays. Eliminating that round-trip is
what `v3_conditional` would do on the GPU.

## Design notes

- **Device memory:** the large buffers (`edges`, `edge_table`, `result_set`, the
  two frontiers) are device-only via `omp_target_alloc`, reached through
  `is_device_ptr(...)` -- the direct analog of the reference's `cudaMalloc`, with
  no host mirror of the (possibly multi-GB) result set. On the CPU backend the
  same pointers are `malloc` and the kernels become host tasks.
- **`new_count` is the exception:** it is the one scalar the host reads *every*
  iteration, so it is pinned host memory from the shared allocator
  (`../alloc.h`, `host_alloc`/`host_free`) with a device copy created by
  `map(alloc:)` and reached from the kernels with `map(present:)` -- exactly how
  the Krylov solvers handle their scalars. `k_writeback` refreshes the host side
  with an `omp target update from(...) nowait` recorded *inside* the taskgraph, so
  the timed loop contains no blocking `omp_target_memcpy`. The remaining scalars
  (`frontier_size`, `result_count`, `overflow`) stay device-only and are read once
  after the fixpoint.
- **The `taskwait` after `TASKGRAPH_END` is required.** Under `USE_TASKGRAPH=1`
  the region is already effectively blocking (xkomp does an implicit taskwait
  while recording, and replay is synchronous), so it is free; but with
  `USE_TASKGRAPH=0` the macros vanish and the `nowait` tasks would still be in
  flight, so without it the loop would read a stale `new_count` and stop early.
- **Atomics:** the open-addressing result set and edge table are built with a
  compare-and-swap (`#pragma omp atomic compare capture`, OpenMP 5.1); the append
  counters use fetch-add (`#pragma omp atomic capture`). One portable code path
  for host and device.
  **Caveat:** on-device `omp atomic compare capture` codegen must be confirmed on
  your XKOMP/clang + NVPTX toolchain (this is the one construct I could not verify
  without a GPU). A reduction cannot replace it -- an open-addressing insert is a
  *slot claim* ("insert iff empty, report novelty"), not an associative combine.

## Build & run

```shell
make USE_TARGET=1 USE_TASKGRAPH=1        # GPU, replayed graph (v2_cudagraph)
make USE_TARGET=1 USE_TASKGRAPH=0        # GPU, host loop      (v1_baseline)
make                                     # CPU tasks (default), for correctness

./tc.x <data.bin> [capacity_mult] [frontier_slots]
make test                                # data_7035.bin (TC=146120, 64 rounds)
```

The fixpoint runs **once**; the number of rounds is determined by the dataset, not
by a command-line knob.

Environment: `TC_WARMUP=<n>` untimed, ungraphed warm-up rounds before round 0
(default 3); `TC_WORKERS=<n>` overrides the GPU grid-stride worker count (see
below); `TC_VERIFY=1` reads the edge table back after the fill and after the
build and reports the occupied-slot counts (a device-side sanity check, off by
default); `TC_TRACE=1` prints one flushed stderr line per kernel dispatch (see
"Debugging"); `TC_WRITE=1` writes `<input>_<version>_tc.bin` (off by default so
sweeps stay clean); `TC_DUMP=<f>` writes a `src dst` text dump; `TC_CSV=<f>`
writes the reference's 15-column metric row.

## Metrics

The measured unit is one fixpoint **round**, reported in the same shape the Krylov
drivers use for iterations (`krylov/common/driver.cpp`):

```
MNMGDatalog TC (transitive closure)
  backend    : GPU (omp target, device-resident buffers)
  exec mode  : asynchronous (tasks)
  taskgraph  : on (record once, replay)
  version    : cudagraph
  input      : MNMGDatalog-reference/data/data_7035.bin
  size       : 7035 edges  ->  TC = 146120 tuples in 64 rounds
  geometry   : 524288 grid-stride workers
  warm-up    : 3 untimed rounds (ungraphed)
  peak memory: 6.12 MB
Statistics
  total time (end-to-end)     :     18.346 ms
    file IO                   :      0.123 ms
    H2D transfer              :      0.012 ms
    setup                     :      1.234 ms
    compute                   :      4.532 ms
    D2H transfer              :      0.500 ms
  round 0 (record)            :      0.345 ms
  round 1 (1st replay)        :      0.120 ms
  rounds 2..63 (avg)          :      0.065 ms   (62 rounds)
  rounds 2..63 (stddev)       :      0.031 ms
```

* **round 0** is where the task graph is *recorded* (XKOMP `rc == 1`).
* **round 1** is the first replay, and where the command graph is built and
  optimized (XKOMP `rc == 2`, `xkomp/src/xkomp/taskgraph.cc`).
* **rounds 2..N-1** are steady state; with fewer than 3 rounds the window degrades
  gracefully (2 -> round 1 alone, 1 -> round 0 alone).
* The `TC_WARMUP` rounds before round 0 run the same kernels with the taskgraph
  wrapper **disabled**, so OpenMP team creation and device bring-up (context,
  module load, kernel JIT, first touch) are paid up front without consuming the
  record pass.

> **Caveat:** unlike a Krylov iteration, TC rounds do very different amounts of
> work -- the frontier grows for the first rounds and then collapses. The
> `rounds 2..N-1` stddev therefore mostly reflects that frontier-size profile, not
> run-to-run jitter. The useful comparison is round 0 and round 1 against the
> steady mean, which isolates the record and graph-build overheads.

**total time (end-to-end)** is the MNMGDatalog paper's metric and per-phase
breakdown (`MNMGDatalog-paper`, Table "End-to-end total time (ms)" and Fig. "TC
per-phase total time breakdown"): `file IO + H2D + setup + compute + D2H`, where
*compute* is the whole measured fixpoint plus the one-shot result compaction.
Unlike the CUDA reference -- which captures and instantiates the graph in a
separate, separately-timed `Build` phase (`tc_v2.cu` `tc_build`) -- XKOMP records
and builds *inside* the loop, so there is no separate build phase to time: that
cost sits in compute and is visible as rounds 0 and 1. Every printed number is
measured, none is extrapolated. The same numbers go into the 15-column `TC_CSV`
row, which keeps the reference `tc_benchmark` schema (`build` = 0, `compute_min` =
`compute`, `repeats` = 1).

## Device-resident sizes and the loop bound

The two size-driven fixpoint kernels (`k_expand`, `k_promote`) are **grid-stride
loops over a fixed host-constant worker count**, not loops bounded by the
device-resident size. That is not a stylistic choice: for a combined
`target teams distribute parallel for`, clang evaluates the loop trip count on
the *host*, inside the target task, to fill the `LoopTripCount` argument of
`__tgt_target_kernel` (`SizeEmitter` in `CGStmtOpenMP.cpp` ->
`CGOpenMPRuntime::emitTargetNumIterationsCall`). Writing
`for (i = 0; i < d_frontier_size[0]; i++)` over an `is_device_ptr` buffer makes
the host load a device address and segfault; with a host-resident scalar it
instead bakes a stale size into the recorded launch.

So the OpenMP bound is `ctx.n_workers` (fixed once in `tc_setup`, hence an
identical launch on every replay) and the size is read from device memory inside
the body -- exactly the reference's fixed `<<<32*numSM, 512>>>` geometry with
`int n = *frontier_size;` read in the kernel. Default is `1<<21`
(`-DTC_WORKERS_DEFAULT=<n>` at build time, `TC_WORKERS=<n>` at run time), clamped
to `frontier_cap`. On the CPU backend it is 1 and the nest collapses to the plain
loop.

## Capacity -- read this before running anything but the two small graphs

The result set is sized `next_pow2(n_edges * capacity_mult)` and must hold
**>= ~2x the TC size**. TC is a property of the *graph*, not of the edge count, so
`capacity_mult` cannot be derived from the input size: the default of 64 is
sufficient for `data_7035` and `data_23874` **and for nothing else**.

| dataset | file | edges | rounds | TC | TC/round | `capacity_mult` | result set |
|---|---|---:|---:|---:|---:|---:|---:|
| OL.cedge | `data_7035.bin` | 7 035 | 64 | 146 120 | 2 283 | **64** | 4 MiB |
| TG.cedge | `data_23874.bin` | 23 874 | 58 | 481 121 | 8 295 | **64** | 16 MiB |
| p2p-Gnutella31 | `data_147892.bin` | 147 892 | 31 | 884 179 859 | 28 522 576 | **8192** | 16 GiB |
| usroad | `data_165435.bin` | 165 435 | 606 | 871 365 688 | 1 437 840 | **8192** | 16 GiB |
| fe_ocean | `data_409593.bin` | 409 593 | 247 | 1 669 750 513 | 6 760 526 | **8192** | 32 GiB |
| vsp_finan | `vsp_finan512_scagr7-2c_rlfddd.bin` | 552 020 | 520 | 910 070 918 | 1 750 136 | **2048** | 16 GiB |
| com-dblp | `com-dblpungraph.bin` | 1 049 866 | 31 | 1 911 754 892 | 61 670 160 | **2048** | 32 GiB |

```shell
./tc.x MNMGDatalog-reference/data/data_147892.bin 8192      # 16 GiB result set
```

Undersizing is detected and reported, not silently wrong: every insert path sets
a device-side overflow flag, `k_writeback` returns it to the host together with
`new_count` in the same async D2H, and the fixpoint aborts on the next
convergence test with the offending capacity printed. The frontier readers
(`k_promote`, `k_set_sizes`) clamp to `frontier_cap`, so an overflowing round can
no longer walk off the end of the frontier buffers -- which used to surface as
`cuStreamSynchronize ... an illegal memory access was encountered`, because
`new_count` counts *every* new fact while only the first `frontier_cap` are
stored.

Frontier buffers are decoupled: `min(result_cap, 2^28)` slots each, i.e. 2 GiB
apiece at the cap. `TC/round` above is the average; for com-dblp (61.7 M
facts/round) the peak may approach the 2^28 default, in which case raise
`frontier_slots` (arg 3).

## Correctness

Known reference sizes: `data_10` -> TC 18 / 3 rounds, `data_7035` -> 146 120 / 64,
`data_23874` -> 481 121 / 58, and the table above for the rest. A build with
`USE_TASKGRAPH=0` and one with `USE_TASKGRAPH=1` must produce the identical TC
size, round count, and (via `TC_DUMP`) tuple set.

## Debugging: the open GPU fault on inputs above ~24 K edges

**Status: unresolved.** `data_7035` and `data_23874` complete on the GPU; every
larger dataset aborts with
`cuStreamSynchronize / cuEventSynchronize failed with an illegal memory access
was encountered (700)` (and, at `-O3`, `Invalid access of peer GPU memory over
nvlink (226)`). The CPU backend is correct on every input. This section records
what has already been ruled out so it is not re-derived.

### Ruled out

| hypothesis | how it was eliminated |
|---|---|
| taskgraph record/replay | fails with `USE_TASKGRAPH=0` |
| async nowait target tasks, xkrt command queue | fails with `USE_SYNC=1` (no tasks at all) |
| `depend` clauses / dependence tracking | `USE_SYNC=1` emits none |
| `-O0` device codegen, device stack | fails at `-O3` too |
| allocator / `omp_target_alloc` | a 16 GiB `fill_result_set` completes and reads back correctly |
| edge table construction | `TC_VERIFY=1` reports fill -> 0 occupied, build -> exactly `n_edges` |
| result-set / frontier capacity | fails at `capacity_mult=8192` (set >= 2x TC), and the overflow flag never trips |
| **buffer size** | `./tc.x data_49152.bin 16` -> every buffer **2^20**, smaller than the working `data_23874` run at 2^21, still faults |
| algorithm | CPU backend produces the correct TC |

### The one live lead

`TC_WORKERS=4096 ./tc.x data_147892.bin 8192` **progresses**, where the default
`TC_WORKERS` (2^21) faults. `n_workers` feeds only `k_expand` and `k_promote`, so
the fault is in one of those two and scales with the grid-stride worker count --
*not* with the buffer sizes. Note `data_23874` works at the same 2^21 workers, so
it is an interaction between worker count and the amount of work per round, not
either alone.

### Tools

```shell
# name the faulting kernel: blocking launches + a flushed line per dispatch,
# so the LAST line printed is the kernel that faulted
make USE_TARGET=1 USE_SYNC=1
TC_TRACE=1 ./tc.x MNMGDatalog-reference/data/data_49152.bin 16

# check the edge table really was built on the device
TC_VERIFY=1 ./tc.x ...

# sweep the worker count (the live lead)
for w in 4096 65536 1048576 2097152; do TC_WORKERS=$w ./tc.x ... ; done

# memory-model A/B: pinned host + map(present:), i.e. what krylov/lulesh do.
# tc.cpp is the only app here using omp_target_alloc + is_device_ptr, and the
# only one that faults. Diagnostic only -- it mirrors every buffer on the host,
# so use it on a small configuration (e.g. data_49152 with capacity_mult 16).
make USE_TARGET=1 TC_MAPPED_MEM=1
```

### Standalone reproducer

`tc_repro.cpp` isolates the three constructs used by tc.cpp and by no other app
in `apps/openmp` (`omp_target_alloc`+`is_device_ptr`, device
`omp atomic compare capture`, device `omp atomic capture`) in `k_expand`'s
grid-stride shape, with no `tasking.h`, no `alloc.h` and no xkomp:

```shell
make repro USE_TARGET=1
./tc_repro.x [keys] [slots] [workers] [mode] [rounds]

# the two sweeps that matter
for w in 4096 65536 1048576 2097152; do ./tc_repro.x 1000000 4194304 $w; done
for m in 0 1 2 3; do ./tc_repro.x 1000000 4194304 2097152 $m; done   # bisect the atomics
make repro-mapped USE_TARGET=1 && ./tc_repro_mapped.x                # memory-model A/B
```

`mode` is a bitmask: bit 0 = atomic CAS insert, bit 1 = atomic fetch-add append,
`0` = plain stores. If the reproducer faults it is a self-contained toolchain bug
report; if it does not, the difference from `k_expand` is the next thing to look
at.

## Evaluation harness

Registered as app `mnmg` in `../scripts/appspecs.py`; datasets are `data_<N>.bin`
where `N` = edge count, so `evaluate.py --apps mnmg --sizes 7035,23874` maps sizes
to files. The per-dataset `capacity_mult` lives in `_MNMG_MULT` there and must be
kept in sync with the table above. `--iters` is ignored for `mnmg` (the round count comes from the data;
use `TC_WARMUP` for the warm-up rounds). `avg_ms`/`stddev_ms` come from the
steady-state rounds, `iter0_ms` from round 0 (the record round) and `elapsed_s`
from the end-to-end total. `evaluate.py` sweeps the synchronous / no-taskgraph /
taskgraph configs
just like the other apps.
