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
# GPU builds must be -O3 (the ../common.mk default) -- see "Debugging".

./tc.x <data.bin> [capacity_mult] [frontier_slots]
make test                                # data_7035.bin (TC=146120, 64 rounds)
```

`capacity_mult` is **optional** -- omit it and the closure size is discovered and
cached automatically (see "Capacity is discovered, not configured").

The fixpoint runs **once**; the number of rounds is determined by the dataset, not
by a command-line knob.

Environment: `TC_CACHE=<f>` capacity cache file (default `./tc_capacity.cache`);
`TC_WARMUP=<n>` untimed, ungraphed warm-up rounds before round 0 (default 3); `TC_WORKERS=<n>` overrides the GPU grid-stride worker count (see
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

## Capacity is discovered, not configured

The result set is a fixed pre-allocated open-addressing table -- that is what
makes the per-round kernel sequence loop-invariant and therefore replayable, and
it is the whole point of the fused design. The *original* MNMGDatalog engine
(`MNMGDatalog-reference/tc.cu`) needs no capacity at all because it
`cudaMalloc`s and re-sorts the entire relation every round -- precisely the cost
this design removes.

A fixed table needs a size up front, and that size **cannot be derived from the
input**: the TC/edge ratio spans 21x (`OL.cedge`) to 5977x (`p2p-Gnutella31`).
The CUDA benchmark this is ported from resolves that with a hand-maintained
per-dataset lookup table and a mandatory `capacity_mult` argument
(`tc_benchmark/tests/benchmark.sh:42-67`). We discover it instead:

```shell
./tc.x MNMGDatalog-reference/data/data_147892.bin        # just works
```

On the first run for a dataset, an **untimed, ungraphed probe solve** runs the
real fixpoint once on a provisional table, learns the exact closure size and the
peak per-round frontier, then reallocates to
`next_pow2(2*TC)` / `next_pow2(2*peak)` for the measured run:

```
# sizing pass: closure size unknown for data_147892.bin -- probing (untimed, cached
#              in tc_capacity.cache for later runs; pass capacity_mult to skip).
# sizing pass: TC=884179859, peak frontier=61234567 -> result_cap=2147483648, ...
```

The result is cached in `tc_capacity.cache` (override with `TC_CACHE=<file>`),
keyed on dataset name and edge count, so every later run and every sweep skips
the probe. Delete the file to re-discover. The probe reports its wall time as a
separate `sizing pass` statistic and is **excluded** from
`total time (end-to-end)`.

The probe must run before anything is recorded, because the taskgraph captures
buffer pointers -- growing the table mid-solve would invalidate a recorded graph,
and xkomp has no graph reset.

### Overriding

Passing `capacity_mult` (arg 2) skips discovery entirely and reproduces the
reference's `next_pow2(edges * mult)` sizing, for fidelity comparisons:

| dataset | edges | rounds | TC | reference `capacity_mult` |
|---|---:|---:|---:|---:|
| OL.cedge (`data_7035`) | 7 035 | 64 | 146 120 | 64 |
| TG.cedge (`data_23874`) | 23 874 | 58 | 481 121 | 64 |
| p2p-Gnutella31 (`data_147892`) | 147 892 | 31 | 884 179 859 | 12288 |
| usroad (`data_165435`) | 165 435 | 606 | 871 365 688 | 8192 |
| fe_ocean (`data_409593`) | 409 593 | 247 | 1 669 750 513 | 8192 |
| vsp_finan | 552 020 | 520 | 910 070 918 | 3456 |
| com-dblp | 1 049 866 | 31 | 1 911 754 892 | 3800 |

`frontier_slots` (arg 3) likewise overrides the discovered frontier size.

Note that auto-sizing generally allocates **less** than the reference: the
hardcoded mults over-provision on most datasets, and the frontier is sized from
the observed peak rather than a flat `min(result_cap, 2^28)` (2 GiB each). That
shrinks `setup` and the `compact` scan, which is folded into `compute`.

Undersizing (only reachable via an explicit `capacity_mult`) is detected, not
silently wrong: every insert path sets a device-side overflow flag, `k_writeback`
returns it with `new_count` in the same async D2H, and the fixpoint aborts on the
next convergence test with the offending capacity printed. The frontier readers
(`k_promote`, `k_set_sizes`) clamp to `frontier_cap`, so an overflowing round
cannot walk off the end of the frontier buffers.

## Correctness

Known reference sizes: `data_10` -> TC 18 / 3 rounds, `data_7035` -> 146 120 / 64,
`data_23874` -> 481 121 / 58, and the table above for the rest. A build with
`USE_TASKGRAPH=0` and one with `USE_TASKGRAPH=1` must produce the identical TC
size, round count, and (via `TC_DUMP`) tuple set.

Verified on GH200 (sm_90, CUDA 13.3.1) at `-O3`, both taskgraph settings:

| dataset | edges | rounds | TC |
|---|---:|---:|---:|
| `data_7035.bin` (OL.cedge) | 7 035 | 64 | 146 120 |
| `data_23874.bin` (TG.cedge) | 23 874 | 58 | 481 121 |
| `data_147892.bin` (p2p-Gnutella31) | 147 892 | 31 | 884 179 859 |

## Debugging

### GPU builds must be `-O3`

`-O0` **breaks GPU offload** on this toolchain. A `target teams distribute
parallel for` whose body makes device function calls -- anything the optimizer
would otherwise have inlined -- faults with `CUDA_ERROR_ILLEGAL_ADDRESS` (700)
once roughly 10^5 threads are concurrently inside the call chain. It is not a
stack-size problem (`LIBOMPTARGET_STACK_SIZE` up to 256 KiB changes nothing) and
not an application bug: [`tc_repro.cpp`](tc_repro.cpp) reproduces it with no
atomics, no tasking and provably in-bounds indexing, and the same source passes
at `-O3`. Full matrix and filing notes in [REPRODUCER.md](REPRODUCER.md).

Two mitigations are in place, so this should not bite again:

* every `declare target` helper in `tc.cpp` is `TC_DEVFN`
  (`static inline __attribute__((always_inline))`), so no device call chain
  survives to codegen at any `-O` level;
* `../common.mk` defaults to `OPT ?= -O3` and warns if `-O0` is combined with
  `USE_TARGET=1`.

This cost weeks of misdiagnosis, because at `-O0` the fault looks exactly like an
application memory bug: it is input-size dependent (only kernels with call chains
fault, and only once the frontier is large enough), it moves between kernels as
the code changes, and `compute-sanitizer` reports nothing.

### Tools

```shell
# TC_TRACE: one flushed stderr line per kernel dispatch. With USE_SYNC=1 every
# launch is blocking, so after an abort the LAST line names the faulting kernel.
make USE_TARGET=1 USE_SYNC=1
TC_TRACE=1 ./tc.x MNMGDatalog-reference/data/data_49152.bin 16

# TC_VERIFY: read the edge table back and check the fill / build ran on device
TC_VERIFY=1 ./tc.x ...

# TC_WORKERS: grid-stride worker count of k_expand / k_promote
for w in 4096 65536 1048576 2097152; do TC_WORKERS=$w ./tc.x ... ; done

# TC_MAPPED_MEM: memory-model A/B. tc.cpp is the only app here using
# omp_target_alloc + is_device_ptr; this switches every buffer to pinned host +
# map(present:), i.e. what krylov/lulesh/HPCCG do. Diagnostic only -- it mirrors
# every buffer on the host, so use a small configuration.
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

for w in 4096 65536 1048576 2097152; do ./tc_repro.x 1000000 4194304 $w; done
for m in 0 1 2 3; do ./tc_repro.x 1000000 4194304 2097152 $m; done
make repro-mapped USE_TARGET=1 && ./tc_repro_mapped.x
make OPT="-O0 -g" repro USE_TARGET=1     # the failing variant
```

`mode` is a bitmask: bit 0 = atomic CAS insert, bit 1 = atomic fetch-add append,
`0` = plain stores.

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
