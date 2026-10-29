"""
Application registry for the apps/openmp evaluation harness.

Each AppSpec describes, for one benchmark app, how to (i) build a given
configuration, (ii) run one problem size, and (iii) parse the metrics it prints.
scripts/evaluate.py drives the sweep (app x variant x config x size) from these.

Configurations
--------------
Every app is always compared across these reference configurations plus one per
user-supplied CGIR optimization combo:

  * synchronous   : USE_SYNC=1, USE_TASKGRAPH=0, one task/kernel per loop
                    -- the "current practice" blocking baseline.
  * no-taskgraph  : USE_SYNC=0, USE_TASKGRAPH=0
                    -- plain OpenMP tasks/target (taskgraph overhead reference).
  * taskgraph:none: USE_TASKGRAPH=1, OMP_TASKGRAPH_OPT=""  (no CGIR pass).
  * taskgraph:<opt>: USE_TASKGRAPH=1, OMP_TASKGRAPH_OPT="<opt>" for each combo.

The synchronous / no-taskgraph / taskgraph split is a *compile-time* choice
(USE_SYNC / USE_TASKGRAPH), so evaluate.py rebuilds per configuration; the CGIR
pass within a taskgraph build is a *run-time* choice (OMP_TASKGRAPH_OPT).
The GPU vs CPU backend (USE_TARGET) is orthogonal and chosen once (--target).
"""

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# CGIR optimization combos compared on top of the references (each -> one
# taskgraph:<opt> configuration). Names are XKOMP OMP_TASKGRAPH_OPT passes
# (comma or space separated): copy-normalize copy-fuse reduce-node reduce-edge
# jit prog-fuse sequence batch.
DEFAULT_OPTS = [
    "reduce-node,reduce-edge",
    "reduce-node,reduce-edge,batch",
]


@dataclass
class Config:
    label: str                 # legend label, e.g. "taskgraph:reduce-node,reduce-edge"
    build: Dict[str, str]      # make variables, e.g. {"USE_SYNC": "1", ...}
    opt: Optional[str]         # OMP_TASKGRAPH_OPT value (None if no taskgraph)
    grain1: bool = False       # run with one task/kernel per loop (sync baseline)


def default_configs(opts: List[str]) -> List[Config]:
    cfgs = [
        Config("synchronous",    {"USE_SYNC": "1", "USE_TASKGRAPH": "0"}, None, grain1=True),
        Config("no-taskgraph",   {"USE_SYNC": "0", "USE_TASKGRAPH": "0"}, None),
        Config("taskgraph:none", {"USE_SYNC": "0", "USE_TASKGRAPH": "1"}, ""),
    ]
    for o in opts:
        cfgs.append(Config(f"taskgraph:{o}", {"USE_SYNC": "0", "USE_TASKGRAPH": "1"}, o))
    return cfgs


# --------------------------------------------------------------------------- #
# Metric parsing helpers.
# --------------------------------------------------------------------------- #
_F = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _grab(text, pattern, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (ValueError, IndexError):
        return None


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None)
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return (m, s)


def _parse_krylov(text):
    return {
        "avg_ms":    _grab(text, r"(?:iterations|restarts) 2\.\..*\(avg\)\s*:\s*" + _F + r"\s*ms"),
        "stddev_ms": _grab(text, r"(?:iterations|restarts) 2\.\..*\(stddev\)\s*:\s*" + _F + r"\s*ms"),
        "iter0_ms":  _grab(text, r"(?:iteration|restart) 0.*:\s*" + _F + r"\s*ms"),
        "elapsed_s": _grab(text, r"total solve time\s*:\s*" + _F),
        "flops":     _grab(text, r"theoretical flops\s*:\s*" + _F),
        "gflops":    _grab(text, r"performance\s*:\s*" + _F),
        "residual":  _grab(text, r"relative residual\s*:\s*" + _F),
        "error":     _grab(text, r"relative error\s*:\s*" + _F),
    }


def _parse_lulesh(text):
    return {
        "avg_ms":    _grab(text, r"iterations 2\.\..*\(avg\)\s*:\s*" + _F + r"\s*ms"),
        "stddev_ms": _grab(text, r"iterations 2\.\..*\(stddev\)\s*:\s*" + _F + r"\s*ms"),
        "iter0_ms":  _grab(text, r"iteration 0\s*:\s*" + _F + r"\s*ms"),
        "elapsed_s": _grab(text, r"Elapsed time\s*=\s*" + _F),
        "fom":       _grab(text, r"FOM\s*=\s*" + _F + r"\s*\(z/s\)"),
    }


def _parse_llmc(text):
    # llm.c prints one "Iteration runtime : X ms" per step (to stderr); compute
    # the avg/stddev over the steady steps (drop step 0 = warmup/first build).
    runs = [float(x) for x in re.findall(r"Iteration runtime\s*:\s*" + _F + r"\s*ms", text)]
    steady = runs[1:] if len(runs) > 1 else runs
    avg, std = _mean_std(steady)
    return {
        "avg_ms":    avg,
        "stddev_ms": std,
        "iter0_ms":  runs[0] if runs else None,
        "elapsed_s": _grab(text, r"Took\s+" + _F + r"\s*s"),
    }


def _parse_harness(text):
    # Shared summary emitted by the polybench / miniWeather / sw4lite / cloverleaf
    # ports (same lines as lulesh): iteration 0 / iterations 2.. (avg|stddev),
    # Elapsed time, and a FOM line (unit varies per app, so match loosely).
    return {
        "avg_ms":    _grab(text, r"iterations 2\.\..*\(avg\)\s*:\s*" + _F + r"\s*ms"),
        "stddev_ms": _grab(text, r"iterations 2\.\..*\(stddev\)\s*:\s*" + _F + r"\s*ms"),
        "iter0_ms":  _grab(text, r"iteration 0\s*:\s*" + _F + r"\s*ms"),
        "elapsed_s": _grab(text, r"Elapsed time\s*=\s*" + _F),
        "fom":       _grab(text, r"FOM\s*=\s*" + _F),
    }


# --------------------------------------------------------------------------- #
# App registry.
# --------------------------------------------------------------------------- #
@dataclass
class AppSpec:
    name: str
    directory: str                       # relative to apps/openmp/
    variants: List[str]                  # [""] when a single binary
    make_target: Callable[[str], str]    # variant -> make target
    binary: Callable[[str], str]         # variant -> ./binary (run cwd = directory)
    run_args: Callable                   # (variant, size, iters, cfg) -> [args]
    parse: Callable[[str], dict]         # stdout+stderr -> metrics dict
    work: Callable[[int], tuple]         # size -> (value, label) for the top axis
    sizes: List[int]
    iters: int
    rebuild_per_size: bool = False       # size is a compile-time macro (rebuild)
    llmc_defs: Optional[Callable] = None # (size, iters, batch, grain) -> defs string
    batch: int = 4                       # llm.c BATCH_SIZE (for tokens = B*T)
    defs_var: str = "LLMC_DEFS"          # make variable the defs string is passed as
                                         # (PB_DEFS / MW_DEFS / SW4_DEFS for the
                                         #  rebuild-per-size stencil apps)


# ---- krylov: grid n, matrix N=n^3, work ~ n^3; -t/-s = task counts (0=threads) --
# grain is a positional list [T1, T2]: T1 -> -t (tasks per vector op), T2 -> -s
# (SpMV sub-tasks per block, default 1 so every loop has T1 tasks). None -> the
# app default (-t 0 -s 0 = auto, i.e. omp threads). Sync is always 1/loop (-t 1 -s 1).
def _krylov_run(variant, size, iters, cfg, grain):
    if cfg.grain1:
        t, s = "1", "1"
    elif grain:
        t = str(grain[0])
        s = str(grain[1]) if len(grain) > 1 else "1"
    else:
        t, s = "0", "0"
    return ["-n", str(size), "-i", str(iters), "-t", t, "-s", s, "-S", "27"]

KRYLOV = AppSpec(
    name="krylov",
    directory="krylov",
    variants=["cg", "cr", "bicgstab", "minres", "gmres"],
    make_target=lambda v: v,
    binary=lambda v: f"./{v}.x",
    run_args=_krylov_run,
    parse=_parse_krylov,
    work=lambda n: (float(n) ** 3, "n\u00b3 (\u221d FLOPs)"),
    sizes=[32, 48, 64],
    iters=50,
)

# ---- lulesh: mesh side s, zones = s^3; -nb = tasks per loop ---------------------
# grain is a positional list [nb]: nb -> -nb (tasks per loop). None -> the app
# default (-nb 32). The synchronous config is always 1 task/loop (-nb 1).
def _lulesh_run(variant, size, iters, cfg, grain):
    nb = "1" if cfg.grain1 else (str(grain[0]) if grain else "32")
    return ["-i", str(iters), "-s", str(size), "-r", "11", "-b", "1", "-c", "1", "-nb", nb]

LULESH = AppSpec(
    name="lulesh",
    directory="lulesh",
    variants=[""],
    make_target=lambda v: "lulesh",
    binary=lambda v: "./lulesh",
    run_args=_lulesh_run,
    parse=_parse_lulesh,
    work=lambda s: (float(s) ** 3, "zones (s\u00b3)"),
    sizes=[16, 32, 48, 64],
    iters=30,
)

# ---- llm.c: SEQUENCE_SIZE T is a compile-time macro -> rebuild per size ---------
# grain is a positional list mapping to the compile-time granularity macros
# [GRAN_TMP, OC_SPLIT, OC_BACK_SPLIT] (extras ignored); injected at build time
# rather than as run arguments. None -> the source defaults.
_LLMC_GRAIN_MACROS = ["GRAN_TMP", "OC_SPLIT", "OC_BACK_SPLIT"]

def _llmc_run(variant, size, iters, cfg, grain):
    return []  # no runtime args; size/steps/grain are compiled in

def _llmc_defs(size, iters, batch, grain):
    defs = f"-DSEQUENCE_SIZE={size} -DNB_STEPS={iters} -DBATCH_SIZE={batch}"
    for name, val in zip(_LLMC_GRAIN_MACROS, grain or []):
        defs += f" -D{name}={val}"
    return defs

LLMC = AppSpec(
    name="llm.c",
    directory="llm.c",
    variants=[""],
    make_target=lambda v: "train_gpt2",
    binary=lambda v: "./train_gpt2",
    run_args=_llmc_run,
    parse=_parse_llmc,
    work=lambda t: (4.0 * float(t), "tokens (B\u00b7T)"),  # batch default 4
    sizes=[64, 128, 256],
    iters=10,
    rebuild_per_size=True,
    llmc_defs=_llmc_defs,
)

# ---- polybench stencils: N / TSTEPS are compile-time macros -> rebuild per size -
# Four fully-parallel stencils (jacobi-1d/2d, heat-3d, fdtd-2d) as variants, like
# krylov's solvers; each builds <variant>.x. -nb = tasks per loop; -i = TSTEPS cap.
# Size sweeps the spatial dimension (TSTEPS held at the default per build).
_PB_TSTEPS = 100

def _pb_run(variant, size, iters, cfg, grain):
    nb = "1" if cfg.grain1 else (str(grain[0]) if grain else "32")
    return ["-i", str(iters), "-nb", nb]

def _pb_defs(size, iters, batch, grain):
    # 2D/3D stencils take N; fdtd-2d takes NX/NY; set all so one spec serves all.
    n = size
    return (f"-DPB_N={n} -DPB_TSTEPS={_PB_TSTEPS} "
            f"-DPB_NX={n} -DPB_NY={n} -DPB_TMAX={_PB_TSTEPS}")

POLYBENCH = AppSpec(
    name="polybench",
    directory="polybench",
    variants=["jacobi-1d", "jacobi-2d", "heat-3d", "fdtd-2d"],
    make_target=lambda v: v,
    binary=lambda v: f"./{v}.x",
    run_args=_pb_run,
    parse=_parse_harness,
    # work axis: lattice sites * timesteps (N^2 * TSTEPS is a reasonable common
    # proxy; heat-3d is N^3 and jacobi-1d is N, but N^2 keeps the plot readable).
    work=lambda n: (float(n) ** 2 * _PB_TSTEPS, "N\u00b2\u00b7TSTEPS"),
    sizes=[128, 256, 512],
    iters=_PB_TSTEPS,
    rebuild_per_size=True,
    llmc_defs=_pb_defs,
    defs_var="PB_DEFS",
)

# ---- miniWeather: NX/NZ/SIM_TIME are compile-time -> rebuild per size ------------
# size -> NX (NZ = NX/2, as the domain is 2:1). -nb = tasks per loop; -i caps steps.
def _mw_run(variant, size, iters, cfg, grain):
    nb = "1" if cfg.grain1 else (str(grain[0]) if grain else "32")
    return ["-nb", nb, "-i", str(iters)]

def _mw_defs(size, iters, batch, grain):
    nx = size
    nz = max(size // 2, 1)
    return (f"-D_NX={nx} -D_NZ={nz} -D_SIM_TIME=400 -D_OUT_FREQ=-1 "
            f"-D_DATA_SPEC=DATA_SPEC_THERMAL")

MINIWEATHER = AppSpec(
    name="miniweather",
    directory="miniWeather",
    variants=[""],
    make_target=lambda v: "miniweather",
    binary=lambda v: "./miniweather",
    run_args=_mw_run,
    parse=_parse_harness,
    work=lambda n: (float(n) * max(n // 2, 1), "cells (NX\u00b7NZ)"),
    sizes=[100, 200, 400],
    iters=50,
    rebuild_per_size=True,
    llmc_defs=_mw_defs,
    defs_var="MW_DEFS",
)

# ---- sw4lite proxy: SW4_N/SW4_NZ are compile-time -> rebuild per size ------------
# size -> N (cubic box, NZ = N). -nb = tasks per loop; -i = #timesteps.
def _sw4_run(variant, size, iters, cfg, grain):
    nb = "1" if cfg.grain1 else (str(grain[0]) if grain else "32")
    return ["-i", str(iters), "-nb", nb]

def _sw4_defs(size, iters, batch, grain):
    return f"-DSW4_N={size} -DSW4_NZ={size}"

SW4LITE = AppSpec(
    name="sw4lite",
    directory="sw4lite",
    variants=[""],
    make_target=lambda v: "sw4lite",
    binary=lambda v: "./sw4lite",
    run_args=_sw4_run,
    parse=_parse_harness,
    work=lambda n: (float(n) ** 3, "grid points (N\u00b3)"),
    sizes=[32, 48, 64],
    iters=50,
    rebuild_per_size=True,
    llmc_defs=_sw4_defs,
    defs_var="SW4_DEFS",
)

# ---- cloverleaf: mesh from an input deck (runtime), not compile-time ------------
# size -> x_cells = y_cells via a generated deck written to the app dir. Granularity
# is deck-driven (tiles_per_chunk); the harness grain maps to tiles_per_chunk.
# end_step caps the run (= iters). test_problem left 0 (no reference KE check in the
# sweep). The deck path is passed with --file and the host device with --device 0.
def _cloverleaf_run(variant, size, iters, cfg, grain):
    tiles = "1" if cfg.grain1 else (str(grain[0]) if grain else "1")
    # write a deck sized `size` with `iters` steps into the app directory.
    import os
    deck = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "cloverleaf", f"clover_sweep_{size}_{iters}_{tiles}.in")
    with open(deck, "w") as fh:
        fh.write(
            "*clover\n"
            " state 1 density=0.2 energy=1.0\n"
            " state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0\n"
            f" x_cells={size}\n y_cells={size}\n"
            " xmin=0.0\n ymin=0.0\n xmax=10.0\n ymax=10.0\n"
            " initial_timestep=0.04\n timestep_rise=1.5\n max_timestep=0.04\n"
            f" end_time=100.0\n end_step={iters}\n"
            f" tiles_per_chunk {tiles}\n"
            "*endclover\n")
    return ["--file", os.path.basename(deck), "--device", "0"]

CLOVERLEAF = AppSpec(
    name="cloverleaf",
    directory="cloverleaf",
    variants=[""],
    make_target=lambda v: "cloverleaf",
    binary=lambda v: "./cloverleaf",
    run_args=_cloverleaf_run,
    parse=_parse_harness,
    work=lambda n: (float(n) ** 2, "cells (x\u00b7y)"),
    sizes=[128, 256, 512],
    iters=30,
)


APPS: Dict[str, AppSpec] = {a.name: a for a in (
    KRYLOV, LULESH, LLMC, POLYBENCH, MINIWEATHER, SW4LITE, CLOVERLEAF)}
