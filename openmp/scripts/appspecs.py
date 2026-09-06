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
pass within a taskgraph build is a *run-time* choice (OMP_TASKGRAPH_OPT for
XKOMP, NODES_TASKITER_CGIR_OPT for NODES). The backend is orthogonal and chosen
once (--target cpu | gpu | ompss); see BACKENDS below.

Unrolling
---------
A taskgraph instance carries an implicit taskgroup, so consecutive instances
cannot overlap. `unroll` (evaluate.py --unroll) folds that many iterations into
ONE recorded instance, recovering the cross-iteration overlap inside the graph
and paying the barrier once per `unroll` iterations. It is a run argument for
krylov / lulesh / mnmg and a compile-time macro (UNROLL) for llm.c, hence part of
that app's build key. It only does anything under USE_TASKGRAPH, so evaluate.py
sweeps it for the taskgraph configurations only.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# CGIR optimization combos compared on top of the references (each -> one
# taskgraph:<opt> configuration). Names are CGIR pass names, comma or space
# separated: copy-fuse reduce-node transitive-reduction prog-fuse jit sequence
# batch. They are passed to XKOMP as OMP_TASKGRAPH_OPT and to NODES as
# NODES_TASKITER_CGIR_OPT (the same string drives both runtimes).
#
# The default is the INCREMENTAL pipeline of the paper: each combo adds one of
# the passes of the paper's pass table to the previous one, in the order the
# library applies them. That order is what makes the resulting bars readable as
# "what does this pass add on top of the ones before it".
#
#   +reduce     the two graph-reduction passes (control-node elimination and
#               transitive reduction)
#   +jit        recompiling each task body at -O3 in the runtime, WITHOUT fusing
#               -- isolated on purpose, so the fusion bar below cannot be
#               credited with a gain that is only recompilation
#   +prog-fuse  fusing serial chains of programs before compiling them
#   +packing    the sequence/batch packing passes (host super-tasks and vendor
#               command graphs)
#
# copy-fuse is deliberately absent: the pass exists in the library but is out of
# the paper's scope, and a configuration nobody reports is machine time wasted.
DEFAULT_OPTS = [
    "reduce-node,transitive-reduction",
    "reduce-node,transitive-reduction,jit",
    "reduce-node,transitive-reduction,jit,prog-fuse",
    "reduce-node,transitive-reduction,jit,prog-fuse,sequence,batch",
]

# Short legend label for each pipeline, keyed by the LAST pass added. Keeps the
# figures readable: the full pass list is still written to runs.csv, and the
# `opt` column remains the ground truth.
OPT_LABELS = {
    "reduce-node,transitive-reduction":                             "+reduce",
    "reduce-node,transitive-reduction,jit":                          "+jit",
    "reduce-node,transitive-reduction,jit,prog-fuse":                "+prog-fuse",
    "reduce-node,transitive-reduction,jit,prog-fuse,sequence,batch": "+packing",
}


# --------------------------------------------------------------------------- #
# Backends. A backend fixes the tasking runtime and the device the work runs on;
# it is orthogonal to the configuration (which fixes the schedule and the CGIR
# pass set) and is chosen once per sweep with --target.
#
# `opt_env` is the environment variable through which that runtime takes its CGIR
# pass list. XKOMP and NODES accept the SAME pass names (both call
# cgir::command_graph_pass_set_from_str), which is what lets one --opts string
# drive an OpenMP and an OmpSs-2 sweep and makes the two comparable.
#
# `env` is applied to every run of the backend. NODES only reaches the CGIR path
# when taskiter.opt.use_cgir is on, and that option is a NODES config variable,
# not an environment variable, hence the NODES_CONFIG_OVERRIDE. (Its list
# separator is ',', so no comma-valued option may be set through it -- which is
# exactly why the pass list has an env var of its own.)
# --------------------------------------------------------------------------- #
@dataclass
class Backend:
    label: str                  # value of the `backend` column in runs.csv
    build: Dict[str, str]       # make variables
    opt_env: str                # env var carrying the CGIR pass list
    env: Dict[str, str] = field(default_factory=dict)


BACKENDS: Dict[str, Backend] = {
    "cpu":   Backend("cpu",   {"USE_TARGET": "0", "USE_OMPSS": "0"}, "OMP_TASKGRAPH_OPT"),
    "gpu":   Backend("gpu",   {"USE_TARGET": "1", "USE_OMPSS": "0"}, "OMP_TASKGRAPH_OPT"),
    "ompss": Backend("ompss", {"USE_TARGET": "0", "USE_OMPSS": "1"}, "NODES_TASKITER_CGIR_OPT",
                     {"NODES_CONFIG_OVERRIDE": "taskiter.opt.use_cgir=true"}),
}


@dataclass
class Config:
    label: str                 # legend label, e.g. "taskgraph:reduce-node,transitive-reduction"
    build: Dict[str, str]      # make variables, e.g. {"USE_SYNC": "1", ...}
    opt: Optional[str]         # CGIR pass list (None if no taskgraph)
    grain1: bool = False       # run with one task/kernel per loop (sync baseline)

    @property
    def taskgraph(self) -> bool:
        """Whether this configuration records a taskgraph -- i.e. whether it has
        the per-instance barrier that --unroll exists to amortize. Elsewhere the
        unroll is inert by construction (the apps group iterations identically in
        every configuration), so sweeping it there would only burn machine time."""
        return self.build.get("USE_TASKGRAPH") == "1"


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


# All three C/C++ apps print the same four timing lines, in the same words, so
# one parser serves them. They are per taskgraph INSTANCE -- a group of `unroll`
# iterations -- and always in ms per iteration, and they map onto the runtime's
# own phases: instance 0 records the graph (XKOMP rc == 1), instance 1 builds and
# optimizes the command graph and runs the first replay (rc == 2), instances 2..
# are steady replays.
#
# `instance` vs `instances` disambiguates the singular lines from the plural ones
# without a lookahead: "instances 2..31" can never match "instance 1", because
# the 's' sits where the space would be. [^:\n] rather than [^:] so a line that
# is absent cannot reach forward to a colon several lines below.
def _parse_instances(text):
    return {
        "iter0_ms":  _grab(text, r"instance 0[^:\n]*:\s*" + _F + r"\s*ms"),
        "iter1_ms":  _grab(text, r"instance 1[^:\n]*:\s*" + _F + r"\s*ms"),
        "avg_ms":    _grab(text, r"instances \d+\.\.\d+[^:\n]*\(avg\)[^:\n]*:\s*" + _F + r"\s*ms"),
        "stddev_ms": _grab(text, r"instances \d+\.\.\d+[^:\n]*\(stddev\)[^:\n]*:\s*" + _F + r"\s*ms"),
    }


def _parse_krylov(text):
    m = _parse_instances(text)
    m.update({
        "elapsed_s": _grab(text, r"total solve time\s*:\s*" + _F),
        "flops":     _grab(text, r"theoretical flops\s*:\s*" + _F),
        "gflops":    _grab(text, r"performance\s*:\s*" + _F),
        "residual":  _grab(text, r"relative residual\s*:\s*" + _F),
        "error":     _grab(text, r"relative error\s*:\s*" + _F),
    })
    return m


def _parse_lulesh(text):
    m = _parse_instances(text)
    m.update({
        "elapsed_s": _grab(text, r"Elapsed time\s*=\s*" + _F),
        "fom":       _grab(text, r"FOM\s*=\s*" + _F + r"\s*\(z/s\)"),
    })
    return m


def _parse_mnmg(text):
    # Same four instance lines as krylov / lulesh; here they are ms per fixpoint
    # ROUND. The extra work is the fallback: the round count is data-dependent
    # (the fixpoint runs to convergence), so a small graph -- or a large -u --
    # can leave fewer than three instances and no steady-state window at all.
    # "total time (end-to-end)" is the MNMGDatalog paper's metric (file IO + H2D +
    # setup + compute + D2H) and is printed in ms, unlike the other apps' seconds.
    m = _parse_instances(text)
    if m["avg_ms"] is None:
        m["avg_ms"] = m["iter1_ms"] if m["iter1_ms"] is not None else m["iter0_ms"]
        m["stddev_ms"] = 0.0 if m["avg_ms"] is not None else None
    total_ms = _grab(text, r"total time \(end-to-end\)\s*:\s*" + _F + r"\s*ms")
    m["elapsed_s"] = (total_ms / 1000.0) if total_ms is not None else None
    return m


def _parse_llmc(text):
    # llm.c prints one "Iteration runtime : X ms" per taskgraph INSTANCE (to
    # stderr), already divided by UNROLL, so the unit is per step at any unroll.
    # Instance 0 records the graph and instance 1 builds the command graph and
    # runs the first replay, so the steady window starts at 2 -- as in the krylov
    # / lulesh / mnmg parsers, and as in llm.c's own printed average. This matters
    # once unrolling makes the instances few: at NB_STEPS=12, UNROLL=4 there are
    # only three, and keeping instance 1 would report the build cost as steady.
    runs = [float(x) for x in re.findall(r"Iteration runtime\s*:\s*" + _F + r"\s*ms", text)]
    steady = runs[2:] if len(runs) > 2 else runs[-1:]
    avg, std = _mean_std(steady)
    return {
        "avg_ms":    avg,
        "stddev_ms": std,
        "iter0_ms":  runs[0] if len(runs) > 0 else None,   # record
        "iter1_ms":  runs[1] if len(runs) > 1 else None,   # build + 1st replay
        "elapsed_s": _grab(text, r"Took\s+" + _F + r"\s*s"),
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
    run_args: Callable                   # (variant, size, iters, cfg, grain, unroll) -> [args]
    parse: Callable[[str], dict]         # stdout+stderr -> metrics dict
    work: Callable[[int], tuple]         # size -> (value, label) for the top axis
    sizes: List[int]
    iters: int
    rebuild_per_size: bool = False       # llm.c: size is a compile-time macro
    llmc_defs: Optional[Callable] = None # (size, iters, batch, grain, unroll) -> LLMC_DEFS
    batch: int = 4                       # llm.c BATCH_SIZE (for tokens = B*T)
    # Number of granularity knobs this app takes, i.e. how many ':'-separated
    # components one --grain entry may hold (0 = the app has no knob). Used by
    # evaluate.py to reject malformed / stale grain specs.
    grain_arity: int = 0
    # Backends the app has a port for. Only llm.c has an OmpSs-2 source path
    # (a `#pragma oss taskiter` over the training loop), so an --target ompss
    # sweep silently skips the others rather than building binaries whose task
    # constructs vanish.
    backends: List[str] = field(default_factory=lambda: ["cpu", "gpu"])
    # Unroll values the app accepts on a given backend. llm.c compiles UNROLL in
    # and #errors on anything but 1 under OmpSs-2 (its taskiter has no epilogue
    # to fold), so the harness pins it instead of failing every build.
    max_unroll: Dict[str, int] = field(default_factory=dict)
    # Presentation metadata for the paper table (plot.py --latex-table): the name
    # and application class to print. Kept next to the app rather than in the
    # plotting script, so one file describes each app completely.
    pretty: str = ""
    klass: str = ""
    # What the panel of the paper figure puts on its x axis: "size" (the problem
    # size sweep) or "variant" (the app's variants at a single size). Krylov earns
    # "variant": five solvers at one size say more about generality than one
    # solver at three sizes, and the paper has room for exactly one panel each.
    panel_x: str = "size"


# ---- krylov: grid n, matrix N=n^3, work ~ n^3; -t/-s = task counts (0=threads) --
# grain is THIS size's entry, written "s:t" (alphabetical, so the first component
# is -s = SpMV sub-tasks per block, the second is -t = tasks per vector op). A
# single-component entry sets -s and leaves -t at 1. None -> the app default
# (-t 0 -s 0 = auto, i.e. omp threads). Sync is always 1/loop (-t 1 -s 1).
#
# -u is the iterations folded into one taskgraph instance. GMRES ignores it (it
# is a restarted solver: each restart ends with a host least-squares solve whose
# result the next restart consumes, so two restarts cannot share an instance).
def _krylov_run(variant, size, iters, cfg, grain, unroll):
    if cfg.grain1:
        t, s = "1", "1"
    elif grain:
        s = str(grain[0])
        t = str(grain[1]) if len(grain) > 1 else "1"
    else:
        t, s = "0", "0"
    return ["-n", str(size), "-i", str(iters), "-t", t, "-s", s, "-S", "27",
            "-u", str(unroll)]

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
    grain_arity=2,          # "s:t"
    pretty="Krylov",
    klass="Iterative solvers",
    panel_x="variant",
)

# ---- lulesh: mesh side s, zones = s^3; -nb = tasks per loop ---------------------
# grain is THIS size's entry, a single component "nb" -> -nb (tasks per loop).
# None -> the app default (-nb 32). The synchronous config is always 1 (-nb 1).
def _lulesh_run(variant, size, iters, cfg, grain, unroll):
    nb = "1" if cfg.grain1 else (str(grain[0]) if grain else "32")
    return ["-i", str(iters), "-s", str(size), "-r", "11", "-b", "1", "-c", "1",
            "-nb", nb, "-u", str(unroll)]

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
    grain_arity=1,          # "nb"
    pretty="LULESH",
    klass="PDE time stepping",
)

# ---- llm.c: SEQUENCE_SIZE T is a compile-time macro -> rebuild per size ---------
# grain is THIS size's entry, written "GRAN_TMP:OC_SPLIT:OC_BACK_SPLIT" (a shorter
# entry sets only the leading macros); injected at build time rather than as run
# arguments. None -> the source defaults.
_LLMC_GRAIN_MACROS = ["GRAN_TMP", "OC_SPLIT", "OC_BACK_SPLIT"]

def _llmc_run(variant, size, iters, cfg, grain, unroll):
    return []  # no runtime args; size/steps/grain/unroll are compiled in

# llm.c takes no arguments, so the unroll is the compile-time macro UNROLL and
# thus part of the build key. It must divide NB_STEPS (the source rejects a
# shorter trailing instance: it is a different graph and would not replay);
# evaluate.py rounds the requested iteration count up to a multiple.
def _llmc_defs(size, iters, batch, grain, unroll):
    defs = (f"-DSEQUENCE_SIZE={size} -DNB_STEPS={iters} -DBATCH_SIZE={batch}"
            f" -DUNROLL={unroll}")
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
    grain_arity=3,          # "GRAN_TMP:OC_SPLIT:OC_BACK_SPLIT"
    backends=["cpu", "gpu", "ompss"],
    max_unroll={"ompss": 1},
    pretty="llm.c",
    klass="AI/ML training",
)

# ---- mnmg: Datalog transitive closure; dataset data_<N>.bin, N = #edges --------
# The "size" selects the input MNMGDatalog-reference/data/data_<size>.bin
# (size = edge count) and the x-axis work is that edge count. capacity_mult sizes
# the result set (next_pow2(edges * mult); must be >= ~2x TC or the run aborts
# with an overflow message).
# `iters` is NOT a knob here: the fixpoint runs until it converges, so the number
# of rounds is determined by the dataset (data_7035 -> 64, data_23874 -> 58) and
# each round is timed individually. Likewise TC has no task-count knob, so grain /
# the synchronous 1-task/loop are irrelevant. (The untimed warm-up rounds before
# round 0 are set with the TC_WARMUP env var; the default of 3 is used here.)
_MNMG_DATA = "MNMGDatalog-reference/data"
_MNMG_MULT = {7035: 64, 23874: 64}     # verified small graphs (TC 146120 / 481121)
_MNMG_MULT_DEFAULT = 4096              # generous default; raise via a larger set

# -u also gates the convergence test, which is then only evaluated every `unroll`
# rounds: the fixpoint may run up to unroll-1 extra (empty, harmless) rounds.
def _mnmg_run(variant, size, iters, cfg, grain, unroll):
    mult = _MNMG_MULT.get(size, _MNMG_MULT_DEFAULT)
    return [f"{_MNMG_DATA}/data_{size}.bin", str(mult), "-u", str(unroll)]

MNMG = AppSpec(
    name="mnmg",
    directory="MNMGDatalog",
    variants=[""],
    make_target=lambda v: "tc",
    binary=lambda v: "./tc.x",
    run_args=_mnmg_run,
    parse=_parse_mnmg,
    work=lambda n: (float(n), "edges"),
    sizes=[7035, 23874],
    iters=0,                           # unused: round count comes from the data
    pretty="MNMG",
    klass="Graph analytics",
)

APPS: Dict[str, AppSpec] = {a.name: a for a in (KRYLOV, LULESH, LLMC, MNMG)}
