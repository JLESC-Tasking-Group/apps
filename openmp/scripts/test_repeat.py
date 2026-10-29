#!/usr/bin/env python3
"""Regression test for --repeat against a `make clean` that wipes its directory.

Builds a fake app whose Makefile cleans with `rm -f *.x` -- krylov's, exactly --
and two variants that therefore destroy each other's binary. Then sweeps it with
--repeat 3 and checks every run still found something to execute.

Needs no compiler and no GPU: the "binaries" are shell scripts, and the harness
neither knows nor cares.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import appspecs                                        # noqa: E402
from appspecs import AppSpec                           # noqa: E402

MAKEFILE = """\
# Mimics krylov/Makefile: clean takes every binary in the directory, not just
# the one about to be rebuilt.
alpha beta:
\tprintf '#!/bin/sh\\necho "avg 1.5 ms"\\necho "answer 42.0"\\n' > $@.x
\tchmod +x $@.x

clean:
\trm -f *.x

.PHONY: alpha beta clean
"""


def parse(out):
    avg = ans = None
    for line in out.splitlines():
        if line.startswith("avg "):
            avg = float(line.split()[1])
        elif line.startswith("answer "):
            ans = float(line.split()[1])
    return {"avg_ms": avg, "answer": ans}


def main():
    root = Path(tempfile.mkdtemp(prefix="repeat-stash-"))
    appdir = root / "fakeapp"
    appdir.mkdir()
    (appdir / "Makefile").write_text(MAKEFILE)

    # evaluate.py builds with `make -C <directory>` from APPS_OPENMP, and runs
    # with cwd = APPS_OPENMP/<directory>, so the app has to live under it.
    import evaluate
    real = evaluate.APPS_OPENMP
    link = real / "_faketest"
    if link.is_symlink() or link.exists():
        (link.unlink() if link.is_symlink() else shutil.rmtree(link))
    link.symlink_to(appdir)

    appspecs.APPS.clear()
    appspecs.APPS["fake"] = AppSpec(
        name="fake",
        pretty="Fake",
        directory="_faketest",
        variants=["alpha", "beta"],          # two binaries, one directory
        make_target=lambda v: v,
        binary=lambda v: f"./{v}.x",
        run_args=lambda v, size, iters, cfg, grain, unroll: [],
        parse=parse,
        work=lambda n: (float(n), "n"),
        sizes=[1],
        iters=1,
    )
    evaluate.APPS = appspecs.APPS

    try:
        return run(evaluate, root, out=root / "results")
    finally:
        link.unlink(missing_ok=True)
        shutil.rmtree(root, ignore_errors=True)


def run(evaluate, root, out):
    """The sweep and its assertions, with the fixture cleaned up either way."""
    argv = ["evaluate.py", "--apps", "fake", "--opts", "jit", "--repeat", "3",
            "--outdir", str(out), "--no-stats", "--omit", "synchronous"]
    sys.argv = argv
    rc = evaluate.main()

    import csv
    rows = list(csv.DictReader(open(out / "runs.csv")))
    builds = 0
    ok = [r for r in rows if r["status"] == "ok"]
    reps = sorted({r["rep"] for r in rows})
    bad = [r for r in rows if r["status"] != "ok"]

    print()
    print(f"  rows={len(rows)} ok={len(ok)} repeats={reps} rc={rc}")
    for r in bad:
        print(f"  FAILED: rep={r['rep']} {r['config']} {r['variant']}: {r['status']}")

    # 2 variants x 2 configs (no-taskgraph, taskgraph:none, taskgraph:jit minus
    # the omitted synchronous) x 3 repeats
    assert rows, "no rows written"
    assert reps == ["1", "2", "3"], f"expected 3 repeats, got {reps}"
    assert not bad, f"{len(bad)} run(s) did not complete -- the stash did not hold"
    assert len(ok) == len(rows), "some run failed"
    per_rep = {r: len([x for x in rows if x["rep"] == r]) for r in reps}
    assert len(set(per_rep.values())) == 1, f"uneven repeats: {per_rep}"

    # The stash is removed once a sweep completes cleanly.
    assert not (out / ".binstash").exists(), "stash left behind after a clean sweep"

    print(f"  PASS: {per_rep} runs per repeat, all ok, stash cleaned up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
