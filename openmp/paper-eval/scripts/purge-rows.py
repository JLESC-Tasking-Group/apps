#!/usr/bin/env python3
"""purge-rows.py <results dir> --apps a,b:size:size [--configs c1;c2] [--tag T]

Remove rows from <dir>/runs.csv, and the cgstats.csv / jitstats.csv rows that
join on them (run_id == tag), before those runs are redone. Keeps a copy of
each file as <file>.bak-<timestamp>. --configs takes evaluate.py config labels
separated by ';' (e.g. 'taskgraph:reduce-node,transitive-reduction'); --tag
selects the runs.csv tag column ('' = the untagged main sweep, the default).
An --apps entry `app:s1:s2` removes only those sizes of the app."""
import argparse, csv, shutil, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--apps", required=True)
ap.add_argument("--configs", default=""); ap.add_argument("--tag", default="")
a = ap.parse_args()
d = Path(a.dir); apps = {}
for e in a.apps.split(","):
    n, *sz = e.split(":"); apps[n] = set(sz); cfgs = set(filter(None, a.configs.split(";")))
ts = time.strftime("%Y%m%d-%H%M%S")

def rewrite(path, keep):
    if not path.exists():
        return 0, 0
    with open(path, newline="") as fh:
        r = csv.reader(fh); head = next(r); rows = list(r)
    kept = [x for x in rows if keep(dict(zip(head, x)))]
    shutil.copy2(path, f"{path}.bak-{ts}")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(kept)
    return len(rows), len(rows) - len(kept)

gone = set()
def keep_run(r):
    hit = (r["app"] in apps and (not apps[r["app"]] or r["size"] in apps[r["app"]])
           and r["tag"] == a.tag and (not cfgs or r["config"] in cfgs))
    if hit:
        gone.add(r["run_id"])
    return not hit
n, k = rewrite(d / "runs.csv", keep_run); print(f"runs.csv: {k} of {n} rows removed")
for f in ("cgstats.csv", "jitstats.csv"):
    n, k = rewrite(d / f, lambda r: r.get("tag") not in gone); print(f"{f}: {k} of {n} rows removed")
