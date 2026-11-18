"""Copy run directories into results/ for publication, leaving out the traces
of arms whose counterparts were conditioned on licensed persona data.

    python scripts/curate_results.py SRC_RUNS_DIR results --withhold-arm usersim

For each run: the published report (report.md, report.json, joint-report.md),
the claim-validation folders, and the traces of every arm not withheld are
copied. The manifest keeps its run row and the rows of published cells; the
withheld arm's rows are dropped, so `parley-sim report` on the copy works and
reproduces every comparison that does not involve that arm. The original
report, with the withheld arm, is kept as published.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

RUNS = ("equiv-sd3", "family-sd4")
KEEP_FILES = ("report.md", "report.json", "joint-report.md")


def curate(src: Path, dst: Path, withhold: set[str]) -> dict[str, int]:
    dst.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (src / "manifest.jsonl").read_text().splitlines()]
    kept, dropped = [], 0
    for row in rows:
        if row["type"] == "cell" and row.get("arm") in withhold:
            dropped += 1
            continue
        kept.append(row)
        if row["type"] == "cell" and row.get("trace"):
            (dst / row["trace"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / row["trace"], dst / row["trace"])
    (dst / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                                                for r in kept))
    for name in KEEP_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    for folder in src.glob("claim-validation-*"):
        shutil.copytree(folder, dst / folder.name, dirs_exist_ok=True)
    return {"kept_cells": sum(r["type"] == "cell" for r in kept), "withheld_cells": dropped}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--withhold-arm", action="append", default=[])
    args = p.parse_args()
    for run in RUNS:
        print(run, curate(Path(args.src) / run, Path(args.dst) / run, set(args.withhold_arm)))


if __name__ == "__main__":
    main()
