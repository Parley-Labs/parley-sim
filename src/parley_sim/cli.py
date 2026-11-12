from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from parley_sim import envs
from parley_sim.matched import MatchedConfig, load_config, run_matched
from parley_sim.metrics import build_report, claims_kappa, export_claims, render
from parley_sim.runtime.episode import verify
from parley_sim.trace import read_trace


def smoke_config(seeds: str) -> MatchedConfig:
    """Scripted policies on every seat, played through the model path, so the
    whole pipeline runs offline. The AOC arm here is not adaptive."""
    return MatchedConfig(
        run_id="smoke", seeds=seeds,
        lineup=[{"name": "scripted-agent", "provider": "scripted", "model": "agent_scripted"},
                {"name": "scripted-customer", "provider": "scripted", "model": "conceder"}],
        focal_models=["scripted-agent"],
        scripted_counterparts=["hardliner", "reasonable", "conceder"],
        adaptive_counterparts=["scripted-customer"],
    )


def _verify_dir(out: Path) -> list[str]:
    problems = []
    for trace in sorted((out / "traces").glob("*.jsonl")):
        lines = read_trace(trace)
        env = envs.get(lines[0]["env"]["name"]).env()
        problems += [f"{trace.name}: {p}" for p in verify(env, lines)]
    return problems


def cmd_smoke(args: argparse.Namespace) -> int:
    out = Path(args.out or tempfile.mkdtemp(prefix="parley-sim-smoke-"))
    manifest = asyncio.run(run_matched(smoke_config(args.seeds), out))
    problems = _verify_dir(out)
    report = build_report(out)
    print(render(report))
    errored = report["cells"].get("errored", 0)
    print(f"traces: {out / 'traces'}")
    print(f"replay check: {'ok' if not problems else problems}")
    print(f"manifest: {manifest}")
    return 1 if problems or errored else 0


def cmd_matched(args: argparse.Namespace) -> int:
    manifest = asyncio.run(run_matched(load_config(args.config), args.out))
    print(render(build_report(args.out)))
    print(f"manifest: {manifest}")
    halted = [json.loads(l) for l in manifest.read_text().splitlines()
              if '"skipped_halted"' in l or "ProviderHalted" in l or '"type": "halt"' in l]
    if halted:
        reason = next((r.get("reason") or r.get("error") for r in halted
                       if r.get("reason") or r.get("error")), "unknown")
        print(f"\nRUN HALTED to protect the provider account: {reason}\n"
              f"{len(halted)} cells did not run. Find the cause before re-running.")
        return 2
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    margins = dict((k, float(v)) for k, v in (kv.split("=") for kv in args.margins.split(",")
                                              if kv)) if args.margins else None
    report = build_report(*args.out, margins=margins)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    exclude = set()
    for path in args.exclude or []:
        import csv
        with open(path) as f:
            exclude |= {r["id"] for r in csv.DictReader(f)}
    blind, key, n = export_claims(args.out, args.prefix, args.sample, args.seed, exclude)
    print(f"{n} sentences. Label {blind} (human_label: understated, overstated, accurate "
          f"or none) without looking at {key}, then run: parley-sim kappa {blind} {key}")
    return 0


def cmd_kappa(args: argparse.Namespace) -> int:
    res = claims_kappa(args.blind, args.key)
    print(json.dumps(res, indent=2))
    return 1 if res["invalid"] else 0


def cmd_verify(args: argparse.Namespace) -> int:
    bad = 0
    for path in args.traces:
        lines = read_trace(path)
        problems = verify(envs.get(args.env or lines[0]["env"]["name"]).env(), lines)
        bad += bool(problems)
        print(f"{path}: {'ok' if not problems else '; '.join(problems)}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="parley-sim")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="offline end-to-end check, no API keys")
    s.add_argument("--seeds", default="0:6")
    s.add_argument("--out")
    s.set_defaults(func=cmd_smoke)
    m = sub.add_parser("matched", help="run a matched SOC/AOC grid from a YAML config")
    m.add_argument("config")
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_matched)
    r = sub.add_parser("report", help="paired report across arms for one or more run directories")
    r.add_argument("out", nargs="+")
    r.add_argument("--margins", help="equivalence margins, e.g. focal_surplus_share=0.08,turns=0.8")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_report)
    c = sub.add_parser("claims", help="export a blind sample of agent claims for labelling")
    c.add_argument("out")
    c.add_argument("--prefix", required=True)
    c.add_argument("--sample", type=int, default=200)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--exclude", action="append",
                   help="CSV with an id column (e.g. an earlier round's key) whose sentences to skip")
    c.set_defaults(func=cmd_claims)
    k = sub.add_parser("kappa", help="agreement between the claim rules and human labels")
    k.add_argument("blind")
    k.add_argument("key")
    k.set_defaults(func=cmd_kappa)
    v = sub.add_parser("verify", help="replay traces through the environment")
    v.add_argument("traces", nargs="+")
    v.add_argument("--env", help="environment name or import path, if not built in "
                                 "(e.g. my_pkg.envs:ENTRY)")
    v.set_defaults(func=cmd_verify)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
