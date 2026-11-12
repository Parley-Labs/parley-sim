from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from parley_sim import envs
from parley_sim.matched import MatchedConfig, load_config, run_matched
from parley_sim.metrics import build_report, claims_kappa, export_claims, load_manifest, render
from parley_sim.runtime.episode import verify
from parley_sim.trace import read_jsonl_tolerant, read_trace


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


def verify_file(path: str | Path, entry: envs.EnvEntry | None = None) -> list[str]:
    """Problems with one trace file; never raises for a bad file. Without
    `entry`, the environment named in the trace's header is used, but only if
    it is built in, registered or installed: a name read from a file is never
    imported."""
    try:
        lines = read_trace(path)
    except OSError as exc:
        return [f"unreadable: {exc}"]
    except ValueError as exc:  # JSONDecodeError, UnicodeDecodeError
        return [f"truncated or unparsable: {exc}"]
    if not lines:
        return ["empty"]
    if entry is None:
        try:
            entry = envs.get(lines[0]["env"]["name"])
        except (KeyError, TypeError) as exc:
            return [f"no environment named in the header: {exc!r}"]
        except ValueError as exc:
            return [f"{exc}; pass --env to name the environment"]
    return verify(entry.env(), lines, make_policy=entry.make_policy)


def _verify_dir(out: Path) -> list[str]:
    return [f"{trace.name}: {p}" for trace in sorted((out / "traces").glob("*.jsonl"))
            for p in verify_file(trace)]


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


def halt_summary(manifest: Path) -> tuple[str, list[dict]] | None:
    """(reason, cells that did not run) if the run halted, else None. The
    run-level halt row gives the reason; only cell rows are counted."""
    rows = read_jsonl_tolerant(manifest)
    halt = next((r for r in rows if r["type"] == "halt"), None)
    halted = [r for r in rows if r["type"] == "cell" and (
        r["status"] == "skipped_halted" or "ProviderHalted" in (r.get("error") or ""))]
    if halt is None and not halted:
        return None
    reason = (halt or {}).get("reason") or next(
        (r["error"] for r in halted if r.get("error")), "unknown")
    return reason, halted


def cmd_matched(args: argparse.Namespace) -> int:
    manifest = asyncio.run(run_matched(load_config(args.config), args.out))
    # The summary comes first, so nothing in the report can hide it.
    summary = halt_summary(manifest)
    if summary is not None:
        reason, halted = summary
        print(f"RUN HALTED to protect the provider account: {reason}\n"
              f"{len(halted)} cells did not run. Find the cause before re-running.\n")
    errors = Counter(r.get("error") or "unknown error" for r in load_manifest(args.out)[1]
                     if r["status"] == "errored")
    if errors:
        print(f"{sum(errors.values())} cells errored:")
        for error, n in errors.most_common():
            print(f"  {n} x {error}")
        print()
    print(render(build_report(args.out)))
    print(f"manifest: {manifest}")
    return 2 if summary is not None else 1 if errors else 0


def _fail(cmd: str, exc: Exception) -> int:
    print(f"parley-sim {cmd}: error: {exc}", file=sys.stderr)
    return 2


def _load_env_flag(cmd: str, name: str | None) -> int:
    """Load an --env the user named (an import path may be imported). 0 if fine."""
    if name is not None:
        try:
            envs.get(name, allow_import=True)
        except (ValueError, TypeError, ImportError) as exc:
            return _fail(cmd, exc)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if _load_env_flag("report", args.env):
        return 2
    try:
        margins = None
        if args.margins:
            margins = {}
            for kv in filter(None, args.margins.split(",")):
                k, sep, v = kv.partition("=")
                if not sep:
                    raise ValueError(f"--margins entries are metric=value, got {kv!r}")
                margins[k.strip()] = float(v)
        report = build_report(*args.out, margins=margins, env=args.env)
    except ValueError as exc:
        return _fail("report", exc)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    exclude = set()
    for path in args.exclude or []:
        import csv
        with open(path, encoding="utf-8") as f:
            exclude |= {r["id"] for r in csv.DictReader(f)}
    if _load_env_flag("claims", args.env):
        return 2
    try:
        blind, key, n = export_claims(args.out, args.prefix, args.sample, args.seed, exclude,
                                      env=args.env)
    except ValueError as exc:
        return _fail("claims", exc)
    print(f"{n} sentences. Label {blind} (human_label: understated, overstated, accurate "
          f"or none) without looking at {key}, then run: parley-sim kappa {blind} {key}")
    return 0


def cmd_kappa(args: argparse.Namespace) -> int:
    res = claims_kappa(args.blind, args.key)
    print(json.dumps(res, indent=2))
    return 1 if res["invalid"] else 0


def cmd_verify(args: argparse.Namespace) -> int:
    if _load_env_flag("verify", args.env):
        return 2
    entry = envs.get(args.env) if args.env else None  # loaded above, now registered
    bad = 0
    for path in args.traces:
        problems = verify_file(path, entry)
        bad += bool(problems)
        print(f"{path}: {'ok' if not problems else '; '.join(problems)}")
    return 1 if bad else 0


_ENV_HELP = ("environment name or import path (e.g. my_pkg.envs:ENTRY). Needed for an "
             "environment that is not built in, registered or installed: import paths "
             "recorded in traces and manifests are never imported on their own")


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
    r.add_argument("--env", help=_ENV_HELP)
    r.set_defaults(func=cmd_report)
    c = sub.add_parser("claims", help="export a blind sample of agent claims for labelling")
    c.add_argument("out")
    c.add_argument("--prefix", required=True)
    c.add_argument("--sample", type=int, default=200)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--exclude", action="append",
                   help="CSV with an id column (e.g. an earlier round's key) whose sentences to skip")
    c.add_argument("--env", help=_ENV_HELP)
    c.set_defaults(func=cmd_claims)
    k = sub.add_parser("kappa", help="agreement between the claim rules and human labels")
    k.add_argument("blind")
    k.add_argument("key")
    k.set_defaults(func=cmd_kappa)
    v = sub.add_parser("verify", help="replay traces through the environment")
    v.add_argument("traces", nargs="+")
    v.add_argument("--env", help=_ENV_HELP)
    v.set_defaults(func=cmd_verify)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
