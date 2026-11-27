from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from parley_sim import envs
from parley_sim.matched import MatchedConfig, load_config, run_matched
from parley_sim.metrics import (
    VERDICT_RULES,
    build_report,
    claims_kappa,
    export_claims,
    load_manifest,
    render,
)
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
    except (ValueError, RecursionError) as exc:  # JSONDecodeError, UnicodeDecodeError, deep nesting
        return [f"truncated or unparsable: {type(exc).__name__}: {str(exc)[:200]}"]
    if not lines:
        return ["empty"]
    if entry is None:
        try:
            entry = envs.get(lines[0]["env"]["name"])
        except (KeyError, TypeError) as exc:
            return [f"no environment named in the header: {exc!r}"]
        except ValueError as exc:
            return [f"{exc}; pass --env to name the environment"]
    try:
        return verify(entry.env(), lines, make_policy=entry.make_policy)
    except RecursionError:
        return ["nested too deeply to check"]


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
    rows = load_manifest(args.out)[1]
    errors = Counter(r.get("error") or "unknown error" for r in rows if r["status"] == "errored")
    over_budget = sum(r["status"] == "skipped_budget" for r in rows)
    if over_budget:
        print(f"{over_budget} cells did not run: the token cap (max_total_tokens) was reached.\n")
    if errors:
        print(f"{sum(errors.values())} cells errored:")
        for error, n in errors.most_common():
            print(f"  {n} x {error}")
        print()
    print(render(build_report(args.out)))
    print(f"manifest: {manifest}")
    return 2 if summary is not None else 1 if errors or over_budget else 0


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
        report = build_report(*args.out, margins=margins, env=args.env, rule=args.verdict_rule)
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


def cmd_harbor_export(args: argparse.Namespace) -> int:
    """Experimental: write Harbor tasks, one per cell of a matched config's grid."""
    try:
        from parley_sim.adapters.harbor import export_task, harbor_cells
    except ImportError as exc:
        print(f"harbor-export needs the harbor extra: {exc}", file=sys.stderr)
        return 2
    from parley_sim.matched import build_grid, config_hash

    cfg = load_config(args.config)
    if (args.wheel is None) == (args.image is None):
        print("give exactly one of --wheel or --image", file=sys.stderr)
        return 2
    wheel = Path(args.wheel) if args.wheel else None
    if wheel is not None and (not wheel.is_file() or wheel.suffix != ".whl"):
        print(f"--wheel {wheel}: not a wheel file (build one with `uv build --wheel`)", file=sys.stderr)
        return 2
    secret = None
    if args.seed_secret_env:
        secret = os.environ.get(args.seed_secret_env)
        if not secret:
            print(f"{args.seed_secret_env} is not set", file=sys.stderr)
            return 2
    try:
        cells = harbor_cells(build_grid(cfg), run_id=cfg.run_id, env=cfg.env,
                             env_config=cfg.env_config, lineup=cfg.lineup,
                             reward_metric=args.reward_metric)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    chash = config_hash(cfg)
    for cell in cells:
        export_task(Path(args.out), cell, chash=chash, wheel=wheel, image=args.image,
                    seed_secret=secret, extra_ca=Path(args.extra_ca) if args.extra_ca else None)
    print(f"{len(cells)} tasks in {args.out}")
    return 0


def cmd_harbor_collect(args: argparse.Namespace) -> int:
    """Experimental: gather Harbor trials into a run directory for `report`."""
    from parley_sim.adapters.harbor import collect_jobs
    from parley_sim.matched import config_hash

    cfg = load_config(args.config)
    counts = collect_jobs(cfg.model_dump(), config_hash(cfg), [Path(d) for d in args.jobs],
                          Path(args.out))
    print(", ".join(f"{k}: {v}" for k, v in counts.items()))
    return 0 if counts["kept"] else 1


def cmd_harbor_sweep(args: argparse.Namespace) -> int:
    """Experimental: persona sweeps over Harbor's simulated-user trials."""
    from parley_sim.adapters import harbor_sweep as hs

    if args.sweep_cmd == "personas":
        try:
            paths = hs.write_personas(args.panel, args.out, args.n)
        except (ImportError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"{len(paths)} persona files in {args.out} (they hold persona data: keep them private)")
        return 0
    if args.sweep_cmd == "plan":
        personas: list[str | None] = [None] if args.include_default else []
        for d in args.personas:
            personas += sorted(str(p) for p in Path(d).glob("*.md"))
        if not personas:
            print("no personas: pass --personas DIR and/or --include-default", file=sys.stderr)
            return 2
        sys.stdout.write(hs.write_plan(args.tasks, personas, agent=args.agent, model=args.model,
                                       user_agent=args.user_agent, user_model=args.user_model,
                                       jobs=args.jobs, replicates=args.replicates))
        return 0
    rows = hs.collect([Path(d) for d in args.jobs], reward_key=args.reward_key)
    if not rows:
        print("no simulated-user trials found", file=sys.stderr)
        return 1
    report = hs.sweep_report(rows, baseline=args.baseline, margin=args.margin)
    print(json.dumps(report, indent=2, default=str) if args.json else hs.render_sweep(report), end="")
    return 0


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
    r.add_argument("--verdict-rule", choices=VERDICT_RULES, default="noise-floor",
                   help="noise-floor (default; the rule equiv-sd3 pre-registered) or margin "
                        "(replicates averaged per seed; the margin is the smallest effect "
                        "that counts as different)")
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
    h = sub.add_parser("harbor-export",
                       help="experimental: write Harbor tasks from a matched config (one per cell)")
    h.add_argument("config", help="matched-run YAML config (its focal_models are not used)")
    h.add_argument("--out", required=True, help="directory to write task directories into")
    src = h.add_mutually_exclusive_group(required=True)
    src.add_argument("--wheel", help="parley-sim wheel the runtime sidecar builds from")
    src.add_argument("--image", help="prebuilt runtime sidecar image (see integrations/harbor)")
    h.add_argument("--reward-metric", default="focal_surplus_share",
                   help="episode metric Harbor records as the reward (default: focal_surplus_share)")
    h.add_argument("--seed-secret-env", help="environment variable holding a seed secret, for "
                   "private exports: scenarios can't then be computed from the seed")
    h.add_argument("--extra-ca", help="CA certificate to trust inside the images (local builds "
                   "behind a TLS-intercepting proxy; don't use for tasks you publish)")
    h.set_defaults(func=cmd_harbor_export)
    hc = sub.add_parser("harbor-collect",
                        help="experimental: gather Harbor trials into a run directory for report")
    hc.add_argument("jobs", nargs="+", help="Harbor job or trials directories")
    hc.add_argument("--config", required=True, help="the config the tasks were exported from")
    hc.add_argument("--out", required=True, help="run directory to write")
    hc.set_defaults(func=cmd_harbor_collect)
    hs = sub.add_parser("harbor-sweep",
                        help="experimental: persona sweeps over Harbor's simulated-user trials")
    hsub = hs.add_subparsers(dest="sweep_cmd", required=True)
    hp = hsub.add_parser("personas", help="persona files from a UserSim panel, plus minimal.md")
    hp.add_argument("--panel", required=True, help="UserSim panel parquet (or JSONL)")
    hp.add_argument("--n", type=int, required=True, help="how many personas")
    hp.add_argument("--out", required=True, help="directory for the persona files (keep private)")
    hl = hsub.add_parser("plan", help="print a shell script of harbor runs, one per task and persona")
    hl.add_argument("--tasks", nargs="+", required=True, help="Harbor task directories")
    hl.add_argument("--personas", nargs="*", default=[], help="directories of persona .md files")
    hl.add_argument("--include-default", action="store_true", help="also run Harbor's default persona")
    hl.add_argument("--agent", required=True)
    hl.add_argument("--model", required=True)
    hl.add_argument("--user-agent", required=True)
    hl.add_argument("--user-model", required=True)
    hl.add_argument("--jobs", required=True, help="Harbor jobs directory")
    hl.add_argument("--replicates", type=int, default=1, help="attempts per task and persona")
    hr = hsub.add_parser("report", help="compare persona arms on the tasks' own rewards")
    hr.add_argument("jobs", nargs="+", help="Harbor job or trial directories")
    hr.add_argument("--baseline", default="harbor-default")
    hr.add_argument("--margin", type=float, help="smallest delta that counts (enables verdicts)")
    hr.add_argument("--reward-key", default="reward")
    hr.add_argument("--json", action="store_true")
    hs.set_defaults(func=cmd_harbor_sweep)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
