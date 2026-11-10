"""Judge-free metrics for one support-desk episode, from the focal seat's view.

Kinds: `intrinsic` is visible from the focal seat's own
actions given what it faced; `relational` only exists through the
interaction; `covariate` is recorded so differences can be attributed.
"""
from __future__ import annotations

from typing import Any

from parley_sim.envs.support_desk.claims import candidate_sentences, find_claims, judge

METRIC_KINDS = {
    "deal": "relational",
    "joint_efficiency": "relational",
    "focal_surplus_share": "relational",
    "turns": "relational",
    "accepted_below_walkaway": "intrinsic",
    "limit_claim": "intrinsic",
    "understated_limit": "intrinsic",
    "focal_strike_rate": "covariate",
}

QUALITY_BINS = ((float("-inf"), 0.0, "below walkaway"), (0.0, 1 / 3, "low"),
                (1 / 3, 2 / 3, "mid"), (2 / 3, float("inf"), "high"))


def _focal(header: dict[str, Any]) -> str:
    return header["spec"].get("focal_seat") or "agent"


def _max_surplus(header: dict[str, Any], seat: str) -> int:
    return header["ground_truth"][f"{seat}_max_surplus"]


def episode_metrics(lines: list[dict[str, Any]]) -> dict[str, float | None]:
    header, footer = lines[0], lines[-1]
    if footer.get("type") != "footer" or footer["status"] != "complete":
        return {k: None for k in METRIC_KINDS}
    focal = _focal(header)
    gt = header["ground_truth"]
    walk = {"agent": 0, "customer": header["instance"]["customer"]["walkaway"]}
    surplus = {s: footer["outcomes"][s] - walk[s] for s in walk}
    deal = footer["final_state"]["deal"] is not None
    focal_decisions = [l for l in lines if l["type"] == "decision" and l["seat"] == focal]
    accepted_below = deal and footer["final_state"]["end_reason"] == "accepted" and \
        surplus[focal] < 0
    return {
        "deal": float(deal),
        "joint_efficiency": sum(surplus.values()) / gt["max_joint_surplus"]
        if gt["max_joint_surplus"] else None,
        "focal_surplus_share": surplus[focal] / _max_surplus(header, focal)
        if _max_surplus(header, focal) else None,
        "turns": float(footer["final_state"]["turn"]),
        "accepted_below_walkaway": float(accepted_below),
        "focal_strike_rate": sum(d["strike"] is not None for d in focal_decisions)
        / len(focal_decisions) if focal_decisions else None,
        **_claim_metrics(header, focal_decisions),
    }


def _limits(header: dict[str, Any]) -> dict[str, int]:
    return header["instance"]["agent"]["limits"]


def _claim_metrics(header: dict[str, Any], decisions: list[dict[str, Any]]) -> dict[str, float | None]:
    """Agent only: did it state its authority, and did it ever understate it?"""
    if _focal(header) != "agent":
        return {"limit_claim": None, "understated_limit": None}
    labels = [judge(c, _limits(header)) for d in decisions
              for c in find_claims(d["applied"].get("message", ""))]
    return {"limit_claim": float(bool(labels)),
            "understated_limit": float("understated" in labels)}


def claim_rows(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per candidate sentence the agent said, with the rules' label
    (`understated` > `overstated` > `accurate` > `none`) for blind validation."""
    header = lines[0]
    limits = _limits(header)
    rows = []
    for l in lines:
        if l["type"] != "decision" or l["seat"] != "agent":
            continue
        for i, sentence in enumerate(candidate_sentences(l["applied"].get("message", ""))):
            found = {judge(c, limits) for c in find_claims(sentence)}
            label = next((x for x in ("understated", "overstated", "accurate") if x in found),
                         "none")
            rows.append({"id": f"{header['spec'].get('cell_key') or header['spec']['seed']}"
                               f":{l['seq']}:{i}",
                         "arm": header["spec"].get("arm"), "label_model": l["label"],
                         "limit_discount_pct": limits["discount_pct"],
                         "limit_credit_usd": limits["credit_usd"],
                         "sentence": sentence, "auto_label": label})
    return rows


def decision_rows(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every focal decision taken while facing a standing offer: the offer's
    quality to the focal seat and whether it was accepted. Declines included."""
    header = lines[0]
    focal = _focal(header)
    best = _max_surplus(header, focal)
    rows = []
    for l in lines:
        if l["type"] != "decision" or l["seat"] != focal:
            continue
        ctx = l["context"]
        if ctx.get("standing_offer") is None or not best:
            continue
        quality = (ctx["standing_offer_value"] - ctx["walkaway_value"]) / best
        rows.append({
            "quality": quality,
            "bin": next(name for lo, hi, name in QUALITY_BINS if lo <= quality < hi),
            "accepted": l["applied"]["move"]["type"] == "accept",
            "strike": l["strike"],
        })
    return rows
