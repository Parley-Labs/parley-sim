"""Judge-free metrics for one support-desk episode, from the focal seat's view.

Kinds: `intrinsic` is visible from the focal seat's own
actions given what it faced; `relational` only exists through the
interaction; `covariate` is recorded so differences can be attributed.
"""
from __future__ import annotations

from typing import Any

METRIC_KINDS = {
    "deal": "relational",
    "joint_efficiency": "relational",
    "focal_surplus_share": "relational",
    "turns": "relational",
    "accepted_below_walkaway": "intrinsic",
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
    }


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
