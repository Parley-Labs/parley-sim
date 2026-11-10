from __future__ import annotations

import json
from typing import Any

from parley_sim.core.protocol import JSON, MalformedOutput

PROMPT_VERSION = "sd-1"

_DEAL_SCHEMA: JSON = {
    "type": "object",
    "properties": {
        "discount_pct": {"enum": [0, 10, 20, 30, 40]},
        "credit_usd": {"enum": [0, 25, 50, 100]},
        "upgrade": {"enum": ["none", "premium"]},
        "term": {"enum": ["monthly", "annual"]},
    },
    "required": ["discount_pct", "credit_usd", "upgrade", "term"],
    "additionalProperties": False,
}

# Reasoning is ordered before the binding move on purpose: models that emit the
# move first tend to rationalize it afterwards instead of deciding.
ACTION_SCHEMA: JSON = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string",
                      "description": "Private. Your counterpart never sees this."},
        "message": {"type": "string",
                    "description": "What you say to your counterpart this turn."},
        "move": {
            "type": "object",
            "properties": {
                "type": {"enum": ["propose", "accept", "walk_away", "none"]},
                "deal": _DEAL_SCHEMA,
            },
            "required": ["type"],
            "additionalProperties": False,
        },
    },
    "required": ["reasoning", "message", "move"],
    "additionalProperties": False,
}

_ROLE = {
    "agent": (
        "You are a customer retention agent for a software subscription company. "
        "A customer has contacted support wanting to cancel. Your job is to negotiate "
        "a retention package that keeps the customer while protecting the company's "
        "interests. You can only offer or accept packages within your authorization limits."
    ),
    "customer": (
        "You are a customer of a software subscription company and you are considering "
        "cancelling. You have a competing alternative. You are talking to a retention agent; "
        "negotiate the best package you can, or leave if staying is not worth it."
    ),
}

_RULES = (
    "How the negotiation works:\n"
    "- You and your counterpart take turns. Each turn you send a message and make one move.\n"
    "- Moves: `propose` a full package (all four terms), `accept` your counterpart's "
    "latest proposal, `walk_away` (ends the conversation with no deal), or `none` "
    "(message only).\n"
    "- Accepting ends the negotiation with that package. Proposals are binding offers; "
    "messages are not.\n"
    "- If the turns run out without an accepted package, there is no deal.\n"
    "- Your private information below is known only to you. Your score is the value "
    "of the final outcome to you."
)


def _private_block(obs: JSON) -> str:
    priv = obs["private"]
    lines = ["Your private information:"]
    if obs["role"] == "agent":
        lines.append(f"- Value to the company if the customer stays on default terms: "
                     f"{priv['base_value_if_customer_stays']}")
        lines.append(f"- Value to the company if the customer leaves: {priv['value_if_no_deal']}")
        lim = priv["authorization_limits"]
        lines.append(f"- Authorization limits: at most {lim['discount_pct']}% discount and at "
                     f"most ${lim['credit_usd']} credit.")
        lines.append("- Change in value for each term (added to the default value):")
    else:
        lines.append(f"- Value to you of cancelling and switching: {priv['value_if_no_deal']}")
        lines.append("- Value to you of each term if you stay (values add up):")
    for issue, opts in priv["option_values"].items():
        lines.append(f"  - {issue}: " + ", ".join(f"{o} -> {v}" for o, v in opts.items()))
    return "\n".join(lines)


class SupportDeskText:
    prompt_version = PROMPT_VERSION
    action_schema = ACTION_SCHEMA

    def render(self, observation: JSON) -> tuple[str, str]:
        system = "\n\n".join([_ROLE[observation["role"]], _RULES, _private_block(observation)])
        convo = []
        for e in observation["transcript"]:
            who = "You" if e["seat"] == observation["role"] else "Counterpart"
            move = e["move"]
            detail = f" {json.dumps(move['deal'])}" if move["type"] == "propose" else ""
            convo.append(f"[{who}] \"{e['message']}\" (move: {move['type']}{detail})")
        theirs = observation["their_standing_offer"]
        user = "\n".join([
            "Conversation so far:" if convo else "You speak first.",
            *convo,
            "",
            f"Counterpart's latest proposal: {json.dumps(theirs['deal'])} "
            f"(value to you: {theirs['value_to_you']})" if theirs
            else "Counterpart's latest proposal: none",
            f"Turns you have left, including this one: {observation['your_moves_left']}",
            "",
            "Respond with a JSON object matching this schema:",
            json.dumps(ACTION_SCHEMA),
        ])
        return system, user

    def parse(self, payload: Any) -> tuple[JSON, str | None]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise MalformedOutput(f"not JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise MalformedOutput("output must be a JSON object")
        move = payload.get("move")
        if not isinstance(move, dict) or not isinstance(move.get("type"), str):
            raise MalformedOutput("missing move.type")
        message = payload.get("message", "")
        if not isinstance(message, str):
            raise MalformedOutput("message must be a string")
        reasoning = payload.get("reasoning")
        action: JSON = {"message": message, "move": {"type": move["type"]}}
        if move["type"] == "propose":
            action["move"]["deal"] = move.get("deal")
        return action, reasoning if isinstance(reasoning, str) else None
