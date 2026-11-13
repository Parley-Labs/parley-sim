from __future__ import annotations

import json
from typing import Any

from parley_sim.core.protocol import JSON, MalformedOutput
from parley_sim.envs.support_desk import scenario as sc

PROMPT_VERSION = "sd-2"

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

# No field asks the model to write out its reasoning: current Claude models
# decline that as reasoning extraction. Deliberation is read from the provider's
# own thinking summaries; the prompt asks only for a one-sentence explanation.
ACTION_SCHEMA: JSON = {
    "type": "object",
    "properties": {
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
        "explanation": {"type": "string",
                        "description": "One sentence on why you chose this move."},
    },
    "required": ["message", "move", "explanation"],
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


def _describe_entry(e: JSON) -> str:
    move = e["move"]
    detail = f" {json.dumps(move['deal'])}" if move["type"] == "propose" else ""
    return f"\"{e['message']}\" (move: {move['type']}{detail})"


def _same_move(entry: JSON, submitted: JSON) -> bool:
    """Whether a recorded transcript entry is the action the seat submitted."""
    move = submitted.get("move") or {}
    if move.get("type") != entry["move"]["type"]:
        return False
    if move.get("type") == "propose":
        try:
            if sc.deal_to_json(sc.deal_from_json(move.get("deal"))) != entry["move"]["deal"]:
                return False
        except (ValueError, TypeError):
            return False
    # Compared in full: a message the environment truncated counts as changed.
    return entry["message"] == str(submitted.get("message") or "")


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

    def render_update(self, previous: JSON, observation: JSON,
                      submitted: JSON | None = None) -> str:
        """The next user message of a conversation seat: what happened since
        its previous turn, its own standing proposal, the counterpart's, and
        the turns left.

        The seat's own replies are already in the conversation, so its own
        moves are restated only when the record differs from what it last
        submitted: an illegal move replaced by the fallback, a normalised or
        truncated action, or a turn with no usable reply (`submitted` None)."""
        new = observation["transcript"][len(previous["transcript"]):]
        own = [e for e in new if e["seat"] == observation["role"]]
        lines = []
        for e in new:
            if e["seat"] != observation["role"]:
                lines.append(f"[Counterpart] {_describe_entry(e)}")
            elif e is not own[-1]:
                lines.append(f"[You] Recorded for you on an earlier turn: {_describe_entry(e)}")
            elif submitted is None:
                lines.append("[You] Your last turn produced no usable reply, so it was recorded "
                             f"as: {_describe_entry(e)}")
            elif not _same_move(e, submitted):
                lines.append("[You] Your last reply could not be applied as sent (an illegal "
                             "move becomes no move, and a long message is cut short), so it "
                             f"was recorded as: {_describe_entry(e)}")
        theirs, mine = observation["their_standing_offer"], observation["your_standing_offer"]
        return "\n".join([
            *(lines or ["The counterpart has not moved since your last turn."]),
            "",
            f"Your standing proposal: {json.dumps(mine['deal'])}" if mine
            else "Your standing proposal: none",
            f"Counterpart's latest proposal: {json.dumps(theirs['deal'])} "
            f"(value to you: {theirs['value_to_you']})" if theirs
            else "Counterpart's latest proposal: none",
            f"Turns you have left, including this one: {observation['your_moves_left']}",
        ])

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
        explanation = payload.get("explanation")
        action: JSON = {"message": message, "move": {"type": move["type"]}}
        if move["type"] == "propose":
            action["move"]["deal"] = move.get("deal")
        return action, explanation if isinstance(explanation, str) else None
