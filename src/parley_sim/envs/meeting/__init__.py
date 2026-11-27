"""Meeting scheduling: two people with private calendars agree on a time.

The `assistant` (the focal seat) books a one-hour meeting for its manager with
an outside `contact` (the counterpart). Each side sees only its own calendar:
which of the week's 40 working hours are free, and how much it likes each free
hour. Both want the meeting to happen (no meeting is worth 0 to each), but
their preferences differ, so it is mixed-motive: cooperative about meeting,
competitive about when.

Moves: `propose` a slot you are free for, `accept` the other side's standing
proposal (if you are free then), `none` (message only), or `give_up`. Proposing
or accepting a slot you are busy for is illegal: each side knows its own
calendar. The episode ends at an acceptance, a give-up, or after `max_turns`.

Every outcome is exact: each side's value for the agreed slot, compared with
the best slot both are free for (`ground_truth`). No judge is involved.
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from parley_sim.core.protocol import JSON, IllegalAction, MalformedOutput, Seat

ASSISTANT, CONTACT = "assistant", "contact"
SEATS = (CONTACT, ASSISTANT)  # the contact opens: they asked for the meeting
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")
HOURS = tuple(range(9, 17))  # meetings start 09:00 to 16:00
SLOTS = tuple(f"{d} {h:02d}:00" for d in DAYS for h in HOURS)
MOVES = ("propose", "accept", "none", "give_up")
PROMPT_VERSION = "meeting-2"


@dataclass(frozen=True)
class Instance:
    seed: int
    max_turns: int
    # seat -> {slot: value 1..10} for its free slots; slots not listed are busy
    free: dict[str, dict[str, int]]


@dataclass(frozen=True)
class State:
    inst: Instance
    turn: int = 0
    proposals: tuple[tuple[str, str], ...] = ()  # (seat, slot), in order
    transcript: tuple[tuple[str, str], ...] = ()  # (seat, message)
    slot: str | None = None  # the agreed slot
    end_reason: str | None = None  # accepted | gave_up | deadline


def _mutual(inst: Instance) -> list[str]:
    return [s for s in SLOTS if s in inst.free[ASSISTANT] and s in inst.free[CONTACT]]


class MeetingScheduling:
    name = "meeting_scheduling"
    version = "1"

    # --- the scenario -------------------------------------------------------
    def generate(self, seed: int, config: Mapping[str, Any]) -> Instance:
        """Each side is busy for about half the week; at least four hours are
        free for both. Preferences are drawn so the sides tend to disagree:
        the contact's value is pulled toward the opposite of the assistant's."""
        r = random.Random(seed)
        busy = float(config.get("busy", 0.5))
        while True:
            free_a = [s for s in SLOTS if r.random() >= busy]
            free_c = [s for s in SLOTS if r.random() >= busy]
            if len(set(free_a) & set(free_c)) >= 4:
                break
        va = {s: r.randint(1, 10) for s in free_a}
        vc = {}
        for s in free_c:
            opposite = 11 - va[s] if s in va else r.randint(1, 10)
            vc[s] = max(1, min(10, round(0.6 * opposite + 0.4 * r.randint(1, 10))))
        return Instance(seed=seed, max_turns=int(config.get("max_turns", 10)),
                        free={ASSISTANT: va, CONTACT: vc})

    def instance_to_json(self, inst: Instance) -> JSON:
        return {"seed": inst.seed, "max_turns": inst.max_turns,
                "free": {seat: dict(v) for seat, v in inst.free.items()}}

    def ground_truth(self, inst: Instance) -> JSON:
        mutual = _mutual(inst)
        joint = {s: inst.free[ASSISTANT][s] + inst.free[CONTACT][s] for s in mutual}
        return {"mutual_slots": mutual,
                "assistant_best": max(inst.free[ASSISTANT][s] for s in mutual),
                "contact_best": max(inst.free[CONTACT][s] for s in mutual),
                "max_joint": max(joint.values())}

    def seats(self, inst: Instance) -> tuple[Seat, ...]:
        return (ASSISTANT, CONTACT)

    # --- the game -----------------------------------------------------------
    def initial_state(self, inst: Instance) -> State:
        return State(inst=inst)

    def to_act(self, state: State) -> Seat | None:
        if state.end_reason is not None:
            return None
        return SEATS[state.turn % 2]

    def turn(self, state: State) -> int:
        return state.turn

    def _standing(self, state: State, seat: Seat) -> str | None:
        """The other side's latest proposal."""
        return next((s for who, s in reversed(state.proposals) if who != seat), None)

    def _mine(self, state: State, seat: Seat) -> str | None:
        return next((s for who, s in reversed(state.proposals) if who == seat), None)

    def observe(self, state: State, seat: Seat) -> JSON:
        mine = state.inst.free[seat]
        theirs = self._standing(state, seat)
        return {"role": seat,
                "calendar": {s: mine.get(s) for s in SLOTS},  # None = busy
                "their_proposal": theirs,
                "their_proposal_value": mine.get(theirs) if theirs else None,
                "my_last_proposal": self._mine(state, seat),
                "turns_left": state.inst.max_turns - state.turn,
                "transcript": [{"seat": s, "message": m} for s, m in state.transcript],
                "proposals": [{"seat": s, "slot": slot} for s, slot in state.proposals]}

    def decision_context(self, state: State, seat: Seat) -> JSON:
        theirs = self._standing(state, seat)
        return {"standing_proposal": theirs,
                "value_to_me": state.inst.free[seat].get(theirs) if theirs else None}

    def apply(self, state: State, seat: Seat, action: JSON) -> tuple[State, list[JSON]]:
        if seat != self.to_act(state):
            raise IllegalAction(f"it is not {seat}'s turn")
        if not isinstance(action, dict) or not isinstance(action.get("move"), dict):
            raise IllegalAction("an action is an object with a move object")
        move, message = action["move"], action.get("message", "")
        if not isinstance(message, str):
            raise IllegalAction("message must be a string")
        kind = move.get("type")
        mine = state.inst.free[seat]
        nxt = replace(state, turn=state.turn + 1, transcript=(*state.transcript, (seat, message[:2000])))
        if kind == "propose":
            slot = move.get("slot")
            if slot not in SLOTS:
                raise IllegalAction(f"slot must be one of the week's slots, like {SLOTS[0]!r}")
            if slot not in mine:
                raise IllegalAction(f"you are busy at {slot}")
            nxt = replace(nxt, proposals=(*state.proposals, (seat, slot)))
            applied = {"type": "propose", "slot": slot}
        elif kind == "accept":
            theirs = self._standing(state, seat)
            if theirs is None:
                raise IllegalAction("there is no proposal to accept")
            if theirs not in mine:
                raise IllegalAction(f"you are busy at {theirs}")
            nxt = replace(nxt, slot=theirs, end_reason="accepted")
            applied = {"type": "accept"}
        elif kind == "give_up":
            nxt = replace(nxt, end_reason="gave_up")
            applied = {"type": "give_up"}
        elif kind == "none":
            applied = {"type": "none"}
        else:
            raise IllegalAction(f"unknown move {kind!r} (one of {', '.join(MOVES)})")
        if nxt.end_reason is None and nxt.turn >= state.inst.max_turns:
            nxt = replace(nxt, end_reason="deadline")
        return nxt, [{"event_type": "move", "seat": seat, "move": applied,
                      "message": message[:2000]}]

    def fallback_action(self, state: State, seat: Seat) -> JSON:
        """Neutral: a turn with no move, so a malformed reply gives nothing away."""
        return {"message": "", "move": {"type": "none"}}

    def outcome(self, state: State) -> dict[Seat, float]:
        if state.slot is None:
            return {ASSISTANT: 0.0, CONTACT: 0.0}
        return {seat: float(state.inst.free[seat][state.slot]) for seat in (ASSISTANT, CONTACT)}

    def describe_end(self, state: State, seat: Seat) -> str:
        """How the episode ended, for the seat (said to tool-using agents)."""
        if state.slot is not None:
            return f"Booked: {state.slot} (value to you: {state.inst.free[seat][state.slot]:g})."
        return {"gave_up": "No meeting was booked: a side gave up.",
                "deadline": "No meeting was booked: the turns ran out."}.get(
                    state.end_reason or "", "No meeting was booked.")

    def state_to_json(self, state: State) -> JSON:
        return {"turn": state.turn, "slot": state.slot, "end_reason": state.end_reason,
                "proposals": [list(p) for p in state.proposals],
                "transcript": [list(t) for t in state.transcript]}


# --- scripted counterparts ----------------------------------------------------
class Scheduler:
    """Proposes its free slots in order of its own preference, and accepts a
    proposal worth at least a threshold that falls from `start` to `floor`
    over the episode. `pick` decides what to propose: its own favourite
    (`self`), or the favourite it hasn't proposed yet (`cycle`)."""

    def __init__(self, name: str, seat: Seat, start: int, floor: int, pick: str):
        self.name, self.seat = name, seat
        self.params = {"start": start, "floor": floor, "pick": pick}

    def decide(self, obs: JSON) -> JSON:
        free = {s: v for s, v in obs["calendar"].items() if v is not None}
        done = len(obs["transcript"])
        progress = done / max(done + obs["turns_left"] - 1, 1)
        need = self.params["start"] - (self.params["start"] - self.params["floor"]) * progress
        theirs, value = obs["their_proposal"], obs["their_proposal_value"]
        if theirs is not None and value is not None and value >= need:
            return {"message": f"{theirs} works for me.", "move": {"type": "accept"}}
        ranked = sorted(free, key=lambda s: (-free[s], SLOTS.index(s)))
        if self.params["pick"] == "cycle":
            asked = {p["slot"] for p in obs["proposals"] if p["seat"] == self.seat}
            ranked = [s for s in ranked if s not in asked] or ranked
        slot = ranked[0]
        if theirs is not None and value is None:
            text = f"I'm busy at {theirs}. How about {slot}?"
        else:
            text = f"How about {slot}?"
        return {"message": text, "move": {"type": "propose", "slot": slot}}


POLICIES = {
    "accommodating": dict(start=6, floor=1, pick="cycle"),
    "picky": dict(start=10, floor=6, pick="cycle"),
    "stubborn": dict(start=10, floor=8, pick="self"),
}


def make_policy(name: str, seat: Seat) -> Scheduler:
    if name == "assistant_scripted":  # a scripted focal seat, for offline runs
        return Scheduler(name, seat, start=8, floor=2, pick="cycle")
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r} (known: {sorted(POLICIES)}, assistant_scripted)")
    return Scheduler(name, seat, **POLICIES[name])


# --- how a model plays a seat --------------------------------------------------
ACTION_SCHEMA: JSON = {
    "type": "object", "additionalProperties": False, "required": ["message", "move"],
    "properties": {
        "message": {"type": "string", "description": "What you say to the other person this turn."},
        "move": {"type": "object", "additionalProperties": False, "required": ["type"],
                 "properties": {"type": {"enum": list(MOVES)},
                                "slot": {"enum": list(SLOTS),
                                         "description": "For propose: a slot you are free for."}}}}}

_ROLE = {
    ASSISTANT: ("You are an executive assistant booking a one-hour meeting for your manager with an "
                "outside contact. Your manager would much rather meet at some times than others; "
                "the values below say how much (10 is ideal). Book the best time you can, but a "
                "meeting at any time you are free beats no meeting."),
    CONTACT: ("You want a one-hour meeting with someone's manager and are arranging it with their "
              "assistant. Some times suit you far better than others; the values below say how "
              "much (10 is ideal). Get a time that suits you, but a meeting at any time you are "
              "free beats no meeting."),
}
_RULES = ("How it works: you take turns. Each turn you send a message and make one move: `propose` a "
          "slot you are free for, `accept` the other side's latest proposal (only if you are free "
          "then), `none` (message only), or `give_up` (no meeting). Accepting books exactly the "
          "slot shown as their latest proposal, whatever either message says. To book a slot they "
          "only mentioned in a message, `propose` it. "
          "Your calendar is private: the other side sees only what you say and propose.")


class MeetingText:
    prompt_version = PROMPT_VERSION
    action_schema = ACTION_SCHEMA

    def render(self, obs: JSON) -> tuple[str, str]:
        free = {s: v for s, v in obs["calendar"].items() if v is not None}
        days = []
        for d in DAYS:
            hours = [f"{s[4:]}={free[s]}" for s in SLOTS if s.startswith(d) and s in free]
            days.append(f"  {d}: " + (", ".join(hours) if hours else "busy all day"))
        system = "\n\n".join([_ROLE[obs["role"]], _RULES,
                              "Your free slots and how much you like each (other times you are busy):\n"
                              + "\n".join(days)])
        convo = [f"[{'You' if t['seat'] == obs['role'] else 'Other side'}] {t['message']}"
                 for t in obs["transcript"]]
        theirs = obs["their_proposal"]
        if theirs is None:
            standing = "Their latest proposal: none"
        elif obs["their_proposal_value"] is None:
            standing = f"Their latest proposal: {theirs} (you are busy then)"
        else:
            standing = f"Their latest proposal: {theirs} (value to you: {obs['their_proposal_value']})"
        user = "\n".join(["Conversation so far:" if convo else "You speak first.", *convo, "",
                          standing, f"Turns left, both sides together: {obs['turns_left']}", "",
                          "Respond with a JSON object matching this schema:",
                          json.dumps(ACTION_SCHEMA)])
        return system, user

    def parse(self, payload: Any) -> tuple[JSON, str | None]:
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError as exc:
            raise MalformedOutput(f"not JSON: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("move"), dict):
            raise MalformedOutput("expected an object with a move object")
        move = data["move"]
        if move.get("type") not in MOVES:
            raise MalformedOutput(f"move.type must be one of {', '.join(MOVES)}")
        clean = {"type": move["type"]}
        if move["type"] == "propose":
            clean["slot"] = move.get("slot")
        return {"message": str(data.get("message", "")), "move": clean}, None


# --- metrics --------------------------------------------------------------------
METRIC_KINDS = {
    "deal": "relational",
    "focal_surplus_share": "relational",
    "joint_efficiency": "relational",
    "turns": "relational",
    "focal_strike_rate": "covariate",
    "cp_opens_with_offer": "counterpart",
    "cp_first_offer_share": "counterpart",
}


def episode_metrics(lines: list[JSON]) -> dict[str, float | None]:
    """Shares are of the best mutually free slot: 1.0 means the agreed slot
    was the best that side could have had in a meeting both could attend."""
    header, footer = lines[0], lines[-1]
    if footer.get("type") != "footer" or footer.get("status") != "complete":
        return dict.fromkeys(METRIC_KINDS)
    focal = header["spec"].get("focal_seat") or ASSISTANT
    other = CONTACT if focal == ASSISTANT else ASSISTANT
    gt, free = header["ground_truth"], header["instance"]["free"]
    slot = footer["final_state"]["slot"]
    deal = slot is not None
    decisions = [l for l in lines if l["type"] == "decision"]
    focal_d = [d for d in decisions if d["seat"] == focal]
    cp_moves = [d["applied"]["move"] for d in decisions if d["seat"] == other]
    first = next((m for m in cp_moves if m["type"] == "propose"), None)
    return {
        "deal": float(deal),
        "focal_surplus_share": free[focal][slot] / gt[f"{focal}_best"] if deal else 0.0,
        "joint_efficiency": (free[ASSISTANT][slot] + free[CONTACT][slot]) / gt["max_joint"] if deal else 0.0,
        "turns": float(footer["final_state"]["turn"]),
        "focal_strike_rate": (sum(d["strike"] is not None for d in focal_d) / len(focal_d)
                              if focal_d else None),
        "cp_opens_with_offer": float(cp_moves[0]["type"] == "propose") if cp_moves else None,
        "cp_first_offer_share": (free[other][first["slot"]] / max(free[other].values())
                                 if first else None),
    }


def decision_rows(lines: list[JSON]) -> list[JSON]:
    return []


def counterpart_opening(lines: list[JSON]) -> str | None:
    focal = lines[0]["spec"].get("focal_seat") or ASSISTANT
    return next((l["applied"].get("message") for l in lines
                 if l["type"] == "decision" and l["seat"] != focal), None)


# --- chat leaks (Harbor simulated-user trials) ------------------------------------

# A rating stated as a number: "worth 10", "value 8", "rates it 9", "(9)", "= 10", "7/10".
_RATING = re.compile(r"(?:\b(?:worth|valued?|values?|rat(?:es?|ed|ing)|scor\w*)\D{0,20}?|\(\s*|=\s*)"
                     r"(10|[1-9])(?![\d:])|\b(10|[1-9])\s*(?:/\s*10|out of 10)\b", re.I)


def chat_leak(lines: list[JSON], seat: Seat, text: str) -> bool:
    """Whether chat text outside the episode states one of the seat's private
    slot ratings as a number. Rankings without numbers ("my best slot") and
    busy times are not counted."""
    values = set()
    for line in lines:
        if line.get("type") == "decision" and line.get("seat") == seat:
            values = {int(v) for v in (line["observation"].get("calendar") or {}).values()
                      if v is not None}
            break
    return any(int(m.group(1) or m.group(2)) in values for m in _RATING.finditer(text))
