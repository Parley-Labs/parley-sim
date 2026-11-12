"""A counterpart conditioned on an NVIDIA NeMo UserSim persona.

The persona, OCEAN-derived behavioral profile, disclosure style and
interaction style come from UserSim's own functions (`pip install
parley-sim[usersim]`), derived the way UserSim's episode builder does it
(`usersim.engine.core.episode_input`), and composed in the order of its
`general_open_ended` probe. The negotiation role, rules, private values and
move format that follow are parley-sim's: UserSim users are help-seekers
judged by UserSim's own judges, which have no notion of binding offers.
Label results "UserSim personas and behavior model", not "UserSim".

Personas come from `usersim panel --locale en_US --num-personas N --out
~/.cache/parley-sim/usersim-en_US.parquet`, which samples NVIDIA's Nemotron-Personas datasets without
any model calls. Those rows are licensed for download, not redistribution:
keep the file out of the repository.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from parley_sim.core.protocol import JSON, TextInterface
from parley_sim.core.rng import seed_for

# From UserSim's user-agent template, kept verbatim: the persona-realism rules
# that apply to any conversation, not the help-seeking instructions.
_ROLEPLAY_RULES = """Conversational and stylistic considerations:
- Do NOT reveal you are role-playing.
- Do NOT use stage directions, bracketed actions, or narrator commentary.
- Speak naturally as the character would in their daily life.
- Match your vocabulary and complexity to your persona's education and background.

REMEMBER: YOU ARE THE CHARACTER. Stay in character throughout."""


def load_personas(path: str | Path) -> list[dict[str, Any]]:
    """Persona dicts from a `usersim panel` parquet (or a JSONL of persona dicts)."""
    path = Path(path).expanduser()
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        values = [r.get("persona", r) for r in rows]
    else:
        import pyarrow.parquet as pq
        values = pq.read_table(path, columns=["persona"]).column("persona").to_pylist()
    personas = [json.loads(v) if isinstance(v, str) else v for v in values]
    if not personas:
        raise ValueError(f"no personas in {path}")
    return personas


def persona_for_seed(personas: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """The same persona for a seed in every arm, cell and model."""
    return personas[seed_for(seed, "usersim.persona") % len(personas)]


def usersim_block(persona: dict[str, Any], locale: str = "en_US") -> tuple[str, JSON]:
    """(prompt block, metadata) for one persona, via UserSim's own code."""
    if locale != "en_US":
        raise ValueError("only en_US personas are supported so far")
    from importlib.metadata import version

    from usersim.engine.core.behavioral import (
        compute_behavioral_profile,
        compute_disclosure_style,
        compute_user_interaction_style,
        format_behavioral_profile_for_prompt,
        format_disclosure_instructions,
        format_interaction_style_instructions,
    )
    from usersim.engine.core.persona import format_persona_for_prompt

    profile = compute_behavioral_profile(persona, locale=locale)
    persona_json = json.dumps(persona, sort_keys=True, default=str)
    persona_seed = int(hashlib.sha256(persona_json.encode()).hexdigest(), 16) & 0xFFFFFFFF
    disclosure = compute_disclosure_style(persona_seed=persona_seed)
    style = compute_user_interaction_style(profile)
    parts = [
        "Act as the following character:",
        f"<YOUR_PERSONA>\n{format_persona_for_prompt(persona)}\n</YOUR_PERSONA>",
        format_behavioral_profile_for_prompt(profile, language="English"),
        format_disclosure_instructions(disclosure),
        format_interaction_style_instructions(style),
        _ROLEPLAY_RULES,
    ]
    block = "\n\n".join(p for p in parts if p)
    meta = {"usersim_version": version("usersim"), "persona_seed": persona_seed,
            "interaction_style": style, "disclosure_style": disclosure,
            "ocean": profile["ocean"], "ocean_labels": profile["ocean_labels"]}
    return block, meta


class UserSimPersonaText:
    """Wraps an env's text interface: UserSim persona block, then the env's
    own role, rules, private information and move format."""

    def __init__(self, base: TextInterface, persona: dict[str, Any], locale: str = "en_US"):
        self.base = base
        self.block, self.meta = usersim_block(persona, locale)
        self.prompt_version = f"{base.prompt_version}+usersim-{self.meta['usersim_version']}"
        self.action_schema = base.action_schema

    def render(self, observation: JSON) -> tuple[str, str]:
        system, user = self.base.render(observation)
        return f"{self.block}\n\n{system}", user

    def parse(self, payload: Any) -> tuple[JSON, str | None]:
        return self.base.parse(payload)
