import asyncio
import json

import pytest

pytest.importorskip("usersim")

from parley_sim.cli import smoke_config
from parley_sim.drivers.usersim_persona import (UserSimPersonaText, load_personas,
                                                 persona_for_seed, usersim_block)
from parley_sim.envs.support_desk import SupportDesk, SupportDeskText
from parley_sim.matched import MatchedConfig, run_matched
from parley_sim.trace import read_trace


def trait(t, label, desc):
    return {"t_score": t, "label": label, "description": desc}


PERSONA = {
    "first_name": "Dana", "last_name": "Ruiz", "age": 46, "sex": "Female",
    "occupation": "dental hygienist", "education_level": "bachelors",
    "city": "Tucson", "state": "AZ",
    "persona": "Practical and budget-conscious; keeps careful track of subscriptions.",
    "openness": trait(45, "average", "Prefers the familiar."),
    "conscientiousness": trait(62, "high", "Organized and deliberate."),
    "extraversion": trait(40, "low", "Reserved."),
    "agreeableness": trait(35, "low", "Skeptical and blunt."),
    "neuroticism": trait(68, "high", "Easily stressed and irritated."),
}


def test_block_uses_usersim_behavior_model():
    block, meta = usersim_block(PERSONA)
    assert "<YOUR_PERSONA>" in block and "Dana" in block
    assert meta["interaction_style"] == "confrontational"  # high N, low A
    assert "<INTERACTION_STYLE>" in block
    assert meta["ocean_labels"]["neuroticism"] == "high"
    assert meta["usersim_version"]


def test_persona_text_prepends_block_and_keeps_env_interface():
    env = SupportDesk()
    obs = env.observe(env.initial_state(env.generate(0, {})), "customer")
    base = SupportDeskText()
    text = UserSimPersonaText(base, PERSONA)
    system, user = text.render(obs)
    assert system.startswith("Act as the following character:")
    assert system.endswith(base.render(obs)[0]) and user == base.render(obs)[1]
    assert text.action_schema == base.action_schema
    assert text.prompt_version.startswith("sd-2+usersim-")


def test_personas_load_and_are_fixed_per_seed(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps({"persona": {**PERSONA, "first_name": n}}) + "\n"
                            for n in ["A", "B", "C"]))
    personas = load_personas(path)
    assert len(personas) == 3
    assert persona_for_seed(personas, 7) == persona_for_seed(personas, 7)


def test_usersim_arm_records_persona_in_trace(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"persona": PERSONA}) + "\n")
    base = smoke_config("0:2").model_dump()
    base["lineup"].append({"name": "us-customer", "provider": "scripted", "model": "conceder",
                           "persona": "usersim"})
    cfg = MatchedConfig(**{**base, "extra_arms": {"usersim": ["us-customer"]},
                           "run_arms": ["usersim"], "persona_file": str(path)})
    asyncio.run(run_matched(cfg, tmp_path / "run"))
    trace = next((tmp_path / "run" / "traces").glob("*-usersim-r0.jsonl"))
    seats = [l for l in read_trace(trace) if l["type"] == "decision" and l["seat"] == "customer"]
    assert seats and seats[0]["raw"]["persona"]["interaction_style"] == "confrontational"
    assert "+usersim-" in seats[0]["raw"]["prompt_version"]
