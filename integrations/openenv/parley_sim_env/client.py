"""Client for a parley-sim OpenEnv server.

    >>> from parley_sim_env import ParleySimEnv, ParleySimAction
    >>> with ParleySimEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset(seed=1000)
    ...     while not result.done:
    ...         reply = my_model(result.observation.system_prompt, result.observation.user_prompt)
    ...         result = env.step(ParleySimAction(text=reply))
"""

from parley_sim.adapters.openenv import ParleySimEnv

__all__ = ["ParleySimEnv"]
