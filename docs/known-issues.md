# Known issues

Problems outside parley-sim that affect it, with what parley-sim does about
them.

## OpenEnv 0.12: idle reaper closes sessions mid-step

**Affects:** the OpenEnv adapter (`openenv` 0.12.0, `openenv/core/env_server/http_server.py`).

**Status:** fixed upstream in
[openenv/OpenEnv#1618](https://github.com/openenv/OpenEnv/pull/1618), merged
after 0.13.0 and not yet in a release. parley-sim pins `openenv>=0.12,<0.13`,
so it keeps the workarounds below until it moves to a release with the fix.

OpenEnv's idle reaper closes WebSocket sessions that have been idle longer
than `ConcurrencyConfig.session_timeout`. Two problems:

1. **A session can be reaped while its `reset` or `step` is running.** The
   WebSocket handler stamps activity (`_update_session_activity`) only after
   `reset_async`/`step_async` returns, and the reaper (`_reap_idle_sessions`)
   compares the time since the *previous* activity with the timeout. A step
   that starts near the deadline, or that runs longer than the timeout
   (parley-sim's step waits for an LLM counterpart), is reaped mid-call: the
   environment's `close()` runs while the step is still using it. The reaper
   checks every `max(timeout / 4, 5)` seconds.
2. **A reaped session's slot is released while its connection stays live.**
   The reaper removes the session from `_sessions`, which frees a slot under
   `max_concurrent_envs`, but the WebSocket handler keeps its own reference to
   the environment and keeps serving `step` on it. More environments can then
   be live than `max_concurrent_envs` allows.

Only sessions in `_session_websocket_attachments` are exempt, and a plain
`/ws` session is not added to that set.

### Reproduction (OpenEnv alone)

```python
import asyncio, json, time
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, ConcurrencyConfig, Observation, State

class SlowAction(Action):
    pass

class SlowEnv(Environment[SlowAction, Observation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True
    closed_mid_step = False

    def __init__(self):
        super().__init__()
        self._state, self._busy = State(episode_id="e", step_count=0), False

    def reset(self, seed=None, episode_id=None, **kw):
        return Observation(done=False, reward=None)

    async def step_async(self, action, **kw):
        self._busy = True
        await asyncio.sleep(9.0)  # longer than the timeout, spanning a reaper check
        self._busy = False
        return Observation(done=False, reward=0.0)

    def step(self, action, **kw):
        raise NotImplementedError

    @property
    def state(self):
        return self._state

    def close(self):
        if self._busy:
            SlowEnv.closed_mid_step = True

app = create_fastapi_app(SlowEnv, SlowAction, Observation, env_name="slow",
                         concurrency_config=ConcurrencyConfig(max_concurrent_envs=1,
                                                              session_timeout=2.0))
with TestClient(app) as client, client.websocket_connect("/ws") as ws:
    ws.send_text(json.dumps({"type": "reset", "data": {}})); ws.receive_json()
    time.sleep(1.5)                                 # inside the 2 s timeout
    ws.send_text(json.dumps({"type": "step", "data": {}})); ws.receive_json()
    with client.websocket_connect("/ws") as ws2:    # the cap is 1 session
        ws2.send_text(json.dumps({"type": "reset", "data": {}}))
        print("second session admitted:", ws2.receive_json()["type"])
        ws.send_text(json.dumps({"type": "step", "data": {}}))
        print("first session still steps:", ws.receive_json()["type"])
print("closed while its step was running:", SlowEnv.closed_mid_step)
```

With 0.12.0 this prints `observation`, `observation` and `True`: the first
session was closed mid-step, a second session was admitted over the cap of
one, and the first session kept stepping.

### The upstream fix

- The `/ws` handler counts operations in flight per session and stamps
  activity when each `reset` or `step` starts and ends; the reaper skips a
  session with an operation in flight.
- Before each message, the handler checks that its session still exists. A
  socket whose session was reaped gets `SESSION_ERROR` and is closed, so a
  released slot can't keep serving.

OpenEnv's `/mcp` WebSocket route still stamps activity only after a call
returns. parley-sim's server doesn't serve that route, so this doesn't affect
it.

### What parley-sim does until then

- `ParleySimEnvironment.close()` marks the environment closed and
  `_advance` checks after every counterpart call, so a session reaped mid-step
  ends its episode with a "closed" error instead of returning an observation,
  and every later message on that connection gets the same error.
- The server refuses to start when `PARLEY_SESSION_TIMEOUT` is shorter than
  two attempts of every seated model counterpart's `timeout_s`. That is a
  lower bound: it assumes each counterpart decides once per `reset` or `step`,
  and an environment where a seat acts several times in a row needs more. The
  docs say to keep the timeout well above the longest pause between steps plus
  the longest whole `reset` or `step`.
