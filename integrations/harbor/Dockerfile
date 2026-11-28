# parley-sim runtime sidecar for Harbor tasks (experimental).
#
# Build from the parley-sim repository root:
#   docker build -f integrations/harbor/Dockerfile -t parley-runtime .
# With the UserSim persona arm:
#   docker build -f integrations/harbor/Dockerfile --build-arg EXTRAS=harbor,openai,anthropic,usersim \
#     -t parley-runtime:usersim .
# Then export tasks that use it:
#   parley-sim harbor-export CONFIG.yaml --out tasks --image parley-runtime
# Each task sets the episode (PARLEY_CELL), its result token and any API key
# variables at runtime; nothing secret is baked into the image.
FROM python:3.12-slim

ARG EXTRAS=harbor,openai,anthropic
COPY . /tmp/repo
RUN pip install --no-cache-dir "/tmp/repo[${EXTRAS}]" && rm -rf /tmp/repo

ENV NEMO_TELEMETRY_ENABLED=false
EXPOSE 8000
CMD ["python3", "-c", "from parley_sim.adapters.harbor import serve; serve()"]
