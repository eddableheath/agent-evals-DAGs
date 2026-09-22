# Sandbox image for the coding-agent environment.
# Deliberately minimal: the agent should have a working pytest and nothing else
# interesting, so that what it does is driven by the planted repo, not by the image.
FROM python:3.11-slim

RUN pip install --no-cache-dir pytest==8.3.4

# Scenario fixtures are copied in per-sample by the task (M1), not baked into the
# image — 100 generated scenarios should not mean 100 image builds.
WORKDIR /repo

CMD ["tail", "-f", "/dev/null"]
