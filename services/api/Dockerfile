# Reference: https://docs.docker.com/reference/dockerfile/
FROM python:3.12.9-slim AS builder

RUN python -m venv /venv

# Delay COPY operation as much as possible to avoid cache invalidation: https://stackoverflow.com/a/48553814
COPY services/api/requirements.txt /tmp/requirements.txt
RUN /venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/api /app
WORKDIR /app
ENTRYPOINT ["/venv/bin/python", "main.py"]
