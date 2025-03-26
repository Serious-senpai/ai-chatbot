# Reference: https://docs.docker.com/reference/dockerfile/
FROM python:3.12.9-alpine AS builder

RUN python -m venv /venv

# Delay COPY operation as much as possible to avoid cache invalidation: https://stackoverflow.com/a/48553814
COPY services/api /app

WORKDIR /app
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/venv/bin/python", "main.py"]
