# Reference: https://docs.docker.com/reference/dockerfile/
FROM ubuntu:25.04

RUN apt-get update && apt-get install -y \
    curl \
    git \
    iputils-ping \
    nodejs \
    npm \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv

RUN python3 -m venv /venv
COPY services/api/dev-requirements.txt /tmp/requirements.txt
RUN /venv/bin/pip install -r /tmp/requirements.txt
