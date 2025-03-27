# ai-chatbot
[![Build](https://github.com/Serious-senpai/ai-chatbot/actions/workflows/build.yml/badge.svg)](https://github.com/Serious-senpai/ai-chatbot/actions/workflows/build.yml)
[![Lint](https://github.com/Serious-senpai/ai-chatbot/actions/workflows/lint.yml/badge.svg)](https://github.com/Serious-senpai/ai-chatbot/actions/workflows/lint.yml)

AI chatbot with Ollama backend.

## Deployment

The following steps assume that you are running an Ollama server in a separate machine. For example, let the Ollama host be `http://192.168.1.39:11434` (you can use `http://localhost:11434` when hosting in the same machine).

First, create a `.env` file in the root of repository (see [.example.env](/.example.env) for details) with the following secret(s):
- [Tavily API key](https://tavily.com/)

### Run in development mode

1. Install development dependencies for backend server.
```bash
$ pwd
/workspaces/ai-chatbot/services/api
$ python -m venv /venv-dev
$ source /venv-dev/bin/activate
$ pip install -r dev-requirements.txt
```

2. Run the backend server with CORS enabled (since the backend runs at `localhost:8000` while the frontend is at `localhost:5173`).
```bash
$ pwd
/workspaces/ai-chatbot/services/api
$ python main.py --model qwen2.5:latest --ollama http://192.168.1.39:11434 --cors
```

The example above uses [`qwen2.5:latest`](https://ollama.com/library/qwen2.5) model. You can replace it with any LLMs that support tools provided that they are pulled to the Ollama server.

3. Set the environment variable `VITE_APP_BASE_API_URL` to point to the backend server.
```bash
$ export VITE_APP_BASE_API_URL=http://localhost:8000/api
```

4. Run the frontend in development mode.
```bash
$ pwd
/workspaces/ai-chatbot/services/frontend
$ npm run dev -- --host 0.0.0.0
```

Now the application can be accessed via web browser at `localhost:5173`.

### Run in production mode

Using [Docker](https://www.docker.com/) is the easiest way: modify the `--model` and `--ollama` arguments in [compose.yml](/compose.yml) and simply run `docker compose up -d`.
