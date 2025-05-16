# ai-chatbot
[![Build](https://github.com/Serious-senpai/ai-chatbot/actions/workflows/build.yml/badge.svg)](https://github.com/Serious-senpai/ai-chatbot/actions/workflows/build.yml)
[![Lint](https://github.com/Serious-senpai/ai-chatbot/actions/workflows/lint.yml/badge.svg)](https://github.com/Serious-senpai/ai-chatbot/actions/workflows/lint.yml)

AI chatbot using Groq API.

## Deployment

First, create a `.env` file in the root of repository (see [.example.env](/.example.env) for details) with the following secret(s):
- [Groq API key](https://console.groq.com/)
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

2. Run the backend server with CORS enabled (since the backend runs at `localhost:80` while the frontend is at `localhost:5173`).
```bash
$ pwd
/workspaces/ai-chatbot/services/api
$ python main.py --cors
```

To view all options available, run `python main.py -h`.

3. Set the environment variable `VITE_APP_BASE_API_URL` to point to the backend server.
```bash
$ export VITE_APP_BASE_API_URL=http://localhost:80/api
```

4. Run the frontend in development mode.
```bash
$ pwd
/workspaces/ai-chatbot/services/frontend
$ npm run dev -- --host 0.0.0.0
```

Now the application can be accessed via web browser at `localhost:5173`.

### Run in production mode

Using [Docker](https://www.docker.com/) is the easiest way: simply run `docker compose up -d`.
