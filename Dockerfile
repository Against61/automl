FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/root
ENV PYTHONPATH=/app

RUN apt-get update \
  && apt-get install -y --no-install-recommends nodejs npm git ca-certificates ripgrep \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

RUN node --version \
  && npm --version \
  && npm install -g @openai/codex \
  && codex --version

RUN python - <<'PY'
import pathlib
import tomllib

cfg = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
deps = cfg.get("project", {}).get("dependencies", [])
pathlib.Path("/tmp/requirements.txt").write_text("\n".join(deps))
PY

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY orchestrator ./orchestrator
COPY pilot.py ./pilot.py
COPY streamlit_app.py ./streamlit_app.py

EXPOSE 8080

CMD ["uvicorn", "orchestrator.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
