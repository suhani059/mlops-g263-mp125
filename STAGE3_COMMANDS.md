# Stage 3 - commands (PowerShell, inside the repo folder)

pip install -r requirements.txt
python -m pytest -q

# run the API locally, then open http://127.0.0.1:8000/docs
python -m uvicorn app.main:app --port 8000

# Docker (Docker Desktop must be running)
docker build -t mp125-api:1.0 .
docker run --rm -p 8000:8000 mp125-api:1.0

# Kubernetes files are checked as files; no cluster needed
