# NeuroLens — Phase 0

This repo contains a minimal starter for NeuroLens:
- FastAPI health-check API
- repo skeleton for later ingestion and RAG components

Run the API:
```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
