# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import platform, sys, os, time

app = FastAPI(title="NeuroLens - Phase 0 (Health API)")

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    python_version: str
    platform: str
    path: str

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check for the API and environment"""
    return HealthResponse(
        status="ok",
        timestamp=time.time(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        path=os.getcwd(),
    )

@app.get("/")
def root():
    return {"message": "NeuroLens Phase 0 — API running. Try GET /health"}
