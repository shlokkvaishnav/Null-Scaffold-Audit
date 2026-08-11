"""FastAPI application entrypoint for the equation-discovery service.

Run locally with:
    uvicorn plugins.physics.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from plugins.physics.api.routes import benchmark, datasets, jobs

app = FastAPI(
    title="Equation Discovery Agent",
    description=(
        "Submit tabular data and get back a discovered closed-form equation. "
        "Upload a CSV via POST /datasets, submit a discovery job via POST "
        "/jobs, and poll GET /jobs/{id} for the result."
    ),
    version="0.1.0",
)

app.include_router(datasets.router)
app.include_router(jobs.router)
app.include_router(benchmark.router)


@app.get("/health")
async def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.get("/")
async def root() -> dict:
    """Root endpoint with a short description and links to the docs."""
    return {
        "name": "Equation Discovery Agent",
        "description": "Submit a dataset, get discovered equations back.",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }
