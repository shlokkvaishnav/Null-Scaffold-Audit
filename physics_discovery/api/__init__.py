"""FastAPI service for the equation-discovery agent.

Exposes a minimal HTTP surface: upload a tabular dataset, submit a
discovery job against it, and poll for the resulting equation, fit
metrics, and a confidence score. Job execution uses FastAPI
``BackgroundTasks`` with an in-memory job store -- no external queue or
database, matching the scope of this project (see ``worker.py``).
"""
