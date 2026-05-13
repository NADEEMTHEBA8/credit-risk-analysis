"""
Backward-compatible entry point.

The pipeline logic now lives in src/main.py. This shim exists so anyone
running `python -m src.pipeline` (old command) gets the same result as
`python -m src.main` (new command).
"""
from src.main import run

run()
