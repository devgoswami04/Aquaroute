"""Make the `aquaroute` package importable without installing it.

Adds ``backend/`` to sys.path for tests and ad-hoc scripts. Production runs use
``uvicorn --app-dir backend`` (see Makefile) which handles this the same way.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
