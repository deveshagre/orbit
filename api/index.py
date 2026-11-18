import os
import sys
from pathlib import Path

# Add backend directory and project root to sys.path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
backend_dir = root_dir / "backend"

for d in (str(backend_dir), str(root_dir)):
    if d not in sys.path:
        sys.path.insert(0, d)

from app import create_app

# Vercel Serverless Function entrypoint: exports WSGI application 'app'
app = create_app()
