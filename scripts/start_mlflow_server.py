"""Start the MLflow tracking server.

Usage: uv run python scripts/start_mlflow_server.py
"""

from mlflow.cli import cli

cli(args=["server", "--host", "127.0.0.1", "--port", "5000",
          "--backend-store-uri", "sqlite:///mlflow.db"])
