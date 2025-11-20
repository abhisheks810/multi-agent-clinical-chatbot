# src/langgraph_rag/observability/mlflow_utils.py

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient


def init_mlflow(
    experiment_name: str = "langgraph_rag_e2e",
    tracking_uri: Optional[str] = None,
) -> None:
    """
    Initialize MLflow for the app.

    Call this once at app startup.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def start_query_run(user_query: str) -> tuple[str, float]:
    """
    Start an MLflow run for a single user query.

    Returns:
      - run_id: the MLflow run id (to attach feedback later)
      - start_time: timestamp to compute total latency
    """
    start_time = time.time()
    run = mlflow.start_run(run_name="rag_query")
    run_id = run.info.run_id

    # Log basic context as params
    mlflow.log_param("user_query", user_query[:300])

    return run_id, start_time


def finish_query_run(
    start_time: float,
    result: Dict[str, Any],
) -> None:
    """
    Finish the MLflow run for this query.

    Assumes there is an active run.
    Logs:
      - total latency
      - full multi-agent result as artifacts (optional, but very useful)
    """
    total_latency_ms = (time.time() - start_time) * 1000.0
    mlflow.log_metric("total_latency_ms", total_latency_ms)

    # Optional: log structured agent outputs if present
    oncologist_view = result.get("oncologist_view")
    plan = result.get("plan")
    execution_result = result.get("execution_result")

    if oncologist_view is not None:
        mlflow.log_dict(oncologist_view, "oncologist_view.json")
    if plan is not None:
        mlflow.log_dict(plan, "plan.json")
    if execution_result is not None:
        mlflow.log_dict(execution_result, "execution_result.json")

    # Close the run
    mlflow.end_run()


def log_feedback(
    run_id: str,
    useful: bool,
    comment: Optional[str] = None,
) -> None:
    """
    Attach user feedback to a completed run.

    - useful: True for 👍, False for 👎
    - comment: optional free-text explanation
    """
    client = MlflowClient()

    client.log_metric(
        run_id,
        "user_feedback_useful",
        1.0 if useful else 0.0,
    )

    if comment:
        # Store short comment as a param; truncate if too long
        client.log_param(
            run_id,
            "user_feedback_comment",
            comment[:500],
        )
