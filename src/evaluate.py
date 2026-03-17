"""Evaluation helpers for the ticket classifier demo."""

from __future__ import annotations

import logging
from typing import List

import mlflow

from .config import Config
from .predict import predict, predict_from_inputs
from .scorers import exact_category_match

logger = logging.getLogger(__name__)


def evaluate(
    config: Config,
    data,
    prompt_uri: str | None = None,
    additional_scorers: List = None,
) -> dict:
    """Run a baseline evaluation of the current prompt + model.

    Args:
        config: Runtime configuration.
        data: Evaluation dataset (DataFrame or list of dicts with ``inputs``
            and ``expectations`` keys).
        prompt_uri: Optional explicit MLflow prompt URI, e.g.
            ``prompts:/support-ticket-classifier-prompt/1``.  When omitted the
            latest registered version is used.
        additional_scorers: Extra scorer functions to include alongside the
            default ``exact_category_match`` scorer.
    """
    logger.info(
        f"Starting evaluation with provider={config.llm_provider}, "
        f"model={config.model_name}, prompt_uri={prompt_uri or 'latest from config'}"
    )
    logger.debug(f"Data size: {len(data) if hasattr(data, '__len__') else 'unknown'} samples")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    logger.debug(f"MLflow tracking URI: {config.mlflow_tracking_uri}")

    mlflow.set_experiment(config.experiment_name)
    logger.debug(f"Experiment: {config.experiment_name}")

    scorers = [exact_category_match]
    if additional_scorers:
        logger.debug(f"Adding {len(additional_scorers)} custom scorers")
        scorers.extend(additional_scorers)

    # End any lingering active run so this evaluation is fully isolated.
    mlflow.end_run()

    run_name = f"evaluate-{prompt_uri or 'latest'}"
    logger.debug(f"Starting fresh MLflow run: {run_name}")

    # NOTE: The lambda parameter name *must* match the key in the `inputs` dict
    # of the evaluation dataset ("customer_message") so that MLflow can wire
    # the value through correctly.
    logger.debug(f"Running evaluation with {len(scorers)} scorers")
    with mlflow.start_run(run_name=run_name):
        result = mlflow.genai.evaluate(
            data=data,
            predict_fn=lambda customer_message: predict_from_inputs(
                config, {"customer_message": customer_message}, prompt_uri=prompt_uri
            ),
            scorers=scorers,
        )

    logger.info(f"Evaluation completed successfully")
    logger.debug(f"Metrics: {result.metrics}")
    return result.metrics


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics in a human-friendly format."""
    logger.info("Evaluation metrics:")
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            msg = f"  {k:36} = {v:.4f}"
            print(msg)
            logger.debug(msg)
        else:
            msg = f"  {k:36} = {v}"
            print(msg)
            logger.debug(msg)
