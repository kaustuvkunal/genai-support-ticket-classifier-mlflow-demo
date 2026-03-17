"""Prompt optimization helpers for the ticket classifier demo."""

from __future__ import annotations

import logging
from typing import Optional

import mlflow

from mlflow.genai.optimize.optimizers import GepaPromptOptimizer

from .config import Config
from .scorers import exact_category_match
from .prompt import load_prompt_uri
from .predict import predict_from_inputs

logger = logging.getLogger(__name__)


def optimize_prompt(
    config: Config,
    train_data,
    prompt_uri: str | None = None,
    prompt_version: str = "latest",
    max_metric_calls: int = 400,
    display_progress_bar: bool = True,
) -> str:
    """Optimize the registered prompt via MLflow GenAI prompt optimization.

    Args:
        config: Runtime configuration.
        train_data: Training dataset.
        prompt_uri: Explicit MLflow prompt URI, e.g.
            ``prompts:/support-ticket-classifier-prompt/1``.  When provided,
            ``prompt_version`` is ignored.
        prompt_version: Version alias used when ``prompt_uri`` is not given
            (default: ``"latest"``).
        max_metric_calls: Maximum scorer calls during optimization.
        display_progress_bar: Whether to show a progress bar.

    Returns:
        The URI of the optimized prompt.
    """
    logger.info(
        f"Starting prompt optimization with provider={config.llm_provider}, "
        f"max_metric_calls={max_metric_calls}"
    )
    logger.debug(
        f"Prompt version: {prompt_version}, "
        f"progress_bar: {display_progress_bar}"
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    logger.debug(f"MLflow tracking URI: {config.mlflow_tracking_uri}")
    
    mlflow.set_experiment(config.experiment_name)
    logger.debug(f"Experiment: {config.experiment_name}")

    # Set reflection model based on configured LLM provider
    if config.llm_provider == "groq":
        reflection_model = "groq:/llama-3.3-70b-versatile"
    elif config.llm_provider == "openai":
        reflection_model = "openai:/gpt-4o"
    else:
        reflection_model = f"{config.llm_provider}:/{config.model_name}"
    
    logger.debug(f"Reflection model: {reflection_model}")

    optimizer = GepaPromptOptimizer(
        reflection_model=reflection_model,
        max_metric_calls=max_metric_calls,
        display_progress_bar=display_progress_bar,
    )

    prompt_uri = prompt_uri or load_prompt_uri(config, version=prompt_version)
    logger.debug(f"Input prompt URI: {prompt_uri}")

    # End any lingering active run so this optimization is fully isolated.
    mlflow.end_run()

    run_name = f"optimize-{prompt_uri}"
    logger.debug(f"Starting fresh MLflow run: {run_name}")

    logger.debug(f"Starting optimization process")
    with mlflow.start_run(run_name=run_name):
        opt_result = mlflow.genai.optimize_prompts(
            predict_fn=lambda customer_message: predict_from_inputs(
                config, {"customer_message": customer_message}, prompt_uri=prompt_uri
            ),
            train_data=train_data,
            prompt_uris=[prompt_uri],
            optimizer=optimizer,
            scorers=[exact_category_match],
        )

    optimized_prompt = opt_result.optimized_prompts[0]
    logger.info(f"Optimization completed successfully")
    logger.info(f"Optimized prompt URI: {optimized_prompt.uri}")

    return optimized_prompt.uri
