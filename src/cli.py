"""Command line interface for the GenAI ticket classifier demo."""

from __future__ import annotations

import sys
import logging

import click

from .config import load_config
from .data import load_eval_data
from .evaluate import evaluate, print_metrics
from .optimize import optimize_prompt
from .registry import register_prompt
from .predict import predict

logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """GenAI ticket classifier demo CLI."""
    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.debug("CLI initialized")


@main.command("register-prompt")
@click.option(
    "--commit-message",
    default="Register prompt",
    show_default=True,
    help="Commit message to associate with the prompt registration.",
)
def register_prompt_cmd(commit_message: str) -> None:
    """Register or update the base prompt in the MLflow tracking server."""
    logger.info("Executing register-prompt command")

    try:
        config = load_config()
        logger.debug(f"Config loaded: provider={config.llm_provider}")
        
        prompt = register_prompt(config, commit_message=commit_message)
        click.echo(f"Registered prompt: {prompt.name} (version {prompt.version})")
        click.echo(f"Prompt URI: {prompt.uri}")
        logger.info(f"register-prompt completed successfully")
    except Exception as e:
        logger.error(f"register-prompt failed: {type(e).__name__}: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("evaluate")
@click.option(
    "--prompt-uri",
    default=None,
    show_default=True,
    help="MLflow prompt URI to evaluate, e.g. prompts:/support-ticket-classifier-prompt/1. "
         "Defaults to the latest registered version.",
)
@click.option(
    "--skip-data",
    is_flag=True,
    default=False,
    help="Skip loading the canonical evaluation dataset (useful for debugging).",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    show_default=True,
    help="Evaluate only the first N samples. Useful during development to get faster feedback.",
)
def evaluate_cmd(prompt_uri: str, skip_data: bool, limit: int | None) -> None:
    """Run baseline evaluation on the canonical evaluation dataset."""
    logger.info("Executing evaluate command")
    if prompt_uri:
        logger.info(f"Using prompt URI: {prompt_uri}")

    try:
        config = load_config()
        logger.debug(f"Config loaded: provider={config.llm_provider}")

        if skip_data:
            click.echo("Skipping data load (no evaluation data).")
            logger.info("Evaluation skipped per user request")
            sys.exit(0)

        logger.debug("Loading evaluation data")
        data = load_eval_data(limit=limit)
        logger.debug(f"Evaluation data loaded: {len(data)} samples")

        logger.debug("Starting evaluation")
        metrics = evaluate(config=config, data=data, prompt_uri=prompt_uri)
        print_metrics(metrics)
        logger.info("evaluate completed successfully")
    except Exception as e:
        logger.error(f"evaluate failed: {type(e).__name__}: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("optimize")
@click.option(
    "--prompt-uri",
    default=None,
    show_default=True,
    help="MLflow prompt URI to optimize, e.g. prompts:/support-ticket-classifier-prompt/1. "
         "When provided, --prompt-version is ignored.",
)
@click.option(
    "--prompt-version",
    default="latest",
    show_default=True,
    help="The prompt version to optimize (used when --prompt-uri is not given).",
)
@click.option(
    "--max-metric-calls",
    default=400,
    show_default=True,
    help="Maximum number of scorer calls to make during optimization.",
)
def optimize_cmd(prompt_uri: str | None, prompt_version: str, max_metric_calls: int) -> None:
    """Optimize the registered prompt using MLflow GenAI optimization."""
    logger.info("Executing optimize command")
    logger.debug(f"Options: prompt_version={prompt_version}, max_metric_calls={max_metric_calls}")

    try:
        config = load_config()
        logger.debug(f"Config loaded: provider={config.llm_provider}")
        
        logger.debug("Loading training data")
        data = load_eval_data()
        logger.debug(f"Training data loaded: {len(data)} samples")

        logger.debug("Starting prompt optimization")
        optimized_uri = optimize_prompt(
            config=config,
            train_data=data,
            prompt_uri=prompt_uri,
            prompt_version=prompt_version,
            max_metric_calls=max_metric_calls,
        )

        click.echo(f"Optimized prompt URI: {optimized_uri}")
        logger.info("optimize completed successfully")
    except Exception as e:
        logger.error(f"optimize failed: {type(e).__name__}: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("predict")
@click.argument("message", nargs=-1)
@click.option(
    "--prompt-uri",
    default=None,
    show_default=True,
    help="MLflow prompt URI to use, e.g. prompts:/support-ticket-classifier-prompt/1. "
         "Defaults to the latest registered version, falling back to prompts/finalise_prompt.py.",
)
def predict_cmd(message: tuple[str, ...], prompt_uri: str | None) -> None:
    """Predict the category for a single customer message."""
    logger.info("Executing predict command")

    try:
        if not message:
            logger.error("No message provided")
            raise click.UsageError("Provide a customer message to classify.")

        customer_message = " ".join(message).strip()
        logger.debug(f"Message to predict: {len(customer_message)} chars")

        config = load_config()
        logger.debug(f"Config loaded: provider={config.llm_provider}")

        if prompt_uri:
            click.echo(f"Prompt : {prompt_uri}")
        else:
            from .prompt import _FINALISE_PROMPT_PATH
            click.echo(f"Prompt : {_FINALISE_PROMPT_PATH} (local fallback)")

        logger.debug("Starting prediction")
        prediction = predict(config, customer_message, prompt_uri=prompt_uri)

        click.echo(f"Result : {prediction}")
        logger.info(f"Prediction: {prediction}")
    except Exception as e:
        logger.error(f"predict failed: {type(e).__name__}: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
