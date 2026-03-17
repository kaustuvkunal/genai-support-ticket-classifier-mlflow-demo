"""MLflow prompt registration helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow

from .config import Config
from .prompt import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredPrompt:
    """Represents a prompt registered with MLflow."""

    name: str
    uri: str
    version: int


def register_prompt(config: Config, commit_message: str = "Register prompt") -> RegisteredPrompt:
    """Register or update the prompt in the MLflow tracking server."""
    logger.info(f"Registering prompt: {config.prompt_template_name}")
    logger.debug(f"Commit message: {commit_message}")
    logger.debug(f"MLflow tracking URI: {config.mlflow_tracking_uri}")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    logger.debug(f"Set MLflow tracking URI")
    
    mlflow.set_experiment(config.experiment_name)
    logger.debug(f"Set experiment: {config.experiment_name}")

    logger.debug(f"Registering prompt with template size: {len(PROMPT_TEMPLATE)} chars")
    prompt = mlflow.genai.register_prompt(
        name=config.prompt_template_name,
        template=PROMPT_TEMPLATE,
        commit_message=commit_message,
    )
    
    logger.info(f"Prompt registered successfully: {prompt.name} (version {prompt.version})")
    logger.debug(f"Prompt URI: {prompt.uri}")

    return RegisteredPrompt(name=prompt.name, uri=prompt.uri, version=prompt.version)
