"""Configuration helpers for the ticket classifier demo."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the ticket classifier demo.

    API keys are intentionally excluded to prevent accidental exposure via
    MLflow traces, logging, or repr serialization.  They are read directly
    from environment variables at the point of use.
    """

    llm_provider: str
    mlflow_tracking_uri: str
    model_name: str
    prompt_template_name: str
    experiment_name: str


def load_config(env_path: Optional[str] = None) -> Config:
    """Load configuration from environment variables.

    It loads environment variables from a `.env` file if present.
    Supports multiple LLM providers: groq, openai, etc.
    """
    logger.debug(f"Loading configuration from env_path={env_path}")

    # Prefer an explicit env file, otherwise look in the repo root.
    env_path = env_path or os.environ.get("TICKET_CLASSIFIER_ENV_PATH")
    if env_path:
        logger.info(f"Loading environment from explicit path: {env_path}")
        load_dotenv(env_path)
    else:
        # Load from .env in the current working directory for convenience.
        candidate = Path.cwd() / ".env"
        if candidate.exists():
            logger.info(f"Loading environment from: {candidate}")
            load_dotenv(candidate)
        else:
            logger.debug("No .env file found in current directory")

    # Get LLM provider (default to groq)
    llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()
    logger.info(f"LLM Provider configured: {llm_provider}")
    
    # Validate and get API keys based on provider
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if llm_provider == "groq" and not groq_api_key:
        logger.error("GROQ_API_KEY is not set but Groq provider is selected")
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please set it in .env file or environment variables. "
            "See .env.example for guidance."
        )
    elif llm_provider == "openai" and not openai_api_key:
        logger.error("OPENAI_API_KEY is not set but OpenAI provider is selected")
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in .env file or environment variables. "
            "See .env.example for guidance."
        )
    elif llm_provider not in ["groq", "openai"]:
        logger.error(f"Unsupported LLM provider: {llm_provider}")
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER: {llm_provider}. "
            "Supported providers: groq, openai"
        )

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    # model name depends on provider, but can be overriddenuse if else here
    #. e.g. if LLM_PROVIDER=openai MODEL_NAME=gpt-4, if LLM_PROVIDER=groq MODEL_NAME=llama-3.3-70b-versatile
    
    if llm_provider == "openai":
        model_name = os.getenv("MODEL_NAME") or "gpt-4"
    elif llm_provider == "groq":
        model_name = os.getenv("MODEL_NAME") or "llama-3.3-70b-versatile"
    else:
        model_name = os.getenv("MODEL_NAME")

    prompt_template_name = os.getenv("PROMPT_NAME") or "support-ticket-classifier-prompt"
    experiment_name = os.getenv("MLFLOW_EXPERIMENT") or "Support_Ticket_Classification_project"

    logger.info("Configuration loaded successfully")
    logger.debug(
        f"Config: provider={llm_provider}, model={model_name}, "
        f"prompt_name={prompt_template_name}, experiment={experiment_name}"
    )

    # API keys are NOT stored in Config — they are read from environment
    # at the point of use inside _get_llm_client() to prevent leakage via
    # MLflow traces, logging, or any other serialization path.
    return Config(
        llm_provider=llm_provider,
        mlflow_tracking_uri=mlflow_tracking_uri,
        model_name=model_name,
        prompt_template_name=prompt_template_name,
        experiment_name=experiment_name,
    )
