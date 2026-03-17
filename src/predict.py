"""Prediction logic for the ticket classifier demo."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow

from .config import Config
from .prompt import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def _build_messages(prompt_template: str, customer_message: str) -> list[dict[str, str]]:
    """Build the chat message history for the LLM chat completion API."""
    logger.debug(f"Building messages with prompt length={len(prompt_template)}")
    return [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": customer_message},
    ]


def _get_prompt_template(
    config: Config,
    prompt_uri: str | None = None,
    prompt_template: str | None = None,
) -> str:
    """Get the prompt template.

    Priority order:
    1. ``prompt_template`` — raw string passed directly (no MLflow needed)
    2. ``prompt_uri`` — fetched from the MLflow registry
    3. Inline ``PROMPT_TEMPLATE`` from ``src/prompt.py``
    """
    if prompt_template:
        logger.debug("Using explicitly provided prompt template")
        return prompt_template
    if prompt_uri:
        logger.debug(f"Loading prompt from MLflow: {prompt_uri}")
        prompt_obj = mlflow.genai.load_prompt(prompt_uri)
        logger.debug(f"Successfully loaded prompt from MLflow (uri={prompt_uri})")
        return prompt_obj.template
    else:
        logger.debug("No prompt URI provided — using inline PROMPT_TEMPLATE from src/prompt.py")
        return PROMPT_TEMPLATE


def _get_llm_client(config: Config):
    """Get the appropriate LLM client based on the configured provider.

    API keys are read directly from environment variables here — they are
    intentionally NOT stored on Config to prevent leakage via MLflow traces,
    logging, or repr serialization.
    """
    logger.debug(f"Creating LLM client for provider: {config.llm_provider}")

    if config.llm_provider == "groq":
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        logger.debug("Initializing Groq client")
        return Groq(api_key=api_key)
    elif config.llm_provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        logger.debug("Initializing OpenAI client")
        return OpenAI(api_key=api_key)
    else:
        logger.error(f"Unsupported LLM provider: {config.llm_provider}")
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


@mlflow.trace(name="ticket_classifier", span_type="LLM")
def predict(
    config: Config,
    customer_message: str,
    prompt_uri: str | None = None,
    prompt_template: str | None = None,
) -> str:
    """Predict the ticket category for a single customer message.

    Args:
        config: Runtime configuration (provider, model, …).
        customer_message: The support ticket text to classify.
        prompt_uri: Optional explicit MLflow prompt URI, e.g.
            ``prompts:/support-ticket-classifier-prompt/1``.
        prompt_template: Optional raw prompt string. When provided, MLflow is
            bypassed entirely and this template is used directly.

    Priority: prompt_template > prompt_uri > inline PROMPT_TEMPLATE.
    """
    try:
        logger.info(
            f"Starting prediction with provider={config.llm_provider}, "
            f"model={config.model_name}, prompt_uri={prompt_uri or 'inline'}"
        )
        logger.debug(f"Customer message length: {len(customer_message)} chars")

        client = _get_llm_client(config)

        tmpl = _get_prompt_template(config, prompt_uri=prompt_uri, prompt_template=prompt_template)
        messages = _build_messages(tmpl, customer_message)

        logger.debug(f"Calling {config.llm_provider} API with model {config.model_name}")
        resp = client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
        )

        result = resp.choices[0].message.content.strip()
        logger.info(f"Prediction successful: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {type(e).__name__}: {e}")
        raise


def predict_batch(config: Config, inputs: list[str], prompt_template: str | None = None) -> list[str]:
    """Predict categories for a batch of messages."""
    return [predict(config, text, prompt_template=prompt_template) for text in inputs]


def predict_from_inputs(
    config: Config,
    inputs: dict[str, str],
    prompt_uri: str | None = None,
) -> str:
    """Wrapper for MLflow GenAI evaluation/optimization.

    MLflow passes the ``inputs`` dict directly to this function, so the
    parameter name ``customer_message`` must match the key used in the
    evaluation dataset.
    """
    customer_message = inputs.get("customer_message")
    if customer_message is None:
        raise ValueError("Missing 'customer_message' in inputs")

    return predict(config, customer_message, prompt_uri=prompt_uri)
