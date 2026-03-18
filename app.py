"""Standalone Gradio app for the GenAI Ticket Classifier.

This file is fully self-contained and can be deployed to Hugging Face Spaces.
It classifies customer support messages into categories using a configurable LLM client.

Supported providers: Groq, OpenAI
"""

from __future__ import annotations

import os
import runpy
import logging
from typing import Optional
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv


# ============================================================================
# Configuration
# ============================================================================

# Configure logging with formatted output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.debug("Initializing Gradio app")


def _load_environment() -> Path | None:
    """Load environment variables from the first available env file."""
    env_candidates = []
    for env_name in (".env", ".env.example"):
        env_candidates.append(Path.cwd() / env_name)
        env_candidates.append(Path(__file__).parent / env_name)

    seen_paths: set[Path] = set()
    for env_path in env_candidates:
        resolved_path = env_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)

        if env_path.exists():
            load_dotenv(env_path, override=False)
            return env_path

    return None


def _split_csv_env(value: str | None) -> list[str]:
    """Split a comma-separated env var into a cleaned list."""
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _provider_display_name(provider: str) -> str:
    """Return the display name for a provider from env configuration."""
    return os.getenv(
        f"{provider.upper()}_DISPLAY_NAME",
        provider.replace("_", " ").title(),
    )


_LOADED_ENV_PATH = _load_environment()

# Load prompt from prompts/finalise_prompt.py — maintained by the user.
_FINALISE_PROMPT_PATH = Path(__file__).parent / "prompts" / "finalise_prompt.py"
PROMPT_TEMPLATE: str = runpy.run_path(str(_FINALISE_PROMPT_PATH))["PROMPT"]

SUPPORTED_PROVIDER_KEYS = _split_csv_env(os.getenv("SUPPORTED_PROVIDERS"))
DEFAULT_PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower()

if DEFAULT_PROVIDER and DEFAULT_PROVIDER not in SUPPORTED_PROVIDER_KEYS:
    SUPPORTED_PROVIDER_KEYS.append(DEFAULT_PROVIDER)

if not DEFAULT_PROVIDER and SUPPORTED_PROVIDER_KEYS:
    DEFAULT_PROVIDER = SUPPORTED_PROVIDER_KEYS[0]

DEFAULT_MODELS = {
    provider: (
        os.getenv(f"{provider.upper()}_MODEL_NAME")
        or (os.getenv("MODEL_NAME") if provider == DEFAULT_PROVIDER else None)
        or ""
    )
    for provider in SUPPORTED_PROVIDER_KEYS
}

SUPPORTED_PROVIDERS = {
    provider: (
        f"{_provider_display_name(provider)} ({DEFAULT_MODELS[provider]})"
        if DEFAULT_MODELS[provider]
        else _provider_display_name(provider)
    )
    for provider in SUPPORTED_PROVIDER_KEYS
}

PROVIDER_LABEL_TO_KEY = {
    label: provider for provider, label in SUPPORTED_PROVIDERS.items()
}


def load_config() -> dict:
    """Load configuration from environment variables."""
    logger.debug("Loading configuration")

    if _LOADED_ENV_PATH is not None:
        logger.debug("Loaded environment from: %s", _LOADED_ENV_PATH)
    else:
        logger.debug("No .env file found, using environment variables")

    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER or "").strip().lower()
    model = os.getenv("MODEL_NAME")

    # API keys are intentionally NOT stored in the config dict to prevent
    # accidental exposure via logging, tracing, or serialization.
    # They are read from os.environ at the point of use.
    config = {
        "provider": provider,
        "model_name": model,
    }

    # Set default model if not specified
    if not config["model_name"]:
        config["model_name"] = DEFAULT_MODELS.get(config["provider"], "")

    logger.info(
        "Configuration loaded - Provider: %s, Model: %s",
        config["provider"],
        config["model_name"],
    )
    logger.debug(
        "Groq API key present: %s, OpenAI API key present: %s",
        bool(os.getenv("GROQ_API_KEY")),
        bool(os.getenv("OPENAI_API_KEY")),
    )
    return config


# ============================================================================
# LLM Client Abstraction
# ============================================================================

def predict_with_groq(model: str, prompt: str, customer_message: str, api_key_override: str | None = None) -> str:
    """Call Groq API."""
    logger.debug(f"Calling Groq API with model {model}")
    try:
        from groq import Groq
        logger.debug("Groq client imported successfully")

        api_key = api_key_override or os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY not set in environment"
        client = Groq(api_key=api_key)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": customer_message},
        ]
        
        logger.debug(f"Making Groq API request with message length: {len(customer_message)}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"Groq prediction successful: {result}")
        return result
    except ImportError:
        logger.error("groq package not installed")
        return "Error: groq package not installed. Install with: pip install groq"
    except Exception as e:
        logger.error(f"Groq API error: {type(e).__name__}: {e}")
        return f"Error: {str(e)}"


def predict_with_openai(model: str, prompt: str, customer_message: str, api_key_override: str | None = None) -> str:
    """Call OpenAI API."""
    logger.debug(f"Calling OpenAI API with model {model}")
    try:
        from openai import OpenAI
        logger.debug("OpenAI client imported successfully")

        api_key = api_key_override or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY not set in environment"
        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": customer_message},
        ]
        
        logger.debug(f"Making OpenAI API request with message length: {len(customer_message)}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"OpenAI prediction successful: {result}")
        return result
    except ImportError:
        logger.error("openai package not installed")
        return "Error: openai package not installed. Install with: pip install openai"
    except Exception as e:
        logger.error(f"OpenAI API error: {type(e).__name__}: {e}")
        return f"Error: {str(e)}"



 
# ============================================================================
# Prediction Logic
# ============================================================================

def predict(config: dict, customer_message: str, api_key_override: str | None = None) -> str:
    """Predict the ticket category using the configured LLM provider.

    Args:
        config: Configuration dictionary with provider and model (no API keys).
        customer_message: The customer support message to classify.
        api_key_override: Optional API key supplied at runtime (e.g. from the
            Gradio UI).  When omitted the key is read from os.environ.

    Returns:
        The predicted category or error message.
    """
    if not customer_message.strip():
        logger.debug("Empty message received, returning empty string")
        return ""

    provider = config["provider"].lower()
    model = config["model_name"]
    prompt = PROMPT_TEMPLATE
    
    logger.debug(f"Predicting with provider={provider}, model={model}, message_length={len(customer_message)}")

    if provider == "groq":
        logger.info("Using Groq provider for prediction")
        return predict_with_groq(model, prompt, customer_message, api_key_override=api_key_override)

    elif provider == "openai":
        logger.info("Using OpenAI provider for prediction")
        return predict_with_openai(model, prompt, customer_message, api_key_override=api_key_override)

    else:
        error_msg = f"Unsupported provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS.keys())}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


# ============================================================================
# Gradio Interface
# ============================================================================

def _build_auth() -> Optional[tuple[str, str]]:
    """Build authentication credentials from environment variables."""
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWD")
    if username and password:
        logger.info("Gradio authentication enabled")
        return (username, password)
    logger.info("Gradio authentication disabled (USERNAME/PASSWD not set)")
    return None


def create_app(config: Optional[dict] = None) -> gr.Blocks:
    """Create and return the Gradio app.
    
    Args:
        config: Optional configuration dictionary. If not provided, loads from env.
        
    Returns:
        A Gradio Blocks interface
    """
    logger.info("Creating Gradio application")
    config = config or load_config()
    logger.debug("App config: provider=%s, model=%s", config["provider"], config["model_name"])

    def classify(customer_message: str) -> str:
        """Classify using the environment-backed app configuration."""
        if not customer_message.strip():
            return ""

        return predict(config, customer_message)

    logger.debug("Building Gradio interface")
    with gr.Blocks(
        title="GenAI Ticket Classifier",
        css="body { font-family: system-ui; }"
    ) as demo:
        gr.Markdown("""
        # 🎫 GenAI Support Ticket Classifier
        ### Classifies customer messages into one of classes based on the message content and intent
        **Incident** • **Request** • **Problem** • **Change**
        """)

        gr.Markdown("### Classify Message")
        txt = gr.TextArea(
            label="Customer message",
            placeholder="Describe the issue or request...",
            lines=6
        )
        classify_btn = gr.Button("Classify", variant="primary")
        out = gr.Textbox(
            label="Predicted category",
            interactive=False
        )

        txt.change(
            fn=classify,
            inputs=txt,
            outputs=out
        )

        classify_btn.click(
            fn=classify,
            inputs=txt,
            outputs=out
        )

    return demo


# ============================================================================
# Entry Points
# ============================================================================

# This is what Hugging Face Spaces looks for
logger.debug("Creating app instance for HF Spaces")
app = create_app()
auth = _build_auth()

# For local development/testing
if __name__ == "__main__":
    logger.info("Starting Gradio app in local mode")
    logger.info("Launching app on http://localhost:7860")
    app.queue()
    app.launch(auth=auth, share=False)
