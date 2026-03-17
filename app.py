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

# Load prompt from prompts/finalise_prompt.py — maintained by the user.
_FINALISE_PROMPT_PATH = Path(__file__).parent / "prompts" / "finalise_prompt.py"
PROMPT_TEMPLATE: str = runpy.run_path(str(_FINALISE_PROMPT_PATH))["PROMPT"]

# Supported LLM providers
SUPPORTED_PROVIDERS = {
    "groq": "Groq (llama-3.1-8b-instant)",
    "openai": "OpenAI (gpt-3.5-turbo)",
}

DEFAULT_PROVIDER = "groq"
DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-3.5-turbo",
}


def load_config() -> dict:
    """Load configuration from environment variables."""
    logger.debug("Loading configuration")
    
    # Load from .env file if it exists
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        logger.debug(f"Loading environment from: {env_path}")
        load_dotenv(env_path)
    else:
        logger.debug("No .env file found, using environment variables")

    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
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
        config["model_name"] = DEFAULT_MODELS.get(config["provider"])

    logger.info(f"Configuration loaded - Provider: {config['provider']}, Model: {config['model_name']}")
    logger.debug(f"Groq API key present: {bool(os.getenv('GROQ_API_KEY'))}, OpenAI API key present: {bool(os.getenv('OPENAI_API_KEY'))}")
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
    logger.debug(f"App config: provider={config['provider']}, model={config['model_name']}")

    def classify(customer_message: str, selected_provider: str, api_key: str) -> str:
        """Wrapper for predict function with runtime provider/key selection."""
        logger.debug(f"Classify called with selected_provider={selected_provider}")
        
        if not customer_message.strip():
            return ""

        # Update config with runtime selections
        runtime_config = config.copy()
        runtime_config["provider"] = selected_provider

        # Pass the user-provided key as an override (falls back to os.environ
        # inside predict_with_groq / predict_with_openai when not supplied).
        key_override = api_key.strip() or None
        if key_override:
            logger.debug(f"Using provided API key for {selected_provider}")

        return predict(runtime_config, customer_message, api_key_override=key_override)

    logger.debug("Building Gradio interface")
    with gr.Blocks(
        title="GenAI Ticket Classifier",
        css="body { font-family: system-ui; }"
    ) as demo:
        gr.Markdown("""
        # 🎫 Ticket Classifier
        Classifies customer messages into one of:  
        **Incident** • **Request** • **Problem** • **Change**
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                provider_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_PROVIDERS.values()),
                    value=SUPPORTED_PROVIDERS.get(config["provider"], list(SUPPORTED_PROVIDERS.values())[0]),
                    label="LLM Provider",
                    interactive=True
                )
                api_key_input = gr.Textbox(
                    label="API Key (optional - uses .env if empty)",
                    placeholder="Leave empty to use key from .env file",
                    type="password",
                    interactive=True
                )
                model_info = gr.Textbox(
                    label="Model Name",
                    value=config["model_name"],
                    interactive=False
                )

            with gr.Column(scale=2):
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

        # Classification logic
        def update_model_info(provider_label):
            """Update model info when provider changes."""
            provider_key = [k for k, v in SUPPORTED_PROVIDERS.items() if v == provider_label][0]
            return DEFAULT_MODELS.get(provider_key, "")

        provider_dropdown.change(
            fn=update_model_info,
            inputs=provider_dropdown,
            outputs=model_info
        )

        txt.change(
            fn=lambda msg, prov, key: classify(msg, prov.split(" (")[0].lower(), key),
            inputs=[txt, provider_dropdown, api_key_input],
            outputs=out
        )

        classify_btn.click(
            fn=lambda msg, prov, key: classify(msg, prov.split(" (")[0].lower(), key),
            inputs=[txt, provider_dropdown, api_key_input],
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
