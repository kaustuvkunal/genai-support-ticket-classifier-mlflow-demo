"""Prompt helpers for the ticket classifier demo."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import Config


# Absolute path to the canonical finalised prompt maintained by the user.
_FINALISE_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "finalise_prompt.py"

PROMPT_TEMPLATE = """\
Classify the following customer support message into **exactly one** of these categories:

- Incident  : unexpected issue requiring immediate attention
- Request   : routine inquiry or service request
- Problem   : underlying / systemic issue causing multiple incidents
- Change    : planned change, update or configuration request

Customer message:
{{customer_message}}

Return **only** the category name.
Allowed answers: Incident, Request, Problem, Change
No explanation. No extra text.
"""


def load_prompt_uri(config: Config, version: Optional[str] = "latest") -> str:
    """Build the MLflow URI for the registered prompt."""
    return f"prompts:/{config.prompt_template_name}@{version}"
