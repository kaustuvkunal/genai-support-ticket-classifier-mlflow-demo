from __future__ import annotations
import os
import logging
from typing import Optional
import gradio as gr

from .config import Config, load_config
from .predict import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _build_auth() -> Optional[tuple[str, str]]:
    username = os.getenv("HF_APP_USERNAME")
    password = os.getenv("HF_APP_PASSWORD")
    if username and password:
        return (username, password)
    return None

def create_app(config: Optional[Config] = None) -> gr.Blocks:
    config = config or load_config()
    
    with gr.Blocks(
        title="GenAI Ticket Classifier",
        css="body { font-family: system-ui; }"
    ) as demo:
        gr.Markdown("""
        # Ticket Classifier
        Classifies customer messages into one of:  
        **Incident** • **Request** • **Problem** • **Change**
        """)
        
        with gr.Row():
            txt = gr.TextArea(
                label="Customer message",
                placeholder="Describe the issue...",
                lines=6
            )
            out = gr.Textbox(label="Predicted category", interactive=False)
        
        # Live classification feels more modern
        txt.change(fn=_classify, inputs=txt, outputs=out)
        
        # Optional explicit button
        # gr.Button("Classify").click(_classify, inputs=txt, outputs=out)
        
        gr.Markdown("Powered by Groq API • fast LLM-based classification")
        
        gr.Examples(
            examples=[
                "My laptop won't connect to Wi-Fi after the update.",
                "I would like to request a new employee badge.",
                "The server keeps crashing every night at 2 AM.",
                "Can we add SAML SSO support to the app?"
            ],
            inputs=txt
        )

    return demo

# This is what HF Spaces looks for
demo = create_app()

# Only for local testing
if __name__ == "__main__":
    auth = _build_auth()
    demo.launch(auth=auth)