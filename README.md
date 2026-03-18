# GenAI Support-Ticket Classifier

An LLM-based support-ticket classification system.
---

## Use Case

**Automated Support Ticket Routing:** Classify incoming customer support messages into one of four categories using LLM reasoning:

- **Incident** — Unexpected issues requiring immediate attention (system outage, data loss)
- **Request** — Routine inquiries and service requests (password reset, feature request)
- **Problem** — Underlying/systemic issues causing multiple incidents (recurring bug)
- **Change** — Planned updates, configurations, or version upgrades
  
The system leverages the semantic understanding of LLMs to accurately classify customer support tickets.
---

## Key Features
**MLflow Demo** — Register prompts, run evaluations, optimize prompts leveraging MlFlow  
**CLI Tools** — Command line interface for batch predictions and prompt management  
**Environment-based Config** — All settings via `.env` file for easy deployment 
**Flexible LLM Providers** — Switch between LLMs at runtime    
**No Backend Required** — Works standalone—no database or external server needed  

##  [Deployed & hosted on Hugging Face](https://huggingface.co/spaces/kaustuvkunal/genai-support-ticket-classifier)
---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/kaustuvkunal/genai-support-ticket-classifier-mlflow-demo.git
cd genai-support-ticket-classifier-mlflow-demo
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and add your LLM provider key:

```env
LLM_PROVIDER=groq
MODEL_NAME=llama-3.1-8b-instant
GROQ_API_KEY=your_groq_api_key_here
```

Or use OpenAI:
```env
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o
OPENAI_API_KEY=your_openai_api_key_here
```

---

##  MLflow Demo CLI Workflow

Demo of MLflow capabilities - prompt registration, evaluation and optimization

### 1. Start MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 --port 5000
```

### 2. Register a Prompt

```bash
python -m src.cli register-prompt
```


### 3. Evaluate Prompt

```bash
python3 -m src.cli evaluate
```

Uses latest registered prompt (default)
`python3 -m src.cli evaluate`

Evaluate against a specific registered prompt version
`python3 -m src.cli evaluate --prompt-uri prompts:/support-ticket-classifier-prompt/1`


### 4. Make Prediction (via CLI)

Use the inline prompt from `src/prompt.py` (default — no MLflow server required):
```bash
python3 -m src.cli predict "My service is down"
```

Use a specific registered prompt version from the MLflow registry:
```bash
python3 -m src.cli predict "My service is down" --prompt-uri prompts:/support-ticket-classifier-prompt/1
```



### 5. Optimize Prompt

Optimize the latest registered prompt (default):
```bash
python3 -m src.cli optimize
```

Optimize a specific version by number:
```bash
python3 -m src.cli optimize --prompt-version 1
```

Optimize using an explicit prompt URI:
```bash
python3 -m src.cli optimize --prompt-uri prompts:/support-ticket-classifier-prompt/1
```

Limit scorer calls for faster runs during development:
```bash
python3 -m src.cli optimize --max-metric-calls 50
```
---

# Deploy Project as Gradio WebApp

### Deploy on local host : Simple Interactive UI

```bash
python3 app.py
```

Open `http://localhost:7860` in your browser.

The app provides:
- **Interactive classification** — Enter a message and run classification from the UI
- **Provider selection** — Switch between  LLMs Groq and OpenAI at runtime
- **Manual submission** — Use the Classify button to run predictions

---

## Deploy & host on Hugging Face Spaces

### Setup

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with **Gradio** runtime
2. Push this repository to the Space

### Configure Secrets

In **Space Settings** → **Secrets**, add your API keys:

```
LLM_PROVIDER=groq
MODEL_NAME=llama-3.1-8b-instant
GROQ_API_KEY=your_groq_key
# Optional: Add auth
USERNAME=your_username
PASSWD=your_password
```

The Space automatically runs `python app.py` on startup.

---

## Supported LLM Providers

| Provider | Model Example | Setup |
|----------|---------------|-------|
| **Groq** | `llama-3.1-8b-instant` | `GROQ_API_KEY=...` |
| **OpenAI** | `gpt-3.5-turbo` | `OPENAI_API_KEY=...` |

Switch providers at runtime via the UI dropdown.

---

## Project Structure

```
.
├── app.py                    # Standalone Gradio app (deploy this!)
├── src/
│   ├── __init__.py
│   ├── cli.py               # MLflow CLI commands
│   ├── config.py            # Configuration loading
│   ├── predict.py           # Prediction logic
│   ├── prompt.py            # Prompt templates
│   └── ...                  # Other utilities
├── tests/                   # Unit tests
├── dataset/                 # Sample data for evaluation
├── requirements.txt         # Python dependencies
├── .env.example            # Configuration template
└── README.md              # This file
```

---

## Configuration Reference

### Environment Variables

**Gradio App Settings:**
- `LLM_PROVIDER` — Which provider to use (default: `groq`)
- `MODEL_NAME` — Model name (overrides defaults)
- API keys — `GROQ_API_KEY`, `OPENAI_API_KEY`

**Authentication:**
- `USERNAME` — Username for UI auth (optional)
- `PASSWD` — Password for UI auth (optional)

**MLflow (CLI only):**
- `MLFLOW_TRACKING_URI` — MLflow server (e.g., `http://127.0.0.1:5000`)
- `MLFLOW_EXPERIMENT` — Experiment name
- `PROMPT_NAME` — Prompt registry name

---

## Testing

Run unit tests:

```bash
pytest
```

Sample test messages in the app:
- "My laptop won't connect to Wi-Fi after the update." → Incident
- "I would like to request a new employee badge." → Request
- "The server keeps crashing every night at 2 AM." → Problem
- "Can we add SAML SSO support to the app?" → Change

---

## Troubleshooting

**"GROQ_API_KEY not set"**
- Ensure `.env` file exists with your API key
- Run `cp .env.example .env` and add your key

**"Module not found" errors**
- Install missing packages: `pip install -r requirements.txt`
- For specific providers: `pip install openai groq`

**Port 7860 already in use (local)**
- Change port: `python app.py --server_port 7861`
- Or kill existing process: `lsof -i :7860`

---

## License

MIT License — See [LICENSE](LICENSE) file

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

**Built with:**
- [Groq API](https://groq.com/) — Ultra-fast LLM inference
- [OpenAI API](https://openai.com/) — GPT models
- [Gradio](https://gradio.app/) — Web UI framework
- [MLflow](https://mlflow.org/) — ML workflow management
