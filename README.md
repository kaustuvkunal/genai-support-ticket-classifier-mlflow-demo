# GenAI Support-Ticket Classifier

An LLM-based IT support-ticket classification system.
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
- **MLflow Demo** — Register prompts, run evaluations, optimize prompts leveraging MLFlow  
- **CLI Tools** — Command line interface for batch predictions and prompt management  
- **Environment-based Config** — All settings via `.env` file for easy deployment 
- **Flexible LLM Providers** — Switch between LLMs at runtime    
- **No Backend Required** — Works standalone—no database or external server needed  

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

Example :
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
The below command **registers the prompt** (defined in `src/prompt.py`) with MLflow

```bash
python3 -m src.cli register-prompt
```


### 3. Evaluate Prompt

python3 -m src.cli evaluate 


To evaluate the **latest registered prompt** (default behavior):
  ```bash
  python3 -m src.cli evaluate
  ```

To evaluate a **specific registered prompt version**:
  ``` bash
  python3 -m src.cli evaluate --prompt-uri prompts:/support-ticket-classifier-prompt/1
  ```

### 4. Make Prediction  

Uses the **inline prompt*** for prediction from `src/prompt.py` (default — no MLflow server required):

```bash
python3 -m src.cli predict "My service is down"
```

Uses a **specific registered prompt version** for prediction from the MLflow registry:

```bash
python3 -m src.cli predict "My service is down" --prompt-uri prompts:/support-ticket-classifier-prompt/1
```


### 5. Optimize Prompt

To optimize the **latest registered prompt version** (default):
  ```bash
  python3 -m src.cli optimize
  ```

To optimize a **specific version** by number:
```bash
python3 -m src.cli optimize --prompt-version 1
```

Optimize using an **explicit prompt URI**:
```bash
python3 -m src.cli optimize --prompt-uri prompts:/support-ticket-classifier-prompt/1
```

**Limit scorer calls** for faster runs:
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

### Deployed & host on Hugging Face Spaces -[Here](https://huggingface.co/spaces/kaustuvkunal/genai-support-ticket-classifier)

<img width="1361" height="747" alt="Screenshot 2026-03-18 at 6 18 25 PM" src="https://github.com/user-attachments/assets/2026d04d-4b48-4243-bf23-ea3f11b1a05e" />

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
- [MLflow](https://mlflow.org/) — ML workflow management
- [Groq API](https://groq.com/) — Ultra-fast LLM inference
- [OpenAI API](https://openai.com/) — GPT models
- [Gradio](https://gradio.app/) — Web UI framework

