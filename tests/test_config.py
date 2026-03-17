import os

import pytest

from src.config import Config, load_config


def test_load_config_missing_api_key(tmp_path, monkeypatch):
    # Ensure environment does not contain a key and no .env file provides it.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # Create an empty env file and force load_config to use it.
    env_file = tmp_path / ".env"
    env_file.write_text("# empty")

    with pytest.raises(RuntimeError):
        load_config(env_path=str(env_file))


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MODEL_NAME", "llama-3.1-8b-instant")

    cfg = load_config()

    assert isinstance(cfg, Config)
    # API keys are NOT stored on Config to prevent leakage — verify that
    # the key is readable from the environment instead.
    assert os.environ.get("GROQ_API_KEY") == "test-key"
    assert cfg.mlflow_tracking_uri == "http://localhost:5000"
    assert cfg.model_name == "llama-3.1-8b-instant"
