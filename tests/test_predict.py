import importlib
import pytest

predict_module = importlib.import_module("src.predict")
from src.predict import predict_from_inputs


def test_predict_from_inputs_calls_predict(monkeypatch):
    called = {"args": None}

    def fake_predict(config, customer_message: str, prompt_uri=None) -> str:
        called["args"] = (config, customer_message)
        return "incident"

    monkeypatch.setattr(predict_module, "predict", fake_predict)

    class DummyConfig:
        pass

    result = predict_from_inputs(DummyConfig(), {"customer_message": "hello"})

    assert result == "incident"
    assert called["args"][1] == "hello"


def test_predict_from_inputs_missing_message():
    with pytest.raises(ValueError):
        predict_from_inputs(None, {})
