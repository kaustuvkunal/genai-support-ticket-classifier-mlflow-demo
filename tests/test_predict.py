import importlib
import pytest

predict_module = importlib.import_module("src.predict")
from src.predict import predict_from_inputs


def test_predict_from_inputs_calls_predict(monkeypatch):
    called = {"args": None, "kwargs": None}

    def fake_predict(config, customer_message: str, **kwargs) -> str:
        called["args"] = (config, customer_message)
        called["kwargs"] = kwargs
        return "incident"

    monkeypatch.setattr(predict_module, "predict", fake_predict)

    class DummyConfig:
        pass

    result = predict_from_inputs(DummyConfig(), {"customer_message": "hello"})

    assert result == "incident"
    assert called["args"][1] == "hello"
    assert called["kwargs"]["prompt_uri"] is None
    assert called["kwargs"]["prompt_template"] is None
    assert called["kwargs"]["client"] is None


def test_predict_from_inputs_missing_message():
    with pytest.raises(ValueError):
        predict_from_inputs(None, {})
