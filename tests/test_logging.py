import json
import logging

from app.logging_config import JsonFormatter


def test_json_formatter_makes_valid_json_with_extra_fields():
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "prediction", None, None)
    rec.fields = {"prediction": 12.3, "model_version": 1}
    out = json.loads(JsonFormatter().format(rec))
    assert out["message"] == "prediction"
    assert out["level"] == "INFO"
    assert out["prediction"] == 12.3
    assert "time" in out
