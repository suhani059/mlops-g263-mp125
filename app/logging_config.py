"""JSON logging: one JSON object per line, written to the screen (stdout)."""
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Turn a log record into one JSON line. Extra fields passed with
    logger.info(..., extra={"fields": {...}}) are merged into the object."""

    def format(self, record):
        out = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out)


def get_logger(name="mp125-api"):
    logger = logging.getLogger(name)
    if not logger.handlers:                        # do not add a handler twice
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
