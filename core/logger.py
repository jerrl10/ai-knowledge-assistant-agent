import json
import logging
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "time": datetime.now(),
            "level": record.levelname,
            "event": record.name,
            "message": record.getMessage(),
        }
        if record.args:
            log_obj.update(record.args)
        return json.dumps(log_obj)


logger = logging.getLogger("agent")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
