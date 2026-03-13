import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

_lock = threading.Lock()


def append_jsonl(file_path: str, record: Dict[str, Any]) -> Optional[str]:
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return str(path)
    except Exception:
        return None

