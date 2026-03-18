import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict


def _parse_dt(s: str) -> datetime | None:
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def main():
    root = Path(__file__).resolve().parent.parent
    p = root / "logs" / "ai_entry_audit.jsonl"
    if not p.exists():
        print("missing:", p)
        return

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=3)

    n = 0
    by_norm = Counter()
    by_raw = Counter()
    by_codes = Counter()
    latency_none = 0
    raw_json_fail = 0

    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            raw_json_fail += 1
            continue
        if not isinstance(rec, dict) or rec.get("event_type") != "confirm_entry":
            continue

        ts = rec.get("timestamp_utc") or _as_dict(rec.get("response")).get("timestamp_utc")
        if not isinstance(ts, str):
            continue
        dt = _parse_dt(ts)
        if dt is None or dt < start:
            continue

        n += 1
        r = _as_dict(rec.get("response"))
        rr = _as_dict(rec.get("response_raw"))
        by_norm[str(r.get("decision"))] += 1
        by_raw[str(rr.get("decision"))] += 1
        if r.get("latency_ms") is None:
            latency_none += 1
        codes = r.get("reason_codes")
        if isinstance(codes, list):
            for c in codes:
                by_codes[str(c)] += 1

    print("confirm_entry_last_3d:", n)
    print("normalized_decisions:", dict(by_norm))
    print("raw_decisions:", dict(by_raw))
    print("latency_none:", latency_none)
    print("top_reason_codes:", by_codes.most_common(20))
    if raw_json_fail:
        print("json_parse_fail_lines:", raw_json_fail)


if __name__ == "__main__":
    main()

