import os
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory


def _touch(path: Path, mtime: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"model")
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))


def test_cleanup_deletes_inactive_experiment_models_regardless_of_age():
    from bot.model_manager import ModelManager

    with TemporaryDirectory() as tmp:
        mgr = ModelManager.__new__(ModelManager)
        mgr.models_dir = Path(tmp) / "ml_models"
        mgr.models_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        old = now - timedelta(days=10)

        active_exp = mgr.models_dir / "xgb_BTCUSDT_15_15m__exp_1111111111_aaaaaaaa_15m.pkl"
        inactive_exp_young = mgr.models_dir / "xgb_BTCUSDT_15_15m__exp_2222222222_bbbbbbbb_15m.pkl"
        inactive_nonexp_young = mgr.models_dir / "xgb_BTCUSDT_15_15m.pkl"
        inactive_nonexp_old = mgr.models_dir / "rf_BTCUSDT_60_1h.pkl"

        _touch(active_exp, now)
        _touch(inactive_exp_young, now)
        _touch(inactive_nonexp_young, now)
        _touch(inactive_nonexp_old, old)

        active_paths = {str(active_exp.resolve())}

        preview = mgr.cleanup_old_inactive_models(
            active_model_paths=active_paths,
            min_age_days=7,
            dry_run=True,
        )

        assert preview["ok"] is True
        assert inactive_exp_young.name in preview["candidate_files"]
        assert inactive_nonexp_old.name in preview["candidate_files"]
        assert inactive_nonexp_young.name not in preview["candidate_files"]
        assert active_exp.name not in preview["candidate_files"]
        assert preview["candidate_experiment_count"] == 1
        assert preview["candidate_non_experiment_count"] == 1
        assert preview["skipped_young_count"] == 1

        res = mgr.cleanup_old_inactive_models(
            active_model_paths=active_paths,
            min_age_days=7,
            dry_run=False,
        )

        assert res["ok"] is True
        assert not inactive_exp_young.exists()
        assert not inactive_nonexp_old.exists()
        assert inactive_nonexp_young.exists()
        assert active_exp.exists()

