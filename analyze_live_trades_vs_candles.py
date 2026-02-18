import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


def _col_letters_to_index(col: str) -> int:
    col = col.upper()
    idx = 0
    for ch in col:
        if "A" <= ch <= "Z":
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _parse_xlsx_minimal(path: Path) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(path) as z:
            shared_strings: list[str] = []
            try:
                with z.open("xl/sharedStrings.xml") as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    for si in root.iter():
                        if si.tag.endswith("si"):
                            texts = []
                            for t in si.iter():
                                if t.tag.endswith("t") and t.text is not None:
                                    texts.append(t.text)
                            shared_strings.append("".join(texts))
            except KeyError:
                shared_strings = []

            with z.open("xl/worksheets/sheet1.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()

            rows = []
            for row in root.iter():
                if not row.tag.endswith("row"):
                    continue
                cells = {}
                for c in row:
                    if not c.tag.endswith("c"):
                        continue
                    r = c.get("r", "")
                    letters = "".join(ch for ch in r if ch.isalpha())
                    if not letters:
                        continue
                    col_idx = _col_letters_to_index(letters)
                    t = c.get("t")

                    val = ""
                    if t == "inlineStr":
                        texts = []
                        for child in c.iter():
                            if child.tag.endswith("t") and child.text is not None:
                                texts.append(child.text)
                        val = "".join(texts)
                    else:
                        v_elem = None
                        for child in c:
                            if child.tag.endswith("v"):
                                v_elem = child
                                break
                        if v_elem is None or v_elem.text is None:
                            val = ""
                        else:
                            text = v_elem.text
                            if t == "s":
                                try:
                                    s_idx = int(text)
                                    val = shared_strings[s_idx] if 0 <= s_idx < len(shared_strings) else ""
                                except Exception:
                                    val = ""
                            else:
                                val = text
                    cells[col_idx] = val
                if not cells:
                    continue
                max_col = max(cells.keys())
                row_values = [cells.get(i, "") for i in range(max_col + 1)]
                rows.append(row_values)

        if not rows:
            return pd.DataFrame()

        header = rows[0]
        data_rows = rows[1:]
        df = pd.DataFrame(data_rows, columns=header)
        return df
    except Exception as e:
        print("Ошибка парсинга Excel", path, ":", e)
        return pd.DataFrame()


def load_trades_from_excel(trade_dir: Path) -> pd.DataFrame:
    files = sorted(trade_dir.glob("trades_history_*.xlsx"))
    if not files:
        print("Нет файлов сделок в", trade_dir)
        return pd.DataFrame()

    frames = []
    for f in files:
        df = _parse_xlsx_minimal(f)
        if df.empty:
            print("Не удалось прочитать файл (пустой после парсинга)", f)
            continue
        df["_source_file"] = f.name
        frames.append(df)

    if not frames:
        print("Не удалось загрузить ни один Excel-файл сделок")
        return pd.DataFrame()

    trades = pd.concat(frames, ignore_index=True)

    for col in ["Время входа", "Время выхода"]:
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], errors="coerce")

    num_before = len(trades)
    trades = trades[trades["Время входа"].notna()]
    num_after = len(trades)
    if num_after < num_before:
        print(f"Отфильтровано {num_before - num_after} строк без времени входа")

    return trades


def load_trades_from_state(root: Path) -> pd.DataFrame:
    state_path = root / "runtime_state.json"
    if not state_path.exists():
        print("Файл состояния не найден:", state_path)
        return pd.DataFrame()

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Не удалось прочитать runtime_state.json:", e)
        return pd.DataFrame()

    trades_data = data.get("trades", [])
    if not trades_data:
        print("В runtime_state.json нет раздела trades")
        return pd.DataFrame()

    trades = pd.DataFrame(trades_data)

    rename_map = {
        "symbol": "Символ",
        "side": "Направление",
        "entry_price": "Цена входа",
        "exit_price": "Цена выхода",
        "pnl_usd": "PnL (USD)",
        "pnl_pct": "PnL (%)",
        "entry_time": "Время входа",
        "exit_time": "Время выхода",
        "status": "Статус",
        "model_name": "Модель ML",
        "horizon": "Горизонт",
        "entry_reason": "Причина входа",
        "exit_reason": "Причина выхода",
        "confidence": "Уверенность (%)",
        "take_profit": "Take Profit",
        "stop_loss": "Stop Loss",
        "leverage": "Плечо",
        "margin_usd": "Маржа (USD)",
        "signal_strength": "Сила сигнала",
    }

    trades = trades.rename(columns=rename_map)

    for col in ["Время входа", "Время выхода"]:
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], errors="coerce")

    if "Уверенность (%)" in trades.columns:
        trades["Уверенность (%)"] = trades["Уверенность (%)"].astype(float) * 100.0

    num_before = len(trades)
    trades = trades[trades["Время входа"].notna()]
    num_after = len(trades)
    if num_after < num_before:
        print(f"Отфильтровано {num_before - num_after} строк без времени входа (state)")

    print(f"Загружено {len(trades)} сделок из runtime_state.json")
    return trades


def load_trades(root: Path) -> pd.DataFrame:
    trade_dir = root / "trade_history"

    trades_excel = load_trades_from_excel(trade_dir)
    if not trades_excel.empty:
        print(f"Загружено {len(trades_excel)} сделок из Excel")
        return trades_excel

    print("Пробуем загрузить сделки из runtime_state.json...")
    trades_state = load_trades_from_state(root)
    return trades_state


def load_candles(ml_data_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path_15 = ml_data_dir / f"{symbol}_15_cache.csv"
        path_60 = ml_data_dir / f"{symbol}_60_cache.csv"

        df = None
        if path_15.exists():
            df = pd.read_csv(path_15)
        elif path_60.exists():
            df = pd.read_csv(path_60)

        if df is None or df.empty:
            print(f"⚠️ Нет свечных данных для {symbol} (15m/60m)")
            continue

        if "timestamp" not in df.columns:
            print(f"⚠️ В файле свечей для {symbol} нет колонки timestamp")
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(
            f"Свечи {symbol}: {len(df)} строк, "
            f"{df['timestamp'].min()} .. {df['timestamp'].max()}"
        )

        result[symbol] = df
    return result


def compute_mfe_mae_for_trade(row: pd.Series, candles: pd.DataFrame) -> tuple[float, float, float]:
    entry_time = row["Время входа"]
    exit_time = row.get("Время выхода")

    if pd.isna(entry_time) or pd.isna(exit_time):
        return np.nan, np.nan, np.nan

    mask = (candles["timestamp"] >= entry_time) & (candles["timestamp"] <= exit_time)
    path = candles.loc[mask]
    if path.empty:
        return np.nan, np.nan, np.nan

    side = str(row.get("Направление", "")).lower()
    entry_price = row.get("Цена входа")
    if not isinstance(entry_price, (int, float)) or entry_price <= 0:
        return np.nan, np.nan, np.nan

    if {"high", "low"}.issubset(path.columns):
        if "long" in side or "buy" in side:
            max_price = float(path["high"].max())
            min_price = float(path["low"].min())
            mfe_pct = (max_price / entry_price - 1.0) * 100.0
            mae_pct = (min_price / entry_price - 1.0) * 100.0
        elif "short" in side or "sell" in side:
            min_price = float(path["low"].min())
            max_price = float(path["high"].max())
            mfe_pct = (entry_price / min_price - 1.0) * 100.0 if min_price > 0 else np.nan
            mae_pct = (entry_price / max_price - 1.0) * 100.0 if max_price > 0 else np.nan
        else:
            return np.nan, np.nan, np.nan
    else:
        if "close" not in path.columns:
            return np.nan, np.nan, np.nan

        closes = path["close"].astype(float)
        if "long" in side or "buy" in side:
            rel = closes / entry_price - 1.0
        elif "short" in side or "sell" in side:
            rel = entry_price / closes - 1.0
        else:
            return np.nan, np.nan, np.nan

        mfe_pct = rel.max() * 100.0
        mae_pct = rel.min() * 100.0

    pnl_pct = row.get("PnL (%)")
    realized_pct = np.nan
    if pnl_pct is not None and not (isinstance(pnl_pct, float) and np.isnan(pnl_pct)):
        if isinstance(pnl_pct, str):
            txt = pnl_pct.replace("%", "").replace(",", ".").strip()
            try:
                realized_pct = float(txt)
            except Exception:
                realized_pct = np.nan
        else:
            try:
                realized_pct = float(pnl_pct)
            except Exception:
                realized_pct = np.nan

    return mfe_pct, mae_pct, realized_pct


def enrich_with_candle_stats(trades: pd.DataFrame, candle_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trades = trades.copy()

    if "Символ" not in trades.columns:
        print("В данных сделок нет колонки 'Символ'")
        trades["mfe_pct"] = np.nan
        trades["mae_pct"] = np.nan
        trades["realized_pct"] = trades.get("PnL (%)", np.nan)
        return trades

    if "Цена входа" in trades.columns:
        trades["Цена входа"] = (
            trades["Цена входа"]
            .astype(str)
            .str.replace(" ", "")
            .str.replace(",", ".", regex=False)
        )
        trades["Цена входа"] = pd.to_numeric(trades["Цена входа"], errors="coerce")

    print("Подготовка MFE/MAE для сделок:", len(trades))
    if "Время входа" in trades.columns:
        print("  Непустых Время входа:", int(trades["Время входа"].notna().sum()))
    if "Время выхода" in trades.columns:
        print("  Непустых Время выхода:", int(trades["Время выхода"].notna().sum()))

    mfe_list = []
    mae_list = []
    realized_list = []

    for idx, row in trades.iterrows():
        symbol = row["Символ"]
        df_sym = candle_data.get(symbol)
        if df_sym is None:
            mfe_list.append(np.nan)
            mae_list.append(np.nan)
            realized_list.append(row.get("PnL (%)", np.nan))
            continue

        mfe, mae, realized = compute_mfe_mae_for_trade(row, df_sym)
        mfe_list.append(mfe)
        mae_list.append(mae)
        realized_list.append(realized)

    trades["mfe_pct"] = mfe_list
    trades["mae_pct"] = mae_list
    trades["realized_pct"] = realized_list

    if "Время выхода" in trades.columns and "Время входа" in trades.columns:
        trades["holding_min"] = (trades["Время выхода"] - trades["Время входа"]).dt.total_seconds() / 60.0

    return trades


def print_overall_stats(trades: pd.DataFrame) -> None:
    print("\n=== Общая статистика по сделкам ===")
    total = len(trades)
    print("Всего сделок:", total)

    if "PnL (USD)" in trades.columns:
        pnl_usd = trades["PnL (USD)"].astype(float)
        print("Суммарный PnL (USD):", float(pnl_usd.sum()))
        print("Средний PnL (USD):", float(pnl_usd.mean()))
        print("Медианный PnL (USD):", float(pnl_usd.median()))
        win_rate = float((pnl_usd > 0).mean()) if total > 0 else 0.0
        print("Win rate (PnL>0):", win_rate)

    if "Символ" in trades.columns:
        print("\nСделки по символам:")
        print(trades["Символ"].value_counts())

    if "Направление" in trades.columns:
        print("\nСделки по направлению:")
        print(trades["Направление"].value_counts())


def print_candle_based_stats(trades: pd.DataFrame) -> None:
    df = trades.dropna(subset=["mfe_pct", "mae_pct", "realized_pct"])
    if df.empty:
        print("\nНет сделок, для которых удалось посчитать MFE/MAE")
        return

    print("\n=== Статистика с учетом свечей (MFE/MAE) ===")
    print("Число сделок с рассчитанным MFE/MAE:", len(df))

    winners = df[df["realized_pct"] > 0]
    losers = df[df["realized_pct"] <= 0]

    for name, subset in [("Прибыльные", winners), ("Убыточные", losers), ("Все", df)]:
        if subset.empty:
            continue
        print(f"\n{name} сделки: {len(subset)}")
        print("  Средний реализованный результат (%):", float(subset["realized_pct"].mean()))
        print("  Средний MFE (%):", float(subset["mfe_pct"].mean()))
        print("  Медианный MFE (%):", float(subset["mfe_pct"].median()))
        print("  Средний MAE (%):", float(subset["mae_pct"].mean()))
        print("  Медианный MAE (%):", float(subset["mae_pct"].median()))

        diff = subset["mfe_pct"] - subset["realized_pct"]
        print("  Средний запас движения (MFE - realized, п.п.):", float(diff.mean()))
        print(
            "  Доля сделок, где MFE >= realized + 1 п.п.:",
            float((diff >= 1.0).mean()),
        )

    if "holding_min" in df.columns:
        print("\nРаспределение длительности сделок (минуты):")
        print(df["holding_min"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))


def main() -> None:
    root = Path(__file__).resolve().parent
    ml_data_dir = root / "ml_data"

    print("Корень проекта:", root)
    print("Каталог свечей:", ml_data_dir)

    trades = load_trades(root)
    if trades.empty:
        return

    print_overall_stats(trades)

    symbols = sorted(trades["Символ"].dropna().unique().tolist()) if "Символ" in trades.columns else []
    print("\nСимволы в сделках:", symbols)

    candle_data = load_candles(ml_data_dir, symbols)
    if not candle_data:
        print("Не удалось загрузить свечи для каких-либо символов")
        return

    trades_with_candles = enrich_with_candle_stats(trades, candle_data)
    print_candle_based_stats(trades_with_candles)


if __name__ == "__main__":
    main()
