"""
Модуль для сбора исторических данных для обучения ML-моделей.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from bot.exchange.bybit_client import BybitClient
from bot.config import ApiSettings


class DataCollector:
    """
    Собирает исторические данные с Bybit для обучения ML-моделей.
    """
    
    def __init__(self, api_settings: ApiSettings):
        self.client = BybitClient(api_settings)
        self.data_dir = Path(__file__).parent.parent.parent / "ml_data"
        self.data_dir.mkdir(exist_ok=True)

    def _interval_to_timedelta(self, interval: str) -> timedelta:
        """Возвращает timedelta для интервала свечей."""
        try:
            minutes = int(interval)
            return timedelta(minutes=minutes)
        except ValueError:
            # Fallback: если интервал не числовой
            return timedelta(minutes=15)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        """Путь к файлу кеша для символа и интервала."""
        return self.data_dir / f"{symbol}_{interval}_cache.csv"

    def _load_cache(self, symbol: str, interval: str) -> pd.DataFrame:
        """Загружает кешированные данные, если они есть."""
        cache_path = self._cache_path(symbol, interval)
        if not cache_path.exists():
            # Если кеш-файл отсутствует, пытаемся собрать данные из существующих файлов
            pattern = f"{symbol}_{interval}_*.csv"
            candidate_files = [p for p in self.data_dir.glob(pattern) if p.name != cache_path.name]
            if not candidate_files:
                return pd.DataFrame()
            try:
                frames = []
                for file_path in candidate_files:
                    df = pd.read_csv(file_path)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        frames.append(df)
                if not frames:
                    return pd.DataFrame()
                merged = pd.concat(frames, ignore_index=True)
                merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
                print(f"[data_collector] Built cache from {len(candidate_files)} files")
                return merged
            except Exception as e:
                print(f"[data_collector] Failed to build cache from existing files: {e}")
                return pd.DataFrame()
        try:
            df = pd.read_csv(cache_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
            return df
        except Exception as e:
            print(f"[data_collector] Failed to load cache {cache_path}: {e}")
            return pd.DataFrame()

    def _save_cache(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """Сохраняет данные в кеш."""
        cache_path = self._cache_path(symbol, interval)
        try:
            df.to_csv(cache_path, index=False)
            print(f"[data_collector] Cache saved to {cache_path}")
        except Exception as e:
            print(f"[data_collector] Failed to save cache {cache_path}: {e}")

    def _fetch_klines_range(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Загружает свечи из API за заданный диапазон."""
        print(f"[data_collector] Fetching {symbol} {interval} from {start_date.date()} to {end_date.date()}")

        all_data: List[Dict[str, Any]] = []
        current_end = end_date

        while current_end > start_date:
            try:
                end_timestamp = int(current_end.timestamp() * 1000)
                resp = self.client.get_kline(
                    symbol=symbol,
                    interval=interval,
                    end=end_timestamp,
                    limit=min(limit, 200),
                )

                if resp.get("retCode") != 0:
                    print(f"[data_collector] Error: {resp.get('retMsg', 'Unknown error')}")
                    break

                result = resp.get("result", {})
                klines = result.get("list", []) or result.get("rows", [])

                if not klines:
                    print(f"[data_collector] No more data available")
                    break

                for kline in klines:
                    all_data.append({
                        "timestamp": int(kline[0]),
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                        "turnover": float(kline[6]),
                    })

                earliest_timestamp = int(klines[-1][0])
                current_end = datetime.fromtimestamp(earliest_timestamp / 1000)

                print(f"[data_collector] Collected {len(klines)} candles, total: {len(all_data)}, earliest: {current_end.date()}")
                time.sleep(0.2)

            except Exception as e:
                print(f"[data_collector] Error collecting data: {e}")
                time.sleep(1)
                continue

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        return df
    
    def collect_klines(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 200,  # Максимум за один запрос в Bybit
        save_to_file: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Собирает исторические свечи за указанный период.
        
        Args:
            symbol: Торговая пара (например, 'ETHUSDT')
            interval: Таймфрейм ('15', '60', '240' для 15m, 1h, 4h)
            start_date: Начальная дата (если None, берется 6 месяцев назад)
            end_date: Конечная дата (если None, берется текущая дата)
            limit: Количество свечей за запрос (максимум 200 для Bybit)
            save_to_file: Сохранять ли данные в файл
        
        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=180)  # 6 месяцев по умолчанию

        print(f"[data_collector] Collecting {symbol} {interval} from {start_date.date()} to {end_date.date()}")

        interval_delta = self._interval_to_timedelta(interval)

        if use_cache:
            cache_df = self._load_cache(symbol, interval)
            if not cache_df.empty:
                cache_min = cache_df["timestamp"].min()
                cache_max = cache_df["timestamp"].max()

                print(f"[data_collector] Cache found: {len(cache_df)} candles from {cache_min} to {cache_max}")

                frames = [cache_df]

                # Дозагрузка более старых данных
                if start_date < cache_min:
                    older_end = cache_min - interval_delta
                    if older_end > start_date:
                        older_df = self._fetch_klines_range(symbol, interval, start_date, older_end, limit=limit)
                        if not older_df.empty:
                            frames.append(older_df)

                # Дозагрузка более новых данных
                if end_date > cache_max:
                    newer_start = cache_max + interval_delta
                    if end_date > newer_start:
                        newer_df = self._fetch_klines_range(symbol, interval, newer_start, end_date, limit=limit)
                        if not newer_df.empty:
                            frames.append(newer_df)

                df_all = pd.concat(frames, ignore_index=True)
                df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
                df_all = df_all.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

                if save_to_file:
                    self._save_cache(symbol, interval, df_all)

                df_filtered = df_all[(df_all["timestamp"] >= start_date) & (df_all["timestamp"] <= end_date)].reset_index(drop=True)
                print(f"[data_collector] Using cached+delta data: {len(df_filtered)} candles")
                return df_filtered

        # Если кеша нет или он отключен, загружаем полный диапазон
        df = self._fetch_klines_range(symbol, interval, start_date, end_date, limit=limit)
        if df.empty:
            print(f"[data_collector] No data collected")
            return pd.DataFrame()

        print(f"[data_collector] Total collected: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")

        if save_to_file:
            # Сохраняем кеш
            if use_cache:
                self._save_cache(symbol, interval, df)
            # Сохраняем исторический файл по диапазону
            filename = f"{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            filepath = self.data_dir / filename
            df.to_csv(filepath, index=False)
            print(f"[data_collector] Data saved to {filepath}")

        return df
    
    def load_from_file(self, filepath: str) -> pd.DataFrame:
        """Загружает данные из CSV файла."""
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    
    def collect_multiple_timeframes(
        self,
        symbol: str,
        intervals: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Собирает данные для нескольких таймфреймов одновременно.
        
        Returns:
            Словарь {interval: DataFrame}
        """
        data = {}
        for interval in intervals:
            print(f"\n[data_collector] Collecting {interval} timeframe...")
            df = self.collect_klines(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
            )
            data[interval] = df
            time.sleep(1)  # Пауза между таймфреймами
        
        return data


def main():
    """CLI для сбора данных."""
    import argparse
    from bot.config import load_settings
    
    parser = argparse.ArgumentParser(description="Collect historical klines from Bybit")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--interval", type=str, default="15", help="Interval: 15, 60, 240")
    parser.add_argument("--years", type=int, default=2, help="How many years back to collect")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD (optional)")
    parser.add_argument("--save", action="store_true", help="Save to CSV")
    parser.add_argument("--no-cache", action="store_true", help="Disable local cache")
    
    args = parser.parse_args()
    
    settings = load_settings()
    collector = DataCollector(settings.api)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=args.years * 365)
    
    df = collector.collect_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_date=start_date,
        end_date=end_date,
        save_to_file=args.save,
        use_cache=not args.no_cache,
    )
    
    print(f"\nCollected {len(df)} candles")
    if len(df) > 0:
        print(df.head())
        print(df.tail())


if __name__ == "__main__":
    main()

