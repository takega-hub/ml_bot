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
    
    def collect_klines(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 200,  # Максимум за один запрос в Bybit
        save_to_file: bool = True,
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
        
        all_data: List[Dict[str, Any]] = []
        current_end = end_date
        
        # Bybit API работает в обратном направлении (от конца к началу)
        # Нужно собирать данные порциями
        while current_end > start_date:
            try:
                # Конвертируем datetime в timestamp (миллисекунды)
                end_timestamp = int(current_end.timestamp() * 1000)
                
                # Получаем данные через API
                # Используем метод get_kline из BybitClient
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
                
                # Конвертируем в DataFrame
                for kline in klines:
                    all_data.append({
                        "timestamp": int(kline[0]),  # startTime в миллисекундах
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                        "turnover": float(kline[6]),  # turnover (опционально)
                    })
                
                # Обновляем current_end на самую раннюю свечу из полученных
                earliest_timestamp = int(klines[-1][0])  # Последняя свеча в списке - самая ранняя
                current_end = datetime.fromtimestamp(earliest_timestamp / 1000)
                
                print(f"[data_collector] Collected {len(klines)} candles, total: {len(all_data)}, earliest: {current_end.date()}")
                
                # Задержка чтобы не превысить rate limit
                time.sleep(0.2)
                
            except Exception as e:
                print(f"[data_collector] Error collecting data: {e}")
                time.sleep(1)
                continue
        
        if not all_data:
            print(f"[data_collector] No data collected")
            return pd.DataFrame()
        
        # Создаем DataFrame и сортируем по времени
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Удаляем дубликаты
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        
        print(f"[data_collector] Total collected: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Сохраняем в файл
        if save_to_file:
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
            )
            data[interval] = df
            time.sleep(1)  # Пауза между таймфреймами
        
        return data


def main():
    """Пример использования для сбора данных."""
    from bot.config import load_settings
    
    settings = load_settings()
    collector = DataCollector(settings.api)
    
    # Собираем данные для ETH/USDT за последние 6 месяцев
    symbol = "ETHUSDT"
    interval = "15"  # 15 минут
    
    df = collector.collect_klines(
        symbol=symbol,
        interval=interval,
        start_date=None,  # Автоматически 6 месяцев назад
        end_date=None,    # До текущего момента
    )
    
    print(f"\nCollected {len(df)} candles")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()

