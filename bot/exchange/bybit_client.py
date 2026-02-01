import time
from typing import Any, Dict, Optional, List

import pandas as pd
from pybit.unified_trading import HTTP
from pybit.exceptions import FailedRequestError

from bot.config import ApiSettings


class BybitClient:
    """
    Thin wrapper around pybit HTTP client.
    Используется для получения свечей с тестнета/реала.
    """

    def __init__(self, api: ApiSettings):
        # В pybit v5.6.2 для тестнета/реала используется флаг testnet,
        # параметр endpoint больше не принимается.
        testnet = "testnet" in (api.base_url or "").lower()
        # Передаём ключи только если они не пустые
        api_key = api.api_key.strip() if api.api_key else None
        api_secret = api.api_secret.strip() if api.api_secret else None
        
        # Диагностика: проверяем наличие ключей
        if not api_key or not api_secret:
            print(f"[bybit] ⚠️ WARNING: API keys are missing or empty!")
            print(f"[bybit]   API Key: {'SET' if api_key else 'NOT SET'} (length: {len(api_key) if api_key else 0})")
            print(f"[bybit]   API Secret: {'SET' if api_secret else 'NOT SET'} (length: {len(api_secret) if api_secret else 0})")
            print(f"[bybit]   Base URL: {api.base_url}")
            print(f"[bybit]   Testnet mode: {testnet}")
            print(f"[bybit]   Please check your .env file and ensure BYBIT_API_KEY and BYBIT_API_SECRET are set correctly.")
        else:
            # Показываем частично скрытый ключ для диагностики
            api_key_display = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            api_secret_display = f"{api_secret[:8]}...{api_secret[-4:]}" if len(api_secret) > 12 else "***"
            print(f"[bybit] Initializing client: testnet={testnet}, base_url={api.base_url}")
            print(f"[bybit]   API Key: {api_key_display} (length: {len(api_key)})")
            print(f"[bybit]   API Secret: {api_secret_display} (length: {len(api_secret)})")
            if testnet:
                print(f"[bybit]   ⚠️ IMPORTANT: Make sure these are TESTNET API keys, not production keys!")
                print(f"[bybit]   Testnet keys can be created at: https://testnet.bybit.com/app/user/api-management")
        
        # Настройки для решения проблем с синхронизацией времени
        # Увеличиваем recv_window до максимума (120 секунд) для учета задержек сети и расхождений времени
        # Это особенно важно при нестабильном интернете или рассинхронизации системного времени
        # Максимальное значение recv_window в Bybit API - 120000 мс (120 секунд)
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
            recv_window=120000,  # Максимум 120 секунд для надежности
        )
        # Убеждаемся, что recv_window установлен правильно (на случай, если pybit не принял параметр)
        if hasattr(self.session, 'recv_window'):
            self.session.recv_window = 120000
        elif hasattr(self.session, '_http_manager') and hasattr(self.session._http_manager, 'recv_window'):
            self.session._http_manager.recv_window = 120000

    def _retry_request(self, func, max_retries: int = 3, delay: float = 1.0, *args, **kwargs) -> Dict[str, Any]:
        """Повторная попытка запроса при rate limit ошибках и ошибках синхронизации времени."""
        for attempt in range(max_retries):
            try:
                resp = func(*args, **kwargs)
                ret_code = resp.get("retCode")
                ret_msg = resp.get("retMsg", "")
                
                # Проверяем на ошибку авторизации (401) - неверные API ключи
                if ret_code == 401:
                    print(f"[bybit] ❌ ERROR 401: Unauthorized - Invalid API keys or keys not for testnet!")
                    print(f"[bybit]   Error message: {ret_msg}")
                    print(f"[bybit]   Please verify:")
                    print(f"[bybit]   1. API keys are correct in .env file")
                    print(f"[bybit]   2. API keys are for TESTNET (not production)")
                    print(f"[bybit]   3. Testnet keys can be created at: https://testnet.bybit.com/app/user/api-management")
                    print(f"[bybit]   4. Keys have proper permissions (Read, Trade)")
                    # Не повторяем запрос при 401 - это не временная ошибка
                    raise FailedRequestError(
                        request=f"Request failed: Invalid API credentials",
                        status_code=401,
                        time=time.time(),
                        response={"retCode": 401, "retMsg": ret_msg}
                    )
                
                # Проверяем на rate limit (403) или другие временные ошибки
                if ret_code == 403 and ("rate limit" in ret_msg.lower() or "usa" in ret_msg.lower()):
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # экспоненциальная задержка
                        print(f"[bybit] Rate limit hit, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                
                # Проверяем на ошибки синхронизации времени (10002)
                if ret_code == 10002 or ("timestamp" in ret_msg.lower() and "recv_window" in ret_msg.lower()):
                    # Парсим информацию о времени из сообщения об ошибке
                    time_diff_ms = None
                    if "req_timestamp" in ret_msg and "server_timestamp" in ret_msg:
                        try:
                            import re
                            req_match = re.search(r'req_timestamp\[(\d+)\]', ret_msg)
                            server_match = re.search(r'server_timestamp\[(\d+)\]', ret_msg)
                            if req_match and server_match:
                                req_ts = int(req_match.group(1))
                                server_ts = int(server_match.group(1))
                                time_diff_ms = abs(req_ts - server_ts)
                                print(f"[bybit] Time difference detected: {time_diff_ms}ms ({time_diff_ms/1000:.2f}s)")
                        except Exception:
                            pass
                    
                    # Увеличиваем задержку перед повторной попыткой при ошибках синхронизации
                    if attempt < max_retries - 1:
                        # При ошибке синхронизации времени делаем более длинную задержку
                        wait_time = max(delay * (2 ** attempt), 3.0)  # Минимум 3 секунды
                        print(f"[bybit] Time sync error (ErrCode: {ret_code}), retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        if time_diff_ms and time_diff_ms > 500:
                            print(f"[bybit]   WARNING: System time is out of sync by {time_diff_ms/1000:.2f}s!")
                            print(f"[bybit]   Please sync your system time with NTP server:")
                            print(f"[bybit]   Windows: Settings > Time & Language > Date & Time > Sync now")
                        else:
                            print(f"[bybit]   Hint: Check system time synchronization. Consider syncing with NTP server.")
                        time.sleep(wait_time)
                        continue
                
                return resp
            except Exception as e:
                error_str = str(e).lower()
                # Проверяем, не связана ли ошибка с синхронизацией времени
                if "timestamp" in error_str or "recv_window" in error_str or "10002" in error_str:
                    # Парсим информацию о времени из сообщения об ошибке
                    time_diff_ms = None
                    error_msg = str(e)
                    if "req_timestamp" in error_msg and "server_timestamp" in error_msg:
                        try:
                            import re
                            req_match = re.search(r'req_timestamp\[(\d+)\]', error_msg)
                            server_match = re.search(r'server_timestamp\[(\d+)\]', error_msg)
                            if req_match and server_match:
                                req_ts = int(req_match.group(1))
                                server_ts = int(server_match.group(1))
                                time_diff_ms = abs(req_ts - server_ts)
                                print(f"[bybit] Time difference detected: {time_diff_ms}ms ({time_diff_ms/1000:.2f}s)")
                        except Exception:
                            pass
                    
                    if attempt < max_retries - 1:
                        wait_time = max(delay * (2 ** attempt), 3.0)  # Минимум 3 секунды
                        print(f"[bybit] Time sync error: {e}, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        if time_diff_ms and time_diff_ms > 500:
                            print(f"[bybit]   WARNING: System time is out of sync by {time_diff_ms/1000:.2f}s!")
                            print(f"[bybit]   Please sync your system time with NTP server:")
                            print(f"[bybit]   Windows: Settings > Time & Language > Date & Time > Sync now")
                        else:
                            print(f"[bybit]   Hint: Check system time synchronization. Consider syncing with NTP server.")
                        time.sleep(wait_time)
                        continue
                
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"[bybit] Request error: {e}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise
        return resp

    def get_kline(
        self, symbol: str, interval: str, start: Optional[int] = None, end: Optional[int] = None, limit: int = 200
    ) -> Dict[str, Any]:
        return self._retry_request(
            self.session.get_kline,
            category="linear",
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            limit=limit,
        )

    def get_kline_df(
        self, symbol: str, interval: str, limit: int = 200, start: Optional[int] = None, end: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Получить свечи и вернуть в виде DataFrame
        с колонками timestamp, open, high, low, close, volume.
        
        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            interval: Таймфрейм ('15m', '1h', '4h', или '15', '60', '240' для Bybit формата)
            limit: Количество свечей (максимум 200 за запрос, при большем значении данные будут загружены батчами)
            start: Начальное время в миллисекундах (опционально)
            end: Конечное время в миллисекундах (опционально, по умолчанию текущее время)
        """
        from datetime import datetime, timedelta
        
        # Конвертируем интервал в формат Bybit API если необходимо
        # Bybit использует числовые значения: "1", "3", "5", "15", "30", "60", "240" и т.д.
        def _convert_interval_to_bybit(interval_str: str) -> str:
            """Конвертирует '15m' -> '15', '1h' -> '60', и т.д."""
            interval_str = interval_str.lower().strip()
            # Если уже числовой формат, возвращаем как есть
            if interval_str.isdigit():
                return interval_str
            
            # Маппинг для конвертации
            mapping = {
                "1m": "1",
                "3m": "3",
                "5m": "5",
                "15m": "15",
                "30m": "30",
                "1h": "60",
                "2h": "120",
                "4h": "240",
                "6h": "360",
                "12h": "720",
                "1d": "D",
                "1w": "W",
                "1M": "M",
            }
            return mapping.get(interval_str, "15")  # По умолчанию 15 минут
        
        bybit_interval = _convert_interval_to_bybit(interval)
        
        # Максимальный лимит за один запрос к Bybit API
        MAX_LIMIT_PER_REQUEST = 200
        
        # Если limit больше максимума и не указаны start/end, вычисляем временной диапазон
        if limit > MAX_LIMIT_PER_REQUEST and start is None:
            # Вычисляем интервал в миллисекундах для одного бара
            interval_ms_map = {
                '1m': 60 * 1000,
                '3m': 3 * 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '30m': 30 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '2h': 2 * 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '6h': 6 * 60 * 60 * 1000,
                '12h': 12 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000,
                '1w': 7 * 24 * 60 * 60 * 1000,
            }
            interval_ms = interval_ms_map.get(interval.lower(), 15 * 60 * 1000)  # По умолчанию 15m (используем оригинальный interval для расчета)
            
            # Вычисляем start время на основе limit
            if end is None:
                end = int(datetime.now().timestamp() * 1000)
            start = end - (limit * interval_ms)
        
        # Если limit больше максимума, загружаем данные батчами
        if limit > MAX_LIMIT_PER_REQUEST:
            all_rows = []
            current_end = end if end is not None else int(datetime.now().timestamp() * 1000)
            
            # Вычисляем интервал в миллисекундах для одного бара
            interval_ms_map = {
                '1m': 60 * 1000,
                '3m': 3 * 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '30m': 30 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '2h': 2 * 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '6h': 6 * 60 * 60 * 1000,
                '12h': 12 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000,
                '1w': 7 * 24 * 60 * 60 * 1000,
            }
            interval_ms = interval_ms_map.get(interval.lower(), 15 * 60 * 1000)
            
            # Вычисляем start время если не указано
            if start is None:
                start = current_end - (limit * interval_ms)
            
            remaining = limit
            seen_timestamps = set()  # Для предотвращения дубликатов
            is_first_batch = True  # Флаг для первого батча
            
            while remaining > 0:
                batch_limit = min(remaining, MAX_LIMIT_PER_REQUEST)
                
                # Для первого батча пробуем без end параметра (получаем самые свежие данные)
                # Для последующих батчей используем end для получения исторических данных
                if is_first_batch and end is None:
                    # Первый запрос без end - получаем самые свежие данные
                    resp = self.get_kline(
                        symbol=symbol,
                        interval=bybit_interval,
                        limit=batch_limit,
                    )
                else:
                    # Используем end параметр для получения исторических данных
                    resp = self.get_kline(
                        symbol=symbol,
                        interval=bybit_interval,
                        end=current_end,
                        limit=batch_limit,
                    )
                
                # Проверяем код ответа
                ret_code = resp.get("retCode")
                if ret_code != 0:
                    ret_msg = resp.get("retMsg", "Unknown error")
                    raise ValueError(f"Bybit API error (retCode={ret_code}): {ret_msg} for {symbol}")
                
                result = resp.get("result", {})
                raw_list: List[Any] = result.get("list", []) or result.get("rows", [])
                
                if not raw_list:
                    # Если данных больше нет, прекращаем загрузку
                    if is_first_batch:
                        # Если первый батч пустой, это критическая ошибка
                        # Пробуем еще раз с явным end параметром
                        if end is None:
                            # Пробуем с текущим временем как end
                            resp = self.get_kline(
                                symbol=symbol,
                                interval=bybit_interval,
                                end=int(datetime.now().timestamp() * 1000),
                                limit=batch_limit,
                            )
                            ret_code = resp.get("retCode")
                            if ret_code == 0:
                                result = resp.get("result", {})
                                raw_list = result.get("list", []) or result.get("rows", [])
                                if raw_list:
                                    # Успешно получили данные, продолжаем
                                    pass
                                else:
                                    raise ValueError(f"No kline data from Bybit for {symbol} - API returned empty result even with explicit end time")
                            else:
                                raise ValueError(f"Bybit API error (retCode={ret_code}): {resp.get('retMsg', 'Unknown error')} for {symbol}")
                        else:
                            raise ValueError(f"No kline data from Bybit for {symbol} - API returned empty result")
                    else:
                        # Для последующих батчей пустой результат означает конец данных
                        break
                
                is_first_batch = False  # После первого успешного батча сбрасываем флаг
                
                # Конвертируем данные (Bybit возвращает от новых к старым)
                batch_rows = []
                for item in raw_list:
                    if isinstance(item, dict):
                        ts = int(item.get("startTime") or item.get("startTimeMs") or item.get("t"))
                    else:
                        ts = int(item[0])
                    
                    # Пропускаем дубликаты и свечи раньше start времени
                    if ts in seen_timestamps:
                        continue
                    if start is not None and ts < start:
                        continue
                    
                    seen_timestamps.add(ts)
                    
                    if isinstance(item, dict):
                        batch_rows.append({
                            "timestamp": ts,
                            "open": float(item["open"]),
                            "high": float(item["high"]),
                            "low": float(item["low"]),
                            "close": float(item["close"]),
                            "volume": float(item.get("volume") or item.get("turnover") or item.get("v", 0)),
                        })
                    else:
                        batch_rows.append({
                            "timestamp": ts,
                            "open": float(item[1]),
                            "high": float(item[2]),
                            "low": float(item[3]),
                            "close": float(item[4]),
                            "volume": float(item[5]),
                        })
                
                if not batch_rows:
                    # Если все свечи были отфильтрованы (раньше start), прекращаем
                    break
                
                # Добавляем батч к общему списку (Bybit возвращает от новых к старым, поэтому добавляем в начало)
                all_rows = batch_rows + all_rows
                
                # Обновляем current_end для следующего батча (идем назад во времени)
                # Берем самое старое время из текущего батча
                oldest_ts = min(row["timestamp"] for row in batch_rows)
                current_end = oldest_ts - 1  # Минус 1 мс чтобы не дублировать последнюю свечу
                
                remaining -= len(batch_rows)
                
                # Если получили меньше данных, чем запрашивали, значит данных больше нет
                if len(raw_list) < batch_limit:
                    break
                
                # Если достигли start времени, прекращаем
                if start is not None and current_end <= start:
                    break
            
            if not all_rows:
                raise ValueError(f"No kline data from Bybit for {symbol} with limit={limit}")
            
            df = pd.DataFrame(all_rows)
            # Упорядочим по времени от старых к новым
            df = df.sort_values("timestamp").reset_index(drop=True)
            # Ограничиваем количеством запрошенных свечей (берем последние N свечей)
            df = df.tail(limit).reset_index(drop=True)
            return df
        
        # Обычный запрос для limit <= MAX_LIMIT_PER_REQUEST
        resp = self.get_kline(symbol=symbol, interval=bybit_interval, start=start, end=end, limit=min(limit, MAX_LIMIT_PER_REQUEST))
        result = resp.get("result", {})
        raw_list: List[Any] = result.get("list", []) or result.get("rows", [])
        if not raw_list:
            raise ValueError(f"No kline data from Bybit: {resp}")

        # Bybit unified обычно отдаёт список списков, где
        # [0]=startTime(ms), [1]=open, [2]=high, [3]=low, [4]=close, [5]=volume
        rows = []
        for item in raw_list:
            # если пришёл dict
            if isinstance(item, dict):
                ts = int(item.get("startTime") or item.get("startTimeMs") or item.get("t"))
                rows.append(
                    {
                        "timestamp": ts,
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"]),
                        "volume": float(item.get("volume") or item.get("turnover") or item.get("v", 0)),
                    }
                )
            else:
                # предполагаем формат списка
                ts = int(item[0])
                rows.append(
                    {
                        "timestamp": ts,
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                    }
                )

        df = pd.DataFrame(rows)
        # упорядочим по времени от старых к новым
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_instrument_info(self, symbol: str) -> Dict[str, Any]:
        """Получить информацию об инструменте (qtyStep, minQty и т.д.)"""
        try:
            resp = self._retry_request(
                self.session.get_instruments_info,
                category="linear",
                symbol=symbol,
            )
            if resp.get("retCode") == 0:
                result = resp.get("result", {})
                list_data = result.get("list", [])
                if list_data:
                    return list_data[0]  # Возвращаем первый (и единственный) инструмент
            return {}
        except Exception as e:
            print(f"[bybit] Error getting instrument info: {e}")
            return {}
    
    def get_qty_step(self, symbol: str) -> float:
        """Получить qtyStep (шаг количества) для символа."""
        try:
            instrument = self.get_instrument_info(symbol)
            if instrument:
                # Ищем в filters
                filters = instrument.get("lotSizeFilter", {})
                qty_step = filters.get("qtyStep", "0.001")
                return float(qty_step)
            return 0.001  # Дефолтное значение
        except Exception as e:
            print(f"[bybit] Error getting qtyStep: {e}")
            return 0.001  # Дефолтное значение

    def get_orderbook(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Получить текущий ордербук (Level 2) для инструмента.
        Возвращает сырой словарь Bybit unified trading API.
        """
        try:
            resp = self._retry_request(
                self.session.get_orderbook,
                category="linear",
                symbol=symbol,
                limit=limit,
            )
            return resp
        except Exception as e:
            print(f"[bybit] Error getting orderbook for {symbol}: {e}")
            return {}

    def get_recent_trades(self, symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Получить последние сделки (тик-данные) для инструмента.
        Используется для расчёта CVD и упрощённого order flow анализа.
        """
        try:
            # Поддержка разных версий pybit: метод может называться по‑разному
            http = self.session
            method = None
            if hasattr(http, "get_public_trading_history"):
                method = http.get_public_trading_history
            elif hasattr(http, "get_public_trade_history"):
                method = http.get_public_trade_history
            else:
                print(
                    "[bybit] ⚠️ WARNING: Neither get_public_trading_history nor "
                    "get_public_trade_history is available on HTTP client – "
                    "orderflow-based AMT_OF strategy will be disabled."
                )
                return []

            resp = self._retry_request(
                method,
                category="linear",
                symbol=symbol,
                limit=limit,
            )
            result = resp.get("result", {}) if isinstance(resp, dict) else {}
            trades = result.get("list", []) or result.get("rows", []) or []
            return trades
        except Exception as e:
            print(f"[bybit] Error getting recent trades for {symbol}: {e}")
            return []
    
    def get_price_step(self, symbol: str) -> float:
        """Получить priceStep (шаг цены / tick size) для символа."""
        try:
            instrument = self.get_instrument_info(symbol)
            if instrument:
                # Ищем в priceFilter
                price_filter = instrument.get("priceFilter", {})
                tick_size = price_filter.get("tickSize", "0.01")
                return float(tick_size)
            return 0.01  # Дефолтное значение
        except Exception as e:
            print(f"[bybit] Error getting priceStep: {e}")
            return 0.01  # Дефолтное значение
    
    def round_price(self, price: float, symbol: str) -> float:
        """
        Округляет цену до минимального шага (tick size) для символа.
        
        Args:
            price: Цена для округления
            symbol: Торговая пара (например, "BTCUSDT")
        
        Returns:
            Округленная цена
        """
        try:
            tick_size = self.get_price_step(symbol)
            if tick_size > 0:
                # Округляем до ближайшего кратного tick_size
                rounded = round(price / tick_size) * tick_size
                return rounded
            return price
        except Exception as e:
            print(f"[bybit] Error rounding price: {e}")
            return price
    
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        reduce_only: bool = False,
        price: Optional[float] = None,
        time_in_force: str = "GoodTillCancel",
        order_link_id: Optional[str] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Разместить ордер.
        
        Args:
            order_link_id: Уникальный ID ордера для связи с сигналом (опционально)
            take_profit: Цена Take Profit (опционально)
            stop_loss: Цена Stop Loss (опционально)
        """
        # Защита от отправки reduce_only ордера, если по символу нет открытой позиции.
        # Это предотвращает ошибку Bybit: "current position is zero, cannot fix reduce-only order qty (ErrCode: 110017)".
        if reduce_only:
            try:
                pos_info = self.get_position_info(symbol=symbol)
                list_data = pos_info.get("result", {}).get("list", []) if isinstance(pos_info, dict) else []
                has_open = False
                for p in list_data:
                    try:
                        size = float(p.get("size", "0") or "0")
                    except Exception:
                        size = 0.0
                    if size > 0:
                        has_open = True
                        break

                if not has_open:
                    print(f"[bybit] ⚠️ Skipping reduce_only order for {symbol}: no open position (avoids ErrCode 110017)")
                    return {"retCode": 110017, "retMsg": "No open position to reduce; skipped reduce_only order", "result": {}}
            except Exception as e:
                # Если при проверке позиции произошла ошибка, логируем и пробуем отправить ордер —
                # в большинстве случаев get_position_info вернёт корректный ответ, но на всякий случай
                # не хотим ломать основной поток выполнения.
                print(f"[bybit] ⚠️ Warning checking position before reduce_only order for {symbol}: {e}. Proceeding with order.")

        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "orderType": order_type,
            "reduceOnly": reduce_only,
            "timeInForce": time_in_force,
        }
        if price:
            payload["price"] = price
        if order_link_id:
            payload["orderLinkId"] = order_link_id
        if take_profit is not None:
            payload["tpTriggerBy"] = "LastPrice"  # Тип триггера для TP
            payload["takeProfit"] = str(take_profit)
        if stop_loss is not None:
            payload["slTriggerBy"] = "LastPrice"  # Тип триггера для SL
            payload["stopLoss"] = str(stop_loss)
        return self.session.place_order(**payload)

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        """
        Получить баланс кошелька.
        account_type: "UNIFIED" (по умолчанию), "CONTRACT", "SPOT", "INVESTMENT", "OPTION", "FUND"
        """
        return self._retry_request(self.session.get_wallet_balance, accountType=account_type)

    def get_position_info(self, symbol: Optional[str] = None, settle_coin: Optional[str] = None) -> Dict[str, Any]:
        """
        Получить информацию об открытых позициях.
        Для unified account требуется указать symbol или settleCoin.
        """
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        elif settle_coin:
            params["settleCoin"] = settle_coin
        else:
            # По умолчанию для unified account используем USDT
            params["settleCoin"] = "USDT"
        return self._retry_request(self.session.get_positions, **params)

    def get_open_orders(
        self, symbol: Optional[str] = None, order_id: Optional[str] = None, order_link_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получить список открытых ордеров.
        Для unified account требуется указать symbol, settleCoin или baseCoin.
        """
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        elif not order_id and not order_link_id:
            # Если symbol не указан и нет order_id/order_link_id, используем settleCoin для unified account
            params["settleCoin"] = "USDT"
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        return self.session.get_open_orders(**params)

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Установить плечо для символа.
        Для unified account в one-way mode buyLeverage и sellLeverage должны быть одинаковыми.
        """
        return self.session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage),
        )

    def set_trading_stop(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Установить Take Profit и/или Stop Loss для позиции.
        
        Args:
            symbol: символ (например, "BTCUSDT")
            stop_loss: цена стоп-лосса (абсолютное значение)
            take_profit: цена тейк-профита (абсолютное значение)
            trailing_stop: trailing stop в процентах (опционально)
        """
        params = {
            "category": "linear",
            "symbol": symbol,
        }
        if stop_loss is not None:
            # Убеждаемся, что цена - это float, а не строка или другое значение
            if isinstance(stop_loss, str):
                stop_loss = float(stop_loss)
            # Округляем цену до минимального шага (tick size) для символа
            stop_loss = self.round_price(stop_loss, symbol)
            # ВАЖНО: Передаем цену как число (float), а не строку
            # pybit сам преобразует его в правильный формат для API
            params["stopLoss"] = stop_loss
            print(f"[bybit] Setting stopLoss for {symbol}: {stop_loss} (type: {type(stop_loss).__name__}, rounded to tick size)")
        if take_profit is not None:
            # Убеждаемся, что цена - это float, а не строка или другое значение
            if isinstance(take_profit, str):
                take_profit = float(take_profit)
            # Округляем цену до минимального шага (tick size) для символа
            take_profit = self.round_price(take_profit, symbol)
            # ВАЖНО: Передаем цену как число (float), а не строку
            # pybit сам преобразует его в правильный формат для API
            params["takeProfit"] = take_profit
            print(f"[bybit] Setting takeProfit for {symbol}: {take_profit} (type: {type(take_profit).__name__}, rounded to tick size)")
        if trailing_stop is not None:
            params["trailingStop"] = str(trailing_stop)
        return self.session.set_trading_stop(**params)
    
    def get_closed_pnl(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Получить историю закрытых позиций (closed PnL).
        
        Args:
            symbol: Торговая пара (опционально)
            start_time: Начальное время в миллисекундах (опционально)
            end_time: Конечное время в миллисекундах (опционально)
            limit: Максимальное количество записей (по умолчанию 50)
        
        Returns:
            Словарь с результатами запроса
        """
        params = {
            "category": "linear",
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self._retry_request(self.session.get_closed_pnl, **params)
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        order_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Получить историю ордеров (order history).
        
        Args:
            symbol: Торговая пара (опционально)
            start_time: Начальное время в миллисекундах (опционально)
            end_time: Конечное время в миллисекундах (опционально)
            limit: Максимальное количество записей (по умолчанию 50)
            order_status: Статус ордера (Filled, PartiallyFilled, Canceled, Rejected) (опционально)
        
        Returns:
            Словарь с результатами запроса
        """
        params = {
            "category": "linear",
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if order_status:
            params["orderStatus"] = order_status
        return self._retry_request(self.session.get_order_history, **params)
    
    def get_execution_list(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Получить историю исполненных ордеров (execution history).
        
        Args:
            symbol: Торговая пара (опционально)
            start_time: Начальное время в миллисекундах (опционально)
            end_time: Конечное время в миллисекундах (опционально)
            limit: Максимальное количество записей (по умолчанию 50)
        
        Returns:
            Словарь с результатами запроса
        """
        params = {
            "category": "linear",
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        # Используем get_executions из Position API для получения истории исполненных ордеров
        # get_execution_list из Trade API может не включать все поля, нужные для определения Order ID закрытия
        try:
            # Пытаемся использовать get_executions из Position API (более полная информация)
            if hasattr(self.session, 'get_executions'):
                return self._retry_request(self.session.get_executions, **params)
            else:
                # Fallback на get_execution_list из Trade API
                return self._retry_request(self.session.get_execution_list, **params)
        except AttributeError:
            # Если get_executions не доступен, используем get_execution_list
            return self._retry_request(self.session.get_execution_list, **params)
