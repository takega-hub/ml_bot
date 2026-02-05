"""
Risk Manager для расчета TP/SL на основе ATR и уровней поддержки/сопротивления.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Менеджер риска для расчета TP/SL.
    
    Логика:
    - SL: за ближайшим релевантным уровнем S/R + буфер (ATR или фикс %)
    - TP: по RR (risk-reward ratio) от SL
    """
    
    def __init__(
        self,
        default_rr: float = 2.5,
        min_rr: float = 2.0,
        max_rr: float = 3.0,
        sl_buffer_atr_mult: float = 0.2,  # Буфер = 0.2 * ATR
        sl_buffer_pct_min: float = 0.001,  # Минимум 0.1% цены
        lookback_for_levels: int = 60,  # Lookback для поиска уровней
    ):
        """
        Args:
            default_rr: Базовый risk-reward ratio
            min_rr: Минимальный RR
            max_rr: Максимальный RR
            sl_buffer_atr_mult: Множитель ATR для буфера SL
            sl_buffer_pct_min: Минимальный буфер в % от цены
            lookback_for_levels: Lookback период для поиска уровней S/R
        """
        self.default_rr = default_rr
        self.min_rr = min_rr
        self.max_rr = max_rr
        self.sl_buffer_atr_mult = sl_buffer_atr_mult
        self.sl_buffer_pct_min = sl_buffer_pct_min
        self.lookback_for_levels = lookback_for_levels
    
    def calculate_tp_sl(
        self,
        side: str,
        entry_price: float,
        row: pd.Series,
        df_history: Optional[pd.DataFrame] = None,
        confidence: Optional[float] = None,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Рассчитывает TP и SL для позиции.
        
        Args:
            side: "LONG" или "SHORT"
            entry_price: Цена входа
            row: Текущая строка данных (должна содержать ATR и другие индикаторы)
            df_history: История данных для поиска уровней (опционально)
            confidence: Уверенность модели (для динамического RR, опционально)
        
        Returns:
            (sl_price, tp_price, rr, metadata)
        """
        # Получаем ATR
        atr = row.get("atr")
        if atr is None or pd.isna(atr) or atr <= 0:
            # Fallback: используем процент от цены
            atr = entry_price * 0.01  # 1%
            logger.debug(f"ATR not available, using fallback: {atr}")
        
        # Ищем уровни поддержки/сопротивления
        support, resistance = self._find_support_resistance(
            entry_price, row, df_history
        )
        
        # Рассчитываем SL от уровня
        sl_price, sl_source, sl_level = self._calculate_sl_from_levels(
            side, entry_price, atr, support, resistance, row
        )
        
        # Fallback SL если уровни не найдены
        if sl_price is None:
            sl_price, sl_source, sl_level = self._fallback_sl(
                side, entry_price, atr
            )
        
        # Динамический RR на основе уверенности (если предоставлена)
        if confidence is not None and np.isfinite(confidence):
            # RR от 2.0 до 3.0 в зависимости от уверенности
            rr = self.min_rr + (confidence - 0.5) / 0.4 * (self.max_rr - self.min_rr)
            rr = np.clip(rr, self.min_rr, self.max_rr)
        else:
            rr = self.default_rr
        
        # Рассчитываем TP от SL с учетом RR
        if side == "LONG":
            risk = abs(entry_price - sl_price)
            tp_price = entry_price + (risk * rr)
        else:  # SHORT
            risk = abs(sl_price - entry_price)
            tp_price = entry_price - (risk * rr)
        
        # Валидация TP/SL
        if side == "LONG":
            if not (sl_price < entry_price < tp_price):
                logger.warning(f"Invalid TP/SL for LONG: sl={sl_price}, entry={entry_price}, tp={tp_price}")
                # Исправляем с учетом RR
                sl_price = entry_price * 0.99  # 1% SL
                tp_price = entry_price * (1 + 0.01 * rr)  # TP = SL% * RR
        else:  # SHORT
            if not (tp_price < entry_price < sl_price):
                logger.warning(f"Invalid TP/SL for SHORT: sl={sl_price}, entry={entry_price}, tp={tp_price}")
                # Исправляем с учетом RR
                sl_price = entry_price * 1.01  # 1% SL
                tp_price = entry_price * (1 - 0.01 * rr)  # TP = SL% * RR
        
        # Метаданные
        metadata = {
            "sl_source": sl_source,
            "sl_level": sl_level,
            "rr": float(rr),
            "atr": float(atr),
            "atr_pct": float((atr / entry_price) * 100),
            "risk_pct": float((risk / entry_price) * 100),
            "tp_pct": float((abs(tp_price - entry_price) / entry_price) * 100),
            "sl_pct": float((abs(entry_price - sl_price) / entry_price) * 100),
        }
        
        return sl_price, tp_price, rr, metadata
    
    def _find_support_resistance(
        self,
        current_price: float,
        row: pd.Series,
        df_history: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Находит ближайшие уровни поддержки и сопротивления.
        
        Returns:
            (support, resistance)
        """
        support = None
        resistance = None
        
        # Метод 1: Используем готовые колонки из row (если есть)
        if "nearest_support" in row.index and pd.notna(row["nearest_support"]):
            support = float(row["nearest_support"])
        if "nearest_resistance" in row.index and pd.notna(row["nearest_resistance"]):
            resistance = float(row["nearest_resistance"])
        
        # Метод 2: Используем Donchian Channels
        if "donchian_lower" in row.index and pd.notna(row["donchian_lower"]):
            donchian_support = float(row["donchian_lower"])
            if support is None or donchian_support < current_price:
                support = donchian_support
        
        if "donchian_upper" in row.index and pd.notna(row["donchian_upper"]):
            donchian_resistance = float(row["donchian_upper"])
            if resistance is None or donchian_resistance > current_price:
                resistance = donchian_resistance
        
        # Метод 3: Используем Bollinger Bands
        if "bb_lower" in row.index and pd.notna(row["bb_lower"]):
            bb_support = float(row["bb_lower"])
            if support is None or (bb_support < current_price and (support is None or bb_support > support)):
                support = bb_support
        
        if "bb_upper" in row.index and pd.notna(row["bb_upper"]):
            bb_resistance = float(row["bb_upper"])
            if resistance is None or (bb_resistance > current_price and (resistance is None or bb_resistance < resistance)):
                resistance = bb_resistance
        
        # Метод 4: Используем recent_high/low
        if "recent_low" in row.index and pd.notna(row["recent_low"]):
            recent_low = float(row["recent_low"])
            if support is None or (recent_low < current_price and (support is None or recent_low > support)):
                support = recent_low
        
        if "recent_high" in row.index and pd.notna(row["recent_high"]):
            recent_high = float(row["recent_high"])
            if resistance is None or (recent_high > current_price and (resistance is None or recent_high < resistance)):
                resistance = recent_high
        
        # Метод 5: Если есть история, ищем локальные экстремумы
        if df_history is not None and len(df_history) > 0:
            lookback = min(self.lookback_for_levels, len(df_history))
            df_tail = df_history.iloc[-lookback:]
            
            if "low" in df_tail.columns:
                recent_low = df_tail["low"].min()
                if support is None or (recent_low < current_price and (support is None or recent_low > support)):
                    support = recent_low
            
            if "high" in df_tail.columns:
                recent_high = df_tail["high"].max()
                if resistance is None or (recent_high > current_price and (resistance is None or recent_high < resistance)):
                    resistance = recent_high
        
        return support, resistance
    
    def _calculate_sl_from_levels(
        self,
        side: str,
        entry_price: float,
        atr: float,
        support: Optional[float],
        resistance: Optional[float],
        row: pd.Series,
    ) -> Tuple[Optional[float], Optional[str], Optional[float]]:
        """
        Рассчитывает SL от уровней S/R.
        
        Returns:
            (sl_price, sl_source, sl_level)
        """
        # Собираем кандидаты уровней
        candidates = []
        
        if side == "LONG":
            # Для LONG ищем поддержку ниже цены
            if support is not None and support < entry_price:
                candidates.append(("support", support))
            
            # Также проверяем дополнительные уровни
            for level_name in ["bb_lower", "donchian_lower", "recent_low", "sma_20", "ema_26", "ema_12"]:
                if level_name in row.index:
                    level_val = row[level_name]
                    if pd.notna(level_val) and level_val < entry_price:
                        candidates.append((level_name, float(level_val)))
        
            if not candidates:
                return None, None, None
            
            # Выбираем ближайшую поддержку (самую высокую ниже цены)
            selected = max(candidates, key=lambda x: x[1])
        
        else:  # SHORT
            # Для SHORT ищем сопротивление выше цены
            if resistance is not None and resistance > entry_price:
                candidates.append(("resistance", resistance))
            
            # Также проверяем дополнительные уровни
            for level_name in ["bb_upper", "donchian_upper", "recent_high", "sma_20", "ema_26", "ema_12"]:
                if level_name in row.index:
                    level_val = row[level_name]
                    if pd.notna(level_val) and level_val > entry_price:
                        candidates.append((level_name, float(level_val)))
            
            if not candidates:
                return None, None, None
            
            # Выбираем ближайшее сопротивление (самое низкое выше цены)
            selected = min(candidates, key=lambda x: x[1])
        
        level_name, level_price = selected
        
        # Рассчитываем буфер
        buffer_atr = atr * self.sl_buffer_atr_mult
        buffer_pct = entry_price * self.sl_buffer_pct_min
        buffer = max(buffer_atr, buffer_pct)
        
        # SL = уровень - буфер (для LONG) или уровень + буфер (для SHORT)
        if side == "LONG":
            sl_price = level_price - buffer
            if sl_price >= entry_price:
                return None, None, None
            # ОГРАНИЧЕНИЕ: SL не дальше 1% от entry
            max_sl_distance = entry_price * 0.01
            if (entry_price - sl_price) > max_sl_distance:
                sl_price = entry_price - max_sl_distance
                level_name = "capped_1pct"
        else:  # SHORT
            sl_price = level_price + buffer
            if sl_price <= entry_price:
                return None, None, None
            # ОГРАНИЧЕНИЕ: SL не дальше 1% от entry
            max_sl_distance = entry_price * 0.01
            if (sl_price - entry_price) > max_sl_distance:
                sl_price = entry_price + max_sl_distance
                level_name = "capped_1pct"
        
        return sl_price, level_name, level_price
    
    def _fallback_sl(
        self,
        side: str,
        entry_price: float,
        atr: float,
    ) -> Tuple[float, str, None]:
        """
        Fallback SL если уровни не найдены.
        ФИКСИРОВАННЫЙ SL = 1% от цены входа.
        """
        # Фиксированный SL = 1% (для предсказуемого риска)
        sl_pct = 0.01  # 1%
        
        if side == "LONG":
            sl_price = entry_price * (1 - sl_pct)
        else:  # SHORT
            sl_price = entry_price * (1 + sl_pct)
        
        return sl_price, "fallback_1pct", None
