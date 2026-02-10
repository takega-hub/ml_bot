"""
Комбинированная ML-стратегия, использующая 1h для фильтрации и 15m для входа.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from bot.strategy import Action, Bias, Signal
from bot.ml.strategy_ml import MLStrategy

logger = logging.getLogger(__name__)


class MultiTimeframeMLStrategy:
    """
    Комбинированная стратегия:
    - 1h модель: определяет направление тренда (фильтр)
    - 15m модель: определяет точку входа
    
    Логика работы:
    1. 1h модель проверяет общее направление тренда
    2. Если 1h дает сигнал, используем 15m для точного входа
    3. Входим только при совпадении направлений обеих моделей
    """
    
    def __init__(
        self,
        model_1h_path: str,
        model_15m_path: str,
        confidence_threshold_1h: float = 0.50,  # Выше порог для 1h (фильтр)
        confidence_threshold_15m: float = 0.35,  # Ниже порог для 15m (вход)
        min_confidence_difference: float = 0.10,  # Минимальная разница между LONG/SHORT
        require_alignment: bool = True,  # Требовать совпадение направлений
        alignment_mode: str = "strict",  # "strict" или "weighted"
        min_signal_strength_1h: str = "среднее",
        min_signal_strength_15m: str = "слабое",
    ):
        """
        Args:
            model_1h_path: Путь к 1h модели
            model_15m_path: Путь к 15m модели
            confidence_threshold_1h: Минимальная уверенность 1h модели
            confidence_threshold_15m: Минимальная уверенность 15m модели
            min_confidence_difference: Минимальная разница между LONG и SHORT
            require_alignment: Требовать совпадение направлений обеих моделей
            alignment_mode: "strict" - строгое совпадение, "weighted" - взвешенное голосование
            min_signal_strength_1h: Минимальная сила сигнала для 1h модели
            min_signal_strength_15m: Минимальная сила сигнала для 15m модели
        """
        self.model_1h_path = Path(model_1h_path)
        self.model_15m_path = Path(model_15m_path)
        self.confidence_threshold_1h = confidence_threshold_1h
        self.confidence_threshold_15m = confidence_threshold_15m
        self.min_confidence_difference = min_confidence_difference
        self.require_alignment = require_alignment
        self.alignment_mode = alignment_mode
        
        # Загружаем обе модели
        if not self.model_1h_path.exists():
            raise FileNotFoundError(f"1h модель не найдена: {model_1h_path}")
        if not self.model_15m_path.exists():
            raise FileNotFoundError(f"15m модель не найдена: {model_15m_path}")
        
        logger.info(f"[MTF Strategy] Загрузка 1h модели: {model_1h_path}")
        self.strategy_1h = MLStrategy(
            model_path=str(model_1h_path),
            confidence_threshold=confidence_threshold_1h,
            min_signal_strength=min_signal_strength_1h,
            stability_filter=True,
        )
        
        logger.info(f"[MTF Strategy] Загрузка 15m модели: {model_15m_path}")
        self.strategy_15m = MLStrategy(
            model_path=str(model_15m_path),
            confidence_threshold=confidence_threshold_15m,
            min_signal_strength=min_signal_strength_15m,
            stability_filter=True,
        )
        
        logger.info(f"[MTF Strategy] Инициализация завершена")
        logger.info(f"  1h порог: {confidence_threshold_1h}, 15m порог: {confidence_threshold_15m}")
        logger.info(f"  Режим выравнивания: {alignment_mode}, require_alignment: {require_alignment}")
    
    def predict_combined(
        self,
        df_15m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        skip_feature_creation: bool = False,
    ) -> tuple[int, float, Dict[str, Any]]:
        """
        Комбинированное предсказание на основе обеих моделей.
        
        Args:
            df_15m: DataFrame с 15m данными
            df_1h: DataFrame с 1h данными (опционально, если None - агрегируется из 15m)
            skip_feature_creation: Пропустить создание фичей
        
        Returns:
            (prediction, confidence, info) где:
            - prediction: 1 (LONG), -1 (SHORT), 0 (HOLD)
            - confidence: комбинированная уверенность (0-1)
            - info: словарь с деталями предсказаний
        """
        # 1. Получаем предсказание от 1h модели
        if df_1h is None or df_1h.empty:
            # Агрегируем 1h из 15m данных
            if not isinstance(df_15m.index, pd.DatetimeIndex):
                # Пытаемся преобразовать индекс
                if "timestamp" in df_15m.columns:
                    df_15m = df_15m.set_index("timestamp")
                else:
                    df_15m.index = pd.to_datetime(df_15m.index, errors='coerce')
            
            ohlcv_agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            # Агрегируем только OHLCV колонки
            agg_cols = {k: v for k, v in ohlcv_agg.items() if k in df_15m.columns}
            if agg_cols:
                df_1h = df_15m.resample("60min").agg(agg_cols).dropna()
                
                # ОТЛАДКА: Проверяем результат агрегации
                if not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                if self._debug_count < 3:
                    logger.debug(f"[MTF Strategy] Агрегация 1h: {len(df_15m)} свечей 15m -> {len(df_1h)} свечей 1h")
                    if len(df_1h) > 0:
                        logger.debug(f"[MTF Strategy] Первая 1h свеча: {df_1h.index[0]}, последняя: {df_1h.index[-1]}")
            else:
                logger.warning("[MTF Strategy] Не удалось агрегировать 1h данные - нет OHLCV колонок")
                return 0, 0.0, {"reason": "no_ohlcv_columns"}
        
        if df_1h.empty:
            logger.warning("[MTF Strategy] 1h DataFrame пуст после агрегации")
            return 0, 0.0, {"reason": "empty_1h_data"}
        
        # Проверяем, достаточно ли данных для 1h модели (минимум 100 свечей)
        if len(df_1h) < 100:
            if not hasattr(self, '_warned_insufficient_1h'):
                logger.warning(f"[MTF Strategy] Мало 1h данных: {len(df_1h)} свечей (минимум 100 рекомендуется)")
                self._warned_insufficient_1h = True
        
        # Предсказание 1h модели
        # ВАЖНО: Если df_1h был агрегирован из df_15m, он содержит только OHLCV, без фичей
        # Поэтому для агрегированных данных всегда создаем фичи заново
        skip_feature_creation_1h = skip_feature_creation and (df_1h is not None and not df_1h.empty and len(df_1h.columns) > 20)
        # Если агрегировали из 15m, то skip_feature_creation=False (нужно создать фичи)
        
        try:
            pred_1h, conf_1h = self.strategy_1h.predict(df_1h, skip_feature_creation=skip_feature_creation_1h)
            # ОТЛАДКА: Логируем первые несколько предсказаний
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            if self._debug_count < 5:
                logger.debug(f"[MTF Strategy] 1h предсказание #{self._debug_count}: pred={pred_1h}, conf={conf_1h:.3f}, порог={self.confidence_threshold_1h}")
                self._debug_count += 1
        except Exception as e:
            logger.error(f"[MTF Strategy] Ошибка предсказания 1h модели: {e}")
            import traceback
            logger.error(f"[MTF Strategy] Traceback:\n{traceback.format_exc()}")
            return 0, 0.0, {"reason": f"1h_prediction_error: {str(e)[:50]}"}
        
        # 2. Получаем предсказание от 15m модели
        try:
            pred_15m, conf_15m = self.strategy_15m.predict(df_15m, skip_feature_creation=skip_feature_creation)
            # ОТЛАДКА: Логируем первые несколько предсказаний
            if hasattr(self, '_debug_count') and self._debug_count < 5:
                logger.debug(f"[MTF Strategy] 15m предсказание #{self._debug_count-1}: pred={pred_15m}, conf={conf_15m:.3f}, порог={self.confidence_threshold_15m}")
        except Exception as e:
            logger.error(f"[MTF Strategy] Ошибка предсказания 15m модели: {e}")
            import traceback
            logger.error(f"[MTF Strategy] Traceback:\n{traceback.format_exc()}")
            return 0, 0.0, {"reason": f"15m_prediction_error: {str(e)[:50]}"}
        
        # 3. Комбинируем предсказания
        info = {
            "pred_1h": pred_1h,
            "conf_1h": conf_1h,
            "pred_15m": pred_15m,
            "conf_15m": conf_15m,
            "alignment": pred_1h == pred_15m if pred_1h != 0 and pred_15m != 0 else None,
        }
        
        if self.alignment_mode == "strict":
            # Строгий режим: обе модели должны совпадать
            if self.require_alignment:
                # ОТЛАДКА: Логируем причины отклонения
                if hasattr(self, '_debug_count') and self._debug_count <= 5:
                    logger.debug(f"[MTF Strategy] Комбинирование: 1h={pred_1h}({conf_1h:.3f}), 15m={pred_15m}({conf_15m:.3f})")
                
                if pred_1h == 0:
                    # 1h не дает сигнал - не входим
                    if hasattr(self, '_debug_count') and self._debug_count <= 5:
                        logger.debug(f"[MTF Strategy] Отклонено: 1h не дает сигнал (conf_1h={conf_1h:.3f} < порог={self.confidence_threshold_1h})")
                    return 0, 0.0, {**info, "reason": "1h_no_signal"}
                
                if pred_15m == 0:
                    # 15m не дает сигнал - не входим
                    if hasattr(self, '_debug_count') and self._debug_count <= 5:
                        logger.debug(f"[MTF Strategy] Отклонено: 15m не дает сигнал (conf_15m={conf_15m:.3f} < порог={self.confidence_threshold_15m})")
                    return 0, 0.0, {**info, "reason": "15m_no_signal"}
                
                if pred_1h != pred_15m:
                    # Направления не совпадают - не входим
                    if hasattr(self, '_debug_count') and self._debug_count <= 5:
                        logger.debug(f"[MTF Strategy] Отклонено: направления не совпадают (1h={pred_1h}, 15m={pred_15m})")
                    return 0, 0.0, {**info, "reason": "directions_mismatch"}
                
                # Обе модели согласны - используем 15m для входа
                combined_confidence = (conf_1h * 0.4 + conf_15m * 0.6)  # 1h=40%, 15m=60%
                if hasattr(self, '_debug_count') and self._debug_count <= 5:
                    logger.debug(f"[MTF Strategy] ✅ Сигнал принят: {pred_15m}, комбинированная уверенность={combined_confidence:.3f}")
                return pred_15m, combined_confidence, {**info, "reason": "aligned"}
            else:
                # Не требуем совпадения - используем взвешенное голосование
                return self._weighted_vote(pred_1h, conf_1h, pred_15m, conf_15m, info)
        
        elif self.alignment_mode == "weighted":
            # Взвешенное голосование
            return self._weighted_vote(pred_1h, conf_1h, pred_15m, conf_15m, info)
        
        else:
            # По умолчанию - строгий режим
            return self._weighted_vote(pred_1h, conf_1h, pred_15m, conf_15m, info)
    
    def _weighted_vote(
        self,
        pred_1h: int,
        conf_1h: float,
        pred_15m: int,
        conf_15m: float,
        info: Dict[str, Any],
    ) -> tuple[int, float, Dict[str, Any]]:
        """
        Взвешенное голосование двух моделей.
        
        Веса:
        - 1h: 40% (направление/тренд)
        - 15m: 60% (точка входа)
        """
        # Если одна из моделей дает HOLD, проверяем другую
        if pred_1h == 0 and pred_15m == 0:
            return 0, 0.0, {**info, "reason": "both_hold"}
        
        if pred_1h == 0:
            # Только 15m дает сигнал - используем его, но с пониженной уверенностью
            return pred_15m, conf_15m * 0.7, {**info, "reason": "15m_only"}
        
        if pred_15m == 0:
            # Только 1h дает сигнал - не входим (нужна точка входа от 15m)
            return 0, 0.0, {**info, "reason": "1h_only_no_entry"}
        
        # Обе модели дают сигнал
        if pred_1h == pred_15m:
            # Направления совпадают - сильный сигнал
            combined_confidence = (conf_1h * 0.4 + conf_15m * 0.6)
            return pred_15m, combined_confidence, {**info, "reason": "aligned_weighted"}
        else:
            # Направления не совпадают - конфликт
            # Используем более уверенную модель, но с пониженной уверенностью
            if conf_1h > conf_15m:
                # 1h более уверена, но нужна точка входа от 15m
                # Если 15m дает противоположный сигнал, не входим
                return 0, 0.0, {**info, "reason": "conflict_1h_stronger"}
            else:
                # 15m более уверена - используем ее, но с пониженной уверенностью
                return pred_15m, conf_15m * 0.6, {**info, "reason": "conflict_15m_stronger"}
    
    def generate_signal(
        self,
        row: pd.Series,
        df_15m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        has_position: Optional[Bias] = None,
        current_price: float = None,
        leverage: int = 10,
        target_profit_pct_margin: float = 25.0,
        max_loss_pct_margin: float = 10.0,
        skip_feature_creation: bool = False,
    ) -> Signal:
        """
        Генерирует комбинированный сигнал.
        
        Args:
            row: Текущий бар 15m (pd.Series)
            df_15m: DataFrame с 15m данными
            df_1h: DataFrame с 1h данными (опционально)
            has_position: Текущая позиция
            current_price: Текущая цена
            leverage: Плечо
            target_profit_pct_margin: Целевая прибыль от маржи в %
            max_loss_pct_margin: Максимальный убыток от маржи в %
            skip_feature_creation: Пропустить создание фичей
        
        Returns:
            Signal объект
        """
        if current_price is None:
            current_price = row.get("close", 0.0)
        
        try:
            # Получаем комбинированное предсказание
            prediction, confidence, info = self.predict_combined(
                df_15m=df_15m,
                df_1h=df_1h,
                skip_feature_creation=skip_feature_creation,
            )
            
            # Если нет сигнала, возвращаем HOLD
            if prediction == 0:
                return Signal(
                    timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                    action=Action.HOLD,
                    reason=f"mtf_{info.get('reason', 'no_signal')}",
                    price=current_price,
                    indicators_info={
                        "strategy": "MTF_ML",
                        "prediction": "HOLD",
                        "confidence": round(confidence, 4),
                        "1h_pred": info.get("pred_1h"),
                        "1h_conf": round(info.get("conf_1h", 0), 4),
                        "15m_pred": info.get("pred_15m"),
                        "15m_conf": round(info.get("conf_15m", 0), 4),
                        "reason": info.get("reason"),
                    }
                )
            
            # Есть сигнал - используем 15m модель для генерации полного сигнала с TP/SL
            # (15m модель уже имеет логику расчета TP/SL)
            signal_15m = self.strategy_15m.generate_signal(
                row=row,
                df=df_15m,
                has_position=has_position,
                current_price=current_price,
                leverage=leverage,
                target_profit_pct_margin=target_profit_pct_margin,
                max_loss_pct_margin=max_loss_pct_margin,
                skip_feature_creation=skip_feature_creation,
            )
            
            # Обновляем информацию о комбинированной стратегии
            if signal_15m.indicators_info:
                signal_15m.indicators_info["strategy"] = "MTF_ML"
                signal_15m.indicators_info["mtf_confidence"] = round(confidence, 4)
                signal_15m.indicators_info["1h_pred"] = info.get("pred_1h")
                signal_15m.indicators_info["1h_conf"] = round(info.get("conf_1h", 0), 4)
                signal_15m.indicators_info["15m_pred"] = info.get("pred_15m")
                signal_15m.indicators_info["15m_conf"] = round(info.get("conf_15m", 0), 4)
                signal_15m.indicators_info["alignment"] = info.get("alignment")
                signal_15m.indicators_info["mtf_reason"] = info.get("reason")
            else:
                signal_15m.indicators_info = {
                    "strategy": "MTF_ML",
                    "mtf_confidence": round(confidence, 4),
                    "1h_pred": info.get("pred_1h"),
                    "1h_conf": round(info.get("conf_1h", 0), 4),
                    "15m_pred": info.get("pred_15m"),
                    "15m_conf": round(info.get("conf_15m", 0), 4),
                    "alignment": info.get("alignment"),
                    "mtf_reason": info.get("reason"),
                }
            
            # Обновляем reason с информацией о комбинированной стратегии
            action_str = "LONG" if prediction == 1 else "SHORT"
            signal_15m.reason = f"mtf_{action_str}_1h{info.get('pred_1h')}_15m{info.get('pred_15m')}_conf{int(confidence*100)}%"
            
            return signal_15m
        
        except Exception as e:
            logger.error(f"[MTF Strategy] Ошибка генерации сигнала: {e}")
            import traceback
            traceback.print_exc()
            return Signal(
                timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                action=Action.HOLD,
                reason=f"mtf_error_{str(e)[:20]}",
                price=current_price
            )
