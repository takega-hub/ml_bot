"""
Unit tests for leverage-based TP/SL recalculation.

Tests the dynamic recalculation of percentage parameters when leverage changes,
including trailing stop and breakeven activation thresholds.
"""

import unittest
from dataclasses import dataclass, field
from typing import Optional
import math


DEFAULT_LEVERAGE = 10


def calculate_price_pct_from_margin_pct(margin_pct: float, leverage: int) -> float:
    """
    Конвертирует процент от маржи в процент от цены.
    
    Пример: при leverage=10, 10% прибыли от маржи = 1% прибыли от цены
    
    Args:
        margin_pct: Процент прибыли от маржи (например, 10.0 для 10%)
        leverage: Плечо
        
    Returns:
        Процент от цены (decimal, например 0.01 для 1%)
    """
    if leverage <= 0:
        leverage = DEFAULT_LEVERAGE
    return margin_pct / leverage / 100.0


def calculate_margin_pct_from_price_pct(price_pct: float, leverage: int) -> float:
    """
    Конвертирует процент от цены в процент от маржи.
    
    Пример: при leverage=10, 1% прибыли от цены = 10% прибыли от маржи
    
    Args:
        price_pct: Процент от цены (decimal, например 0.01 для 1%)
        leverage: Плечо
        
    Returns:
        Процент от маржи (decimal, например 0.1 для 10%)
    """
    if leverage <= 0:
        leverage = DEFAULT_LEVERAGE
    return price_pct * leverage * 100.0


def recalculate_tp_sl_for_leverage(
    entry_price: float,
    side: str,
    leverage: int,
    target_profit_pct_margin: float = 18.0,
    max_loss_pct_margin: float = 10.0,
    tick_size: float = 0.01
) -> tuple:
    """
    Пересчитывает TP и SL на основе плеча и процентов от маржи.
    
    Args:
        entry_price: Цена входа
        side: "Buy" или "Sell"
        leverage: Плечо
        target_profit_pct_margin: Целевая прибыль от маржи в %
        max_loss_pct_margin: Максимальный убыток от маржи в %
        tick_size: Размер тика для округления
        
    Returns:
        (tp_price, sl_price)
    """
    if leverage <= 0:
        leverage = DEFAULT_LEVERAGE
    
    sl_ratio = max_loss_pct_margin / leverage / 100.0
    tp_ratio = target_profit_pct_margin / leverage / 100.0
    
    if side == "Buy":
        sl = entry_price * (1 - sl_ratio)
        tp = entry_price * (1 + tp_ratio)
    else:
        sl = entry_price * (1 + sl_ratio)
        tp = entry_price * (1 - tp_ratio)
    
    if tick_size > 0:
        places = max(0, int(-math.log10(tick_size)))
        sl = round(sl, places)
        tp = round(tp, places)
    
    return tp, sl


@dataclass
class MockRiskSettings:
    trailing_activation_mode: str = "price"
    trailing_activation_pct_margin: float = 7.0
    trailing_stop_activation_pct: float = 0.007
    trailing_stop_distance_pct: float = 0.003
    breakeven_activation_mode: str = "price"
    breakeven_level1_activation_pct_margin: float = 5.0
    breakeven_level2_activation_pct_margin: float = 10.0
    breakeven_level1_activation_pct: float = 0.005
    breakeven_level2_activation_pct: float = 0.01
    breakeven_level1_sl_pct: float = 0.001
    breakeven_level2_sl_pct: float = 0.005
    dca_drawdown_mode: str = "price"
    dca_drawdown_pct_margin: float = 3.0
    dca_drawdown_pct: float = 0.003
    partial_close_mode: str = "price"
    max_position_usd: float = 200.0
    max_margin_usd: float = None


@dataclass
class MockSettings:
    risk: MockRiskSettings = field(default_factory=MockRiskSettings)


def get_trailing_activation_pct(settings: MockSettings, leverage: int = None) -> float:
    if leverage is None or leverage <= 0:
        leverage = DEFAULT_LEVERAGE
    
    mode = getattr(settings.risk, 'trailing_activation_mode', 'price')
    
    if mode == "margin":
        margin_pct = getattr(settings.risk, 'trailing_activation_pct_margin', 7.0)
        return calculate_price_pct_from_margin_pct(margin_pct, leverage)
    else:
        return settings.risk.trailing_stop_activation_pct


def get_breakeven_activation_pct(settings: MockSettings, level: int, leverage: int = None) -> float:
    if leverage is None or leverage <= 0:
        leverage = DEFAULT_LEVERAGE
    
    mode = getattr(settings.risk, 'breakeven_activation_mode', 'price')
    
    if mode == "margin":
        if level == 1:
            margin_pct = getattr(settings.risk, 'breakeven_level1_activation_pct_margin', 5.0)
        else:
            margin_pct = getattr(settings.risk, 'breakeven_level2_activation_pct_margin', 10.0)
        return calculate_price_pct_from_margin_pct(margin_pct, leverage)
    else:
        if level == 1:
            return settings.risk.breakeven_level1_activation_pct
        else:
            return settings.risk.breakeven_level2_activation_pct


def validate_leverage_settings(leverage: int) -> tuple:
    if leverage < 1:
        return False, "Leverage must be at least 1"
    if leverage > 100:
        return False, "Leverage cannot exceed 100"
    return True, ""


def get_dca_drawdown_pct(settings: MockSettings, leverage: int = None) -> float:
    if leverage is None or leverage <= 0:
        leverage = DEFAULT_LEVERAGE
    
    mode = getattr(settings.risk, 'dca_drawdown_mode', 'price')
    
    if mode == "margin":
        margin_pct = getattr(settings.risk, 'dca_drawdown_pct_margin', 3.0)
        return calculate_price_pct_from_margin_pct(margin_pct, leverage)
    else:
        return settings.risk.dca_drawdown_pct


def get_partial_close_progress_pct(
    settings: MockSettings,
    current_progress: float,
    leverage: int = None
) -> float:
    if leverage is None or leverage <= 0:
        leverage = DEFAULT_LEVERAGE
    
    partial_close_mode = getattr(settings.risk, 'partial_close_mode', 'price')
    
    if partial_close_mode == "margin":
        return current_progress * leverage
    else:
        return current_progress


def get_max_margin_usd(settings: MockSettings, leverage: int = None) -> float:
    if leverage is None or leverage <= 0:
        leverage = DEFAULT_LEVERAGE
    
    max_margin = getattr(settings.risk, 'max_margin_usd', None)
    if max_margin and max_margin > 0:
        return float(max_margin)
    
    max_position = getattr(settings.risk, 'max_position_usd', None)
    if max_position and max_position > 0:
        return float(max_position) / leverage
    
    return None


def calculate_margin_based_rr(
    tp_pct: float,
    sl_pct: float,
    leverage: int,
    mode: str = "price"
) -> float:
    if not tp_pct or not sl_pct or sl_pct <= 0:
        return None
    
    if mode == "margin":
        return (tp_pct * leverage) / (sl_pct * leverage)
    else:
        return tp_pct / sl_pct


class TestMarginToPriceConversion(unittest.TestCase):
    """Test conversion between margin percentage and price percentage."""
    
    def test_margin_to_price_at_default_leverage(self):
        result = calculate_price_pct_from_margin_pct(10.0, 10)
        self.assertAlmostEqual(result, 0.01, places=5)
    
    def test_margin_to_price_at_higher_leverage(self):
        result = calculate_price_pct_from_margin_pct(10.0, 20)
        self.assertAlmostEqual(result, 0.005, places=5)
    
    def test_margin_to_price_at_lower_leverage(self):
        result = calculate_price_pct_from_margin_pct(10.0, 5)
        self.assertAlmostEqual(result, 0.02, places=5)
    
    def test_price_to_margin_at_default_leverage(self):
        result = calculate_margin_pct_from_price_pct(0.001, 10)
        self.assertAlmostEqual(result, 1.0, places=5)
    
    def test_price_to_margin_at_higher_leverage(self):
        result = calculate_margin_pct_from_price_pct(0.001, 20)
        self.assertAlmostEqual(result, 2.0, places=5)
    
    def test_zero_leverage_uses_default(self):
        result = calculate_price_pct_from_margin_pct(10.0, 0)
        self.assertAlmostEqual(result, 0.01, places=5)
    
    def test_negative_leverage_uses_default(self):
        result = calculate_price_pct_from_margin_pct(10.0, -5)
        self.assertAlmostEqual(result, 0.01, places=5)


class TestTrailingActivationWithLeverage(unittest.TestCase):
    """Test trailing stop activation percentage calculation with different leverage values."""
    
    def setUp(self):
        self.settings = MockSettings()
    
    def test_price_mode_fixed_at_default_leverage(self):
        self.settings.risk.trailing_activation_mode = "price"
        result = get_trailing_activation_pct(self.settings, 10)
        self.assertAlmostEqual(result, 0.007, places=5)
    
    def test_price_mode_fixed_at_different_leverage(self):
        self.settings.risk.trailing_activation_mode = "price"
        result = get_trailing_activation_pct(self.settings, 20)
        self.assertAlmostEqual(result, 0.007, places=5)
    
    def test_margin_mode_scales_with_leverage(self):
        self.settings.risk.trailing_activation_mode = "margin"
        self.settings.risk.trailing_activation_pct_margin = 7.0
        
        result_10x = get_trailing_activation_pct(self.settings, 10)
        result_20x = get_trailing_activation_pct(self.settings, 20)
        result_5x = get_trailing_activation_pct(self.settings, 5)
        
        self.assertAlmostEqual(result_10x, 0.007, places=5)
        self.assertAlmostEqual(result_20x, 0.0035, places=5)
        self.assertAlmostEqual(result_5x, 0.014, places=5)
    
    def test_margin_mode_with_different_margin_values(self):
        self.settings.risk.trailing_activation_mode = "margin"
        self.settings.risk.trailing_activation_pct_margin = 14.0
        
        result = get_trailing_activation_pct(self.settings, 10)
        self.assertAlmostEqual(result, 0.014, places=5)


class TestBreakevenActivationWithLeverage(unittest.TestCase):
    """Test breakeven activation percentage calculation with different leverage values."""
    
    def setUp(self):
        self.settings = MockSettings()
    
    def test_price_mode_level1_fixed(self):
        self.settings.risk.breakeven_activation_mode = "price"
        result = get_breakeven_activation_pct(self.settings, 1, 10)
        self.assertAlmostEqual(result, 0.005, places=5)
    
    def test_price_mode_level2_fixed(self):
        self.settings.risk.breakeven_activation_mode = "price"
        result = get_breakeven_activation_pct(self.settings, 2, 10)
        self.assertAlmostEqual(result, 0.01, places=5)
    
    def test_margin_mode_level1_scales_with_leverage(self):
        self.settings.risk.breakeven_activation_mode = "margin"
        self.settings.risk.breakeven_level1_activation_pct_margin = 5.0
        
        result_10x = get_breakeven_activation_pct(self.settings, 1, 10)
        result_20x = get_breakeven_activation_pct(self.settings, 1, 20)
        result_5x = get_breakeven_activation_pct(self.settings, 1, 5)
        
        self.assertAlmostEqual(result_10x, 0.005, places=5)
        self.assertAlmostEqual(result_20x, 0.0025, places=5)
        self.assertAlmostEqual(result_5x, 0.01, places=5)
    
    def test_margin_mode_level2_scales_with_leverage(self):
        self.settings.risk.breakeven_activation_mode = "margin"
        self.settings.risk.breakeven_level2_activation_pct_margin = 10.0
        
        result_10x = get_breakeven_activation_pct(self.settings, 2, 10)
        result_20x = get_breakeven_activation_pct(self.settings, 2, 20)
        
        self.assertAlmostEqual(result_10x, 0.01, places=5)
        self.assertAlmostEqual(result_20x, 0.005, places=5)


class TestTPSLCalculation(unittest.TestCase):
    """Test TP/SL price calculation with different leverage values."""
    
    def test_tp_sl_at_default_leverage_long(self):
        tp, sl = recalculate_tp_sl_for_leverage(
            entry_price=50000.0,
            side="Buy",
            leverage=10,
            target_profit_pct_margin=18.0,
            max_loss_pct_margin=10.0
        )
        
        expected_tp = 50000 * (1 + 18.0 / 10 / 100)
        expected_sl = 50000 * (1 - 10.0 / 10 / 100)
        
        self.assertAlmostEqual(tp, expected_tp, places=2)
        self.assertAlmostEqual(sl, expected_sl, places=2)
    
    def test_tp_sl_at_higher_leverage_long(self):
        tp, sl = recalculate_tp_sl_for_leverage(
            entry_price=50000.0,
            side="Buy",
            leverage=20,
            target_profit_pct_margin=18.0,
            max_loss_pct_margin=10.0
        )
        
        expected_tp = 50000 * (1 + 18.0 / 20 / 100)
        expected_sl = 50000 * (1 - 10.0 / 20 / 100)
        
        self.assertAlmostEqual(tp, expected_tp, places=2)
        self.assertAlmostEqual(sl, expected_sl, places=2)
    
    def test_tp_sl_at_lower_leverage_long(self):
        tp, sl = recalculate_tp_sl_for_leverage(
            entry_price=50000.0,
            side="Buy",
            leverage=5,
            target_profit_pct_margin=18.0,
            max_loss_pct_margin=10.0
        )
        
        expected_tp = 50000 * (1 + 18.0 / 5 / 100)
        expected_sl = 50000 * (1 - 10.0 / 5 / 100)
        
        self.assertAlmostEqual(tp, expected_tp, places=2)
        self.assertAlmostEqual(sl, expected_sl, places=2)
    
    def test_tp_sl_short_side(self):
        tp, sl = recalculate_tp_sl_for_leverage(
            entry_price=50000.0,
            side="Sell",
            leverage=10,
            target_profit_pct_margin=18.0,
            max_loss_pct_margin=10.0
        )
        
        expected_tp = 50000 * (1 - 18.0 / 10 / 100)
        expected_sl = 50000 * (1 + 10.0 / 10 / 100)
        
        self.assertAlmostEqual(tp, expected_tp, places=2)
        self.assertAlmostEqual(sl, expected_sl, places=2)
    
    def test_tp_sl_with_tick_size_rounding(self):
        tp, sl = recalculate_tp_sl_for_leverage(
            entry_price=50000.0,
            side="Buy",
            leverage=10,
            target_profit_pct_margin=18.0,
            max_loss_pct_margin=10.0,
            tick_size=0.5
        )
        
        self.assertEqual(tp % 0.5, 0)
        self.assertEqual(sl % 0.5, 0)
    
    def test_tp_sl_zero_leverage_uses_default(self):
        tp, sl = recalculate_tp_sl_for_leverage(
            entry_price=50000.0,
            side="Buy",
            leverage=0,
            target_profit_pct_margin=18.0,
            max_loss_pct_margin=10.0
        )
        
        expected_tp = 50000 * (1 + 18.0 / 10 / 100)
        expected_sl = 50000 * (1 - 10.0 / 10 / 100)
        
        self.assertAlmostEqual(tp, expected_tp, places=2)
        self.assertAlmostEqual(sl, expected_sl, places=2)


class TestLeverageValidation(unittest.TestCase):
    """Test leverage validation."""
    
    def test_valid_leverage_range(self):
        for lev in [1, 10, 50, 100]:
            is_valid, msg = validate_leverage_settings(lev)
            self.assertTrue(is_valid)
            self.assertEqual(msg, "")
    
    def test_leverage_below_minimum(self):
        is_valid, msg = validate_leverage_settings(0)
        self.assertFalse(is_valid)
        self.assertIn("at least 1", msg)
    
    def test_leverage_above_maximum(self):
        is_valid, msg = validate_leverage_settings(101)
        self.assertFalse(is_valid)
        self.assertIn("cannot exceed 100", msg)
    
    def test_negative_leverage(self):
        is_valid, msg = validate_leverage_settings(-5)
        self.assertFalse(is_valid)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_very_high_leverage(self):
        result = calculate_price_pct_from_margin_pct(100.0, 100)
        self.assertAlmostEqual(result, 0.01, places=5)
    
    def test_very_low_leverage(self):
        result = calculate_price_pct_from_margin_pct(1.0, 1)
        self.assertAlmostEqual(result, 0.01, places=5)
    
    def test_zero_margin_pct(self):
        result = calculate_price_pct_from_margin_pct(0.0, 10)
        self.assertEqual(result, 0.0)
    
    def test_tp_sl_very_small_entry_price(self):
        tp, sl = recalculate_tp_sl_for_leverage(
            entry_price=0.0001,
            side="Buy",
            leverage=10,
            target_profit_pct_margin=18.0,
            max_loss_pct_margin=10.0,
            tick_size=0.00000001
        )
        self.assertGreater(tp, 0.0001)
        self.assertLess(sl, 0.0001)
    
    def test_tp_sl_very_large_entry_price(self):
        tp, sl = recalculate_tp_sl_for_leverage(
            entry_price=1000000.0,
            side="Buy",
            leverage=10
        )
        self.assertGreater(tp, 1000000)
        self.assertLess(sl, 1000000)


if __name__ == "__main__":
    unittest.main()


class TestDCADrawdownWithLeverage(unittest.TestCase):
    """Test DCA drawdown calculation with different leverage values."""
    
    def setUp(self):
        self.settings = MockSettings()
    
    def test_price_mode_fixed(self):
        self.settings.risk.dca_drawdown_mode = "price"
        result = get_dca_drawdown_pct(self.settings, 10)
        self.assertAlmostEqual(result, 0.003, places=5)
    
    def test_margin_mode_scales_with_leverage(self):
        self.settings.risk.dca_drawdown_mode = "margin"
        self.settings.risk.dca_drawdown_pct_margin = 3.0
        
        result_10x = get_dca_drawdown_pct(self.settings, 10)
        result_20x = get_dca_drawdown_pct(self.settings, 20)
        result_5x = get_dca_drawdown_pct(self.settings, 5)
        
        self.assertAlmostEqual(result_10x, 0.003, places=5)
        self.assertAlmostEqual(result_20x, 0.0015, places=5)
        self.assertAlmostEqual(result_5x, 0.006, places=5)


class TestPartialCloseProgressWithLeverage(unittest.TestCase):
    """Test partial close progress calculation with different leverage values."""
    
    def setUp(self):
        self.settings = MockSettings()
    
    def test_price_mode_unchanged(self):
        self.settings.risk.partial_close_mode = "price"
        
        result = get_partial_close_progress_pct(self.settings, 0.5, 10)
        self.assertEqual(result, 0.5)
    
    def test_margin_mode_scales_with_leverage(self):
        self.settings.risk.partial_close_mode = "margin"
        
        result_10x = get_partial_close_progress_pct(self.settings, 0.5, 10)
        result_20x = get_partial_close_progress_pct(self.settings, 0.5, 20)
        result_5x = get_partial_close_progress_pct(self.settings, 0.5, 5)
        
        self.assertEqual(result_10x, 5.0)
        self.assertEqual(result_20x, 10.0)
        self.assertEqual(result_5x, 2.5)


class TestMaxMarginUSD(unittest.TestCase):
    """Test max margin USD calculation."""
    
    def setUp(self):
        self.settings = MockSettings()
    
    def test_explicit_max_margin(self):
        self.settings.risk.max_margin_usd = 100.0
        result = get_max_margin_usd(self.settings, 10)
        self.assertEqual(result, 100.0)
    
    def test_calculated_from_max_position(self):
        self.settings.risk.max_position_usd = 200.0
        self.settings.risk.max_margin_usd = None
        
        result_10x = get_max_margin_usd(self.settings, 10)
        result_20x = get_max_margin_usd(self.settings, 20)
        
        self.assertEqual(result_10x, 20.0)
        self.assertEqual(result_20x, 10.0)
    
    def test_no_limit_returns_none(self):
        self.settings.risk.max_position_usd = None
        self.settings.risk.max_margin_usd = None
        
        result = get_max_margin_usd(self.settings, 10)
        self.assertIsNone(result)


class TestMarginBasedRR(unittest.TestCase):
    """Test margin-based risk-reward calculation."""
    
    def test_price_mode_unchanged(self):
        result = calculate_margin_based_rr(0.02, 0.01, 10, "price")
        self.assertEqual(result, 2.0)
    
    def test_margin_mode_same_as_price(self):
        result = calculate_margin_based_rr(0.02, 0.01, 10, "margin")
        self.assertEqual(result, 2.0)
    
    def test_invalid_inputs(self):
        result = calculate_margin_based_rr(None, 0.01, 10, "price")
        self.assertIsNone(result)
        
        result = calculate_margin_based_rr(0.02, 0, 10, "price")
        self.assertIsNone(result)