"""
Стресс-тестирование лучших ML моделей на различных сценариях.
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from backtest_ml_strategy import run_exact_backtest

# Лучшие модели по результатам бэктеста
BEST_MODELS = [
    ("BTCUSDT", "ml_models/rf_BTCUSDT_15_mtf.pkl"),  # Лучшая: Win Rate 60%, PF 3.36
    ("BTCUSDT", "ml_models/quad_ensemble_BTCUSDT_15_mtf.pkl"),  # Хорошая: PF 2.62
    ("BTCUSDT", "ml_models/xgb_BTCUSDT_15_mtf.pkl"),  # Стабильная: PF 2.27
]

# Сценарии стресс-теста
SCENARIOS = [
    {
        'name': 'Baseline',
        'description': 'Стандартные параметры (0.06% комиссия, 20% депозита)',
        'commission': 0.0006,
        'days': 14,
    },
    {
        'name': 'High Commission',
        'description': 'Комиссия x2 (0.12% вместо 0.06%)',
        'commission': 0.0012,
        'days': 14,
    },
    {
        'name': 'Longer Period',
        'description': '21 день (проверка на разных условиях)',
        'commission': 0.0006,
        'days': 21,
    },
    {
        'name': 'Shorter Period',
        'description': '7 дней (более актуальные условия)',
        'commission': 0.0006,
        'days': 7,
    },
    {
        'name': 'Forward Test',
        'description': 'Последние 3 дня (полностью out-of-sample)',
        'commission': 0.0006,
        'days': 3,
    },
    {
        'name': 'High Volatility',
        'description': 'Период с высокой волатильностью (21 день, проверка устойчивости)',
        'commission': 0.0006,
        'days': 21,
    },
    {
        'name': 'Low Balance',
        'description': 'Низкий начальный баланс ($50 вместо $100)',
        'commission': 0.0006,
        'days': 14,
        'initial_balance': 50.0,
    },
    {
        'name': 'High Leverage',
        'description': 'Высокое плечо (20x вместо 10x)',
        'commission': 0.0006,
        'days': 14,
        'leverage': 20,
    },
]

def main():
    print("=" * 80)
    print("🧪 СТРЕСС-ТЕСТИРОВАНИЕ ЛУЧШИХ МОДЕЛЕЙ")
    print("=" * 80)
    
    all_results = []
    
    for symbol, model_path in BEST_MODELS:
        model_name = Path(model_path).stem
        print(f"\n{'='*80}")
        print(f"📊 Модель: {model_name}")
        print(f"{'='*80}\n")
        
        model_results = {
            'model': model_name,
            'symbol': symbol,
            'scenarios': [],
        }
        
        for scenario in SCENARIOS:
            print(f"\n🔬 Сценарий: {scenario['name']}")
            print(f"   {scenario['description']}")
            
            try:
                # Запускаем бэктест с параметрами из сценария
                initial_balance = scenario.get('initial_balance', 100.0)
                leverage = scenario.get('leverage', 10)
                
                metrics = run_exact_backtest(
                    model_path=model_path,
                    symbol=symbol,
                    days_back=scenario['days'],
                    interval="15",
                    initial_balance=initial_balance,
                    risk_per_trade=0.02,
                    leverage=leverage,
                )
                
                if metrics:
                    scenario_result = {
                        'scenario_name': scenario['name'],
                        'win_rate': metrics.win_rate,
                        'profit_factor': metrics.profit_factor,
                        'total_pnl_pct': metrics.total_pnl_pct,
                        'total_trades': metrics.total_trades,
                        'max_drawdown_pct': metrics.max_drawdown_pct,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'signals_with_tp_sl_pct': metrics.signals_with_tp_sl_pct,
                    }
                    
                    model_results['scenarios'].append(scenario_result)
                    
                    print(f"   ✅ Win Rate: {metrics.win_rate:.1f}%")
                    print(f"   ✅ Profit Factor: {metrics.profit_factor:.2f}")
                    print(f"   ✅ PnL: {metrics.total_pnl_pct:+.2f}%")
                    print(f"   ✅ Trades: {metrics.total_trades}")
                    print(f"   ✅ Max DD: {metrics.max_drawdown_pct:.2f}%")
                else:
                    print(f"   ❌ Бэктест не вернул метрики")
                    
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        all_results.append(model_results)
    
    # Итоговый отчет
    print("\n" + "=" * 80)
    print("📊 ИТОГОВЫЙ ОТЧЕТ СТРЕСС-ТЕСТА")
    print("=" * 80)
    
    for model_result in all_results:
        print(f"\n📈 {model_result['model']}:")
        print(f"{'Сценарий':<20} | {'Win Rate':<10} | {'PF':<8} | {'PnL %':<10} | {'Trades':<8} | {'Max DD %':<10}")
        print("-" * 85)
        
        for scenario in model_result['scenarios']:
            print(f"{scenario['scenario_name']:<20} | {scenario['win_rate']:>8.1f}% | {scenario['profit_factor']:>6.2f} | {scenario['total_pnl_pct']:>8.2f}% | {scenario['total_trades']:>6d} | {scenario['max_drawdown_pct']:>8.2f}%")
    
    # Сохраняем результаты
    output_dir = Path("backtest_reports")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f"stress_test_results_{timestamp}.json"
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены в: {json_file}")
    
    # Анализ устойчивости
    print("\n" + "=" * 80)
    print("🔍 АНАЛИЗ УСТОЙЧИВОСТИ")
    print("=" * 80)
    
    for model_result in all_results:
        baseline = next((s for s in model_result['scenarios'] if s['scenario_name'] == 'Baseline'), None)
        if not baseline:
            continue
        
        print(f"\n📊 {model_result['model']}:")
        
        # Проверяем критерии успеха
        criteria = {
            'PnL > 0 в Baseline': baseline['total_pnl_pct'] > 0,
            'Win Rate > 45%': baseline['win_rate'] > 45,
            'Profit Factor > 1.5': baseline['profit_factor'] > 1.5,
            'Max DD < 20%': baseline['max_drawdown_pct'] < 20,
        }
        
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")
        
        # Проверяем устойчивость к изменениям
        high_commission = next((s for s in model_result['scenarios'] if s['scenario_name'] == 'High Commission'), None)
        if high_commission:
            pnl_change = high_commission['total_pnl_pct'] - baseline['total_pnl_pct']
            if pnl_change < -50:
                print(f"   ⚠️  КРИТИЧНО: PnL падает на {abs(pnl_change):.1f}% при удвоенной комиссии")
            elif pnl_change < -20:
                print(f"   ⚠️  PnL падает на {abs(pnl_change):.1f}% при удвоенной комиссии")
            else:
                print(f"   ✅ Устойчива к высоким комиссиям (изменение: {pnl_change:+.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ СТРЕСС-ТЕСТ ЗАВЕРШЕН")
    print("=" * 80)

if __name__ == "__main__":
    main()
