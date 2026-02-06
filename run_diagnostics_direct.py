"""
Скрипт для запуска диагностики всех 15m моделей.
Импортирует функцию напрямую, чтобы избежать проблем с командной строкой.
"""
import sys
import os
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from backtest_ml_strategy import run_exact_backtest

# Модели для тестирования (15m, не MTF)
models = [
    ("BTCUSDT", "xgb_BTCUSDT_15_15m.pkl"),
    ("BTCUSDT", "rf_BTCUSDT_15_15m.pkl"),
    ("BTCUSDT", "ensemble_BTCUSDT_15_15m.pkl"),
    ("BTCUSDT", "triple_ensemble_BTCUSDT_15_15m.pkl"),
    ("ETHUSDT", "xgb_ETHUSDT_15_15m.pkl"),
    ("SOLUSDT", "rf_SOLUSDT_15_15m.pkl"),
]

def main():
    print("="*80)
    print("ДИАГНОСТИКА КАЧЕСТВА СИГНАЛОВ - БЭКТЕСТЫ ВСЕХ 15M МОДЕЛЕЙ")
    print("="*80)
    print("Период: 14 дней")
    print("Размер позиции: 20% депозита (из config.py)")
    print("="*80)
    
    results = []
    for symbol, model_name in models:
        model_path = f"ml_models/{model_name}"
        
        print(f"\n{'='*80}")
        print(f"Тестирование: {model_name}")
        print(f"Символ: {symbol}")
        print(f"{'='*80}\n")
        
        try:
            metrics = run_exact_backtest(
                model_path=model_path,
                symbol=symbol,
                days_back=14,
                interval="15",
                initial_balance=1000.0,
                risk_per_trade=0.02,
                leverage=10,
            )
            
            if metrics:
                results.append((symbol, model_name, True, metrics))
                print(f"\n✅ Бэктест завершен для {model_name}")
            else:
                results.append((symbol, model_name, False, None))
                print(f"\n❌ Бэктест не удался для {model_name}")
        except Exception as e:
            print(f"\n❌ Ошибка при бэктесте {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((symbol, model_name, False, None))
    
    # Итоговый отчет
    print("\n" + "="*80)
    print("ИТОГИ ДИАГНОСТИКИ")
    print("="*80)
    
    successful = []
    failed = []
    
    for symbol, model_name, success, metrics in results:
        if success and metrics:
            successful.append((symbol, model_name, metrics))
            print(f"\n✅ {model_name} ({symbol}):")
            print(f"   Сделок: {metrics.total_trades}")
            print(f"   Win Rate: {metrics.win_rate:.1f}%")
            print(f"   Profit Factor: {metrics.profit_factor:.2f}")
            print(f"   Net PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:+.2f}%)")
            print(f"   Сигналов с TP/SL: {metrics.signals_with_tp_sl_pct:.1f}%")
            print(f"   Сигналов с SL=1%: {metrics.signals_with_correct_sl_pct:.1f}%")
        else:
            failed.append((symbol, model_name))
            print(f"\n❌ {model_name} ({symbol}): Ошибка")
    
    print("\n" + "="*80)
    print(f"Успешно: {len(successful)}/{len(models)}")
    print(f"Ошибок: {len(failed)}/{len(models)}")
    print("="*80)
    
    # Сохраняем результаты в файл
    if successful:
        import json
        from datetime import datetime
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "period_days": 14,
            "models": []
        }
        
        for symbol, model_name, metrics in successful:
            report["models"].append({
                "symbol": symbol,
                "model": model_name,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_pnl": metrics.total_pnl,
                "total_pnl_pct": metrics.total_pnl_pct,
                "signals_with_tp_sl_pct": metrics.signals_with_tp_sl_pct,
                "signals_with_correct_sl_pct": metrics.signals_with_correct_sl_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "sharpe_ratio": metrics.sharpe_ratio,
            })
        
        report_file = f"backtest_reports/diagnostics_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("backtest_reports", exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Отчет сохранен: {report_file}")

if __name__ == "__main__":
    main()
