"""
Скрипт для тестирования новых MTF моделей (включая QuadEnsemble).
"""
import sys
import os
import pandas as pd
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from backtest_ml_strategy import run_exact_backtest

# Новые MTF модели для тестирования
models = [
    ("BTCUSDT", "rf_BTCUSDT_15_mtf.pkl"),
    ("BTCUSDT", "xgb_BTCUSDT_15_mtf.pkl"),
    ("BTCUSDT", "ensemble_BTCUSDT_15_mtf.pkl"),
    ("BTCUSDT", "triple_ensemble_BTCUSDT_15_mtf.pkl"),
    ("BTCUSDT", "quad_ensemble_BTCUSDT_15_mtf.pkl"),  # Самая важная новинка
]

def main():
    print("="*80)
    print("ТЕСТИРОВАНИЕ НОВЫХ MTF МОДЕЛЕЙ (14 дней, 20% депозита)")
    print("="*80)
    
    results = []
    
    for symbol, model_name in models:
        model_path = f"ml_models/{model_name}"
        
        # Проверяем наличие файла
        if not os.path.exists(model_path):
            print(f"⚠️ Модель {model_name} не найдена, пропускаем.")
            continue
            
        print(f"\n{'='*80}")
        print(f"Тестирование: {model_name}")
        print(f"{'='*80}\n")
        
        try:
            # Запускаем точный бэктест
            metrics = run_exact_backtest(
                model_path=model_path,
                symbol=symbol,
                days_back=14,
                interval="15",
                initial_balance=1000.0,
                risk_per_trade=0.02, # В коде берется из config (20%), это значение может игнорироваться
                leverage=10,
            )
            
            if metrics:
                results.append((model_name, metrics))
                print(f"\n✅ Бэктест завершен для {model_name}")
            else:
                print(f"\n❌ Бэктест не вернул метрики для {model_name}")
                
        except Exception as e:
            print(f"\n❌ Ошибка при бэктесте {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Итоговый отчет
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ (14 дней)")
    print("="*80)
    print(f"{'Модель':<35} | {'Win Rate':<8} | {'PnL':<10} | {'Trades':<6} | {'PF':<6} | {'TP/SL %':<8}")
    print("-" * 85)
    
    report_data = []
    
    for model_name, m in results:
        # Сокращаем имя для таблицы
        short_name = model_name.replace("_BTCUSDT_15_mtf.pkl", "").replace("ensemble", "ens")
        
        print(f"{short_name:<35} | {m.win_rate:>6.1f}% | {m.total_pnl_pct:>8.2f}% | {m.total_trades:>6d} | {m.profit_factor:>6.2f} | {m.signals_with_tp_sl_pct:>7.1f}%")
        
        report_data.append({
            "model": model_name,
            "win_rate": m.win_rate,
            "total_pnl_pct": m.total_pnl_pct,
            "total_pnl": m.total_pnl,
            "total_trades": m.total_trades,
            "profit_factor": m.profit_factor,
            "signals_with_tp_sl_pct": m.signals_with_tp_sl_pct,
            "max_drawdown_pct": m.max_drawdown_pct,
            "sharpe_ratio": m.sharpe_ratio
        })
        
    print("="*80)
    
    # Сохраняем в JSON
    import json
    from datetime import datetime
    
    report_filename = f"backtest_reports/mtf_models_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("backtest_reports", exist_ok=True)
    
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n💾 Полный отчет сохранен в: {report_filename}")

if __name__ == "__main__":
    main()
