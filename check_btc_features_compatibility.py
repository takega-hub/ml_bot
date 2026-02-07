"""
Скрипт для проверки совместимости моделей с BTC фичами.

Проверяет:
1. Была ли модель обучена с BTC фичами
2. Совпадает ли количество фичей
3. Есть ли BTC фичи в feature_names модели
"""
import pickle
import sys
from pathlib import Path
import argparse

def check_model_btc_compatibility(model_path: str) -> dict:
    """
    Проверяет совместимость модели с BTC фичами.
    
    Returns:
        Словарь с результатами проверки
    """
    model_file = Path(model_path)
    if not model_file.exists():
        return {"error": f"Модель не найдена: {model_path}"}
    
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        feature_names = model_data.get("feature_names", [])
        metadata = model_data.get("metadata", {})
        
        # Проверяем наличие BTC фичей
        btc_feature_names = [
            "relative_return",
            "relative_strength_5", "relative_strength_10", "relative_strength_20",
            "relative_strength_ratio_5", "relative_strength_ratio_10", "relative_strength_ratio_20",
            "btc_correlation_10", "btc_correlation_20", "btc_correlation_50",
            "btc_rsi", "btc_rsi_diff", "btc_macd", "btc_adx", "btc_atr_pct",
            "volatility_ratio_vs_btc",
            "alt_btc_lag_1_corr", "alt_btc_lag_2_corr", "alt_btc_lag_3_corr",
        ]
        
        found_btc_features = [f for f in btc_feature_names if f in feature_names]
        
        result = {
            "model": model_file.name,
            "total_features": len(feature_names),
            "btc_features_found": len(found_btc_features),
            "btc_features_list": found_btc_features,
            "has_btc_features": len(found_btc_features) > 0,
            "expected_btc_features": len(btc_feature_names),
            "model_type": metadata.get("model_type", "unknown"),
            "feature_names_sample": feature_names[:10] if feature_names else []
        }
        
        # Определяем символ
        model_name = model_file.name
        if "_" in model_name:
            parts = model_name.replace(".pkl", "").split("_")
            if len(parts) >= 3:
                result["symbol"] = parts[1].upper()
            elif len(parts) >= 2:
                result["symbol"] = parts[1].upper()
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(
        description="Проверка совместимости моделей с BTC фичами"
    )
    parser.add_argument('--model', type=str, help='Путь к модели для проверки')
    parser.add_argument('--symbol', type=str, help='Проверить все модели для символа')
    parser.add_argument('--models-dir', type=str, default='ml_models',
                       help='Директория с моделями')
    
    args = parser.parse_args()
    
    if args.model:
        # Проверяем одну модель
        result = check_model_btc_compatibility(args.model)
        print(f"\n{'='*80}")
        print(f"Проверка модели: {result.get('model', 'unknown')}")
        print(f"{'='*80}")
        
        if "error" in result:
            print(f"❌ Ошибка: {result['error']}")
            return
        
        print(f"Всего фичей: {result['total_features']}")
        print(f"BTC фичей найдено: {result['btc_features_found']}/{result['expected_btc_features']}")
        print(f"Модель обучена с BTC фичами: {'✅ ДА' if result['has_btc_features'] else '❌ НЕТ'}")
        print(f"Тип модели: {result['model_type']}")
        
        if result['has_btc_features']:
            print(f"\n✅ BTC фичи найдены:")
            for feat in result['btc_features_list']:
                print(f"   - {feat}")
        else:
            print(f"\n⚠️  ВНИМАНИЕ: Модель НЕ была обучена с BTC фичами!")
            print(f"   Если BTC фичи включены в strategy_ml.py, это может вызвать ошибки.")
            print(f"   Решение: переобучить модель с BTC фичами или отключить BTC фичи.")
        
        print(f"\nПримеры фичей модели:")
        for feat in result['feature_names_sample']:
            print(f"   - {feat}")
            
    elif args.symbol:
        # Проверяем все модели для символа
        from run_all_backtests import find_models_for_symbol
        
        models = find_models_for_symbol(args.symbol, args.models_dir)
        if not models:
            print(f"❌ Не найдено моделей для {args.symbol}")
            return
        
        print(f"\n{'='*80}")
        print(f"Проверка моделей для {args.symbol}")
        print(f"{'='*80}\n")
        
        results = []
        for model_path in models:
            result = check_model_btc_compatibility(model_path)
            if "error" not in result:
                results.append(result)
        
        # Группируем по наличию BTC фичей
        with_btc = [r for r in results if r['has_btc_features']]
        without_btc = [r for r in results if not r['has_btc_features']]
        
        print(f"Всего моделей: {len(results)}")
        print(f"С BTC фичами: {len(with_btc)}")
        print(f"Без BTC фичей: {len(without_btc)}")
        
        if without_btc:
            print(f"\n⚠️  МОДЕЛИ БЕЗ BTC ФИЧЕЙ (нужно переобучить или отключить BTC фичи):")
            for r in without_btc:
                print(f"   - {r['model']} ({r['total_features']} фичей)")
        
        if with_btc:
            print(f"\n✅ МОДЕЛИ С BTC ФИЧАМИ:")
            for r in with_btc:
                print(f"   - {r['model']} ({r['total_features']} фичей, {r['btc_features_found']} BTC фичей)")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
