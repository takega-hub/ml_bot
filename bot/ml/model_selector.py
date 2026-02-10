"""
Утилита для автоматического выбора лучших моделей для MTF стратегии.

Приоритет выбора:
1. Файл best_strategies_*.json (последний по времени)
2. comparison_15m_vs_1h.csv
3. ml_models_comparison_*.csv (последний файл)
4. Fallback на первые найденные модели
"""
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


def load_best_strategies_from_file(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Загружает лучшие стратегии из файла best_strategies_*.json.
    
    Returns:
        Словарь с информацией о стратегии или None
    """
    # Ищем все файлы best_strategies_*.json
    strategy_files = sorted(
        Path(".").glob("best_strategies_*.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True
    )
    
    # Также проверяем в optimization_results
    optimization_dir = Path("optimization_results")
    if optimization_dir.exists():
        strategy_files.extend(
            sorted(
                optimization_dir.glob("best_strategies_*.json"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True
            )
        )
    
    if not strategy_files:
        return None
    
    # Загружаем последний файл
    try:
        with open(strategy_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        symbols = data.get('symbols', {})
        if symbol.upper() in symbols:
            strategy = symbols[symbol.upper()]
            strategy['_source_file'] = str(strategy_files[0])
            return strategy
    except Exception as e:
        logger.warning(f"Ошибка загрузки best_strategies из {strategy_files[0]}: {e}")
    
    return None


def find_best_models_from_comparison(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Находит лучшие модели из результатов сравнения.
    Сначала ищет в comparison_15m_vs_1h.csv, затем в ml_models_comparison_*.csv.
    
    Returns:
        (model_1h_path, model_15m_path) или (None, None) если не найдены
    """
    models_dir = Path("ml_models")
    symbol_upper = symbol.upper()
    
    # 1. Сначала ищем в comparison_15m_vs_1h.csv (если есть)
    comparison_15m_1h = Path("comparison_15m_vs_1h.csv")
    if comparison_15m_1h.exists():
        try:
            df = pd.read_csv(comparison_15m_1h)
            symbol_data = df[df['symbol'] == symbol_upper]
            if not symbol_data.empty:
                best_row = symbol_data.iloc[0]
                best_15m_name = best_row.get('best_15m_model', '')
                best_1h_name = best_row.get('best_1h_model', '')
                
                if best_15m_name and best_1h_name:
                    model_15m_path = models_dir / f"{best_15m_name}.pkl"
                    model_1h_path = models_dir / f"{best_1h_name}.pkl"
                    
                    if model_15m_path.exists() and model_1h_path.exists():
                        return str(model_1h_path), str(model_15m_path)
        except Exception as e:
            logger.debug(f"Ошибка загрузки из comparison_15m_vs_1h.csv: {e}")
    
    # 2. Ищем в ml_models_comparison_*.csv
    comparison_files = sorted(
        Path(".").glob("ml_models_comparison_*.csv"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True
    )
    
    if not comparison_files:
        return None, None
    
    try:
        # Загружаем последний файл сравнения
        df = pd.read_csv(comparison_files[0])
        
        # Ищем лучшие модели для символа, раздельно для 1h и 15m
        symbol_data = df[df['symbol'] == symbol_upper]
        if symbol_data.empty:
            return None, None
        
        # Лучшая 1h модель (mode_suffix == '1h')
        symbol_1h = symbol_data[symbol_data.get('mode_suffix', '') == '1h']
        if symbol_1h.empty:
            # Пробуем найти по имени файла
            symbol_1h = symbol_data[symbol_data['model_filename'].str.contains('_60_|_1h', na=False)]
        
        # Лучшая 15m модель (mode_suffix == '15m')
        symbol_15m = symbol_data[symbol_data.get('mode_suffix', '') == '15m']
        if symbol_15m.empty:
            # Пробуем найти по имени файла
            symbol_15m = symbol_data[symbol_data['model_filename'].str.contains('_15_|_15m', na=False)]
        
        if symbol_1h.empty or symbol_15m.empty:
            return None, None
        
        # Сортируем по total_pnl_pct и берем лучшие
        best_1h = symbol_1h.sort_values('total_pnl_pct', ascending=False).iloc[0]
        best_15m = symbol_15m.sort_values('total_pnl_pct', ascending=False).iloc[0]
        
        # Извлекаем имена моделей
        best_1h_name = best_1h.get('model_name', '') or best_1h.get('model_filename', '').replace('.pkl', '')
        best_15m_name = best_15m.get('model_name', '') or best_15m.get('model_filename', '').replace('.pkl', '')
        
        if not best_1h_name or not best_15m_name:
            return None, None
        
        # Ищем файлы моделей
        model_1h_path = models_dir / f"{best_1h_name}.pkl"
        model_15m_path = models_dir / f"{best_15m_name}.pkl"
        
        if model_1h_path.exists() and model_15m_path.exists():
            return str(model_1h_path), str(model_15m_path)
    except Exception as e:
        logger.warning(f"Ошибка загрузки лучших моделей из сравнения: {e}")
    
    return None, None


def find_all_models_for_symbol(symbol: str) -> Tuple[List[str], List[str]]:
    """
    Находит ВСЕ модели 1h и 15m для символа.
    
    Returns:
        (list_1h_models, list_15m_models)
    """
    models_dir = Path("ml_models")
    if not models_dir.exists():
        return [], []
    
    # Ищем 1h модели
    models_1h = list(models_dir.glob(f"*_{symbol}_60_*.pkl"))
    if not models_1h:
        models_1h = list(models_dir.glob(f"*_{symbol}_*1h*.pkl"))
    
    # Ищем 15m модели
    models_15m = list(models_dir.glob(f"*_{symbol}_15_*.pkl"))
    if not models_15m:
        models_15m = list(models_dir.glob(f"*_{symbol}_*15m*.pkl"))
    
    # Сортируем по имени (для стабильности)
    models_1h = sorted([str(m) for m in models_1h])
    models_15m = sorted([str(m) for m in models_15m])
    
    return models_1h, models_15m


def select_best_models(
    symbol: str,
    use_best_from_comparison: bool = True,
    model_1h_name: Optional[str] = None,
    model_15m_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Выбирает лучшие модели для символа с учетом всех источников.
    
    Args:
        symbol: Торговая пара
        use_best_from_comparison: Использовать лучшие модели из результатов сравнения
        model_1h_name: Конкретное имя 1h модели (если указано, используется оно)
        model_15m_name: Конкретное имя 15m модели (если указано, используется оно)
    
    Returns:
        (model_1h_path, model_15m_path, info_dict) где info_dict содержит информацию об источнике
    """
    models_dir = Path("ml_models")
    info = {
        "source": "unknown",
        "model_1h": None,
        "model_15m": None,
        "strategy_type": None,
        "confidence_threshold_1h": None,
        "confidence_threshold_15m": None,
        "alignment_mode": None,
        "require_alignment": None,
    }
    
    if not models_dir.exists():
        logger.error(f"Директория {models_dir} не существует")
        return None, None, info
    
    # 1. Проверяем best_strategies_*.json (высший приоритет)
    best_strategy = load_best_strategies_from_file(symbol)
    if best_strategy and best_strategy.get('strategy_type') == 'mtf':
        model_1h_name_from_strategy = best_strategy.get('model_1h', '').replace('.pkl', '')
        model_15m_name_from_strategy = best_strategy.get('model_15m', '').replace('.pkl', '')
        
        model_1h_path = models_dir / f"{model_1h_name_from_strategy}.pkl"
        model_15m_path = models_dir / f"{model_15m_name_from_strategy}.pkl"
        
        if model_1h_path.exists() and model_15m_path.exists():
            info.update({
                "source": "best_strategies_json",
                "model_1h": model_1h_name_from_strategy,
                "model_15m": model_15m_name_from_strategy,
                "strategy_type": "mtf",
                "confidence_threshold_1h": best_strategy.get('confidence_threshold_1h', 0.50),
                "confidence_threshold_15m": best_strategy.get('confidence_threshold_15m', 0.35),
                "alignment_mode": best_strategy.get('alignment_mode', 'strict'),
                "require_alignment": best_strategy.get('require_alignment', True),
                "metrics": best_strategy.get('metrics', {}),
                "source_file": best_strategy.get('_source_file', ''),
            })
            logger.info(f"[{symbol}] ✅ Используются модели из best_strategies.json")
            logger.info(f"   1h: {model_1h_name_from_strategy}")
            logger.info(f"   15m: {model_15m_name_from_strategy}")
            return str(model_1h_path), str(model_15m_path), info
    
    # 2. Если указаны конкретные модели, используем их
    if model_1h_name:
        model_1h_path = models_dir / f"{model_1h_name}.pkl"
        if model_1h_path.exists():
            model_1h = str(model_1h_path)
        else:
            logger.warning(f"Указанная 1h модель не найдена: {model_1h_name}")
            model_1h = None
    else:
        model_1h = None
    
    if model_15m_name:
        model_15m_path = models_dir / f"{model_15m_name}.pkl"
        if model_15m_path.exists():
            model_15m = str(model_15m_path)
        else:
            logger.warning(f"Указанная 15m модель не найдена: {model_15m_name}")
            model_15m = None
    else:
        model_15m = None
    
    # Если обе модели найдены, возвращаем их
    if model_1h and model_15m:
        info.update({
            "source": "manual",
            "model_1h": Path(model_1h).stem,
            "model_15m": Path(model_15m).stem,
        })
        return model_1h, model_15m, info
    
    # 3. Ищем лучшие из сравнения
    if use_best_from_comparison and (not model_1h or not model_15m):
        best_1h, best_15m = find_best_models_from_comparison(symbol)
        if best_1h and best_15m:
            info.update({
                "source": "comparison_csv",
                "model_1h": Path(best_1h).stem,
                "model_15m": Path(best_15m).stem,
            })
            logger.info(f"[{symbol}] ✅ Используются лучшие модели из сравнения")
            logger.info(f"   1h: {Path(best_1h).name}")
            logger.info(f"   15m: {Path(best_15m).name}")
            return best_1h, best_15m, info
    
    # 4. Если не нашли лучшие, ищем любые доступные
    models_1h_list, models_15m_list = find_all_models_for_symbol(symbol)
    
    if not model_1h and models_1h_list:
        model_1h = models_1h_list[0]
        info["model_1h"] = Path(model_1h).stem
        logger.info(f"[{symbol}] 📦 Используется 1h модель: {Path(model_1h).name}")
    
    if not model_15m and models_15m_list:
        model_15m = models_15m_list[0]
        info["model_15m"] = Path(model_15m).stem
        logger.info(f"[{symbol}] 📦 Используется 15m модель: {Path(model_15m).name}")
    
    if not model_1h:
        logger.warning(f"1h модель для {symbol} не найдена")
        if models_1h_list:
            logger.info(f"   Доступные 1h модели: {[Path(m).name for m in models_1h_list]}")
    if not model_15m:
        logger.warning(f"15m модель для {symbol} не найдена")
        if models_15m_list:
            logger.info(f"   Доступные 15m модели: {[Path(m).name for m in models_15m_list]}")
    
    if model_1h and model_15m:
        if info["source"] == "unknown":
            info["source"] = "fallback_first_found"
    
    return model_1h, model_15m, info
