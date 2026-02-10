"""
Модуль для экспорта истории сделок в Excel для анализа.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging

from bot.state import TradeRecord

logger = logging.getLogger(__name__)


def export_trades_to_excel(
    trades: List[TradeRecord],
    output_dir: str = "trade_history",
    filename: Optional[str] = None
) -> str:
    """
    Экспортирует историю сделок в Excel файл.
    
    Args:
        trades: Список сделок для экспорта
        output_dir: Директория для сохранения файла
        filename: Имя файла (если не указано, генерируется автоматически)
    
    Returns:
        Путь к созданному Excel файлу
    """
    if not trades:
        logger.warning("No trades to export")
        return ""
    
    logger.info(f"📊 Starting export of {len(trades)} trade(s) to Excel...")
    
    # Создаем директорию если не существует
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_path}")
    
    # Генерируем имя файла если не указано
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trades_history_{timestamp}.xlsx"
    
    filepath = output_path / filename
    
    # Подготавливаем данные для экспорта
    data = []
    for trade in trades:
        # Извлекаем информацию о модели
        model_name = trade.model_name or ""
        model_type = ""
        if model_name:
            # Пытаемся определить тип модели из имени файла
            if "ensemble" in model_name.lower():
                if "triple" in model_name.lower():
                    model_type = "TripleEnsemble"
                elif "quad" in model_name.lower():
                    model_type = "QuadEnsemble"
                else:
                    model_type = "Ensemble"
            elif "rf_" in model_name.lower() or "random" in model_name.lower():
                model_type = "RandomForest"
            elif "xgb_" in model_name.lower() or "xgboost" in model_name.lower():
                model_type = "XGBoost"
            elif "lgb_" in model_name.lower() or "lightgbm" in model_name.lower():
                model_type = "LightGBM"
            else:
                model_type = "Unknown"
        
        # Извлекаем параметры из signal_parameters
        signal_params = trade.signal_parameters or {}
        tp_pct = signal_params.get('take_profit_pct', None)
        sl_pct = signal_params.get('stop_loss_pct', None)
        rr_ratio = signal_params.get('risk_reward_ratio', None)
        
        # Вычисляем длительность сделки
        duration_minutes = None
        if trade.entry_time and trade.exit_time:
            try:
                entry_dt = datetime.fromisoformat(trade.entry_time.replace('Z', '+00:00'))
                exit_dt = datetime.fromisoformat(trade.exit_time.replace('Z', '+00:00'))
                duration = exit_dt - entry_dt
                duration_minutes = duration.total_seconds() / 60
            except Exception as e:
                logger.debug(f"Could not calculate duration: {e}")
        
        # Вычисляем размер позиции в USD
        position_size_usd = trade.entry_price * trade.qty if trade.entry_price and trade.qty else 0.0
        
        # Формируем строку данных
        row = {
            # Основная информация
            "Время входа": trade.entry_time,
            "Время выхода": trade.exit_time or "",
            "Длительность (мин)": duration_minutes,
            "Символ": trade.symbol,
            "Направление": trade.side,
            "Статус": trade.status,
            
            # Цены
            "Цена входа": trade.entry_price,
            "Цена выхода": trade.exit_price or "",
            "Take Profit": trade.take_profit or "",
            "Stop Loss": trade.stop_loss or "",
            
            # Размеры и PnL
            "Количество": trade.qty,
            "Размер позиции (USD)": position_size_usd,
            "Маржа (USD)": trade.margin_usd,
            "Плечо": trade.leverage,
            "PnL (USD)": trade.pnl_usd,
            "PnL (%)": trade.pnl_pct,
            
            # ML модель
            "Модель ML": model_name,
            "Тип модели": model_type,
            "Горизонт": trade.horizon,
            
            # Сигнал
            "Причина входа": trade.entry_reason,
            "Причина выхода": trade.exit_reason,
            "Уверенность (%)": trade.confidence * 100 if trade.confidence else 0,
            "Сила сигнала": trade.signal_strength,
            
            # Параметры сигнала
            "TP (%)": tp_pct * 100 if tp_pct else "",
            "SL (%)": sl_pct * 100 if sl_pct else "",
            "Risk/Reward": rr_ratio,
            
            # Дополнительно
            "DCA счетчик": trade.dca_count,
        }
        
        data.append(row)
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Сортируем по времени входа (новые сверху)
    if "Время входа" in df.columns:
        df = df.sort_values("Время входа", ascending=False)
    
    try:
        # Экспортируем в Excel с форматированием
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Сделки', index=False)
            
            # Получаем workbook и worksheet для форматирования
            workbook = writer.book
            worksheet = writer.sheets['Сделки']
            
            # Автоматическая ширина колонок
            from openpyxl.utils import get_column_letter
            for idx, col in enumerate(df.columns, 1):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                )
                col_letter = get_column_letter(idx)
                worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)
            
            # Форматирование числовых колонок
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.styles.numbers import FORMAT_PERCENTAGE_00, FORMAT_NUMBER_00
            
            # Заголовки
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF")
            
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Форматирование PnL - зеленый для прибыли, красный для убытка
            pnl_col_idx = None
            for idx, col in enumerate(df.columns, 1):
                if col == "PnL (USD)":
                    pnl_col_idx = idx
                    break
            
            if pnl_col_idx:
                pnl_fill_profit = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                pnl_fill_loss = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                
                for row_idx in range(2, len(df) + 2):
                    cell = worksheet.cell(row=row_idx, column=pnl_col_idx)
                    pnl_value = df.iloc[row_idx - 2]["PnL (USD)"]
                    if isinstance(pnl_value, (int, float)):
                        if pnl_value > 0:
                            cell.fill = pnl_fill_profit
                        elif pnl_value < 0:
                            cell.fill = pnl_fill_loss
            
            # Форматирование процентов
            for idx, col in enumerate(df.columns, 1):
                if "%" in col or col == "Уверенность (%)":
                    for row_idx in range(2, len(df) + 2):
                        cell = worksheet.cell(row=row_idx, column=idx)
                        if cell.value and isinstance(cell.value, (int, float)):
                            cell.number_format = FORMAT_PERCENTAGE_00
        
        logger.info(f"✅ Экспортировано {len(trades)} сделок в {filepath}")
        return str(filepath)
    
    except ImportError:
        # Если openpyxl не установлен, экспортируем без форматирования
        logger.warning("openpyxl not installed, exporting without formatting")
        df.to_excel(filepath, index=False)
        return str(filepath)
    except Exception as e:
        logger.error(f"Error exporting trades to Excel: {e}", exc_info=True)
        # Пробуем экспортировать в CSV как fallback
        csv_path = filepath.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Exported to CSV instead: {csv_path}")
        return str(csv_path)


def export_closed_trades_to_excel(
    trades: List[TradeRecord],
    output_dir: str = "trade_history",
    filename: Optional[str] = None
) -> str:
    """
    Экспортирует только закрытые сделки в Excel файл.
    
    Args:
        trades: Список всех сделок
        output_dir: Директория для сохранения файла
        filename: Имя файла (если не указано, генерируется автоматически)
    
    Returns:
        Путь к созданному Excel файлу
    """
    closed_trades = [t for t in trades if t.status == "closed"]
    return export_trades_to_excel(closed_trades, output_dir, filename)
