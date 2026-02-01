"""
LSTM модель для предсказания движения цены на основе временных рядов.
Использует последовательности свечей для улавливания долгосрочных зависимостей.
"""
import warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from bot.ml.feature_engineering import FeatureEngineer


class CryptoSequenceDataset(Dataset):
    """Dataset для временных рядов крипто данных."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Массив последовательностей (n_samples, seq_length, n_features)
            targets: Массив целей (n_samples,) со значениями -1, 0, 1
        """
        self.sequences = torch.FloatTensor(sequences)
        # Преобразуем targets: -1,0,1 -> 0,1,2 для классификации
        self.targets = torch.LongTensor(targets + 1)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """
    LSTM модель для предсказания движения цены.
    
    Архитектура:
    - LSTM слои для анализа последовательностей
    - Dropout для регуляризации
    - Dense слои для финального предсказания
    """
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        """
        Args:
            input_size: Количество фичей на свечу
            hidden_size: Размер скрытого слоя LSTM
            num_layers: Количество LSTM слоев
            dropout: Dropout для регуляризации
            num_classes: Количество классов (3: SHORT, HOLD, LONG)
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Dropout после LSTM
        self.dropout = nn.Dropout(dropout)
        
        # Финальный слой классификации
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов для лучшей сходимости."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Устанавливаем forget gate bias в 1 для лучшей сходимости
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Входные данные (batch_size, seq_length, input_size)
        
        Returns:
            Логиты для классов (batch_size, num_classes)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Берем последний выход последовательности
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # Финальный слой
        output = self.fc(last_output)
        
        return output


class LSTMTrainer:
    """
    Тренер для LSTM модели.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 50,
        device: Optional[str] = None,
    ):
        """
        Args:
            sequence_length: Длина последовательности (количество свечей)
            hidden_size: Размер скрытого слоя LSTM
            num_layers: Количество LSTM слоев
            dropout: Dropout для регуляризации
            learning_rate: Скорость обучения
            batch_size: Размер батча
            num_epochs: Количество эпох обучения
            device: Устройство для обучения ('cuda' или 'cpu')
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Определяем устройство
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[LSTM Trainer] Using device: {self.device}")
        
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.scaler = StandardScaler()  # Для нормализации фичей
        self.feature_names = None
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает последовательности из DataFrame.
        
        Args:
            df: DataFrame с фичами и таргетом
            target_col: Название колонки с таргетом
        
        Returns:
            (sequences, targets) где:
            - sequences: (n_samples, sequence_length, n_features)
            - targets: (n_samples,) со значениями -1, 0, 1
        """
        # Выбираем фичи (исключаем OHLCV и таргет)
        feature_cols = [
            col for col in df.columns
            if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', target_col]
            and not col.startswith('target_')
        ]
        
        # Используем ВСЕ доступные фичи для лучшей производительности
        # (вместо ограниченного набора из 12 фичей)
        available_features = feature_cols
        
        # Если фичей слишком много, берем первые 50 (для производительности)
        if len(available_features) > 50:
            print(f"[LSTM Trainer] Too many features ({len(available_features)}), using first 50")
            available_features = feature_cols[:50]
        
        # Сохраняем используемые фичи для последующего использования при предсказании
        self.feature_names = available_features.copy()
        
        print(f"[LSTM Trainer] Using {len(available_features)} features")
        print(f"[LSTM Trainer] Feature examples: {available_features[:10]}")
        
        # Создаем последовательности
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(df)):
            # Последовательность из последних sequence_length свечей
            sequence = df[available_features].iloc[i - self.sequence_length:i].values
            
            # Таргет для текущей свечи
            target = df[target_col].iloc[i]
            
            # Пропускаем NaN
            if np.isnan(sequence).any() or np.isnan(target):
                continue
            
            sequences.append(sequence)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"[LSTM Trainer] Created {len(sequences)} sequences")
        print(f"[LSTM Trainer] Sequence shape: {sequences.shape}")
        target_counts = np.bincount(targets.astype(int) + 1)
        print(f"[LSTM Trainer] Target distribution:")
        for i, count in enumerate(target_counts):
            target_name = {-1: "SHORT", 0: "HOLD", 1: "LONG"}.get(i - 1, f"CLASS_{i-1}")
            pct = (count / len(targets)) * 100 if len(targets) > 0 else 0
            print(f"   {target_name}: {count} ({pct:.1f}%)")
        
        return sequences, targets
    
    def train(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2,
    ) -> Tuple[LSTMModel, Dict[str, Any]]:
        """
        Обучает LSTM модель.
        
        Args:
            df: DataFrame с фичами и таргетом
            validation_split: Доля данных для валидации
        
        Returns:
            (model, metrics) - обученная модель и метрики
        """
        print(f"\n[LSTM Trainer] Starting training...")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.num_epochs}")
        
        # Подготавливаем последовательности
        sequences, targets = self.prepare_sequences(df)
        
        if len(sequences) == 0:
            raise ValueError("No sequences created. Check data and sequence_length.")
        
        # Разделяем на train/validation
        split_idx = int(len(sequences) * (1 - validation_split))
        train_sequences = sequences[:split_idx]
        train_targets = targets[:split_idx]
        val_sequences = sequences[split_idx:]
        val_targets = targets[split_idx:]
        
        # Нормализуем последовательности (критично для LSTM!)
        print(f"\n[LSTM Trainer] Normalizing features...")
        n_samples_train, seq_length, n_features = train_sequences.shape
        n_samples_val = val_sequences.shape[0]
        
        # Reshape для нормализации
        train_flat = train_sequences.reshape(-1, n_features)
        val_flat = val_sequences.reshape(-1, n_features)
        
        # Fit scaler только на train данных
        self.scaler.fit(train_flat)
        
        # Transform обеих выборок
        train_flat_scaled = self.scaler.transform(train_flat)
        val_flat_scaled = self.scaler.transform(val_flat)
        
        # Reshape обратно
        train_sequences = train_flat_scaled.reshape(n_samples_train, seq_length, n_features)
        val_sequences = val_flat_scaled.reshape(n_samples_val, seq_length, n_features)
        
        print(f"[LSTM Trainer] Features normalized using StandardScaler")
        
        print(f"  Train samples: {len(train_sequences)}")
        print(f"  Validation samples: {len(val_sequences)}")
        
        # Создаем datasets
        train_dataset = CryptoSequenceDataset(train_sequences, train_targets)
        val_dataset = CryptoSequenceDataset(val_sequences, val_targets)
        
        # Создаем data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        # Создаем модель
        input_size = sequences.shape[2]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        
        # Вычисляем веса классов для балансировки (критично!)
        unique_classes, class_counts = np.unique(train_targets + 1, return_counts=True)  # +1 для 0,1,2
        total_samples = len(train_targets)
        
        # Вычисляем веса классов (обратно пропорционально частоте)
        class_weights = {}
        for cls, count in zip(unique_classes, class_counts):
            if count > 0:
                # Увеличиваем вес для редких классов (LONG/SHORT), уменьшаем для HOLD
                base_weight = total_samples / (len(unique_classes) * count)
                if cls == 1:  # HOLD (класс 1 после +1)
                    weight = base_weight * 0.5  # Уменьшаем вес HOLD
                else:  # LONG (2) или SHORT (0)
                    weight = base_weight * 2.0  # Увеличиваем вес редких классов
                class_weights[int(cls)] = weight
        
        print(f"\n[LSTM Trainer] Class weights: {class_weights}")
        
        # Создаем tensor весов для loss функции
        weight_tensor = torch.FloatTensor([
            class_weights.get(0, 1.0),  # SHORT
            class_weights.get(1, 1.0),  # HOLD
            class_weights.get(2, 1.0),  # LONG
        ]).to(self.device)
        
        # Loss с весами классов
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7  # Увеличили patience
        )
        
        # Обучение
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 15  # Увеличили patience для большего количества эпох
        
        for epoch in range(self.num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch_sequences, batch_targets in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_sequences, batch_targets in val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_sequences)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_targets.size(0)
                    correct += (predicted == batch_targets).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Обновляем learning rate
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"    Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Метрики
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'final_val_acc': val_accuracies[-1] if val_accuracies else 0.0,
            'num_epochs_trained': len(train_losses),
            'input_size': input_size,
            'sequence_length': self.sequence_length,
            'class_weights': {k: float(v) for k, v in class_weights.items()},
        }
        
        print(f"\n[LSTM Trainer] Training completed!")
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
        print(f"  Final validation accuracy: {metrics['final_val_acc']:.4f}")
        
        return self.model, metrics
    
    def predict(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Делает предсказания на последовательностях.
        
        Args:
            sequences: Массив последовательностей (n_samples, seq_length, n_features)
        
        Returns:
            (predictions, probabilities) где:
            - predictions: (n_samples,) со значениями -1, 0, 1
            - probabilities: (n_samples, 3) вероятности для каждого класса
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        dataset = CryptoSequenceDataset(sequences, np.zeros(len(sequences)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_sequences, _ in loader:
                batch_sequences = batch_sequences.to(self.device)
                outputs = self.model(batch_sequences)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        predictions = np.array(all_predictions) - 1  # 0,1,2 -> -1,0,1
        probabilities = np.array(all_probabilities)
        
        return predictions, probabilities
    
    def save_model(
        self,
        filepath: str,
        feature_names: list,
        metrics: Dict[str, Any],
        symbol: str = "BTCUSDT",
        interval: str = "15",
    ):
        """Сохраняет модель и метаданные."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.fc.in_features,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            },
            'trainer_config': {
                'sequence_length': self.sequence_length,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'class_weights': metrics.get('class_weights', None),
            },
            'scaler': self.scaler,  # Сохраняем scaler для нормализации при предсказании
            'feature_names': feature_names,
            'metrics': metrics,
            'metadata': {
                'symbol': symbol,
                'interval': interval,
                'trained_at': datetime.now().isoformat(),
                'model_type': 'lstm',
                'device': str(self.device),
            },
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[LSTM Trainer] Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Tuple[LSTMModel, Dict[str, Any], Optional[StandardScaler]]:
        """Загружает модель и метаданные."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        config = model_data['model_config']
        model = LSTMModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
        )
        
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        # Загружаем scaler если есть
        scaler = model_data.get('scaler', None)
        
        return model, model_data, scaler
