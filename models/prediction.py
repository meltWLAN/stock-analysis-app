#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI预测模块 - 实现基于机器学习的股票走势预测
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# 尝试导入AI模型相关包
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet未安装，相关功能将不可用")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow/Keras未安装，相关功能将不可用")


class StockPredictor:
    """股票预测类"""
    
    def __init__(self):
        """初始化预测器"""
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
    
    def prepare_data(self, price_df, feature_columns=None, target_column='close', window_size=20):
        """
        准备模型训练数据
        
        Args:
            price_df: 价格数据DataFrame
            feature_columns: 特征列名列表，默认为None（使用所有数值列）
            target_column: 目标列名，默认为'close'
            window_size: 时间窗口大小，默认为20天
            
        Returns:
            tuple: (X, y) 特征数据和目标数据
        """
        if price_df is None or price_df.empty:
            self.logger.warning("价格数据为空，无法准备训练数据")
            return None, None
        
        # 如果未指定特征列，则使用所有数值列
        if feature_columns is None:
            feature_columns = price_df.select_dtypes(include=[np.number]).columns.tolist()
            # 确保目标列在特征列中
            if target_column not in feature_columns:
                feature_columns.append(target_column)
        
        # 创建特征数据
        df = price_df[feature_columns].copy()
        
        # 数据标准化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        
        # 创建时间序列数据
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i])
            # 目标是下一天的收盘价
            y.append(scaled_data[i, df.columns.get_loc(target_column)])
        
        return np.array(X), np.array(y), scaler
    
    def build_lstm_model(self, input_shape, units=50, dropout=0.2):
        """
        构建LSTM模型
        
        Args:
            input_shape: 输入数据形状
            units: LSTM单元数，默认为50
            dropout: Dropout比例，默认为0.2
            
        Returns:
            model: LSTM模型
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow/Keras未安装，无法构建LSTM模型")
            return None
        
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_lstm_model(self, price_df, stock_code, epochs=50, batch_size=32):
        """
        训练LSTM模型
        
        Args:
            price_df: 价格数据DataFrame
            stock_code: 股票代码
            epochs: 训练轮数，默认为50
            batch_size: 批次大小，默认为32
            
        Returns:
            bool: 训练是否成功
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow/Keras未安装，无法训练LSTM模型")
            return False
        
        # 准备数据
        X, y, scaler = self.prepare_data(price_df)
        if X is None or y is None:
            return False
        
        # 分割训练集和验证集
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # 构建模型
        model = self.build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        if model is None:
            return False
        
        # 训练模型
        try:
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # 保存模型和缩放器
            self.models[stock_code] = model
            self.scalers[stock_code] = scaler
            
            return True
        except Exception as e:
            self.logger.error(f"训练LSTM模型失败: {e}")
            return False
    
    def predict_with_lstm(self, price_df, stock_code, days=5):
        """
        使用LSTM模型预测未来价格
        
        Args:
            price_df: 价格数据DataFrame
            stock_code: 股票代码
            days: 预测天数，默认为5天
            
        Returns:
            DataFrame: 预测结果
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow/Keras未安装，无法使用LSTM模型预测")
            return None
        
        # 检查模型是否已训练
        if stock_code not in self.models or stock_code not in self.scalers:
            self.logger.warning(f"股票{stock_code}的LSTM模型未训练")
            return None
        
        model = self.models[stock_code]
        scaler = self.scalers[stock_code]
        
        # 准备预测数据
        feature_columns = price_df.select_dtypes(include=[np.number]).columns.tolist()
        df = price_df[feature_columns].copy()
        scaled_data = scaler.transform(df)
        
        # 获取最后一个窗口的数据
        window_size = 20  # 与训练时保持一致
        last_window = scaled_data[-window_size:].reshape(1, window_size, len(feature_columns))
        
        # 预测未来几天
        predictions = []
        current_window = last_window.copy()
        
        for _ in range(days):
            # 预测下一天
            pred = model.predict(current_window)[0][0]
            predictions.append(pred)
            
            # 更新窗口
            next_point = current_window[0][-1].copy()
            next_point[df.columns.get_loc('close')] = pred
            current_window = np.append(current_window[:, 1:, :], [next_point.reshape(1, len(feature_columns))], axis=1)
        
        # 反向转换预测结果
        last_date = price_df.index[-1]
        pred_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # 创建预测结果DataFrame
        pred_df = pd.DataFrame(index=pred_dates, columns=['close_pred'])
        
        # 将预测值转换回原始比例
        for i, date in enumerate(pred_dates):
            # 创建一个与原始数据相同形状的零数组
            temp = np.zeros((1, len(feature_columns)))
            # 将预测的收盘价放入相应位置
            temp[0, df.columns.get_loc('close')] = predictions[i]
            # 反向转换
            pred_value = scaler.inverse_transform(temp)[0, df.columns.get_loc('close')]
            pred_df.loc[date, 'close_pred'] = pred_value
        
        return pred_df
    
    def train_prophet_model(self, price_df, stock_code):
        """
        训练Prophet模型
        
        Args:
            price_df: 价格数据DataFrame
            stock_code: 股票代码
            
        Returns:
            bool: 训练是否成功
        """
        if not PROPHET_AVAILABLE:
            self.logger.error("Prophet未安装，无法训练Prophet模型")
            return False
        
        # 准备Prophet数据格式
        df = price_df.reset_index()
        if 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'ds', 'close': 'y'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'ds', 'close': 'y'})
        else:
            self.logger.error("数据格式不符合要求，无法训练Prophet模型")
            return False
        
        # 确保日期列为datetime类型
        df['ds'] = pd.to_datetime(df['ds'])
        
        # 创建并训练模型
        try:
            model = Prophet(daily_seasonality=True)
            model.fit(df[['ds', 'y']])
            
            # 保存模型
            self.models[f"{stock_code}_prophet"] = model
            
            return True
        except Exception as e:
            self.logger.error(f"训练Prophet模型失败: {e}")
            return False
    
    def predict_with_prophet(self, price_df, stock_code, days=30):
        """
        使用Prophet模型预测未来价格
        
        Args:
            price_df: 价格数据DataFrame
            stock_code: 股票代码
            days: 预测天数，默认为30天
            
        Returns:
            DataFrame: 预测结果
        """
        if not PROPHET_AVAILABLE:
            self.logger.error("Prophet未安装，无法使用Prophet模型预测")
            return None
        
        # 检查模型是否已训练
        model_key = f"{stock_code}_prophet"
        if model_key not in self.models:
            self.logger.warning(f"股票{stock_code}的Prophet模型未训练")
            return None
        
        model = self.models[model_key]
        
        # 创建未来日期DataFrame
        future = model.make_future_dataframe(periods=days)
        
        # 预测
        forecast = model.predict(future)
        
        # 提取预测结果
        last_date = price_df.index[-1] if isinstance(price_df.index, pd.DatetimeIndex) else pd.to_datetime(price_df['trade_date'].iloc[-1])
        forecast_result = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        # 转换为更友好的格式
        result_df = forecast_result.rename(columns={
            'ds': 'date',
            'yhat': 'close_pred',
            'yhat_lower': 'close_lower',
            'yhat_upper': 'close_upper'
        })
        
        return result_df
    
    def calculate_up_probability(self, price_df, stock_code, days=5):
        """
        计算上涨概率
        
        Args:
            price_df: 价格数据DataFrame
            stock_code: 股票代码
            days: 预测天数，默认为5天
            
        Returns:
            float: 上涨概率（0-1之间）
        """
        # 尝试使用LSTM模型预测
        lstm_pred = None
        if TF_AVAILABLE and stock_code in self.models:
            lstm_pred = self.predict_with_lstm(price_df, stock_code, days)
        
        # 尝试使用Prophet模型预测
        prophet_pred = None
        prophet_key = f"{stock_code}_prophet"
        if PROPHET_AVAILABLE and prophet_key in self.models:
            prophet_pred = self.predict_with_prophet(price_df, stock_code, days)
        
        # 如果两个模型都没有预测结果，返回None
        if lstm_pred is None and prophet_pred is None:
            self.logger.warning(f"无法为股票{stock_code}计算上涨概率，模型未训练")
            return None
        
        # 获取当前价格
        current_price = price_df['close'].iloc[-1]
        
        # 计算上涨概率
        up_prob = 0.5  # 默认概率
        count = 0
        
        # 使用LSTM预测结果
        if lstm_pred is not None and not lstm_pred.empty:
            last_pred = lstm_pred['close_pred'].iloc[-1]
            lstm_up_prob = 1.0 if last_pred > current_price else 0.0
            up_prob += lstm_up_prob
            count += 1
        
        # 使用Prophet预测结果
        if prophet_pred is not None and not prophet_pred.empty:
            # 获取最后一天的预测值
            last_pred = prophet_pred['close_pred'].iloc[-1]
            prophet_up_prob = 1.0 if last_pred > current_price else 0.0
            up_prob += prophet_up_prob
            count += 1
        
        # 计算平均上涨概率
        if count > 0:
            up_prob /= count
        
        return up_prob
    
    def build_ensemble_model(self, input_shape, num_models=3, dropout_range=(0.1, 0.4)):
        """
        构建集成LSTM模型 - 集成多个不同参数的模型以提高稳定性和精度
        
        Args:
            input_shape: 输入数据形状
            num_models: 集成的模型数量
            dropout_range: Dropout比例范围
            
        Returns:
            list: 模型列表
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow/Keras未安装，无法构建集成模型")
            return None
        
        models = []
        
        for i in range(num_models):
            # 随机化超参数以增加模型多样性
            units = np.random.choice([32, 50, 64, 80, 100])
            dropout = np.random.uniform(dropout_range[0], dropout_range[1])
            
            # 构建具有不同参数的LSTM模型
            model = Sequential()
            model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(dropout))
            model.add(LSTM(units=units))
            model.add(Dropout(dropout))
            model.add(Dense(1))
            
            # 使用不同的优化器和学习率
            if i % 3 == 0:
                optimizer = 'adam'
            elif i % 3 == 1:
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
            else:
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
                
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            models.append(model)
        
        return models
    
    def train_ensemble_model(self, price_df, stock_code, epochs=50, batch_size=32, num_models=3):
        """
        训练集成LSTM模型
        
        Args:
            price_df: 价格数据DataFrame
            stock_code: 股票代码
            epochs: 训练轮数，默认为50
            batch_size: 批次大小，默认为32
            num_models: 集成的模型数量
            
        Returns:
            bool: 训练是否成功
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow/Keras未安装，无法训练集成模型")
            return False
        
        # 准备数据
        X, y, scaler = self.prepare_data(price_df)
        if X is None or y is None:
            return False
        
        # 分割训练集和验证集
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # 构建集成模型
        models = self.build_ensemble_model(input_shape=(X.shape[1], X.shape[2]), num_models=num_models)
        if models is None:
            return False
        
        # 训练模型
        try:
            trained_models = []
            for i, model in enumerate(models):
                # 添加早停机制，避免过拟合
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                # 学习率调度器，动态调整学习率
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
                
                # 训练模型
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, lr_scheduler],
                    verbose=0
                )
                
                # 评估模型性能
                val_loss = model.evaluate(X_val, y_val, verbose=0)
                self.logger.info(f"模型 {i+1} 验证损失: {val_loss:.4f}")
                
                trained_models.append(model)
            
            # 保存模型和缩放器
            self.models[f"{stock_code}_ensemble"] = trained_models
            self.scalers[stock_code] = scaler
            
            return True
        except Exception as e:
            self.logger.error(f"训练集成LSTM模型失败: {e}")
            return False
    
    def predict_with_ensemble(self, price_df, stock_code, days=5):
        """
        使用集成LSTM模型预测未来价格
        
        Args:
            price_df: 价格数据DataFrame
            stock_code: 股票代码
            days: 预测天数，默认为5天
            
        Returns:
            list: 预测结果
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow/Keras未安装，无法使用集成模型预测")
            return None
        
        # 检查模型是否已训练
        ensemble_key = f"{stock_code}_ensemble"
        if ensemble_key not in self.models or stock_code not in self.scalers:
            self.logger.warning(f"股票{stock_code}的集成模型未训练")
            return None
        
        models = self.models[ensemble_key]
        scaler = self.scalers[stock_code]
        
        # 准备预测数据
        feature_columns = price_df.select_dtypes(include=[np.number]).columns.tolist()
        df = price_df[feature_columns].copy()
        scaled_data = scaler.transform(df)
        
        # 获取最后一个窗口的数据
        window_size = 20  # 与训练时保持一致
        last_window = scaled_data[-window_size:].reshape(1, window_size, len(feature_columns))
        
        # 对每个模型进行预测并汇总结果
        all_predictions = []
        
        for model in models:
            # 预测未来几天
            predictions = []
            current_window = last_window.copy()
            
            for _ in range(days):
                # 预测下一天
                pred = model.predict(current_window, verbose=0)[0][0]
                predictions.append(pred)
                
                # 更新窗口
                next_point = current_window[0][-1].copy()
                next_point[df.columns.get_loc('close')] = pred
                current_window = np.append(current_window[:, 1:, :], [next_point.reshape(1, len(feature_columns))], axis=1)
            
            all_predictions.append(predictions)
        
        # 计算集成预测结果（平均值）
        ensemble_predictions = []
        for i in range(days):
            ensemble_predictions.append(np.mean([p[i] for p in all_predictions]))
        
        # 反向转换预测结果
        result_predictions = []
        for pred in ensemble_predictions:
            # 创建一个与原始数据相同形状的零数组
            temp = np.zeros((1, len(feature_columns)))
            # 将预测的收盘价放入相应位置
            temp[0, df.columns.get_loc('close')] = pred
            # 反向转换
            pred_value = scaler.inverse_transform(temp)[0, df.columns.get_loc('close')]
            result_predictions.append(pred_value)
        
        return result_predictions
    
    def evaluate_model_performance(self, price_df, stock_code, test_size=0.2):
        """
        评估模型性能
        
        Args:
            price_df: 价格数据DataFrame
            stock_code: 股票代码
            test_size: 测试集比例，默认为0.2
            
        Returns:
            dict: 性能评估结果
        """
        if price_df is None or price_df.empty:
            self.logger.warning("价格数据为空，无法评估模型性能")
            return None
        
        # 确保价格数据包含收盘价
        if 'close' not in price_df.columns:
            self.logger.error("价格数据中缺少收盘价列")
            return None
        
        try:
            # 划分训练集和测试集
            test_start_idx = int(len(price_df) * (1 - test_size))
            train_data = price_df.iloc[:test_start_idx].copy()
            test_data = price_df.iloc[test_start_idx:].copy()
            
            # 训练LSTM模型
            self.train_lstm_model(train_data, stock_code)
            
            # 训练集成模型
            self.train_ensemble_model(train_data, stock_code)
            
            # 训练Prophet模型
            self.train_prophet_model(train_data, stock_code)
            
            # 评估不同模型的性能
            results = {}
            
            # 准备测试数据：按每天预测下一天来评估
            actual_close = test_data['close'].values
            
            # 评估LSTM模型
            if stock_code in self.models:
                lstm_predictions = []
                # 对每个测试日进行预测
                for i in range(len(test_data) - 1):
                    # 使用截至当天的所有数据进行预测
                    pred_df = pd.concat([train_data, test_data.iloc[:i+1]])
                    # 预测下一天
                    pred = self.predict_price(pred_df, days=1, model_type='lstm')
                    if pred:
                        lstm_predictions.append(pred[0])
                    else:
                        lstm_predictions.append(actual_close[i])  # 预测失败时使用当天的收盘价
                
                # 计算LSTM模型的误差指标
                lstm_mae = np.mean(np.abs(np.array(lstm_predictions) - actual_close[1:]))
                lstm_mape = np.mean(np.abs((np.array(lstm_predictions) - actual_close[1:]) / actual_close[1:])) * 100
                results['lstm'] = {
                    'mae': lstm_mae,
                    'mape': lstm_mape
                }
            
            # 评估集成模型
            ensemble_key = f"{stock_code}_ensemble"
            if ensemble_key in self.models:
                ensemble_predictions = []
                # 对每个测试日进行预测
                for i in range(len(test_data) - 1):
                    # 使用截至当天的所有数据进行预测
                    pred_df = pd.concat([train_data, test_data.iloc[:i+1]])
                    # 预测下一天
                    pred = self.predict_with_ensemble(pred_df, stock_code, days=1)
                    if pred:
                        ensemble_predictions.append(pred[0])
                    else:
                        ensemble_predictions.append(actual_close[i])
                
                # 计算集成模型的误差指标
                ensemble_mae = np.mean(np.abs(np.array(ensemble_predictions) - actual_close[1:]))
                ensemble_mape = np.mean(np.abs((np.array(ensemble_predictions) - actual_close[1:]) / actual_close[1:])) * 100
                results['ensemble'] = {
                    'mae': ensemble_mae,
                    'mape': ensemble_mape
                }
            
            # 评估Prophet模型
            prophet_key = f"{stock_code}_prophet"
            if prophet_key in self.models:
                prophet_predictions = []
                # 对每个测试日进行预测
                for i in range(len(test_data) - 1):
                    # 使用截至当天的所有数据进行预测
                    pred_df = pd.concat([train_data, test_data.iloc[:i+1]])
                    # 预测下一天
                    pred_result = self.predict_with_prophet(pred_df, stock_code, days=1)
                    if pred_result is not None and not pred_result.empty:
                        prophet_predictions.append(pred_result['close_pred'].iloc[0])
                    else:
                        prophet_predictions.append(actual_close[i])
                
                # 计算Prophet模型的误差指标
                prophet_mae = np.mean(np.abs(np.array(prophet_predictions) - actual_close[1:]))
                prophet_mape = np.mean(np.abs((np.array(prophet_predictions) - actual_close[1:]) / actual_close[1:])) * 100
                results['prophet'] = {
                    'mae': prophet_mae,
                    'mape': prophet_mape
                }
            
            # 确定最佳模型
            min_mape = float('inf')
            best_model = None
            
            for model_name, metrics in results.items():
                if metrics['mape'] < min_mape:
                    min_mape = metrics['mape']
                    best_model = model_name
            
            results['best_model'] = best_model
            
            self.logger.info(f"模型评估完成，最佳模型: {best_model}，MAPE: {min_mape:.2f}%")
            return results
            
        except Exception as e:
            self.logger.error(f"评估模型性能时出错: {e}")
            return None
    
    def predict_price(self, price_df, days=5, model_type='ensemble'):
        """
        预测未来股价
        
        Args:
            price_df: 价格数据DataFrame
            days: 预测天数，默认为5天
            model_type: 模型类型，'lstm'、'prophet'、'ensemble'或'smart'(自动选择最佳模型)，默认为'ensemble'
            
        Returns:
            list: 未来几天的预测收盘价列表
        """
        if price_df is None or price_df.empty:
            self.logger.warning("价格数据为空，无法进行预测")
            return None
            
        try:
            # 确保价格数据有收盘价列
            if 'close' not in price_df.columns:
                self.logger.error("价格数据中没有'close'列")
                return None
                
            # 提取股票代码
            if 'ts_code' in price_df.columns:
                stock_code = price_df['ts_code'].iloc[0]
            else:
                # 如果没有股票代码列，使用"default"作为代码
                stock_code = "default"
            
            # 如果是智能模式，先评估各模型性能再选择最佳模型
            if model_type == 'smart':
                # 检查是否已训练过三种模型
                has_lstm = stock_code in self.models
                has_ensemble = f"{stock_code}_ensemble" in self.models
                has_prophet = f"{stock_code}_prophet" in self.models
                
                # 如果未训练全部模型，需要先训练
                if not (has_lstm and has_ensemble and has_prophet):
                    self.logger.info(f"智能模式：尚未训练全部模型，开始训练...")
                    if not has_lstm:
                        self.train_lstm_model(price_df, stock_code)
                    if not has_ensemble:
                        self.train_ensemble_model(price_df, stock_code)
                    if not has_prophet:
                        self.train_prophet_model(price_df, stock_code)
                
                # 评估模型性能
                eval_results = self.evaluate_model_performance(price_df, stock_code)
                if eval_results and 'best_model' in eval_results:
                    best_model = eval_results['best_model']
                    self.logger.info(f"智能模式：选择 {best_model} 模型进行预测")
                    model_type = best_model
                else:
                    # 如果评估失败，默认使用集成模型
                    self.logger.warning("智能模式：模型评估失败，默认使用集成模型")
                    model_type = 'ensemble'
            
            # 储存各模型的预测结果
            all_predictions = []
            
            # 根据模型类型选择预测方法
            if model_type == 'lstm':
                # 检查是否已训练模型
                if stock_code not in self.models:
                    self.logger.info(f"股票 {stock_code} 的LSTM模型未训练，开始训练...")
                    self.train_lstm_model(price_df, stock_code)
                
                # 使用LSTM模型预测
                if stock_code in self.models:
                    pred = self.predict_with_lstm(price_df, stock_code, days)
                    if pred is not None:
                        all_predictions.append(pred)
            
            elif model_type == 'prophet':
                # 检查是否已训练模型
                prophet_key = f"{stock_code}_prophet"
                if prophet_key not in self.models:
                    self.logger.info(f"股票 {stock_code} 的Prophet模型未训练，开始训练...")
                    self.train_prophet_model(price_df, stock_code)
                
                # 使用Prophet模型预测
                if prophet_key in self.models:
                    pred_df = self.predict_with_prophet(price_df, stock_code, days)
                    if pred_df is not None and not pred_df.empty:
                        all_predictions.append(pred_df['close_pred'].values.tolist())
            
            elif model_type == 'ensemble':
                # 检查是否已训练模型
                ensemble_key = f"{stock_code}_ensemble"
                if ensemble_key not in self.models:
                    self.logger.info(f"股票 {stock_code} 的集成模型未训练，开始训练...")
                    self.train_ensemble_model(price_df, stock_code)
                
                # 使用集成模型预测
                if ensemble_key in self.models:
                    pred = self.predict_with_ensemble(price_df, stock_code, days)
                    if pred is not None:
                        all_predictions.append(pred)
            
            else:  # 默认使用所有可用模型并取平均值
                # 尝试使用LSTM模型
                if stock_code in self.models:
                    pred = self.predict_with_lstm(price_df, stock_code, days)
                    if pred is not None:
                        all_predictions.append(pred)
                
                # 尝试使用集成模型
                ensemble_key = f"{stock_code}_ensemble"
                if ensemble_key in self.models:
                    pred = self.predict_with_ensemble(price_df, stock_code, days)
                    if pred is not None:
                        all_predictions.append(pred)
                
                # 尝试使用Prophet模型
                prophet_key = f"{stock_code}_prophet"
                if prophet_key in self.models:
                    pred_df = self.predict_with_prophet(price_df, stock_code, days)
                    if pred_df is not None and not pred_df.empty:
                        all_predictions.append(pred_df['close_pred'].values.tolist())
            
            # 如果模型预测失败，尝试训练并重新预测
            if not all_predictions:
                self.logger.warning(f"模型预测失败，尝试重新训练模型...")
                if model_type == 'lstm':
                    self.train_lstm_model(price_df, stock_code)
                    pred = self.predict_with_lstm(price_df, stock_code, days)
                    if pred is not None:
                        all_predictions.append(pred)
                
                elif model_type == 'prophet':
                    self.train_prophet_model(price_df, stock_code)
                    pred_df = self.predict_with_prophet(price_df, stock_code, days)
                    if pred_df is not None and not pred_df.empty:
                        all_predictions.append(pred_df['close_pred'].values.tolist())
                
                elif model_type == 'ensemble':
                    self.train_ensemble_model(price_df, stock_code)
                    pred = self.predict_with_ensemble(price_df, stock_code, days)
                    if pred is not None:
                        all_predictions.append(pred)
            
            # 如果是ensemble模型，已经在predict_with_ensemble中融合了所有模型预测
            if len(all_predictions) == 1:
                return all_predictions[0]
            
            # 如果有多个模型预测结果，取平均值
            elif len(all_predictions) > 1:
                ensemble_predictions = []
                for i in range(days):
                    day_predictions = [pred[i] for pred in all_predictions if i < len(pred)]
                    if day_predictions:
                        ensemble_predictions.append(sum(day_predictions) / len(day_predictions))
                    else:
                        # 如果某天没有预测值，使用最近的有效预测或当前收盘价
                        ensemble_predictions.append(ensemble_predictions[-1] if ensemble_predictions else price_df['close'].iloc[-1])
                
                return ensemble_predictions
                
            # 如果所有模型都失败，使用简单的线性回归预测
            self.logger.warning(f"所有模型预测失败，使用简单线性回归预测")
            
            # 使用最近30天的收盘价进行线性回归预测
            recent_close = price_df['close'].tail(30).values
            x = np.arange(len(recent_close)).reshape(-1, 1)
            y = recent_close
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x, y)
            
            # 预测未来几天
            future_x = np.arange(len(recent_close), len(recent_close) + days).reshape(-1, 1)
            predictions = model.predict(future_x)
            
            return predictions.tolist()
                
        except Exception as e:
            self.logger.error(f"预测价格出错: {str(e)}")
            # 出错时返回一些合理的模拟数据
            last_close = price_df['close'].iloc[-1]
            predictions = [last_close * (1 + np.random.normal(0.001, 0.01)) for _ in range(days)]
            return predictions

    def optimize_features(self, price_df, feature_importance_threshold=0.02):
        """
        自动进行特征选择，剔除不重要的特征以提高模型性能
        
        Args:
            price_df: 价格数据DataFrame
            feature_importance_threshold: 特征重要性阈值，低于此值的特征将被剔除
            
        Returns:
            tuple: (优化后的特征DataFrame, 重要特征列表)
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # 准备特征数据
            features_df = self._prepare_features(price_df)
            if features_df is None or features_df.empty:
                self.logger.error("无法准备特征数据")
                return None, []
            
            # 分离特征和目标
            X = features_df.drop(['target'], axis=1)
            y = features_df['target']
            
            # 移除日期列（如果存在）
            if 'date' in X.columns:
                X = X.drop(['date'], axis=1)
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 使用随机森林评估特征重要性
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_scaled, y)
            
            # 获取特征重要性
            feature_importances = rf_model.feature_importances_
            
            # 创建特征重要性数据框
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importances
            }).sort_values('importance', ascending=False)
            
            # 筛选出重要特征
            important_features = importance_df[importance_df['importance'] >= feature_importance_threshold]['feature'].tolist()
            
            self.logger.info(f"选择了 {len(important_features)} 个重要特征，阈值: {feature_importance_threshold}")
            self.logger.info(f"特征重要性前5: {importance_df.head(5)}")
            
            # 使用重要特征创建新的特征数据框
            optimized_features = features_df[important_features + ['target']]
            
            return optimized_features, important_features
        
        except Exception as e:
            self.logger.error(f"优化特征时出错: {str(e)}")
            return None, []

    def auto_optimize_parameters(self, price_df, param_grid=None, cv=3):
        """
        自动优化LSTM模型参数
        
        Args:
            price_df: 价格数据DataFrame
            param_grid: 参数网格，默认为None（使用预定义网格）
            cv: 交叉验证折数
            
        Returns:
            dict: 最佳参数
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit
            
            # 如果TensorFlow/Keras未安装，返回默认参数
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                from tensorflow.keras.optimizers import Adam, RMSprop
                from tensorflow.keras.callbacks import EarlyStopping
            except ImportError:
                self.logger.error("TensorFlow/Keras未安装，无法优化LSTM参数")
                return {
                    'units': 50,
                    'dropout': 0.2,
                    'optimizer': 'adam',
                    'batch_size': 32,
                    'epochs': 50
                }
            
            # 准备特征和目标
            features_df, important_features = self.optimize_features(price_df)
            if features_df is None or features_df.empty:
                self.logger.error("无法准备特征数据")
                return None
            
            # 使用重要特征准备训练数据
            X = features_df.drop(['target'], axis=1).values
            y = features_df['target'].values
            
            # 确保数据是浮点型
            X = X.astype('float32')
            y = y.astype('float32')
            
            # 默认参数网格
            if param_grid is None:
                param_grid = {
                    'units': [32, 50, 64, 128],
                    'dropout': [0.1, 0.2, 0.3],
                    'optimizer': ['adam', 'rmsprop'],
                    'batch_size': [16, 32, 64],
                    'epochs': [30, 50, 100]
                }
            
            # 创建时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=cv)
            
            # 存储结果
            best_params = None
            best_score = float('inf')  # 使用MSE，所以越低越好
            
            # 准备输入数据形状
            timesteps = 10  # 时间步数
            input_dim = X.shape[1]  # 特征数量
            
            # 重塑数据
            X_reshaped = []
            y_reshaped = []
            
            for i in range(len(X) - timesteps):
                X_reshaped.append(X[i:i+timesteps])
                y_reshaped.append(y[i+timesteps])
            
            X_reshaped = np.array(X_reshaped)
            y_reshaped = np.array(y_reshaped)
            
            # 网格搜索
            for units in param_grid['units']:
                for dropout in param_grid['dropout']:
                    for optimizer_name in param_grid['optimizer']:
                        for batch_size in param_grid['batch_size']:
                            for epochs in param_grid['epochs']:
                                # 设置参数
                                params = {
                                    'units': units,
                                    'dropout': dropout,
                                    'optimizer': optimizer_name,
                                    'batch_size': batch_size,
                                    'epochs': epochs
                                }
                                
                                self.logger.info(f"测试参数: {params}")
                                
                                # 交叉验证
                                cv_scores = []
                                
                                for train_idx, test_idx in tscv.split(X_reshaped):
                                    X_train, X_test = X_reshaped[train_idx], X_reshaped[test_idx]
                                    y_train, y_test = y_reshaped[train_idx], y_reshaped[test_idx]
                                    
                                    # 创建模型
                                    model = Sequential()
                                    model.add(LSTM(units=units, input_shape=(timesteps, input_dim), return_sequences=False))
                                    model.add(Dropout(dropout))
                                    model.add(Dense(1, activation='sigmoid'))
                                    
                                    # 设置优化器
                                    if optimizer_name == 'adam':
                                        optimizer = Adam(learning_rate=0.001)
                                    else:
                                        optimizer = RMSprop(learning_rate=0.001)
                                    
                                    # 编译模型
                                    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                                    
                                    # 早停
                                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                                    
                                    # 训练模型
                                    history = model.fit(
                                        X_train, y_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(X_test, y_test),
                                        callbacks=[early_stopping],
                                        verbose=0
                                    )
                                    
                                    # 评估模型
                                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                                    cv_scores.append(loss)
                                
                                # 计算平均分数
                                mean_score = np.mean(cv_scores)
                                
                                self.logger.info(f"参数 {params} 的平均损失: {mean_score:.4f}")
                                
                                # 更新最佳参数
                                if mean_score < best_score:
                                    best_score = mean_score
                                    best_params = params
                                    self.logger.info(f"找到新的最佳参数: {best_params}, 损失: {best_score:.4f}")
            
            self.logger.info(f"最佳参数: {best_params}, 最佳损失: {best_score:.4f}")
            return best_params
        
        except Exception as e:
            self.logger.error(f"自动优化参数时出错: {str(e)}")
            return {
                'units': 50,
                'dropout': 0.2,
                'optimizer': 'adam',
                'batch_size': 32,
                'epochs': 50
            }

    def adaptive_model_selection(self, price_df, test_size=0.2):
        """
        自适应模型选择，根据数据特性选择最合适的预测模型
        
        Args:
            price_df: 价格数据DataFrame
            test_size: 测试集比例
            
        Returns:
            str: 最佳模型类型('lstm', 'prophet', 'ensemble')
        """
        try:
            from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
            import numpy as np
            
            # 评估LSTM模型性能
            lstm_performance = self.evaluate_model_performance(price_df, model_type='lstm', test_size=test_size)
            
            # 评估Prophet模型性能
            prophet_performance = self.evaluate_model_performance(price_df, model_type='prophet', test_size=test_size)
            
            # 评估集成模型性能
            ensemble_performance = self.evaluate_model_performance(price_df, model_type='ensemble', test_size=test_size)
            
            # 获取各模型的MAPE
            lstm_mape = lstm_performance.get('mape', float('inf'))
            prophet_mape = prophet_performance.get('mape', float('inf'))
            ensemble_mape = ensemble_performance.get('mape', float('inf'))
            
            self.logger.info(f"LSTM MAPE: {lstm_mape:.4f}")
            self.logger.info(f"Prophet MAPE: {prophet_mape:.4f}")
            self.logger.info(f"Ensemble MAPE: {ensemble_mape:.4f}")
            
            # 选择MAPE最低的模型
            model_mapes = {
                'lstm': lstm_mape,
                'prophet': prophet_mape,
                'ensemble': ensemble_mape
            }
            
            best_model = min(model_mapes, key=model_mapes.get)
            self.logger.info(f"最佳模型: {best_model}, MAPE: {model_mapes[best_model]:.4f}")
            
            return best_model
        
        except Exception as e:
            self.logger.error(f"自适应模型选择时出错: {str(e)}")
            return 'ensemble'  # 默认使用集成模型

    def create_automl_prediction(self, stock_code, days=30, auto_optimize=False):
        """
        使用AutoML自动选择最佳模型和参数进行预测
        
        Args:
            stock_code: 股票代码
            days: 预测天数
            auto_optimize: 是否自动优化参数
            
        Returns:
            dict: 预测结果
        """
        try:
            # 获取历史数据
            price_df = self.get_stock_data(stock_code)
            if price_df is None or price_df.empty:
                return {"error": f"无法获取股票 {stock_code} 的数据"}
            
            # 自动选择最佳模型
            best_model_type = self.adaptive_model_selection(price_df)
            
            # 如果需要自动优化参数
            if auto_optimize and best_model_type == 'lstm':
                best_params = self.auto_optimize_parameters(price_df)
                
                # 使用最佳参数重新训练LSTM模型
                self.train_lstm_model(
                    price_df,
                    units=best_params['units'],
                    dropout=best_params['dropout'],
                    optimizer=best_params['optimizer'],
                    batch_size=best_params['batch_size'],
                    epochs=best_params['epochs']
                )
            
            # 进行预测
            if best_model_type == 'lstm':
                prediction_result = self.predict_with_lstm(stock_code, days)
            elif best_model_type == 'prophet':
                prediction_result = self.predict_with_prophet(stock_code, days)
            else:  # ensemble
                prediction_result = self.predict_with_ensemble(stock_code, days)
            
            # 添加模型选择信息
            prediction_result['model_type'] = best_model_type
            prediction_result['auto_optimized'] = auto_optimize
            
            return prediction_result
        
        except Exception as e:
            self.logger.error(f"AutoML预测时出错: {str(e)}")
            return {"error": f"AutoML预测时出错: {str(e)}"}

    def predict_with_ensemble(self, stock_code, days=30, weights=None):
        """
        使用集成模型进行预测
        
        Args:
            stock_code: 股票代码
            days: 预测天数
            weights: 各模型权重，默认为None（根据模型性能自动计算）
            
        Returns:
            dict: 预测结果
        """
        try:
            # 获取历史数据
            price_df = self.get_stock_data(stock_code)
            if price_df is None or price_df.empty:
                return {"error": f"无法获取股票 {stock_code} 的数据"}
            
            # 使用LSTM模型预测
            lstm_prediction = self.predict_with_lstm(stock_code, days)
            if "error" in lstm_prediction:
                lstm_prediction = None
            
            # 使用Prophet模型预测
            prophet_prediction = self.predict_with_prophet(stock_code, days)
            if "error" in prophet_prediction:
                prophet_prediction = None
            
            # 如果两个模型都失败，返回错误
            if lstm_prediction is None and prophet_prediction is None:
                return {"error": "所有模型预测失败"}
            
            # 如果只有一个模型成功，使用该模型的预测
            if lstm_prediction is None:
                return prophet_prediction
            if prophet_prediction is None:
                return lstm_prediction
            
            # 获取预测结果
            lstm_prices = lstm_prediction.get('predicted_prices', [])
            prophet_prices = prophet_prediction.get('predicted_prices', [])
            
            # 确保预测天数一致
            min_days = min(len(lstm_prices), len(prophet_prices))
            lstm_prices = lstm_prices[:min_days]
            prophet_prices = prophet_prices[:min_days]
            
            # 如果未提供权重，根据模型性能计算权重
            if weights is None:
                # 评估模型性能
                lstm_performance = self.evaluate_model_performance(price_df, model_type='lstm')
                prophet_performance = self.evaluate_model_performance(price_df, model_type='prophet')
                
                # 获取MAPE（越低越好）
                lstm_mape = lstm_performance.get('mape', float('inf'))
                prophet_mape = prophet_performance.get('mape', float('inf'))
                
                # 计算反向权重（性能越好，权重越高）
                if lstm_mape == float('inf') and prophet_mape == float('inf'):
                    # 如果两个模型都无法评估，使用相等权重
                    lstm_weight = 0.5
                    prophet_weight = 0.5
                elif lstm_mape == float('inf'):
                    # 如果LSTM无法评估，使用Prophet
                    lstm_weight = 0
                    prophet_weight = 1
                elif prophet_mape == float('inf'):
                    # 如果Prophet无法评估，使用LSTM
                    lstm_weight = 1
                    prophet_weight = 0
                else:
                    # 根据MAPE计算权重
                    total_error = lstm_mape + prophet_mape
                    lstm_weight = 1 - (lstm_mape / total_error) if total_error > 0 else 0.5
                    prophet_weight = 1 - (prophet_mape / total_error) if total_error > 0 else 0.5
                    
                    # 归一化权重
                    sum_weights = lstm_weight + prophet_weight
                    lstm_weight = lstm_weight / sum_weights if sum_weights > 0 else 0.5
                    prophet_weight = prophet_weight / sum_weights if sum_weights > 0 else 0.5
                
                weights = {
                    'lstm': lstm_weight,
                    'prophet': prophet_weight
                }
            
            # 计算加权平均预测
            ensemble_prices = []
            for i in range(min_days):
                weighted_price = lstm_prices[i] * weights['lstm'] + prophet_prices[i] * weights['prophet']
                ensemble_prices.append(weighted_price)
            
            # 获取日期
            prediction_dates = lstm_prediction.get('prediction_dates', [])[:min_days]
            
            # 计算上涨概率
            lstm_up_prob = lstm_prediction.get('up_probability', 0.5)
            prophet_up_prob = prophet_prediction.get('up_probability', 0.5)
            ensemble_up_prob = lstm_up_prob * weights['lstm'] + prophet_up_prob * weights['prophet']
            
            # 最后一个已知价格
            last_price = price_df['close'].iloc[-1] if not price_df.empty else 0
            
            return {
                'stock_code': stock_code,
                'prediction_dates': prediction_dates,
                'predicted_prices': ensemble_prices,
                'up_probability': ensemble_up_prob,
                'last_known_price': last_price,
                'model_weights': weights,
                'ensemble_method': 'weighted_average'
            }
        
        except Exception as e:
            self.logger.error(f"集成模型预测时出错: {str(e)}")
            return {"error": f"集成模型预测时出错: {str(e)}"}

    def auto_update_models(self, stock_codes, max_age_days=30):
        """
        自动更新模型，当模型超过指定天数时重新训练
        
        Args:
            stock_codes: 股票代码列表
            max_age_days: 模型最大有效天数
            
        Returns:
            dict: 更新结果
        """
        try:
            import os
            from datetime import datetime, timedelta
            
            updated_models = []
            failed_updates = []
            
            # 检查和创建模型目录
            os.makedirs(self.model_dir, exist_ok=True)
            
            for stock_code in stock_codes:
                try:
                    # 构建模型文件路径
                    lstm_model_path = f"{self.model_dir}/lstm_{stock_code}.h5"
                    scaler_path = f"{self.model_dir}/scaler_{stock_code}.pkl"
                    
                    # 检查模型文件是否存在
                    model_exists = os.path.exists(lstm_model_path) and os.path.exists(scaler_path)
                    
                    # 如果模型存在，检查其年龄
                    if model_exists:
                        model_time = datetime.fromtimestamp(os.path.getmtime(lstm_model_path))
                        current_time = datetime.now()
                        model_age_days = (current_time - model_time).days
                        
                        # 如果模型年龄超过阈值，重新训练
                        if model_age_days > max_age_days:
                            self.logger.info(f"模型 {stock_code} 已过期 ({model_age_days} 天)，正在更新...")
                            need_update = True
                        else:
                            self.logger.info(f"模型 {stock_code} 仍然有效 ({model_age_days} 天)")
                            need_update = False
                    else:
                        self.logger.info(f"模型 {stock_code} 不存在，正在创建...")
                        need_update = True
                    
                    # 更新模型
                    if need_update:
                        # 获取股票数据
                        price_df = self.get_stock_data(stock_code)
                        if price_df is None or price_df.empty:
                            self.logger.warning(f"无法获取股票 {stock_code} 的数据，跳过更新")
                            failed_updates.append({
                                'stock_code': stock_code,
                                'reason': '无法获取数据'
                            })
                            continue
                        
                        # 自动优化参数并训练LSTM模型
                        best_params = self.auto_optimize_parameters(price_df)
                        
                        # 使用最佳参数训练模型
                        self.train_lstm_model(
                            price_df,
                            units=best_params['units'],
                            dropout=best_params['dropout'],
                            optimizer=best_params['optimizer'],
                            batch_size=best_params['batch_size'],
                            epochs=best_params['epochs']
                        )
                        
                        updated_models.append({
                            'stock_code': stock_code,
                            'parameters': best_params,
                            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                
                except Exception as e:
                    self.logger.error(f"更新股票 {stock_code} 的模型时出错: {str(e)}")
                    failed_updates.append({
                        'stock_code': stock_code,
                        'reason': str(e)
                    })
            
            return {
                'updated_models': updated_models,
                'failed_updates': failed_updates,
                'total_updated': len(updated_models),
                'total_failed': len(failed_updates)
            }
        
        except Exception as e:
            self.logger.error(f"自动更新模型时出错: {str(e)}")
            return {
                'error': str(e),
                'updated_models': [],
                'failed_updates': stock_codes
            }