#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据验证工具 - 提供数据质量检查和验证功能
"""

import logging
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class DataValidator:
    """数据验证和质量检查工具类"""
    
    def __init__(self):
        """初始化数据验证器"""
        self.logger = logging.getLogger(__name__)
        # 异常检测模型
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        # 缺失值填充模型
        self.imputer = KNNImputer(n_neighbors=5, weights="distance")
        
    def validate_stock_data(self, df, required_columns=None):
        """
        验证股票数据的有效性
        
        Args:
            df: 股票数据DataFrame
            required_columns: 必需的列列表，默认为None
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if df is None or df.empty:
            return False, "数据为空"
            
        # 检查必需的列
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"缺少必需的列: {', '.join(missing_columns)}"
                
        # 检查日期列是否有效
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            for col in date_cols:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        # 尝试转换为日期格式
                        pd.to_datetime(df[col])
                    except:
                        return False, f"日期列 '{col}' 格式无效"
        
        return True, ""
        
    def detect_anomalies(self, df, columns=None):
        """
        检测数据中的异常值
        
        Args:
            df: 数据DataFrame
            columns: 要检查的列，默认为数值列
            
        Returns:
            DataFrame: 标记了异常的DataFrame，增加了'is_anomaly'列
        """
        if df.empty:
            return df
            
        # 如果未指定列，则使用所有数值列
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
            
        if not columns:
            self.logger.warning("没有数值列用于异常检测")
            df['is_anomaly'] = False
            return df
            
        # 复制数据避免修改原始数据
        result_df = df.copy()
        
        try:
            # 获取用于异常检测的数据
            detection_data = result_df[columns].copy()
            
            # 处理异常检测数据中的缺失值
            detection_data = detection_data.fillna(detection_data.mean())
            
            # 训练模型并预测
            self.anomaly_detector.fit(detection_data)
            predictions = self.anomaly_detector.predict(detection_data)
            
            # 标记异常值 (-1表示异常)
            result_df['is_anomaly'] = predictions == -1
            
            # 记录异常值比例
            anomaly_ratio = result_df['is_anomaly'].mean()
            self.logger.info(f"检测到异常值比例: {anomaly_ratio:.2%}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            result_df['is_anomaly'] = False
            return result_df
            
    def smart_fill_missing_values(self, df, columns=None, method='knn'):
        """
        智能填充缺失值
        
        Args:
            df: 数据DataFrame
            columns: 要填充的列，默认为数值列
            method: 填充方法，可选 'knn', 'mean', 'median', 'mode', 'ffill', 'bfill'
            
        Returns:
            DataFrame: 填充后的数据
        """
        if df.empty:
            return df
            
        # 如果未指定列，则使用所有有缺失值的数值列
        if columns is None:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            columns = [col for col in numeric_cols if df[col].isna().any()]
            
        if not columns:
            return df
            
        # 复制数据避免修改原始数据
        result_df = df.copy()
        
        try:
            if method == 'knn':
                # KNN填充法处理相关列
                data_for_impute = result_df[columns].values
                imputed_data = self.imputer.fit_transform(data_for_impute)
                
                # 更新填充后的数据
                for i, col in enumerate(columns):
                    result_df[col] = imputed_data[:, i]
                    
            elif method == 'mean':
                # 均值填充
                for col in columns:
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                    
            elif method == 'median':
                # 中位数填充
                for col in columns:
                    result_df[col] = result_df[col].fillna(result_df[col].median())
                    
            elif method == 'mode':
                # 众数填充
                for col in columns:
                    result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
                    
            elif method == 'ffill':
                # 前向填充
                result_df[columns] = result_df[columns].ffill()
                
            elif method == 'bfill':
                # 后向填充
                result_df[columns] = result_df[columns].bfill()
                
            else:
                self.logger.warning(f"未知的填充方法: {method}, 使用KNN填充")
                data_for_impute = result_df[columns].values
                imputed_data = self.imputer.fit_transform(data_for_impute)
                
                for i, col in enumerate(columns):
                    result_df[col] = imputed_data[:, i]
                    
            return result_df
            
        except Exception as e:
            self.logger.error(f"缺失值填充失败: {e}")
            return df
            
    def cross_validate_data_sources(self, df1, df2, key_column, compare_columns):
        """
        交叉验证两个数据源的数据一致性
        
        Args:
            df1: 第一个数据源DataFrame
            df2: 第二个数据源DataFrame
            key_column: 用于连接的键列名
            compare_columns: 要比较的列列表
            
        Returns:
            DataFrame: 包含不一致数据的DataFrame
        """
        if df1.empty or df2.empty:
            return pd.DataFrame()
            
        # 确保键列在两个DataFrame中都存在
        if key_column not in df1.columns or key_column not in df2.columns:
            self.logger.error(f"键列 '{key_column}' 不存在于一个或两个DataFrame中")
            return pd.DataFrame()
            
        # 确保比较列在两个DataFrame中都存在
        df1_cols = set(df1.columns)
        df2_cols = set(df2.columns)
        valid_compare_cols = [col for col in compare_columns if col in df1_cols and col in df2_cols]
        
        if not valid_compare_cols:
            self.logger.error("没有共同的列可以比较")
            return pd.DataFrame()
            
        # 合并两个DataFrame
        merged_df = pd.merge(df1, df2, on=key_column, suffixes=('_1', '_2'))
        
        # 检查每一列的差异
        discrepancies = []
        for col in valid_compare_cols:
            # 对数值列计算相对差异，对非数值列直接比较
            if pd.api.types.is_numeric_dtype(merged_df[f"{col}_1"]):
                # 计算相对差异，避免除以零
                epsilon = 1e-10  # 很小的数，避免除以零
                denom = np.maximum(np.abs(merged_df[f"{col}_1"]), epsilon)
                rel_diff = np.abs(merged_df[f"{col}_1"] - merged_df[f"{col}_2"]) / denom
                
                # 找出差异大于阈值的行
                threshold = 0.01  # 1%的差异阈值
                diff_rows = merged_df[rel_diff > threshold]
                
                if not diff_rows.empty:
                    discrepancies.append({
                        'column': col,
                        'count': len(diff_rows),
                        'max_diff': rel_diff.max(),
                        'avg_diff': rel_diff.mean()
                    })
            else:
                # 非数值列直接比较
                diff_mask = merged_df[f"{col}_1"] != merged_df[f"{col}_2"]
                diff_count = diff_mask.sum()
                
                if diff_count > 0:
                    discrepancies.append({
                        'column': col,
                        'count': diff_count,
                        'diff_ratio': diff_count / len(merged_df)
                    })
                    
        # 创建差异报告
        if discrepancies:
            return pd.DataFrame(discrepancies)
        else:
            return pd.DataFrame() 