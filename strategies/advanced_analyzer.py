import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load

# 导入其他策略类
from strategies.trend_strategy import TrendStrategy
from strategies.volume_price_strategy import VolumePriceStrategy
from strategies.market_strategy import MarketStrategy

class AdvancedAnalyzer:
    """
    高级分析器类，整合多种分析方法和机器学习模型以提高预测精确度
    """
    
    def __init__(self, data_source=None, config=None):
        """
        初始化高级分析器
        
        Args:
            data_source: 数据源对象
            config: 配置信息
        """
        self.logger = logging.getLogger(__name__)
        self.data_source = data_source
        self.config = config or {}
        
        # 初始化其他策略类
        self.trend_strategy = TrendStrategy(data_source, config)
        self.volume_price_strategy = VolumePriceStrategy(data_source, config)
        self.market_strategy = MarketStrategy(data_source, config)
        
        # 机器学习模型存储目录
        self.models_dir = self.config.get('models_dir', 'models/saved')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 初始化分类模型
        self.models = {
            'random_forest': None,
            'gradient_boosting': None
        }
    
    def analyze_stock(self, stock_code, days=60):
        """
        对股票进行综合分析
        
        Args:
            stock_code: 股票代码
            days: 分析所需的历史数据天数
            
        Returns:
            dict: 分析结果
        """
        try:
            # 获取股票数据
            df = self.get_stock_data(stock_code, days)
            if df is None or df.empty:
                return {"error": f"无法获取股票 {stock_code} 的数据"}
            
            # 基本分析
            basic_analysis = self._basic_analysis(stock_code, df)
            
            # 技术分析
            technical_analysis = self._technical_analysis(df)
            
            # 机器学习预测
            ml_prediction = self._ml_prediction(stock_code, df)
            
            # 综合评分
            score, details = self._calculate_comprehensive_score(basic_analysis, technical_analysis, ml_prediction)
            
            # 返回结果
            return {
                "stock_code": stock_code,
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "basic_analysis": basic_analysis,
                "technical_analysis": technical_analysis,
                "ml_prediction": ml_prediction,
                "comprehensive_score": score,
                "details": details,
                "recommendation": self._get_recommendation(score)
            }
        except Exception as e:
            self.logger.error(f"分析股票时出错: {str(e)}")
            return {"error": f"分析股票 {stock_code} 时出错: {str(e)}"}
    
    def get_stock_data(self, stock_code, days=60):
        """
        获取股票历史数据
        
        Args:
            stock_code: 股票代码
            days: 历史数据天数
            
        Returns:
            DataFrame: 股票数据
        """
        if self.data_source is None:
            self.logger.error("未提供数据源")
            return None
        
        try:
            # 获取股票数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.data_source.get_daily_bars(
                stock_code,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            if df is None or df.empty:
                self.logger.warning(f"未找到股票 {stock_code} 的数据")
                return None
            
            # 确保数据按日期排序
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            return df
        except Exception as e:
            self.logger.error(f"获取股票数据时出错: {str(e)}")
            return None
    
    def _basic_analysis(self, stock_code, df):
        """
        执行基本分析
        
        Args:
            stock_code: 股票代码
            df: 股票数据DataFrame
            
        Returns:
            dict: 基本分析结果
        """
        try:
            # 基本价格统计
            last_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
            change_pct = (last_close - prev_close) / prev_close * 100 if prev_close > 0 else 0
            
            # 计算20天平均成交量
            volume_col = 'vol' if 'vol' in df.columns else 'volume'
            avg_volume_20d = df[volume_col].tail(20).mean() if volume_col in df.columns else 0
            last_volume = df[volume_col].iloc[-1] if volume_col in df.columns else 0
            volume_ratio = last_volume / avg_volume_20d if avg_volume_20d > 0 else 1
            
            # 计算20天最高和最低价
            high_20d = df['high'].tail(20).max()
            low_20d = df['low'].tail(20).min()
            price_position = (last_close - low_20d) / (high_20d - low_20d) if (high_20d - low_20d) > 0 else 0.5
            
            return {
                "last_close": round(last_close, 2),
                "change_pct": round(change_pct, 2),
                "avg_volume_20d": round(avg_volume_20d, 0),
                "volume_ratio": round(volume_ratio, 2),
                "high_20d": round(high_20d, 2),
                "low_20d": round(low_20d, 2),
                "price_position": round(price_position, 2)
            }
        except Exception as e:
            self.logger.error(f"执行基本分析时出错: {str(e)}")
            return {"error": str(e)}
    
    def _technical_analysis(self, df):
        """
        执行技术分析
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            dict: 技术分析结果
        """
        try:
            # 使用趋势策略进行分析
            trend_evaluation = self.trend_strategy.evaluate_trend_strength(df)
            
            # 使用量价策略进行分析
            volume_price_check = self.volume_price_strategy.check_volume_price_relationship(df)
            
            # 计算布林带
            bollinger_bands = self.trend_strategy.calculate_bollinger_bands(df)
            bollinger_breakout = self.trend_strategy.check_bollinger_breakout(bollinger_bands) if bollinger_bands is not None else False
            
            # 计算RSI指标
            rsi_data = self.trend_strategy.calculate_rsi(df)
            last_rsi = rsi_data['rsi'].iloc[-1] if rsi_data is not None and 'rsi' in rsi_data.columns else 50
            rsi_buy_signal = self.trend_strategy.check_rsi_buy_signal(rsi_data) if rsi_data is not None else False
            
            # 计算MACD指标
            macd_data = self.trend_strategy.calculate_macd(df)
            macd_golden_cross = self.trend_strategy.check_macd_golden_cross(macd_data) if macd_data is not None else False
            
            # 计算风险回报比
            risk_reward_ratio = self.trend_strategy.get_trend_risk_reward_ratio(df)
            
            return {
                "trend_strength": trend_evaluation["trend_strength"],
                "trend_score": trend_evaluation["score"],
                "trend_details": trend_evaluation["details"],
                "volume_price_relationship": volume_price_check,
                "bollinger_breakout": bollinger_breakout,
                "rsi": round(last_rsi, 2),
                "rsi_buy_signal": rsi_buy_signal,
                "macd_golden_cross": macd_golden_cross,
                "risk_reward_ratio": round(risk_reward_ratio, 2)
            }
        except Exception as e:
            self.logger.error(f"执行技术分析时出错: {str(e)}")
            return {"error": str(e)}
    
    def _prepare_features(self, df):
        """
        为机器学习模型准备特征
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            DataFrame: 特征数据
        """
        try:
            # 复制数据以避免修改原始数据
            features_df = df.copy()
            
            # 添加基本技术指标
            # 移动平均线
            features_df['ma5'] = features_df['close'].rolling(window=5).mean()
            features_df['ma10'] = features_df['close'].rolling(window=10).mean()
            features_df['ma20'] = features_df['close'].rolling(window=20).mean()
            
            # 相对位置指标
            features_df['ma5_position'] = (features_df['close'] - features_df['ma5']) / features_df['ma5']
            features_df['ma10_position'] = (features_df['close'] - features_df['ma10']) / features_df['ma10']
            features_df['ma20_position'] = (features_df['close'] - features_df['ma20']) / features_df['ma20']
            
            # 价格动量指标
            features_df['momentum_1d'] = features_df['close'].pct_change(1)
            features_df['momentum_3d'] = features_df['close'].pct_change(3)
            features_df['momentum_5d'] = features_df['close'].pct_change(5)
            
            # 波动性指标
            atr_df = self.trend_strategy.calculate_atr(df)
            if atr_df is not None and 'atr' in atr_df.columns:
                features_df['atr'] = atr_df['atr']
                features_df['atr_ratio'] = features_df['atr'] / features_df['close']
            
            # RSI指标
            rsi_df = self.trend_strategy.calculate_rsi(df)
            if rsi_df is not None and 'rsi' in rsi_df.columns:
                features_df['rsi'] = rsi_df['rsi']
            
            # 布林带指标
            bb_df = self.trend_strategy.calculate_bollinger_bands(df)
            if bb_df is not None:
                features_df['bb_position'] = bb_df['%b'] if '%b' in bb_df.columns else 0.5
                features_df['bb_bandwidth'] = bb_df['bb_bandwidth'] if 'bb_bandwidth' in bb_df.columns else 0.0
            
            # MACD指标
            macd_df = self.trend_strategy.calculate_macd(df)
            if macd_df is not None:
                features_df['macd'] = macd_df['macd'] if 'macd' in macd_df.columns else 0.0
                features_df['macd_signal'] = macd_df['macd_signal'] if 'macd_signal' in macd_df.columns else 0.0
                features_df['macd_hist'] = macd_df['macd_hist'] if 'macd_hist' in macd_df.columns else 0.0
            
            # 成交量相关指标
            volume_col = 'vol' if 'vol' in features_df.columns else 'volume'
            if volume_col in features_df.columns:
                features_df['volume_ma5'] = features_df[volume_col].rolling(window=5).mean()
                features_df['volume_ma10'] = features_df[volume_col].rolling(window=10).mean()
                features_df['volume_ratio'] = features_df[volume_col] / features_df['volume_ma5']
            
            # 创建目标变量：5天后价格是否上涨
            features_df['target'] = features_df['close'].shift(-5) > features_df['close']
            features_df['target'] = features_df['target'].astype(int)
            
            # 删除NaN值
            features_df = features_df.dropna()
            
            return features_df
        except Exception as e:
            self.logger.error(f"准备特征时出错: {str(e)}")
            return None
    
    def train_models(self, stock_codes, days=365, force_retrain=False):
        """
        训练机器学习模型
        
        Args:
            stock_codes: 用于训练的股票代码列表
            days: 训练所需的历史数据天数
            force_retrain: 是否强制重新训练模型
            
        Returns:
            bool: 训练是否成功
        """
        try:
            if not stock_codes:
                self.logger.error("未提供股票代码")
                return False
            
            # 检查模型是否已存在
            rf_model_path = os.path.join(self.models_dir, 'random_forest_model.joblib')
            gb_model_path = os.path.join(self.models_dir, 'gradient_boosting_model.joblib')
            scaler_path = os.path.join(self.models_dir, 'feature_scaler.joblib')
            
            if not force_retrain and os.path.exists(rf_model_path) and os.path.exists(gb_model_path):
                # 加载现有模型
                self.models['random_forest'] = load(rf_model_path)
                self.models['gradient_boosting'] = load(gb_model_path)
                self.feature_scaler = load(scaler_path) if os.path.exists(scaler_path) else StandardScaler()
                self.logger.info("已加载现有模型")
                return True
            
            # 收集训练数据
            all_features = []
            for stock_code in stock_codes:
                df = self.get_stock_data(stock_code, days)
                if df is None or df.empty:
                    self.logger.warning(f"无法获取股票 {stock_code} 的数据")
                    continue
                
                features_df = self._prepare_features(df)
                if features_df is not None and not features_df.empty:
                    all_features.append(features_df)
            
            if not all_features:
                self.logger.error("没有有效的训练数据")
                return False
            
            # 合并所有特征数据
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # 分离特征和目标
            feature_columns = [col for col in combined_features.columns if col not in ['date', 'target']]
            X = combined_features[feature_columns]
            y = combined_features['target']
            
            # 标准化特征
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # 训练随机森林模型
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # 训练梯度提升模型
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb_model.fit(X_train, y_train)
            
            # 评估模型
            rf_pred = rf_model.predict(X_test)
            gb_pred = gb_model.predict(X_test)
            
            rf_accuracy = accuracy_score(y_test, rf_pred)
            gb_accuracy = accuracy_score(y_test, gb_pred)
            
            self.logger.info(f"随机森林模型准确率: {rf_accuracy:.4f}")
            self.logger.info(f"梯度提升模型准确率: {gb_accuracy:.4f}")
            
            # 保存模型
            os.makedirs(self.models_dir, exist_ok=True)
            dump(rf_model, rf_model_path)
            dump(gb_model, gb_model_path)
            dump(self.feature_scaler, scaler_path)
            
            # 更新模型
            self.models['random_forest'] = rf_model
            self.models['gradient_boosting'] = gb_model
            
            return True
        except Exception as e:
            self.logger.error(f"训练模型时出错: {str(e)}")
            return False
    
    def _ml_prediction(self, stock_code, df):
        """
        使用机器学习模型进行预测
        
        Args:
            stock_code: 股票代码
            df: 股票数据DataFrame
            
        Returns:
            dict: 预测结果
        """
        try:
            # 准备特征
            features_df = self._prepare_features(df)
            if features_df is None or features_df.empty:
                return {"error": "无法准备特征数据"}
            
            # 加载模型（如果尚未加载）
            rf_model_path = os.path.join(self.models_dir, 'random_forest_model.joblib')
            gb_model_path = os.path.join(self.models_dir, 'gradient_boosting_model.joblib')
            scaler_path = os.path.join(self.models_dir, 'feature_scaler.joblib')
            
            if (self.models['random_forest'] is None or self.models['gradient_boosting'] is None) and \
               os.path.exists(rf_model_path) and os.path.exists(gb_model_path):
                self.models['random_forest'] = load(rf_model_path)
                self.models['gradient_boosting'] = load(gb_model_path)
                self.feature_scaler = load(scaler_path) if os.path.exists(scaler_path) else StandardScaler()
            
            # 如果模型未加载，返回默认预测
            if self.models['random_forest'] is None or self.models['gradient_boosting'] is None:
                return {
                    "rf_probability": 0.5,
                    "gb_probability": 0.5,
                    "ensemble_probability": 0.5,
                    "trained_models_available": False
                }
            
            # 准备最新数据的特征
            latest_features = features_df.iloc[-1:][features_df.columns.drop(['target', 'date'] if 'date' in features_df.columns else ['target'])]
            
            # 标准化特征
            X_scaled = self.feature_scaler.transform(latest_features)
            
            # 预测
            rf_prob = self.models['random_forest'].predict_proba(X_scaled)[0][1]
            gb_prob = self.models['gradient_boosting'].predict_proba(X_scaled)[0][1]
            
            # 集成预测
            ensemble_prob = (rf_prob + gb_prob) / 2
            
            return {
                "rf_probability": round(rf_prob, 4),
                "gb_probability": round(gb_prob, 4),
                "ensemble_probability": round(ensemble_prob, 4),
                "trained_models_available": True
            }
        except Exception as e:
            self.logger.error(f"机器学习预测时出错: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_comprehensive_score(self, basic_analysis, technical_analysis, ml_prediction):
        """
        计算综合评分
        
        Args:
            basic_analysis: 基本分析结果
            technical_analysis: 技术分析结果
            ml_prediction: 机器学习预测结果
            
        Returns:
            tuple: (综合评分, 详细评分项)
        """
        try:
            score = 0
            details = {}
            
            # 基本分析评分 (最高30分)
            basic_score = 0
            
            # 价格位置评分
            price_position = basic_analysis.get("price_position", 0.5)
            if price_position > 0.8:
                price_position_score = 5  # 接近高点
            elif price_position < 0.2:
                price_position_score = 15  # 接近低点，买入机会
            else:
                price_position_score = 10  # 中间位置
            basic_score += price_position_score
            details["价格位置"] = price_position_score
            
            # 成交量比率评分
            volume_ratio = basic_analysis.get("volume_ratio", 1.0)
            if volume_ratio > 2.0:
                volume_score = 15  # 放量
            elif volume_ratio > 1.2:
                volume_score = 10  # 量能扩大
            else:
                volume_score = 5  # 正常或缩量
            basic_score += volume_score
            details["成交量"] = volume_score
            
            # 技术分析评分 (最高40分)
            tech_score = 0
            
            # 趋势强度评分
            trend_score = technical_analysis.get("trend_score", 0)
            normalized_trend_score = min(25, trend_score / 100 * 25)
            tech_score += normalized_trend_score
            details["趋势强度"] = round(normalized_trend_score, 1)
            
            # 风险回报比评分
            risk_reward = technical_analysis.get("risk_reward_ratio", 1.0)
            if risk_reward > 2.0:
                rr_score = 15  # 优秀风险回报比
            elif risk_reward > 1.5:
                rr_score = 10  # 良好风险回报比
            elif risk_reward > 1.0:
                rr_score = 5  # 中性风险回报比
            else:
                rr_score = 0  # 不良风险回报比
            tech_score += rr_score
            details["风险回报比"] = rr_score
            
            # 机器学习预测评分 (最高30分)
            ml_score = 0
            
            if "error" not in ml_prediction:
                ensemble_prob = ml_prediction.get("ensemble_probability", 0.5)
                if ensemble_prob > 0.75:
                    ml_score = 30  # 强烈看涨
                elif ensemble_prob > 0.65:
                    ml_score = 25  # 看涨
                elif ensemble_prob > 0.55:
                    ml_score = 20  # 略微看涨
                elif ensemble_prob > 0.45:
                    ml_score = 15  # 中性
                elif ensemble_prob > 0.35:
                    ml_score = 10  # 略微看跌
                elif ensemble_prob > 0.25:
                    ml_score = 5  # 看跌
                else:
                    ml_score = 0  # 强烈看跌
                details["机器学习预测"] = ml_score
            else:
                # 如果ML预测失败，使用技术分析的权重
                ml_score = tech_score / 40 * 30
                details["机器学习预测"] = f"{round(ml_score, 1)} (基于技术分析)"
            
            # 计算总分
            score = basic_score + tech_score + ml_score
            
            # 归一化至100分
            normalized_score = min(100, score)
            
            return normalized_score, details
        except Exception as e:
            self.logger.error(f"计算综合评分时出错: {str(e)}")
            return 50, {"错误": str(e)}
    
    def _get_recommendation(self, score):
        """
        根据综合评分提供建议
        
        Args:
            score: 综合评分
            
        Returns:
            str: 投资建议
        """
        if score >= 85:
            return "强烈推荐买入"
        elif score >= 70:
            return "推荐买入"
        elif score >= 60:
            return "建议买入"
        elif score >= 50:
            return "观望"
        elif score >= 40:
            return "谨慎持有"
        elif score >= 30:
            return "建议减持"
        elif score >= 15:
            return "推荐卖出"
        else:
            return "强烈推荐卖出"
    
    def batch_analyze(self, stock_codes, top_n=10):
        """
        批量分析多只股票并排序
        
        Args:
            stock_codes: 股票代码列表
            top_n: 返回前N名股票
            
        Returns:
            list: 分析结果列表，按评分降序排序
        """
        results = []
        
        for stock_code in stock_codes:
            analysis = self.analyze_stock(stock_code)
            if "error" not in analysis:
                results.append({
                    "stock_code": stock_code,
                    "score": analysis.get("comprehensive_score", 0),
                    "recommendation": analysis.get("recommendation", "未知"),
                    "trend_strength": analysis.get("technical_analysis", {}).get("trend_strength", "未知"),
                    "last_close": analysis.get("basic_analysis", {}).get("last_close", 0),
                    "full_analysis": analysis
                })
        
        # 按评分降序排序
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # 返回前N名
        return sorted_results[:top_n] 