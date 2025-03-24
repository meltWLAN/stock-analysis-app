import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入策略模块
from strategies.trend_strategy import TrendStrategy
from strategies.volume_price_strategy import VolumePriceStrategy
from strategies.market_strategy import MarketStrategy
from strategies.advanced_analyzer import AdvancedAnalyzer

class IntegratedStrategy:
    """
    整合策略类，将各种分析方法和策略集成在一起，提供更精准的分析结果
    """
    
    def __init__(self, data_source=None, config=None):
        """
        初始化整合策略
        
        Args:
            data_source: 数据源对象
            config: 配置信息
        """
        self.logger = logging.getLogger(__name__)
        self.data_source = data_source
        self.config = config or {}
        
        # 初始化各个策略组件
        self.trend_strategy = TrendStrategy(data_source, config)
        self.volume_price_strategy = VolumePriceStrategy(data_source, config)
        self.market_strategy = MarketStrategy(data_source, config)
        self.advanced_analyzer = AdvancedAnalyzer(data_source, config)
        
        # 配置参数
        self.lookback_days = self.config.get('lookback_days', 60)
        self.ml_weight = self.config.get('ml_weight', 0.4)
        self.technical_weight = self.config.get('technical_weight', 0.4)
        self.market_weight = self.config.get('market_weight', 0.2)
    
    def analyze_stock(self, stock_code):
        """
        分析单只股票
        
        Args:
            stock_code: 股票代码
            
        Returns:
            dict: 分析结果
        """
        try:
            # 获取市场状况
            market_status = self.analyze_market()
            
            # 分析个股
            stock_analysis = self.advanced_analyzer.analyze_stock(stock_code)
            
            # 检查是否有错误
            if "error" in stock_analysis:
                return stock_analysis
            
            # 整合市场分析和个股分析
            integrated_analysis = self._integrate_analysis(stock_analysis, market_status)
            
            # 检查与市场趋势的一致性
            consistency = self._check_market_consistency(stock_analysis, market_status)
            
            # 添加最终决策
            final_decision = self._make_final_decision(integrated_analysis, consistency)
            
            return {
                "stock_code": stock_code,
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_analysis": market_status,
                "stock_analysis": stock_analysis,
                "integrated_score": integrated_analysis["score"],
                "market_consistency": consistency,
                "final_decision": final_decision,
                "confidence": integrated_analysis["confidence"]
            }
        except Exception as e:
            self.logger.error(f"分析股票时出错: {str(e)}")
            return {"error": f"分析股票 {stock_code} 时出错: {str(e)}"}
    
    def analyze_market(self):
        """
        分析市场状况
        
        Returns:
            dict: 市场分析结果
        """
        try:
            # 获取指数数据
            market_indices = self.config.get('market_indices', ['000001.SH', '399001.SZ', '399006.SZ'])
            market_data = {}
            
            for index_code in market_indices:
                df = self.data_source.get_daily_bars(
                    index_code,
                    start_date=(datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d"),
                    end_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if df is not None and not df.empty:
                    market_data[index_code] = df
            
            # 分析各指数趋势
            trend_results = {}
            for index_code, df in market_data.items():
                trend_eval = self.trend_strategy.evaluate_trend_strength(df)
                trend_results[index_code] = {
                    "trend_strength": trend_eval["trend_strength"],
                    "score": trend_eval["score"]
                }
            
            # 计算市场整体趋势
            overall_score = sum([result["score"] for result in trend_results.values()]) / len(trend_results) if trend_results else 0
            
            # 检查市场是否为上升趋势
            is_uptrend = self.market_strategy.check_index_uptrend()
            
            # 评估市场情绪
            market_sentiment = self.market_strategy.evaluate_market_sentiment()
            
            # 返回市场状况
            return {
                "overall_trend_score": round(overall_score, 2),
                "is_uptrend": is_uptrend,
                "market_sentiment": market_sentiment,
                "index_trends": trend_results
            }
        except Exception as e:
            self.logger.error(f"分析市场时出错: {str(e)}")
            return {"error": f"分析市场时出错: {str(e)}"}
    
    def _integrate_analysis(self, stock_analysis, market_status):
        """
        整合个股分析和市场分析
        
        Args:
            stock_analysis: 个股分析结果
            market_status: 市场分析结果
            
        Returns:
            dict: 整合分析结果
        """
        try:
            # 提取个股分数
            stock_score = stock_analysis.get("comprehensive_score", 50)
            
            # 提取市场分数
            market_score = market_status.get("overall_trend_score", 50)
            if market_score > 75:
                market_modifier = 1.2  # 强势市场加分
            elif market_score > 60:
                market_modifier = 1.1  # 良好市场小幅加分
            elif market_score < 30:
                market_modifier = 0.8  # 弱势市场减分
            elif market_score < 40:
                market_modifier = 0.9  # 较弱市场小幅减分
            else:
                market_modifier = 1.0  # 中性市场不变
            
            # 计算机器学习预测可信度
            ml_prediction = stock_analysis.get("ml_prediction", {})
            ml_confidence = 0.6  # 默认信任度
            
            if "ensemble_probability" in ml_prediction:
                ensemble_prob = ml_prediction["ensemble_probability"]
                # 概率越接近0或1，置信度越高
                ml_confidence = abs(ensemble_prob - 0.5) * 2
            
            # 整合评分
            technical_score = stock_analysis.get("technical_analysis", {}).get("trend_score", 50)
            
            # 加权平均
            integrated_score = (
                technical_score * self.technical_weight +
                (ml_prediction.get("ensemble_probability", 0.5) * 100) * self.ml_weight +
                market_score * self.market_weight
            ) * market_modifier
            
            # 限制在0-100范围内
            integrated_score = max(0, min(100, integrated_score))
            
            return {
                "score": round(integrated_score, 2),
                "confidence": round(ml_confidence, 2),
                "market_modifier": market_modifier
            }
        except Exception as e:
            self.logger.error(f"整合分析时出错: {str(e)}")
            return {"score": 50, "confidence": 0.5, "market_modifier": 1.0, "error": str(e)}
    
    def _check_market_consistency(self, stock_analysis, market_status):
        """
        检查个股趋势与市场趋势的一致性
        
        Args:
            stock_analysis: 个股分析结果
            market_status: 市场分析结果
            
        Returns:
            dict: 一致性评估结果
        """
        try:
            # 提取个股趋势
            stock_trend = stock_analysis.get("technical_analysis", {}).get("trend_strength", "中等")
            
            # 提取市场趋势
            market_trend_score = market_status.get("overall_trend_score", 50)
            if market_trend_score >= 75:
                market_trend = "极强"
            elif market_trend_score >= 60:
                market_trend = "强"
            elif market_trend_score >= 45:
                market_trend = "中强"
            elif market_trend_score >= 30:
                market_trend = "中等"
            else:
                market_trend = "弱"
            
            # 评估一致性
            trend_consistency = "未知"
            consistency_score = 0
            
            # 强趋势股票
            if stock_trend in ["极强", "强"]:
                if market_trend in ["极强", "强", "中强"]:
                    trend_consistency = "高度一致"
                    consistency_score = 0.9
                elif market_trend == "中等":
                    trend_consistency = "中度一致"
                    consistency_score = 0.7
                else:
                    trend_consistency = "逆市强势"
                    consistency_score = 0.5
            
            # 中强趋势股票
            elif stock_trend == "中强":
                if market_trend in ["中强", "中等"]:
                    trend_consistency = "高度一致"
                    consistency_score = 0.8
                elif market_trend in ["极强", "强"]:
                    trend_consistency = "略弱于大盘"
                    consistency_score = 0.6
                else:
                    trend_consistency = "强于大盘"
                    consistency_score = 0.7
            
            # 中等趋势股票
            elif stock_trend == "中等":
                if market_trend == "中等":
                    trend_consistency = "高度一致"
                    consistency_score = 0.7
                elif market_trend in ["极强", "强", "中强"]:
                    trend_consistency = "弱于大盘"
                    consistency_score = 0.5
                else:
                    trend_consistency = "强于大盘"
                    consistency_score = 0.6
            
            # 弱势股票
            else:
                if market_trend in ["弱", "中弱"]:
                    trend_consistency = "同步弱势"
                    consistency_score = 0.8
                else:
                    trend_consistency = "个股弱势"
                    consistency_score = 0.4
            
            return {
                "consistency": trend_consistency,
                "score": consistency_score,
                "stock_trend": stock_trend,
                "market_trend": market_trend
            }
        except Exception as e:
            self.logger.error(f"检查市场一致性时出错: {str(e)}")
            return {"consistency": "未知", "score": 0.5, "error": str(e)}
    
    def _make_final_decision(self, integrated_analysis, consistency):
        """
        根据整合分析和一致性评估做出最终决策
        
        Args:
            integrated_analysis: 整合分析结果
            consistency: 一致性评估结果
            
        Returns:
            dict: 最终决策
        """
        try:
            score = integrated_analysis.get("score", 50)
            confidence = integrated_analysis.get("confidence", 0.5)
            consistency_score = consistency.get("score", 0.5)
            
            # 基于评分的基础决策
            if score >= 85:
                base_decision = "强烈推荐买入"
                action = "buy"
                strength = 5
            elif score >= 70:
                base_decision = "推荐买入"
                action = "buy"
                strength = 4
            elif score >= 60:
                base_decision = "建议买入"
                action = "buy"
                strength = 3
            elif score >= 50:
                base_decision = "观望"
                action = "hold"
                strength = 0
            elif score >= 40:
                base_decision = "谨慎持有"
                action = "hold"
                strength = -1
            elif score >= 30:
                base_decision = "建议减持"
                action = "sell"
                strength = -3
            elif score >= 15:
                base_decision = "推荐卖出"
                action = "sell"
                strength = -4
            else:
                base_decision = "强烈推荐卖出"
                action = "sell"
                strength = -5
            
            # 调整决策强度，基于一致性和置信度
            adjusted_strength = strength * (confidence * 0.7 + consistency_score * 0.3)
            
            # 根据调整后的强度确定最终决策
            if adjusted_strength >= 4:
                final_decision = "强烈推荐买入"
                final_action = "buy"
            elif adjusted_strength >= 2.5:
                final_decision = "推荐买入"
                final_action = "buy"
            elif adjusted_strength >= 1:
                final_decision = "建议买入"
                final_action = "buy"
            elif adjusted_strength >= -1:
                final_decision = "观望"
                final_action = "hold"
            elif adjusted_strength >= -2.5:
                final_decision = "建议减持"
                final_action = "sell"
            elif adjusted_strength >= -4:
                final_decision = "推荐卖出"
                final_action = "sell"
            else:
                final_decision = "强烈推荐卖出"
                final_action = "sell"
            
            return {
                "base_decision": base_decision,
                "final_decision": final_decision,
                "action": final_action,
                "original_strength": strength,
                "adjusted_strength": round(adjusted_strength, 2)
            }
        except Exception as e:
            self.logger.error(f"做出最终决策时出错: {str(e)}")
            return {"final_decision": "无法决策", "action": "hold", "error": str(e)}
    
    def screen_stocks(self, stock_codes, screen_type='strong_trend', top_n=10):
        """
        筛选股票
        
        Args:
            stock_codes: 股票代码列表
            screen_type: 筛选类型，'strong_trend', 'reversal', 'market_leader', 'value'
            top_n: 返回前N名股票
            
        Returns:
            list: 筛选结果列表
        """
        results = []
        
        for stock_code in stock_codes:
            try:
                # 获取股票数据
                df = self.data_source.get_daily_bars(
                    stock_code,
                    start_date=(datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d"),
                    end_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if df is None or df.empty:
                    continue
                
                # 根据筛选类型应用不同的筛选条件
                if screen_type == 'strong_trend':
                    # 寻找强趋势股票
                    trend_eval = self.trend_strategy.evaluate_trend_strength(df)
                    score = trend_eval["score"]
                    if score >= 60:  # 只保留趋势较强的股票
                        results.append({
                            "stock_code": stock_code,
                            "score": score,
                            "trend_strength": trend_eval["trend_strength"],
                            "details": trend_eval["details"]
                        })
                
                elif screen_type == 'reversal':
                    # 寻找反转信号
                    rsi_data = self.trend_strategy.calculate_rsi(df)
                    macd_data = self.trend_strategy.calculate_macd(df)
                    
                    # 计算分数
                    score = 0
                    details = {}
                    
                    # RSI超卖反弹
                    if rsi_data is not None and 'rsi' in rsi_data.columns:
                        last_rsi = rsi_data['rsi'].iloc[-1]
                        if last_rsi < 30:
                            score += 40
                            details["RSI超卖"] = True
                        elif last_rsi < 40:
                            score += 20
                            details["RSI低位"] = True
                        
                        # RSI上升
                        rsi_change = rsi_data['rsi'].diff(1).iloc[-1]
                        if rsi_change > 5:
                            score += 20
                            details["RSI快速上升"] = True
                        elif rsi_change > 0:
                            score += 10
                            details["RSI上升"] = True
                    
                    # MACD金叉或即将金叉
                    if macd_data is not None:
                        macd_golden_cross = self.trend_strategy.check_macd_golden_cross(macd_data)
                        if macd_golden_cross:
                            score += 30
                            details["MACD金叉"] = True
                        
                        # MACD柱状图由负转正
                        if len(macd_data) > 2 and 'macd_hist' in macd_data.columns:
                            hist_1 = macd_data['macd_hist'].iloc[-1]
                            hist_2 = macd_data['macd_hist'].iloc[-2]
                            if hist_2 < 0 and hist_1 > 0:
                                score += 20
                                details["MACD柱状图转正"] = True
                            elif hist_1 > hist_2 and hist_1 < 0:
                                score += 10
                                details["MACD柱状图上升"] = True
                    
                    if score >= 50:  # 只保留反转信号强的股票
                        results.append({
                            "stock_code": stock_code,
                            "score": score,
                            "details": details
                        })
                
                elif screen_type == 'market_leader':
                    # 寻找市场领导者
                    # 获取市场状况
                    market_status = self.analyze_market()
                    
                    # 分析个股
                    stock_analysis = self.advanced_analyzer.analyze_stock(stock_code)
                    
                    if "error" not in stock_analysis:
                        # 检查与市场趋势的一致性
                        consistency = self._check_market_consistency(stock_analysis, market_status)
                        
                        # 计算分数
                        stock_score = stock_analysis.get("comprehensive_score", 50)
                        consistency_score = consistency.get("score", 0.5) * 100
                        
                        # 强于大盘的股票得分更高
                        if consistency.get("consistency", "") == "逆市强势" or consistency.get("consistency", "") == "强于大盘":
                            leader_score = stock_score * 1.2
                        else:
                            leader_score = stock_score * consistency_score / 100
                        
                        if leader_score >= 60:  # 只保留潜在市场领导者
                            results.append({
                                "stock_code": stock_code,
                                "score": leader_score,
                                "stock_score": stock_score,
                                "consistency": consistency.get("consistency", ""),
                                "recommendation": stock_analysis.get("recommendation", "")
                            })
                
                elif screen_type == 'value':
                    # 实现价值投资筛选的逻辑
                    # 此处需要基本面数据，可能需要额外的数据源
                    pass
                
            except Exception as e:
                self.logger.error(f"筛选股票 {stock_code} 时出错: {str(e)}")
                continue
        
        # 按评分降序排序
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # 返回前N名
        return sorted_results[:top_n]
    
    def backtest_strategy(self, stock_code, start_date, end_date):
        """
        回测策略
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict: 回测结果
        """
        try:
            # 获取回测期间的股票数据
            df = self.data_source.get_daily_bars(
                stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                return {"error": f"无法获取股票 {stock_code} 的数据"}
            
            # 确保数据按日期排序
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            # 创建回测结果DataFrame
            backtest_results = pd.DataFrame()
            backtest_results['date'] = df['date'] if 'date' in df.columns else pd.date_range(start=start_date, periods=len(df))
            backtest_results['close'] = df['close']
            
            # 初始化信号、持仓和资金
            backtest_results['signal'] = 0  # 0: 无操作, 1: 买入, -1: 卖出
            backtest_results['position'] = 0  # 0: 无持仓, 1: 持有
            backtest_results['capital'] = 100000  # 初始资金
            backtest_results['shares'] = 0  # 持有股数
            backtest_results['value'] = 0  # 持仓价值
            backtest_results['total_value'] = 0  # 总资产价值
            
            # 设置回测窗口
            window_size = 60  # 使用60天数据进行分析
            
            # 回测每一天
            for i in range(window_size, len(df)):
                # 获取当前日期的历史数据
                history_data = df.iloc[i-window_size:i]
                
                # 分析数据
                trend_eval = self.trend_strategy.evaluate_trend_strength(history_data)
                trend_score = trend_eval["score"]
                
                # 计算MACD
                macd_data = self.trend_strategy.calculate_macd(history_data)
                macd_signal = self.trend_strategy.check_macd_golden_cross(macd_data) if macd_data is not None else False
                
                # 计算RSI
                rsi_data = self.trend_strategy.calculate_rsi(history_data)
                rsi_signal = self.trend_strategy.check_rsi_buy_signal(rsi_data) if rsi_data is not None else False
                
                # 生成信号
                # 买入信号: 趋势强度高，且MACD金叉或RSI买入信号
                if trend_score >= 60 and (macd_signal or rsi_signal):
                    backtest_results.loc[i, 'signal'] = 1
                # 卖出信号: 趋势强度低，或MACD死叉
                elif trend_score < 30 or (macd_data is not None and self.trend_strategy.check_macd_death_cross(macd_data)):
                    backtest_results.loc[i, 'signal'] = -1
                
                # 更新持仓
                prev_position = backtest_results.loc[i-1, 'position']
                current_signal = backtest_results.loc[i, 'signal']
                
                # 处理买入信号
                if current_signal == 1 and prev_position == 0:
                    # 全仓买入
                    price = df.loc[i, 'close']
                    available_capital = backtest_results.loc[i-1, 'capital']
                    shares = int(available_capital / price)
                    cost = shares * price
                    
                    backtest_results.loc[i, 'position'] = 1
                    backtest_results.loc[i, 'shares'] = shares
                    backtest_results.loc[i, 'capital'] = available_capital - cost
                    backtest_results.loc[i, 'value'] = shares * price
                else:
                    # 持仓不变
                    backtest_results.loc[i, 'position'] = prev_position
                    backtest_results.loc[i, 'shares'] = backtest_results.loc[i-1, 'shares']
                    backtest_results.loc[i, 'capital'] = backtest_results.loc[i-1, 'capital']
                    backtest_results.loc[i, 'value'] = backtest_results.loc[i, 'shares'] * df.loc[i, 'close']
                
                # 处理卖出信号
                if current_signal == -1 and prev_position == 1:
                    # 全仓卖出
                    price = df.loc[i, 'close']
                    shares = backtest_results.loc[i, 'shares']
                    value = shares * price
                    
                    backtest_results.loc[i, 'position'] = 0
                    backtest_results.loc[i, 'shares'] = 0
                    backtest_results.loc[i, 'capital'] = backtest_results.loc[i, 'capital'] + value
                    backtest_results.loc[i, 'value'] = 0
                
                # 计算总资产价值
                backtest_results.loc[i, 'total_value'] = backtest_results.loc[i, 'capital'] + backtest_results.loc[i, 'value']
            
            # 计算回测指标
            initial_value = backtest_results['total_value'].iloc[window_size]
            final_value = backtest_results['total_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            # 计算年化收益率
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            annual_return = total_return * 365 / days if days > 0 else 0
            
            # 计算最大回撤
            backtest_results['cummax'] = backtest_results['total_value'].cummax()
            backtest_results['drawdown'] = (backtest_results['cummax'] - backtest_results['total_value']) / backtest_results['cummax'] * 100
            max_drawdown = backtest_results['drawdown'].max()
            
            # 统计交易次数
            buy_signals = backtest_results[backtest_results['signal'] == 1]
            sell_signals = backtest_results[backtest_results['signal'] == -1]
            trade_count = min(len(buy_signals), len(sell_signals))
            
            # 返回回测结果
            return {
                "initial_value": initial_value,
                "final_value": final_value,
                "total_return": total_return,
                "annual_return": annual_return,
                "max_drawdown": max_drawdown,
                "trade_count": trade_count,
                "sharpe_ratio": annual_return / max_drawdown if max_drawdown > 0 else 0,
                "results_df": backtest_results
            }
        except Exception as e:
            self.logger.error(f"回测策略时出错: {str(e)}")
            return {"error": f"回测策略时出错: {str(e)}"} 