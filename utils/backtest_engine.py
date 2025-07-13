"""
Backtesting Engine for portfolio performance simulation and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Comprehensive backtesting engine for portfolio performance analysis
    """
    
    def __init__(self):
        """Initialize the backtesting engine"""
        self.portfolio_history = {}
        self.benchmark_history = {}
        self.performance_metrics = {}
        self.trade_history = []
        
        # Default parameters
        self.default_params = {
            'initial_capital': 100000,
            'rebalance_frequency': 'monthly',  # daily, weekly, monthly, quarterly
            'transaction_cost': 0.001,  # 0.1% transaction cost
            'max_position_size': 0.1,  # Maximum 10% per position
            'min_position_size': 0.01,  # Minimum 1% per position
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'benchmark': 'SPY'  # S&P 500 as benchmark
        }
    
    def run_backtest(self, stock_selections: Dict[str, List[str]], 
                    price_data: Dict[str, pd.DataFrame],
                    start_date: str, end_date: str,
                    strategy_params: Dict = None) -> Dict:
        """
        Run comprehensive backtest simulation
        
        Args:
            stock_selections: Dictionary with dates as keys and selected stocks as values
            price_data: Dictionary of price DataFrames by symbol
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            strategy_params: Strategy parameters
        
        Returns:
            Dictionary with backtest results
        """
        # Merge with default parameters
        params = {**self.default_params, **(strategy_params or {})}
        
        results = {
            'parameters': params,
            'start_date': start_date,
            'end_date': end_date,
            'portfolio_performance': {},
            'benchmark_performance': {},
            'performance_metrics': {},
            'trade_analysis': {},
            'risk_analysis': {}
        }
        
        try:
            logger.info(f"Running backtest from {start_date} to {end_date}")
            
            # Prepare data
            aligned_data = self._align_price_data(price_data, start_date, end_date)
            
            if not aligned_data:
                results['error'] = "No valid price data for backtesting period"
                return results
            
            # Generate trading calendar
            trading_dates = self._generate_trading_calendar(aligned_data, start_date, end_date, params['rebalance_frequency'])
            
            # Simulate portfolio performance
            portfolio_results = self._simulate_portfolio(stock_selections, aligned_data, trading_dates, params)
            results['portfolio_performance'] = portfolio_results
            
            # Calculate benchmark performance
            benchmark_results = self._calculate_benchmark_performance(aligned_data, params['benchmark'], trading_dates, params)
            results['benchmark_performance'] = benchmark_results
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(portfolio_results, benchmark_results, params)
            results['performance_metrics'] = performance_metrics
            
            # Analyze trades
            trade_analysis = self._analyze_trades(self.trade_history, aligned_data)
            results['trade_analysis'] = trade_analysis
            
            # Risk analysis
            risk_analysis = self._perform_risk_analysis(portfolio_results, benchmark_results, params)
            results['risk_analysis'] = risk_analysis
            
            # Store results
            self.portfolio_history = portfolio_results
            self.benchmark_history = benchmark_results
            self.performance_metrics = performance_metrics
            
            logger.info("Backtest completed successfully")
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _align_price_data(self, price_data: Dict[str, pd.DataFrame], 
                         start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Align price data to the backtesting period"""
        aligned_data = {}
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        for symbol, df in price_data.items():
            if df.empty:
                continue
            
            # Filter by date range
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            filtered_df = df.loc[mask].copy()
            
            if len(filtered_df) > 10:  # Need minimum data points
                # Forward fill missing values
                filtered_df = filtered_df.fillna(method='ffill')
                aligned_data[symbol] = filtered_df
        
        return aligned_data
    
    def _generate_trading_calendar(self, price_data: Dict[str, pd.DataFrame], 
                                 start_date: str, end_date: str, frequency: str) -> List[str]:
        """Generate trading calendar based on rebalancing frequency"""
        # Get common trading dates from price data
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index)
        
        common_dates = sorted(list(all_dates))
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter to backtest period
        trading_dates = [date for date in common_dates if start_dt <= date <= end_dt]
        
        if frequency == 'daily':
            return [date.strftime('%Y-%m-%d') for date in trading_dates]
        elif frequency == 'weekly':
            # Take every 5th trading day (approximately weekly)
            return [date.strftime('%Y-%m-%d') for date in trading_dates[::5]]
        elif frequency == 'monthly':
            # Take first trading day of each month
            monthly_dates = []
            current_month = None
            for date in trading_dates:
                if current_month != date.month:
                    monthly_dates.append(date.strftime('%Y-%m-%d'))
                    current_month = date.month
            return monthly_dates
        elif frequency == 'quarterly':
            # Take first trading day of each quarter
            quarterly_dates = []
            current_quarter = None
            for date in trading_dates:
                quarter = (date.month - 1) // 3 + 1
                if current_quarter != quarter:
                    quarterly_dates.append(date.strftime('%Y-%m-%d'))
                    current_quarter = quarter
            return quarterly_dates
        
        return [date.strftime('%Y-%m-%d') for date in trading_dates]
    
    def _simulate_portfolio(self, stock_selections: Dict[str, List[str]], 
                          price_data: Dict[str, pd.DataFrame],
                          trading_dates: List[str], params: Dict) -> Dict:
        """Simulate portfolio performance over time"""
        portfolio_results = {
            'dates': [],
            'portfolio_value': [],
            'cash': [],
            'positions': [],
            'returns': [],
            'cumulative_returns': []
        }
        
        # Initialize portfolio
        current_value = params['initial_capital']
        current_cash = params['initial_capital']
        current_positions = {}
        
        self.trade_history = []
        
        for i, date in enumerate(trading_dates):
            date_dt = pd.to_datetime(date)
            
            # Get stock selection for this date
            selected_stocks = self._get_stock_selection_for_date(stock_selections, date)
            
            if not selected_stocks:
                # No selection available, hold cash
                portfolio_results['dates'].append(date)
                portfolio_results['portfolio_value'].append(current_value)
                portfolio_results['cash'].append(current_cash)
                portfolio_results['positions'].append(current_positions.copy())
                portfolio_results['returns'].append(0.0)
                portfolio_results['cumulative_returns'].append(0.0 if i == 0 else portfolio_results['cumulative_returns'][-1])
                continue
            
            # Calculate current portfolio value
            portfolio_value = current_cash
            for symbol, shares in current_positions.items():
                if symbol in price_data and date_dt in price_data[symbol].index:
                    current_price = price_data[symbol].loc[date_dt, 'close']
                    portfolio_value += shares * current_price
            
            # Rebalance portfolio
            new_positions, new_cash, trades = self._rebalance_portfolio(
                selected_stocks, current_positions, current_cash, 
                portfolio_value, price_data, date_dt, params
            )
            
            # Record trades
            for trade in trades:
                trade['date'] = date
                self.trade_history.append(trade)
            
            # Update portfolio state
            current_positions = new_positions
            current_cash = new_cash
            current_value = portfolio_value
            
            # Calculate returns
            if i == 0:
                daily_return = 0.0
                cumulative_return = 0.0
            else:
                daily_return = (current_value - portfolio_results['portfolio_value'][-1]) / portfolio_results['portfolio_value'][-1]
                cumulative_return = (current_value - params['initial_capital']) / params['initial_capital']
            
            # Record results
            portfolio_results['dates'].append(date)
            portfolio_results['portfolio_value'].append(current_value)
            portfolio_results['cash'].append(current_cash)
            portfolio_results['positions'].append(current_positions.copy())
            portfolio_results['returns'].append(daily_return)
            portfolio_results['cumulative_returns'].append(cumulative_return)
        
        return portfolio_results
    
    def _get_stock_selection_for_date(self, stock_selections: Dict[str, List[str]], target_date: str) -> List[str]:
        """Get the most recent stock selection for a given date"""
        target_dt = pd.to_datetime(target_date)
        
        # Find the most recent selection date
        available_dates = [pd.to_datetime(date) for date in stock_selections.keys()]
        valid_dates = [date for date in available_dates if date <= target_dt]
        
        if not valid_dates:
            return []
        
        most_recent_date = max(valid_dates)
        return stock_selections[most_recent_date.strftime('%Y-%m-%d')]
    
    def _rebalance_portfolio(self, selected_stocks: List[str], current_positions: Dict,
                           current_cash: float, portfolio_value: float,
                           price_data: Dict[str, pd.DataFrame], date: pd.Timestamp,
                           params: Dict) -> Tuple[Dict, float, List[Dict]]:
        """Rebalance portfolio to target allocation"""
        new_positions = {}
        trades = []
        
        # Calculate target allocation
        n_stocks = len(selected_stocks)
        if n_stocks == 0:
            return current_positions, current_cash, trades
        
        target_weight = min(1.0 / n_stocks, params['max_position_size'])
        target_weight = max(target_weight, params['min_position_size'])
        
        # Calculate target values
        available_cash = current_cash
        
        # Sell positions not in new selection
        for symbol, shares in current_positions.items():
            if symbol not in selected_stocks and symbol in price_data and date in price_data[symbol].index:
                current_price = price_data[symbol].loc[date, 'close']
                sell_value = shares * current_price * (1 - params['transaction_cost'])
                available_cash += sell_value
                
                trades.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': shares,
                    'price': current_price,
                    'value': sell_value,
                    'transaction_cost': shares * current_price * params['transaction_cost']
                })
        
        # Buy new positions
        total_target_value = portfolio_value
        
        for symbol in selected_stocks:
            if symbol not in price_data or date not in price_data[symbol].index:
                continue
            
            current_price = price_data[symbol].loc[date, 'close']
            target_value = total_target_value * target_weight
            
            # Calculate shares to buy
            max_shares = int(target_value / current_price)
            transaction_cost = max_shares * current_price * params['transaction_cost']
            total_cost = max_shares * current_price + transaction_cost
            
            if total_cost <= available_cash and max_shares > 0:
                new_positions[symbol] = max_shares
                available_cash -= total_cost
                
                trades.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'shares': max_shares,
                    'price': current_price,
                    'value': max_shares * current_price,
                    'transaction_cost': transaction_cost
                })
        
        return new_positions, available_cash, trades
    
    def _calculate_benchmark_performance(self, price_data: Dict[str, pd.DataFrame],
                                       benchmark_symbol: str, trading_dates: List[str],
                                       params: Dict) -> Dict:
        """Calculate benchmark performance"""
        benchmark_results = {
            'dates': [],
            'values': [],
            'returns': [],
            'cumulative_returns': []
        }
        
        if benchmark_symbol not in price_data:
            # Create a simple market return if benchmark not available
            logger.warning(f"Benchmark {benchmark_symbol} not available, using average market return")
            for i, date in enumerate(trading_dates):
                benchmark_results['dates'].append(date)
                benchmark_results['values'].append(params['initial_capital'] * (1.08 ** (i / 252)))  # 8% annual return
                if i == 0:
                    benchmark_results['returns'].append(0.0)
                    benchmark_results['cumulative_returns'].append(0.0)
                else:
                    daily_return = 0.08 / 252  # Daily return approximation
                    benchmark_results['returns'].append(daily_return)
                    benchmark_results['cumulative_returns'].append((1.08 ** (i / 252)) - 1)
            return benchmark_results
        
        benchmark_data = price_data[benchmark_symbol]
        initial_price = None
        
        for i, date in enumerate(trading_dates):
            date_dt = pd.to_datetime(date)
            
            if date_dt not in benchmark_data.index:
                continue
            
            current_price = benchmark_data.loc[date_dt, 'close']
            
            if initial_price is None:
                initial_price = current_price
                benchmark_value = params['initial_capital']
                daily_return = 0.0
                cumulative_return = 0.0
            else:
                benchmark_value = params['initial_capital'] * (current_price / initial_price)
                if i > 0 and len(benchmark_results['values']) > 0:
                    daily_return = (benchmark_value - benchmark_results['values'][-1]) / benchmark_results['values'][-1]
                else:
                    daily_return = 0.0
                cumulative_return = (current_price / initial_price) - 1
            
            benchmark_results['dates'].append(date)
            benchmark_results['values'].append(benchmark_value)
            benchmark_results['returns'].append(daily_return)
            benchmark_results['cumulative_returns'].append(cumulative_return)
        
        return benchmark_results
    
    def _calculate_performance_metrics(self, portfolio_results: Dict, 
                                     benchmark_results: Dict, params: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        try:
            # Basic performance metrics
            if portfolio_results['portfolio_value']:
                initial_value = params['initial_capital']
                final_value = portfolio_results['portfolio_value'][-1]
                
                metrics['total_return'] = (final_value - initial_value) / initial_value
                metrics['annualized_return'] = self._annualize_return(metrics['total_return'], len(portfolio_results['dates']))
                
                # Volatility
                returns = np.array(portfolio_results['returns'])
                metrics['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
                
                # Sharpe Ratio
                excess_returns = returns - (params['risk_free_rate'] / 252)
                metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                
                # Sortino Ratio
                downside_returns = returns[returns < 0]
                downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
                metrics['sortino_ratio'] = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
                
                # Maximum Drawdown
                cumulative_returns = np.array(portfolio_results['cumulative_returns'])
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max)
                metrics['max_drawdown'] = np.min(drawdown)
                
                # Calmar Ratio
                metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
                
                # Win Rate
                winning_periods = len(returns[returns > 0])
                total_periods = len(returns)
                metrics['win_rate'] = winning_periods / total_periods if total_periods > 0 else 0
                
                # Average Win/Loss
                winning_returns = returns[returns > 0]
                losing_returns = returns[returns < 0]
                metrics['avg_win'] = np.mean(winning_returns) if len(winning_returns) > 0 else 0
                metrics['avg_loss'] = np.mean(losing_returns) if len(losing_returns) > 0 else 0
                metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
                
                # Benchmark comparison
                if benchmark_results['values']:
                    benchmark_total_return = (benchmark_results['values'][-1] - params['initial_capital']) / params['initial_capital']
                    metrics['alpha'] = metrics['total_return'] - benchmark_total_return
                    
                    # Beta calculation
                    if len(benchmark_results['returns']) == len(portfolio_results['returns']):
                        portfolio_returns = np.array(portfolio_results['returns'])
                        benchmark_returns = np.array(benchmark_results['returns'])
                        
                        if np.std(benchmark_returns) > 0:
                            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                            benchmark_variance = np.var(benchmark_returns)
                            metrics['beta'] = covariance / benchmark_variance
                        else:
                            metrics['beta'] = 0
                    
                    # Information Ratio
                    if len(benchmark_results['returns']) == len(portfolio_results['returns']):
                        active_returns = portfolio_returns - benchmark_returns
                        tracking_error = np.std(active_returns)
                        metrics['information_ratio'] = np.mean(active_returns) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
                
                # Risk-adjusted metrics
                metrics['return_to_risk'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _annualize_return(self, total_return: float, num_periods: int) -> float:
        """Convert total return to annualized return"""
        if num_periods <= 0:
            return 0
        
        years = num_periods / 252  # Assuming 252 trading days per year
        if years <= 0:
            return 0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _analyze_trades(self, trade_history: List[Dict], price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze trading activity and performance"""
        analysis = {
            'total_trades': len(trade_history),
            'buy_trades': 0,
            'sell_trades': 0,
            'total_transaction_costs': 0,
            'symbols_traded': set(),
            'trade_frequency': {},
            'largest_trades': []
        }
        
        try:
            for trade in trade_history:
                if trade['action'] == 'buy':
                    analysis['buy_trades'] += 1
                else:
                    analysis['sell_trades'] += 1
                
                analysis['total_transaction_costs'] += trade.get('transaction_cost', 0)
                analysis['symbols_traded'].add(trade['symbol'])
                
                # Track trade frequency by symbol
                symbol = trade['symbol']
                if symbol not in analysis['trade_frequency']:
                    analysis['trade_frequency'][symbol] = 0
                analysis['trade_frequency'][symbol] += 1
            
            # Convert set to list for JSON serialization
            analysis['symbols_traded'] = list(analysis['symbols_traded'])
            analysis['unique_symbols'] = len(analysis['symbols_traded'])
            
            # Find largest trades by value
            sorted_trades = sorted(trade_history, key=lambda x: x.get('value', 0), reverse=True)
            analysis['largest_trades'] = sorted_trades[:10]
            
            # Average trade size
            if trade_history:
                trade_values = [trade.get('value', 0) for trade in trade_history]
                analysis['avg_trade_size'] = np.mean(trade_values)
                analysis['median_trade_size'] = np.median(trade_values)
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {str(e)}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _perform_risk_analysis(self, portfolio_results: Dict, benchmark_results: Dict, params: Dict) -> Dict:
        """Perform comprehensive risk analysis"""
        risk_analysis = {}
        
        try:
            returns = np.array(portfolio_results['returns'])
            
            # Value at Risk (VaR)
            risk_analysis['var_95'] = np.percentile(returns, 5)  # 5th percentile
            risk_analysis['var_99'] = np.percentile(returns, 1)  # 1st percentile
            
            # Conditional Value at Risk (CVaR)
            var_95_threshold = risk_analysis['var_95']
            tail_returns = returns[returns <= var_95_threshold]
            risk_analysis['cvar_95'] = np.mean(tail_returns) if len(tail_returns) > 0 else 0
            
            # Skewness and Kurtosis
            risk_analysis['skewness'] = self._calculate_skewness(returns)
            risk_analysis['kurtosis'] = self._calculate_kurtosis(returns)
            
            # Downside Risk
            downside_returns = returns[returns < 0]
            risk_analysis['downside_frequency'] = len(downside_returns) / len(returns) if len(returns) > 0 else 0
            risk_analysis['downside_deviation'] = np.std(downside_returns) if len(downside_returns) > 0 else 0
            
            # Ulcer Index (measure of downside risk)
            cumulative_returns = np.array(portfolio_results['cumulative_returns'])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) * 100  # Convert to percentage
            risk_analysis['ulcer_index'] = np.sqrt(np.mean(drawdowns ** 2))
            
            # Risk-Return Scatter Data
            if len(returns) > 30:  # Need sufficient data
                # Rolling 30-day risk-return
                rolling_returns = []
                rolling_volatility = []
                
                for i in range(30, len(returns)):
                    period_returns = returns[i-30:i]
                    rolling_returns.append(np.mean(period_returns) * 252)  # Annualized
                    rolling_volatility.append(np.std(period_returns) * np.sqrt(252))  # Annualized
                
                risk_analysis['rolling_risk_return'] = {
                    'returns': rolling_returns,
                    'volatility': rolling_volatility
                }
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            risk_analysis['error'] = str(e)
        
        return risk_analysis
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3  # Excess kurtosis
        return kurtosis
    
    def generate_backtest_report(self, backtest_results: Dict) -> Dict:
        """Generate a comprehensive backtest report"""
        report = {
            'executive_summary': {},
            'performance_summary': {},
            'risk_summary': {},
            'trade_summary': {},
            'recommendations': []
        }
        
        try:
            # Executive Summary
            if 'performance_metrics' in backtest_results:
                metrics = backtest_results['performance_metrics']
                report['executive_summary'] = {
                    'total_return': f"{metrics.get('total_return', 0):.2%}",
                    'annualized_return': f"{metrics.get('annualized_return', 0):.2%}",
                    'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                    'max_drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                    'win_rate': f"{metrics.get('win_rate', 0):.2%}"
                }
            
            # Performance Summary
            report['performance_summary'] = backtest_results.get('performance_metrics', {})
            
            # Risk Summary
            report['risk_summary'] = backtest_results.get('risk_analysis', {})
            
            # Trade Summary
            report['trade_summary'] = backtest_results.get('trade_analysis', {})
            
            # Generate recommendations
            recommendations = self._generate_recommendations(backtest_results)
            report['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error generating backtest report: {str(e)}")
            report['error'] = str(e)
        
        return report
    
    def _generate_recommendations(self, backtest_results: Dict) -> List[str]:
        """Generate recommendations based on backtest results"""
        recommendations = []
        
        try:
            metrics = backtest_results.get('performance_metrics', {})
            risk_analysis = backtest_results.get('risk_analysis', {})
            
            # Performance recommendations
            if metrics.get('sharpe_ratio', 0) < 1.0:
                recommendations.append("Consider improving risk-adjusted returns. Sharpe ratio below 1.0 indicates suboptimal risk-return profile.")
            
            if metrics.get('max_drawdown', 0) < -0.2:  # More than 20% drawdown
                recommendations.append("High maximum drawdown detected. Consider implementing stop-loss mechanisms or position sizing rules.")
            
            if metrics.get('win_rate', 0) < 0.4:  # Less than 40% win rate
                recommendations.append("Low win rate suggests need for better stock selection criteria or timing.")
            
            # Risk recommendations
            if risk_analysis.get('downside_frequency', 0) > 0.6:  # More than 60% negative periods
                recommendations.append("High frequency of negative returns. Consider more defensive positioning or better market timing.")
            
            if abs(risk_analysis.get('skewness', 0)) > 1.0:
                recommendations.append("High skewness in returns distribution. Consider diversification to reduce tail risk.")
            
            # Trade analysis recommendations
            trade_analysis = backtest_results.get('trade_analysis', {})
            if trade_analysis.get('total_transaction_costs', 0) > metrics.get('total_return', 0) * 0.1:
                recommendations.append("High transaction costs relative to returns. Consider reducing trading frequency.")
            
            if not recommendations:
                recommendations.append("Strategy shows reasonable performance characteristics. Continue monitoring and consider minor optimizations.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations

