"""
Portfolio Simulation Module for advanced portfolio management and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

class PortfolioSimulator:
    """
    Advanced portfolio simulator with multiple strategies and optimization
    """
    
    def __init__(self):
        """Initialize the portfolio simulator"""
        self.backtest_engine = BacktestEngine()
        self.simulation_results = {}
        self.optimization_results = {}
        
        # Portfolio strategies
        self.strategies = {
            'equal_weight': self._equal_weight_strategy,
            'market_cap_weight': self._market_cap_weight_strategy,
            'risk_parity': self._risk_parity_strategy,
            'momentum_weight': self._momentum_weight_strategy,
            'fundamental_weight': self._fundamental_weight_strategy
        }
    
    def run_multi_strategy_simulation(self, stock_selections: Dict[str, List[str]],
                                    price_data: Dict[str, pd.DataFrame],
                                    fundamental_data: Dict[str, Dict],
                                    start_date: str, end_date: str,
                                    strategies: List[str] = None) -> Dict:
        """
        Run simulation with multiple portfolio strategies
        
        Args:
            stock_selections: Dictionary with dates as keys and selected stocks as values
            price_data: Dictionary of price DataFrames by symbol
            fundamental_data: Dictionary of fundamental data by symbol
            start_date: Simulation start date
            end_date: Simulation end date
            strategies: List of strategies to test
        
        Returns:
            Dictionary with results for each strategy
        """
        if strategies is None:
            strategies = list(self.strategies.keys())
        
        results = {
            'simulation_parameters': {
                'start_date': start_date,
                'end_date': end_date,
                'strategies_tested': strategies
            },
            'strategy_results': {},
            'comparison': {},
            'best_strategy': None
        }
        
        try:
            logger.info(f"Running multi-strategy simulation with {len(strategies)} strategies")
            
            strategy_performance = {}
            
            for strategy_name in strategies:
                if strategy_name not in self.strategies:
                    logger.warning(f"Strategy {strategy_name} not available")
                    continue
                
                logger.info(f"Testing strategy: {strategy_name}")
                
                # Generate weighted selections for this strategy
                weighted_selections = self._apply_strategy_weights(
                    stock_selections, price_data, fundamental_data, strategy_name
                )
                
                # Run backtest with this strategy
                backtest_results = self.backtest_engine.run_backtest(
                    weighted_selections, price_data, start_date, end_date,
                    {'strategy_name': strategy_name}
                )
                
                results['strategy_results'][strategy_name] = backtest_results
                
                # Extract key performance metrics
                if 'performance_metrics' in backtest_results:
                    metrics = backtest_results['performance_metrics']
                    strategy_performance[strategy_name] = {
                        'total_return': metrics.get('total_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'volatility': metrics.get('volatility', 0)
                    }
            
            # Compare strategies
            comparison = self._compare_strategies(strategy_performance)
            results['comparison'] = comparison
            
            # Identify best strategy
            if strategy_performance:
                # Use Sharpe ratio as primary criterion
                best_strategy = max(strategy_performance.items(), 
                                  key=lambda x: x[1]['sharpe_ratio'])
                results['best_strategy'] = {
                    'name': best_strategy[0],
                    'metrics': best_strategy[1]
                }
            
            self.simulation_results = results
            
        except Exception as e:
            logger.error(f"Error in multi-strategy simulation: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _apply_strategy_weights(self, stock_selections: Dict[str, List[str]],
                              price_data: Dict[str, pd.DataFrame],
                              fundamental_data: Dict[str, Dict],
                              strategy_name: str) -> Dict[str, List[Tuple[str, float]]]:
        """Apply weighting strategy to stock selections"""
        weighted_selections = {}
        
        for date, stocks in stock_selections.items():
            if not stocks:
                weighted_selections[date] = []
                continue
            
            # Apply strategy-specific weighting
            weights = self.strategies[strategy_name](stocks, price_data, fundamental_data, date)
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {stock: weight/total_weight for stock, weight in weights.items()}
            else:
                # Equal weights if strategy fails
                normalized_weights = {stock: 1.0/len(stocks) for stock in stocks}
            
            # Convert to list of tuples
            weighted_selections[date] = [(stock, weight) for stock, weight in normalized_weights.items()]
        
        return weighted_selections
    
    def _equal_weight_strategy(self, stocks: List[str], price_data: Dict[str, pd.DataFrame],
                              fundamental_data: Dict[str, Dict], date: str) -> Dict[str, float]:
        """Equal weight strategy"""
        if not stocks:
            return {}
        
        weight = 1.0 / len(stocks)
        return {stock: weight for stock in stocks}
    
    def _market_cap_weight_strategy(self, stocks: List[str], price_data: Dict[str, pd.DataFrame],
                                   fundamental_data: Dict[str, Dict], date: str) -> Dict[str, float]:
        """Market cap weighted strategy"""
        weights = {}
        market_caps = {}
        
        date_dt = pd.to_datetime(date)
        
        # Calculate market caps (approximated by price * volume)
        for stock in stocks:
            if stock in price_data and date_dt in price_data[stock].index:
                price = price_data[stock].loc[date_dt, 'close']
                volume = price_data[stock].loc[date_dt, 'volume'] if 'volume' in price_data[stock].columns else 1000000
                market_caps[stock] = price * volume
            else:
                market_caps[stock] = 1000000  # Default value
        
        # Calculate weights based on market cap
        total_market_cap = sum(market_caps.values())
        if total_market_cap > 0:
            weights = {stock: market_cap / total_market_cap for stock, market_cap in market_caps.items()}
        else:
            weights = self._equal_weight_strategy(stocks, price_data, fundamental_data, date)
        
        return weights
    
    def _risk_parity_strategy(self, stocks: List[str], price_data: Dict[str, pd.DataFrame],
                             fundamental_data: Dict[str, Dict], date: str) -> Dict[str, float]:
        """Risk parity strategy (inverse volatility weighting)"""
        weights = {}
        volatilities = {}
        
        date_dt = pd.to_datetime(date)
        
        # Calculate historical volatilities
        for stock in stocks:
            if stock in price_data:
                # Get 30-day historical data
                stock_data = price_data[stock]
                end_date = date_dt
                start_date = end_date - timedelta(days=30)
                
                historical_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
                
                if len(historical_data) > 5:
                    returns = historical_data['close'].pct_change().dropna()
                    volatility = returns.std() if len(returns) > 1 else 0.1
                    volatilities[stock] = max(volatility, 0.001)  # Minimum volatility
                else:
                    volatilities[stock] = 0.1  # Default volatility
            else:
                volatilities[stock] = 0.1
        
        # Calculate inverse volatility weights
        inverse_vols = {stock: 1.0 / vol for stock, vol in volatilities.items()}
        total_inverse_vol = sum(inverse_vols.values())
        
        if total_inverse_vol > 0:
            weights = {stock: inv_vol / total_inverse_vol for stock, inv_vol in inverse_vols.items()}
        else:
            weights = self._equal_weight_strategy(stocks, price_data, fundamental_data, date)
        
        return weights
    
    def _momentum_weight_strategy(self, stocks: List[str], price_data: Dict[str, pd.DataFrame],
                                 fundamental_data: Dict[str, Dict], date: str) -> Dict[str, float]:
        """Momentum-based weighting strategy"""
        weights = {}
        momentum_scores = {}
        
        date_dt = pd.to_datetime(date)
        
        # Calculate momentum scores
        for stock in stocks:
            if stock in price_data:
                stock_data = price_data[stock]
                
                # Calculate 1-month and 3-month momentum
                if date_dt in stock_data.index:
                    current_price = stock_data.loc[date_dt, 'close']
                    
                    # 1-month momentum
                    month_ago = date_dt - timedelta(days=21)
                    month_price = None
                    for i in range(30):  # Look back up to 30 days
                        check_date = month_ago - timedelta(days=i)
                        if check_date in stock_data.index:
                            month_price = stock_data.loc[check_date, 'close']
                            break
                    
                    # 3-month momentum
                    quarter_ago = date_dt - timedelta(days=63)
                    quarter_price = None
                    for i in range(30):  # Look back up to 30 days
                        check_date = quarter_ago - timedelta(days=i)
                        if check_date in stock_data.index:
                            quarter_price = stock_data.loc[check_date, 'close']
                            break
                    
                    # Calculate momentum score
                    momentum_1m = (current_price / month_price - 1) if month_price else 0
                    momentum_3m = (current_price / quarter_price - 1) if quarter_price else 0
                    
                    # Combined momentum score
                    momentum_scores[stock] = momentum_1m * 0.6 + momentum_3m * 0.4
                else:
                    momentum_scores[stock] = 0
            else:
                momentum_scores[stock] = 0
        
        # Convert to positive weights (add minimum to handle negative momentum)
        min_momentum = min(momentum_scores.values()) if momentum_scores else 0
        adjusted_scores = {stock: score - min_momentum + 0.1 for stock, score in momentum_scores.items()}
        
        # Calculate weights
        total_score = sum(adjusted_scores.values())
        if total_score > 0:
            weights = {stock: score / total_score for stock, score in adjusted_scores.items()}
        else:
            weights = self._equal_weight_strategy(stocks, price_data, fundamental_data, date)
        
        return weights
    
    def _fundamental_weight_strategy(self, stocks: List[str], price_data: Dict[str, pd.DataFrame],
                                   fundamental_data: Dict[str, Dict], date: str) -> Dict[str, float]:
        """Fundamental-based weighting strategy"""
        weights = {}
        fundamental_scores = {}
        
        # Calculate fundamental scores
        for stock in stocks:
            score = 0
            
            if stock in fundamental_data:
                fund_data = fundamental_data[stock]
                
                # ROE component
                roe = fund_data.get('roe', 0)
                score += min(roe * 10, 2.0)  # Cap at 2.0
                
                # Revenue growth component
                revenue_growth = fund_data.get('revenue_growth_5y', 0)
                score += min(revenue_growth * 5, 1.0)  # Cap at 1.0
                
                # Profitability component
                net_margin = fund_data.get('net_margin', 0)
                score += min(net_margin * 10, 1.0)  # Cap at 1.0
                
                # Financial health component
                current_ratio = fund_data.get('current_ratio', 1)
                if current_ratio > 1:
                    score += min((current_ratio - 1) * 0.5, 0.5)
                
                # Debt component (lower is better)
                debt_ratio = fund_data.get('debt_to_ebitda', 0)
                if debt_ratio < 3:
                    score += (3 - debt_ratio) * 0.1
            
            fundamental_scores[stock] = max(score, 0.1)  # Minimum score
        
        # Calculate weights
        total_score = sum(fundamental_scores.values())
        if total_score > 0:
            weights = {stock: score / total_score for stock, score in fundamental_scores.items()}
        else:
            weights = self._equal_weight_strategy(stocks, price_data, fundamental_data, date)
        
        return weights
    
    def _compare_strategies(self, strategy_performance: Dict[str, Dict]) -> Dict:
        """Compare performance across strategies"""
        comparison = {
            'performance_ranking': {},
            'risk_ranking': {},
            'risk_adjusted_ranking': {},
            'summary_table': []
        }
        
        try:
            # Performance ranking (by total return)
            performance_sorted = sorted(strategy_performance.items(), 
                                      key=lambda x: x[1]['total_return'], reverse=True)
            comparison['performance_ranking'] = {
                strategy: rank + 1 for rank, (strategy, _) in enumerate(performance_sorted)
            }
            
            # Risk ranking (by max drawdown, lower is better)
            risk_sorted = sorted(strategy_performance.items(), 
                               key=lambda x: abs(x[1]['max_drawdown']))
            comparison['risk_ranking'] = {
                strategy: rank + 1 for rank, (strategy, _) in enumerate(risk_sorted)
            }
            
            # Risk-adjusted ranking (by Sharpe ratio)
            sharpe_sorted = sorted(strategy_performance.items(), 
                                 key=lambda x: x[1]['sharpe_ratio'], reverse=True)
            comparison['risk_adjusted_ranking'] = {
                strategy: rank + 1 for rank, (strategy, _) in enumerate(sharpe_sorted)
            }
            
            # Summary table
            for strategy, metrics in strategy_performance.items():
                comparison['summary_table'].append({
                    'strategy': strategy,
                    'total_return': f"{metrics['total_return']:.2%}",
                    'sharpe_ratio': f"{metrics['sharpe_ratio']:.2f}",
                    'max_drawdown': f"{metrics['max_drawdown']:.2%}",
                    'volatility': f"{metrics['volatility']:.2%}",
                    'performance_rank': comparison['performance_ranking'][strategy],
                    'risk_rank': comparison['risk_ranking'][strategy],
                    'sharpe_rank': comparison['risk_adjusted_ranking'][strategy]
                })
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {str(e)}")
            comparison['error'] = str(e)
        
        return comparison
    
    def optimize_portfolio_allocation(self, stock_selections: Dict[str, List[str]],
                                    price_data: Dict[str, pd.DataFrame],
                                    optimization_objective: str = 'sharpe') -> Dict:
        """
        Optimize portfolio allocation using different objectives
        
        Args:
            stock_selections: Stock selections over time
            price_data: Price data
            optimization_objective: 'sharpe', 'return', 'risk', 'calmar'
        
        Returns:
            Dictionary with optimization results
        """
        optimization_results = {
            'objective': optimization_objective,
            'optimal_weights': {},
            'expected_metrics': {},
            'efficient_frontier': []
        }
        
        try:
            # For simplicity, we'll optimize the most recent selection
            if not stock_selections:
                return optimization_results
            
            latest_date = max(stock_selections.keys())
            stocks = stock_selections[latest_date]
            
            if len(stocks) < 2:
                return optimization_results
            
            # Calculate historical returns and covariance
            returns_data = self._calculate_returns_matrix(stocks, price_data, latest_date)
            
            if returns_data.empty:
                return optimization_results
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean()
            cov_matrix = returns_data.cov()
            
            # Optimize based on objective
            if optimization_objective == 'sharpe':
                optimal_weights = self._optimize_sharpe_ratio(expected_returns, cov_matrix)
            elif optimization_objective == 'return':
                optimal_weights = self._optimize_return(expected_returns, cov_matrix)
            elif optimization_objective == 'risk':
                optimal_weights = self._minimize_risk(expected_returns, cov_matrix)
            else:
                # Default to equal weights
                optimal_weights = {stock: 1.0/len(stocks) for stock in stocks}
            
            optimization_results['optimal_weights'] = optimal_weights
            
            # Calculate expected metrics for optimal portfolio
            expected_metrics = self._calculate_expected_metrics(optimal_weights, expected_returns, cov_matrix)
            optimization_results['expected_metrics'] = expected_metrics
            
            # Generate efficient frontier points
            efficient_frontier = self._generate_efficient_frontier(expected_returns, cov_matrix, n_points=10)
            optimization_results['efficient_frontier'] = efficient_frontier
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _calculate_returns_matrix(self, stocks: List[str], price_data: Dict[str, pd.DataFrame], 
                                 end_date: str, lookback_days: int = 252) -> pd.DataFrame:
        """Calculate returns matrix for portfolio optimization"""
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - timedelta(days=lookback_days)
        
        returns_dict = {}
        
        for stock in stocks:
            if stock not in price_data:
                continue
            
            stock_data = price_data[stock]
            period_data = stock_data[(stock_data.index >= start_dt) & (stock_data.index <= end_dt)]
            
            if len(period_data) > 10:
                returns = period_data['close'].pct_change().dropna()
                returns_dict[stock] = returns
        
        if not returns_dict:
            return pd.DataFrame()
        
        # Align all return series
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def _optimize_sharpe_ratio(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for maximum Sharpe ratio (simplified)"""
        # Simplified optimization - in practice would use scipy.optimize
        n_assets = len(expected_returns)
        
        # Start with equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Simple gradient ascent for Sharpe ratio
        for _ in range(100):  # Simple iterations
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std > 0:
                sharpe_ratio = portfolio_return / portfolio_std
                
                # Simple gradient approximation
                gradient = np.zeros(n_assets)
                for i in range(n_assets):
                    weights_plus = weights.copy()
                    weights_plus[i] += 0.001
                    weights_plus = weights_plus / weights_plus.sum()  # Normalize
                    
                    portfolio_return_plus = np.dot(weights_plus, expected_returns)
                    portfolio_variance_plus = np.dot(weights_plus.T, np.dot(cov_matrix, weights_plus))
                    portfolio_std_plus = np.sqrt(portfolio_variance_plus)
                    
                    if portfolio_std_plus > 0:
                        sharpe_plus = portfolio_return_plus / portfolio_std_plus
                        gradient[i] = (sharpe_plus - sharpe_ratio) / 0.001
                
                # Update weights
                weights += 0.01 * gradient
                weights = np.maximum(weights, 0)  # No short selling
                weights = weights / weights.sum()  # Normalize
        
        return {stock: weight for stock, weight in zip(expected_returns.index, weights)}
    
    def _optimize_return(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for maximum return"""
        # Simply weight by expected returns
        positive_returns = expected_returns.copy()
        positive_returns[positive_returns < 0] = 0  # No negative weights
        
        if positive_returns.sum() > 0:
            weights = positive_returns / positive_returns.sum()
        else:
            weights = pd.Series(1.0/len(expected_returns), index=expected_returns.index)
        
        return weights.to_dict()
    
    def _minimize_risk(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Minimize portfolio risk"""
        # Inverse volatility weighting
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        return {stock: weight for stock, weight in zip(expected_returns.index, weights)}
    
    def _calculate_expected_metrics(self, weights: Dict[str, float], 
                                  expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict:
        """Calculate expected portfolio metrics"""
        weight_array = np.array([weights.get(stock, 0) for stock in expected_returns.index])
        
        expected_return = np.dot(weight_array, expected_returns) * 252  # Annualized
        expected_variance = np.dot(weight_array.T, np.dot(cov_matrix, weight_array)) * 252  # Annualized
        expected_volatility = np.sqrt(expected_variance)
        
        expected_sharpe = expected_return / expected_volatility if expected_volatility > 0 else 0
        
        return {
            'expected_return': expected_return,
            'expected_volatility': expected_volatility,
            'expected_sharpe': expected_sharpe
        }
    
    def _generate_efficient_frontier(self, expected_returns: pd.Series, 
                                   cov_matrix: pd.DataFrame, n_points: int = 10) -> List[Dict]:
        """Generate efficient frontier points"""
        frontier_points = []
        
        try:
            # Generate range of target returns
            min_return = expected_returns.min() * 252
            max_return = expected_returns.max() * 252
            target_returns = np.linspace(min_return, max_return, n_points)
            
            for target_return in target_returns:
                # For each target return, find minimum variance portfolio
                # Simplified approach - equal weighting with bias toward higher return stocks
                weights = expected_returns / expected_returns.sum()
                
                portfolio_return = np.dot(weights, expected_returns) * 252
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights)) * 252
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                frontier_points.append({
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                })
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {str(e)}")
        
        return frontier_points
    
    def generate_simulation_report(self, simulation_results: Dict) -> Dict:
        """Generate comprehensive simulation report"""
        report = {
            'executive_summary': {},
            'strategy_comparison': {},
            'recommendations': [],
            'detailed_analysis': {}
        }
        
        try:
            # Executive summary
            if 'best_strategy' in simulation_results:
                best = simulation_results['best_strategy']
                report['executive_summary'] = {
                    'best_strategy': best['name'],
                    'best_return': f"{best['metrics']['total_return']:.2%}",
                    'best_sharpe': f"{best['metrics']['sharpe_ratio']:.2f}",
                    'strategies_tested': len(simulation_results.get('strategy_results', {}))
                }
            
            # Strategy comparison
            if 'comparison' in simulation_results:
                report['strategy_comparison'] = simulation_results['comparison']
            
            # Generate recommendations
            recommendations = self._generate_simulation_recommendations(simulation_results)
            report['recommendations'] = recommendations
            
            # Detailed analysis
            report['detailed_analysis'] = simulation_results.get('strategy_results', {})
            
        except Exception as e:
            logger.error(f"Error generating simulation report: {str(e)}")
            report['error'] = str(e)
        
        return report
    
    def _generate_simulation_recommendations(self, simulation_results: Dict) -> List[str]:
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        try:
            if 'comparison' in simulation_results and 'summary_table' in simulation_results['comparison']:
                summary = simulation_results['comparison']['summary_table']
                
                # Find consistently good performers
                consistent_performers = [
                    s for s in summary 
                    if s['performance_rank'] <= 2 and s['sharpe_rank'] <= 2
                ]
                
                if consistent_performers:
                    best_consistent = consistent_performers[0]['strategy']
                    recommendations.append(f"Strategy '{best_consistent}' shows consistent performance across multiple metrics.")
                
                # Risk analysis
                low_risk_strategies = [s for s in summary if s['risk_rank'] <= 2]
                if low_risk_strategies:
                    recommendations.append(f"For risk-conscious investors, consider '{low_risk_strategies[0]['strategy']}' strategy.")
                
                # High return strategies
                high_return_strategies = [s for s in summary if s['performance_rank'] == 1]
                if high_return_strategies:
                    recommendations.append(f"For maximum returns, '{high_return_strategies[0]['strategy']}' strategy performed best.")
                
                if not recommendations:
                    recommendations.append("Consider diversifying across multiple strategies to balance risk and return.")
            
        except Exception as e:
            logger.error(f"Error generating simulation recommendations: {str(e)}")
        
        return recommendations

