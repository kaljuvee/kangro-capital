"""
Advanced Backtesting Module with Machine Learning Training and Superior Portfolio Detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf

from .backtest_engine import BacktestEngine
from .portfolio_simulator import PortfolioSimulator

logger = logging.getLogger(__name__)

class AdvancedBacktester:
    """
    Advanced backtesting system with ML training and superior portfolio detection
    """
    
    def __init__(self):
        """Initialize the advanced backtester"""
        self.backtest_engine = BacktestEngine()
        self.portfolio_simulator = PortfolioSimulator()
        self.ml_models = {}
        self.training_results = {}
        self.benchmark_data = {}
        
        # Benchmark symbols for comparison
        self.benchmarks = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'IWM': 'Russell 2000',
            'VTI': 'Total Stock Market',
            'EFA': 'International Developed',
            'VWO': 'Emerging Markets',
            'BND': 'Total Bond Market',
            'GLD': 'Gold',
            'VNQ': 'Real Estate'
        }
        
        # Performance thresholds for "superior" classification
        self.superior_thresholds = {
            'min_sharpe_ratio': 1.0,
            'min_annual_return': 0.08,  # 8%
            'max_drawdown': -0.15,  # -15%
            'min_win_rate': 0.55,  # 55%
            'min_alpha_vs_spy': 0.02,  # 2% alpha
            'min_information_ratio': 0.5
        }
    
    def train_portfolio_predictor(self, historical_selections: Dict[str, List[str]],
                                price_data: Dict[str, pd.DataFrame],
                                fundamental_data: Dict[str, Dict],
                                training_period_months: int = 24) -> Dict:
        """
        Train ML models to predict portfolio performance
        
        Args:
            historical_selections: Historical stock selections
            price_data: Price data for all stocks
            fundamental_data: Fundamental data for feature engineering
            training_period_months: Number of months for training data
        
        Returns:
            Training results and model performance
        """
        training_results = {
            'training_period_months': training_period_months,
            'models_trained': [],
            'feature_importance': {},
            'model_performance': {},
            'predictions': {},
            'training_summary': {}
        }
        
        try:
            logger.info(f"Training portfolio predictor with {training_period_months} months of data")
            
            # Prepare training data
            training_data = self._prepare_training_data(
                historical_selections, price_data, fundamental_data, training_period_months
            )
            
            if training_data.empty:
                training_results['error'] = "Insufficient training data"
                return training_results
            
            # Split features and targets
            feature_columns = [col for col in training_data.columns if col.startswith('feature_')]
            target_columns = ['future_return', 'future_sharpe', 'future_max_drawdown']
            
            X = training_data[feature_columns]
            
            # Train models for different targets
            models_trained = []
            
            for target in target_columns:
                if target not in training_data.columns:
                    continue
                
                y = training_data[target]
                
                # Remove NaN values
                valid_mask = ~(X.isna().any(axis=1) | y.isna())
                X_clean = X[valid_mask]
                y_clean = y[valid_mask]
                
                if len(X_clean) < 10:  # Need minimum samples
                    continue
                
                # Train multiple models
                models = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'linear_regression': LinearRegression()
                }
                
                model_performance = {}
                
                for model_name, model in models.items():
                    try:
                        # Split train/test
                        split_idx = int(len(X_clean) * 0.8)
                        X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
                        y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
                        
                        # Scale features for linear models
                        if model_name == 'linear_regression':
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        # Calculate performance metrics
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        model_performance[model_name] = {
                            'mse': mse,
                            'r2': r2,
                            'rmse': np.sqrt(mse)
                        }
                        
                        # Store model
                        model_key = f"{target}_{model_name}"
                        self.ml_models[model_key] = {
                            'model': model,
                            'scaler': scaler if model_name == 'linear_regression' else None,
                            'features': feature_columns,
                            'target': target,
                            'performance': model_performance[model_name]
                        }
                        
                        # Feature importance for tree-based models
                        if hasattr(model, 'feature_importances_'):
                            importance_dict = dict(zip(feature_columns, model.feature_importances_))
                            training_results['feature_importance'][model_key] = importance_dict
                        
                        models_trained.append(model_key)
                        
                    except Exception as e:
                        logger.error(f"Error training {model_name} for {target}: {str(e)}")
                
                training_results['model_performance'][target] = model_performance
            
            training_results['models_trained'] = models_trained
            training_results['training_summary'] = {
                'total_samples': len(training_data),
                'features_used': len(feature_columns),
                'models_successful': len(models_trained),
                'best_models': self._identify_best_models(training_results['model_performance'])
            }
            
            self.training_results = training_results
            
        except Exception as e:
            logger.error(f"Error in portfolio predictor training: {str(e)}")
            training_results['error'] = str(e)
        
        return training_results
    
    def _prepare_training_data(self, historical_selections: Dict[str, List[str]],
                              price_data: Dict[str, pd.DataFrame],
                              fundamental_data: Dict[str, Dict],
                              training_period_months: int) -> pd.DataFrame:
        """Prepare training data with features and targets"""
        training_data = []
        
        # Sort dates
        sorted_dates = sorted(historical_selections.keys())
        
        for i, date in enumerate(sorted_dates[:-3]):  # Leave room for future returns
            try:
                stocks = historical_selections[date]
                if not stocks:
                    continue
                
                # Calculate features for this portfolio
                features = self._calculate_portfolio_features(stocks, price_data, fundamental_data, date)
                
                # Calculate future performance (3 months ahead)
                future_date_idx = min(i + 3, len(sorted_dates) - 1)
                future_date = sorted_dates[future_date_idx]
                
                future_performance = self._calculate_future_performance(
                    stocks, price_data, date, future_date
                )
                
                # Combine features and targets
                row_data = {**features, **future_performance, 'date': date}
                training_data.append(row_data)
                
            except Exception as e:
                logger.error(f"Error preparing training data for {date}: {str(e)}")
                continue
        
        return pd.DataFrame(training_data)
    
    def _calculate_portfolio_features(self, stocks: List[str], 
                                    price_data: Dict[str, pd.DataFrame],
                                    fundamental_data: Dict[str, Dict],
                                    date: str) -> Dict:
        """Calculate features for portfolio at given date"""
        features = {}
        date_dt = pd.to_datetime(date)
        
        try:
            # Portfolio size
            features['feature_portfolio_size'] = len(stocks)
            
            # Fundamental features (averages across portfolio)
            fund_metrics = ['roe', 'current_ratio', 'gross_margin', 'net_margin', 
                          'revenue_growth_5y', 'debt_to_ebitda']
            
            for metric in fund_metrics:
                values = []
                for stock in stocks:
                    if stock in fundamental_data and metric in fundamental_data[stock]:
                        values.append(fundamental_data[stock][metric])
                
                features[f'feature_avg_{metric}'] = np.mean(values) if values else 0
                features[f'feature_std_{metric}'] = np.std(values) if len(values) > 1 else 0
            
            # Technical features
            momentum_1m = []
            momentum_3m = []
            volatilities = []
            
            for stock in stocks:
                if stock not in price_data:
                    continue
                
                stock_data = price_data[stock]
                
                # Get price at date
                if date_dt not in stock_data.index:
                    continue
                
                current_price = stock_data.loc[date_dt, 'close']
                
                # 1-month momentum
                month_ago = date_dt - timedelta(days=21)
                month_price = self._get_closest_price(stock_data, month_ago)
                if month_price:
                    momentum_1m.append((current_price / month_price) - 1)
                
                # 3-month momentum
                quarter_ago = date_dt - timedelta(days=63)
                quarter_price = self._get_closest_price(stock_data, quarter_ago)
                if quarter_price:
                    momentum_3m.append((current_price / quarter_price) - 1)
                
                # Volatility (30-day)
                start_vol = date_dt - timedelta(days=30)
                vol_data = stock_data[(stock_data.index >= start_vol) & (stock_data.index <= date_dt)]
                if len(vol_data) > 5:
                    returns = vol_data['close'].pct_change().dropna()
                    volatilities.append(returns.std())
            
            features['feature_avg_momentum_1m'] = np.mean(momentum_1m) if momentum_1m else 0
            features['feature_avg_momentum_3m'] = np.mean(momentum_3m) if momentum_3m else 0
            features['feature_avg_volatility'] = np.mean(volatilities) if volatilities else 0
            features['feature_portfolio_momentum_dispersion'] = np.std(momentum_1m) if len(momentum_1m) > 1 else 0
            
            # Market environment features
            market_features = self._calculate_market_features(date_dt, price_data)
            features.update(market_features)
            
            # Sector diversification (simplified)
            features['feature_diversification_score'] = min(len(stocks) / 10, 1.0)  # Normalized by 10 stocks
            
        except Exception as e:
            logger.error(f"Error calculating portfolio features: {str(e)}")
        
        return features
    
    def _get_closest_price(self, stock_data: pd.DataFrame, target_date: pd.Timestamp) -> Optional[float]:
        """Get closest available price to target date"""
        try:
            # Look for exact match first
            if target_date in stock_data.index:
                return stock_data.loc[target_date, 'close']
            
            # Look within 10 days
            for i in range(10):
                check_date = target_date - timedelta(days=i)
                if check_date in stock_data.index:
                    return stock_data.loc[check_date, 'close']
            
            return None
        except:
            return None
    
    def _calculate_market_features(self, date: pd.Timestamp, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate market environment features"""
        features = {}
        
        try:
            # Use SPY as market proxy if available
            if 'SPY' in price_data:
                spy_data = price_data['SPY']
                
                if date in spy_data.index:
                    current_price = spy_data.loc[date, 'close']
                    
                    # Market momentum
                    month_ago = date - timedelta(days=21)
                    month_price = self._get_closest_price(spy_data, month_ago)
                    if month_price:
                        features['feature_market_momentum'] = (current_price / month_price) - 1
                    
                    # Market volatility
                    start_vol = date - timedelta(days=30)
                    vol_data = spy_data[(spy_data.index >= start_vol) & (spy_data.index <= date)]
                    if len(vol_data) > 5:
                        returns = vol_data['close'].pct_change().dropna()
                        features['feature_market_volatility'] = returns.std()
            
            # Default values if SPY not available
            if 'feature_market_momentum' not in features:
                features['feature_market_momentum'] = 0
            if 'feature_market_volatility' not in features:
                features['feature_market_volatility'] = 0.01
            
        except Exception as e:
            logger.error(f"Error calculating market features: {str(e)}")
        
        return features
    
    def _calculate_future_performance(self, stocks: List[str], 
                                    price_data: Dict[str, pd.DataFrame],
                                    start_date: str, end_date: str) -> Dict:
        """Calculate future performance metrics for training targets"""
        performance = {}
        
        try:
            # Simple equal-weight portfolio simulation
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            portfolio_returns = []
            
            for stock in stocks:
                if stock not in price_data:
                    continue
                
                stock_data = price_data[stock]
                
                # Get start and end prices
                start_price = self._get_closest_price(stock_data, start_dt)
                end_price = self._get_closest_price(stock_data, end_dt)
                
                if start_price and end_price:
                    stock_return = (end_price / start_price) - 1
                    portfolio_returns.append(stock_return)
            
            if portfolio_returns:
                # Portfolio return (equal weighted)
                portfolio_return = np.mean(portfolio_returns)
                performance['future_return'] = portfolio_return
                
                # Simplified Sharpe ratio (using return/volatility)
                return_volatility = np.std(portfolio_returns) if len(portfolio_returns) > 1 else 0.1
                performance['future_sharpe'] = portfolio_return / return_volatility if return_volatility > 0 else 0
                
                # Simplified max drawdown (using worst stock performance as proxy)
                performance['future_max_drawdown'] = min(portfolio_returns) if portfolio_returns else 0
            else:
                performance['future_return'] = 0
                performance['future_sharpe'] = 0
                performance['future_max_drawdown'] = 0
            
        except Exception as e:
            logger.error(f"Error calculating future performance: {str(e)}")
            performance['future_return'] = 0
            performance['future_sharpe'] = 0
            performance['future_max_drawdown'] = 0
        
        return performance
    
    def _identify_best_models(self, model_performance: Dict) -> Dict:
        """Identify best performing models for each target"""
        best_models = {}
        
        for target, models in model_performance.items():
            if not models:
                continue
            
            # Find model with highest R²
            best_model = max(models.items(), key=lambda x: x[1].get('r2', -1))
            best_models[target] = {
                'model_name': best_model[0],
                'r2_score': best_model[1].get('r2', 0),
                'rmse': best_model[1].get('rmse', 0)
            }
        
        return best_models
    
    def predict_portfolio_performance(self, stocks: List[str],
                                    price_data: Dict[str, pd.DataFrame],
                                    fundamental_data: Dict[str, Dict],
                                    prediction_date: str) -> Dict:
        """Predict portfolio performance using trained models"""
        predictions = {
            'prediction_date': prediction_date,
            'portfolio_stocks': stocks,
            'predictions': {},
            'confidence_scores': {},
            'recommendation': 'HOLD'
        }
        
        try:
            if not self.ml_models:
                predictions['error'] = "No trained models available"
                return predictions
            
            # Calculate features for current portfolio
            features = self._calculate_portfolio_features(stocks, price_data, fundamental_data, prediction_date)
            
            # Make predictions with each model
            for model_key, model_info in self.ml_models.items():
                try:
                    model = model_info['model']
                    feature_columns = model_info['features']
                    target = model_info['target']
                    
                    # Prepare feature vector
                    feature_vector = []
                    for feature in feature_columns:
                        feature_vector.append(features.get(feature, 0))
                    
                    feature_array = np.array(feature_vector).reshape(1, -1)
                    
                    # Scale if needed
                    if model_info['scaler']:
                        feature_array = model_info['scaler'].transform(feature_array)
                    
                    # Make prediction
                    prediction = model.predict(feature_array)[0]
                    predictions['predictions'][model_key] = prediction
                    
                    # Calculate confidence based on model performance
                    r2_score = model_info['performance'].get('r2', 0)
                    confidence = max(0, min(1, r2_score))  # Clamp between 0 and 1
                    predictions['confidence_scores'][model_key] = confidence
                    
                except Exception as e:
                    logger.error(f"Error making prediction with {model_key}: {str(e)}")
            
            # Generate overall recommendation
            recommendation = self._generate_prediction_recommendation(predictions['predictions'])
            predictions['recommendation'] = recommendation
            
        except Exception as e:
            logger.error(f"Error in portfolio performance prediction: {str(e)}")
            predictions['error'] = str(e)
        
        return predictions
    
    def _generate_prediction_recommendation(self, predictions: Dict) -> str:
        """Generate investment recommendation based on predictions"""
        try:
            # Extract predictions for different targets
            return_predictions = [v for k, v in predictions.items() if 'future_return' in k]
            sharpe_predictions = [v for k, v in predictions.items() if 'future_sharpe' in k]
            drawdown_predictions = [v for k, v in predictions.items() if 'future_max_drawdown' in k]
            
            # Average predictions
            avg_return = np.mean(return_predictions) if return_predictions else 0
            avg_sharpe = np.mean(sharpe_predictions) if sharpe_predictions else 0
            avg_drawdown = np.mean(drawdown_predictions) if drawdown_predictions else 0
            
            # Generate recommendation
            if avg_return > 0.05 and avg_sharpe > 0.8 and avg_drawdown > -0.15:
                return 'STRONG_BUY'
            elif avg_return > 0.02 and avg_sharpe > 0.5:
                return 'BUY'
            elif avg_return > -0.02 and avg_drawdown > -0.20:
                return 'HOLD'
            elif avg_return > -0.05:
                return 'SELL'
            else:
                return 'STRONG_SELL'
        
        except:
            return 'HOLD'
    
    def comprehensive_benchmark_comparison(self, portfolio_selections: Dict[str, List[str]],
                                         price_data: Dict[str, pd.DataFrame],
                                         start_date: str, end_date: str) -> Dict:
        """Comprehensive comparison against multiple benchmarks"""
        comparison_results = {
            'analysis_period': {'start': start_date, 'end': end_date},
            'portfolio_performance': {},
            'benchmark_performance': {},
            'relative_performance': {},
            'superiority_analysis': {},
            'risk_adjusted_comparison': {}
        }
        
        try:
            logger.info("Running comprehensive benchmark comparison")
            
            # Download benchmark data
            benchmark_data = self._download_benchmark_data(start_date, end_date)
            
            # Run portfolio backtest
            portfolio_backtest = self.backtest_engine.run_backtest(
                portfolio_selections, price_data, start_date, end_date
            )
            comparison_results['portfolio_performance'] = portfolio_backtest
            
            # Compare against each benchmark
            benchmark_results = {}
            relative_performance = {}
            
            for benchmark_symbol, benchmark_name in self.benchmarks.items():
                if benchmark_symbol in benchmark_data:
                    # Calculate benchmark performance
                    benchmark_perf = self._calculate_benchmark_metrics(
                        benchmark_data[benchmark_symbol], start_date, end_date
                    )
                    benchmark_results[benchmark_symbol] = {
                        'name': benchmark_name,
                        'performance': benchmark_perf
                    }
                    
                    # Calculate relative performance
                    if 'performance_metrics' in portfolio_backtest:
                        portfolio_metrics = portfolio_backtest['performance_metrics']
                        relative_perf = self._calculate_relative_performance(
                            portfolio_metrics, benchmark_perf
                        )
                        relative_performance[benchmark_symbol] = relative_perf
            
            comparison_results['benchmark_performance'] = benchmark_results
            comparison_results['relative_performance'] = relative_performance
            
            # Analyze portfolio superiority
            superiority_analysis = self._analyze_portfolio_superiority(
                portfolio_backtest, benchmark_results
            )
            comparison_results['superiority_analysis'] = superiority_analysis
            
            # Risk-adjusted comparison
            risk_adjusted = self._risk_adjusted_comparison(
                portfolio_backtest, benchmark_results
            )
            comparison_results['risk_adjusted_comparison'] = risk_adjusted
            
        except Exception as e:
            logger.error(f"Error in benchmark comparison: {str(e)}")
            comparison_results['error'] = str(e)
        
        return comparison_results
    
    def _download_benchmark_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download benchmark data from Yahoo Finance"""
        benchmark_data = {}
        
        for symbol in self.benchmarks.keys():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Standardize column names
                    data.columns = [col.lower() for col in data.columns]
                    benchmark_data[symbol] = data
                    
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {str(e)}")
        
        return benchmark_data
    
    def _calculate_benchmark_metrics(self, benchmark_data: pd.DataFrame, 
                                   start_date: str, end_date: str) -> Dict:
        """Calculate performance metrics for benchmark"""
        metrics = {}
        
        try:
            # Calculate returns
            returns = benchmark_data['close'].pct_change().dropna()
            
            # Basic metrics
            total_return = (benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0]) - 1
            metrics['total_return'] = total_return
            
            # Annualized return
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            years = days / 365.25
            metrics['annualized_return'] = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Volatility
            metrics['volatility'] = returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            excess_returns = returns - (0.02 / 252)
            metrics['sharpe_ratio'] = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
            
            # Win rate
            metrics['win_rate'] = (returns > 0).mean()
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            metrics['sortino_ratio'] = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {str(e)}")
        
        return metrics
    
    def _calculate_relative_performance(self, portfolio_metrics: Dict, benchmark_metrics: Dict) -> Dict:
        """Calculate relative performance metrics"""
        relative = {}
        
        try:
            # Alpha (excess return)
            relative['alpha'] = portfolio_metrics.get('total_return', 0) - benchmark_metrics.get('total_return', 0)
            relative['annualized_alpha'] = portfolio_metrics.get('annualized_return', 0) - benchmark_metrics.get('annualized_return', 0)
            
            # Relative Sharpe ratio
            relative['sharpe_difference'] = portfolio_metrics.get('sharpe_ratio', 0) - benchmark_metrics.get('sharpe_ratio', 0)
            
            # Relative volatility
            relative['volatility_difference'] = portfolio_metrics.get('volatility', 0) - benchmark_metrics.get('volatility', 0)
            
            # Relative max drawdown
            relative['drawdown_difference'] = portfolio_metrics.get('max_drawdown', 0) - benchmark_metrics.get('max_drawdown', 0)
            
            # Win rate difference
            relative['win_rate_difference'] = portfolio_metrics.get('win_rate', 0) - benchmark_metrics.get('win_rate', 0)
            
            # Information ratio approximation
            if benchmark_metrics.get('volatility', 0) > 0:
                tracking_error = abs(portfolio_metrics.get('volatility', 0) - benchmark_metrics.get('volatility', 0))
                relative['information_ratio'] = relative['annualized_alpha'] / tracking_error if tracking_error > 0 else 0
            else:
                relative['information_ratio'] = 0
            
        except Exception as e:
            logger.error(f"Error calculating relative performance: {str(e)}")
        
        return relative
    
    def _analyze_portfolio_superiority(self, portfolio_backtest: Dict, benchmark_results: Dict) -> Dict:
        """Analyze if portfolio is superior to benchmarks"""
        superiority = {
            'is_superior': False,
            'superiority_score': 0,
            'criteria_met': {},
            'benchmark_comparison': {},
            'overall_ranking': 0
        }
        
        try:
            if 'performance_metrics' not in portfolio_backtest:
                return superiority
            
            portfolio_metrics = portfolio_backtest['performance_metrics']
            
            # Check against superiority thresholds
            criteria_met = {}
            criteria_met['sharpe_ratio'] = portfolio_metrics.get('sharpe_ratio', 0) >= self.superior_thresholds['min_sharpe_ratio']
            criteria_met['annual_return'] = portfolio_metrics.get('annualized_return', 0) >= self.superior_thresholds['min_annual_return']
            criteria_met['max_drawdown'] = portfolio_metrics.get('max_drawdown', 0) >= self.superior_thresholds['max_drawdown']
            criteria_met['win_rate'] = portfolio_metrics.get('win_rate', 0) >= self.superior_thresholds['min_win_rate']
            
            # Check alpha vs SPY
            spy_alpha = 0
            if 'SPY' in benchmark_results and 'performance' in benchmark_results['SPY']:
                spy_return = benchmark_results['SPY']['performance'].get('annualized_return', 0)
                spy_alpha = portfolio_metrics.get('annualized_return', 0) - spy_return
            criteria_met['alpha_vs_spy'] = spy_alpha >= self.superior_thresholds['min_alpha_vs_spy']
            
            superiority['criteria_met'] = criteria_met
            
            # Calculate superiority score
            score = sum(criteria_met.values()) / len(criteria_met)
            superiority['superiority_score'] = score
            
            # Portfolio is superior if it meets at least 4 out of 5 criteria
            superiority['is_superior'] = score >= 0.8
            
            # Compare against each benchmark
            benchmark_comparison = {}
            wins = 0
            total_comparisons = 0
            
            for benchmark_symbol, benchmark_data in benchmark_results.items():
                if 'performance' not in benchmark_data:
                    continue
                
                benchmark_metrics = benchmark_data['performance']
                comparison = {}
                
                # Compare key metrics
                comparison['return_advantage'] = portfolio_metrics.get('annualized_return', 0) > benchmark_metrics.get('annualized_return', 0)
                comparison['sharpe_advantage'] = portfolio_metrics.get('sharpe_ratio', 0) > benchmark_metrics.get('sharpe_ratio', 0)
                comparison['drawdown_advantage'] = portfolio_metrics.get('max_drawdown', 0) > benchmark_metrics.get('max_drawdown', 0)
                
                # Count wins
                benchmark_wins = sum(comparison.values())
                comparison['wins_out_of_3'] = benchmark_wins
                comparison['beats_benchmark'] = benchmark_wins >= 2
                
                if comparison['beats_benchmark']:
                    wins += 1
                total_comparisons += 1
                
                benchmark_comparison[benchmark_symbol] = comparison
            
            superiority['benchmark_comparison'] = benchmark_comparison
            
            # Overall ranking (percentage of benchmarks beaten)
            if total_comparisons > 0:
                superiority['overall_ranking'] = wins / total_comparisons
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio superiority: {str(e)}")
        
        return superiority
    
    def _risk_adjusted_comparison(self, portfolio_backtest: Dict, benchmark_results: Dict) -> Dict:
        """Risk-adjusted performance comparison"""
        risk_adjusted = {
            'risk_efficiency': {},
            'return_per_unit_risk': {},
            'downside_protection': {},
            'consistency_metrics': {}
        }
        
        try:
            if 'performance_metrics' not in portfolio_backtest:
                return risk_adjusted
            
            portfolio_metrics = portfolio_backtest['performance_metrics']
            
            # Risk efficiency (return / volatility)
            portfolio_risk_efficiency = portfolio_metrics.get('annualized_return', 0) / portfolio_metrics.get('volatility', 1)
            risk_adjusted['risk_efficiency']['portfolio'] = portfolio_risk_efficiency
            
            # Compare with benchmarks
            for benchmark_symbol, benchmark_data in benchmark_results.items():
                if 'performance' not in benchmark_data:
                    continue
                
                benchmark_metrics = benchmark_data['performance']
                benchmark_risk_efficiency = benchmark_metrics.get('annualized_return', 0) / benchmark_metrics.get('volatility', 1)
                risk_adjusted['risk_efficiency'][benchmark_symbol] = benchmark_risk_efficiency
            
            # Return per unit of downside risk
            portfolio_sortino = portfolio_metrics.get('sortino_ratio', 0)
            risk_adjusted['return_per_unit_risk']['portfolio_sortino'] = portfolio_sortino
            
            # Downside protection (max drawdown comparison)
            portfolio_drawdown = abs(portfolio_metrics.get('max_drawdown', 0))
            risk_adjusted['downside_protection']['portfolio_max_drawdown'] = portfolio_drawdown
            
            # Calculate relative downside protection
            for benchmark_symbol, benchmark_data in benchmark_results.items():
                if 'performance' not in benchmark_data:
                    continue
                
                benchmark_drawdown = abs(benchmark_data['performance'].get('max_drawdown', 0))
                protection_ratio = benchmark_drawdown / portfolio_drawdown if portfolio_drawdown > 0 else 1
                risk_adjusted['downside_protection'][f'{benchmark_symbol}_protection_ratio'] = protection_ratio
            
            # Consistency metrics
            portfolio_win_rate = portfolio_metrics.get('win_rate', 0)
            risk_adjusted['consistency_metrics']['portfolio_win_rate'] = portfolio_win_rate
            
            # Sharpe ratio comparison
            portfolio_sharpe = portfolio_metrics.get('sharpe_ratio', 0)
            risk_adjusted['consistency_metrics']['portfolio_sharpe'] = portfolio_sharpe
            
            sharpe_rankings = []
            for benchmark_symbol, benchmark_data in benchmark_results.items():
                if 'performance' not in benchmark_data:
                    continue
                
                benchmark_sharpe = benchmark_data['performance'].get('sharpe_ratio', 0)
                if portfolio_sharpe > benchmark_sharpe:
                    sharpe_rankings.append(1)
                else:
                    sharpe_rankings.append(0)
            
            risk_adjusted['consistency_metrics']['sharpe_ranking'] = np.mean(sharpe_rankings) if sharpe_rankings else 0
            
        except Exception as e:
            logger.error(f"Error in risk-adjusted comparison: {str(e)}")
        
        return risk_adjusted
    
    def generate_superiority_report(self, comparison_results: Dict) -> Dict:
        """Generate comprehensive superiority analysis report"""
        report = {
            'executive_summary': {},
            'superiority_verdict': 'UNKNOWN',
            'key_strengths': [],
            'areas_for_improvement': [],
            'benchmark_rankings': {},
            'recommendations': []
        }
        
        try:
            superiority = comparison_results.get('superiority_analysis', {})
            risk_adjusted = comparison_results.get('risk_adjusted_comparison', {})
            
            # Executive summary
            is_superior = superiority.get('is_superior', False)
            superiority_score = superiority.get('superiority_score', 0)
            overall_ranking = superiority.get('overall_ranking', 0)
            
            report['executive_summary'] = {
                'is_superior_portfolio': is_superior,
                'superiority_score': f"{superiority_score:.1%}",
                'benchmarks_beaten': f"{overall_ranking:.1%}",
                'criteria_met': f"{sum(superiority.get('criteria_met', {}).values())}/5"
            }
            
            # Superiority verdict
            if superiority_score >= 0.8:
                report['superiority_verdict'] = 'SUPERIOR'
            elif superiority_score >= 0.6:
                report['superiority_verdict'] = 'ABOVE_AVERAGE'
            elif superiority_score >= 0.4:
                report['superiority_verdict'] = 'AVERAGE'
            else:
                report['superiority_verdict'] = 'BELOW_AVERAGE'
            
            # Identify key strengths
            criteria_met = superiority.get('criteria_met', {})
            strengths = []
            
            if criteria_met.get('sharpe_ratio', False):
                strengths.append("Excellent risk-adjusted returns (Sharpe ratio > 1.0)")
            if criteria_met.get('annual_return', False):
                strengths.append("Strong absolute returns (> 8% annually)")
            if criteria_met.get('max_drawdown', False):
                strengths.append("Good downside protection (max drawdown < 15%)")
            if criteria_met.get('win_rate', False):
                strengths.append("Consistent performance (win rate > 55%)")
            if criteria_met.get('alpha_vs_spy', False):
                strengths.append("Outperforms market benchmark (alpha > 2%)")
            
            report['key_strengths'] = strengths
            
            # Areas for improvement
            improvements = []
            if not criteria_met.get('sharpe_ratio', False):
                improvements.append("Improve risk-adjusted returns")
            if not criteria_met.get('annual_return', False):
                improvements.append("Increase absolute returns")
            if not criteria_met.get('max_drawdown', False):
                improvements.append("Reduce maximum drawdown")
            if not criteria_met.get('win_rate', False):
                improvements.append("Improve consistency of returns")
            if not criteria_met.get('alpha_vs_spy', False):
                improvements.append("Generate more alpha vs market")
            
            report['areas_for_improvement'] = improvements
            
            # Benchmark rankings
            benchmark_comparison = superiority.get('benchmark_comparison', {})
            rankings = {}
            
            for benchmark, comparison in benchmark_comparison.items():
                rankings[benchmark] = {
                    'beats_benchmark': comparison.get('beats_benchmark', False),
                    'wins_out_of_3': comparison.get('wins_out_of_3', 0),
                    'benchmark_name': self.benchmarks.get(benchmark, benchmark)
                }
            
            report['benchmark_rankings'] = rankings
            
            # Generate recommendations
            recommendations = self._generate_superiority_recommendations(
                superiority, risk_adjusted, criteria_met
            )
            report['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error generating superiority report: {str(e)}")
            report['error'] = str(e)
        
        return report
    
    def _generate_superiority_recommendations(self, superiority: Dict, 
                                            risk_adjusted: Dict, 
                                            criteria_met: Dict) -> List[str]:
        """Generate specific recommendations for portfolio improvement"""
        recommendations = []
        
        try:
            # Performance recommendations
            if not criteria_met.get('sharpe_ratio', False):
                recommendations.append("Focus on improving risk-adjusted returns through better stock selection or position sizing")
            
            if not criteria_met.get('max_drawdown', False):
                recommendations.append("Implement stronger risk management to reduce maximum drawdown")
            
            if not criteria_met.get('win_rate', False):
                recommendations.append("Improve stock selection criteria to increase win rate")
            
            # Risk management recommendations
            risk_efficiency = risk_adjusted.get('risk_efficiency', {})
            portfolio_efficiency = risk_efficiency.get('portfolio', 0)
            
            if portfolio_efficiency < 0.5:
                recommendations.append("Consider reducing portfolio volatility while maintaining returns")
            
            # Diversification recommendations
            overall_ranking = superiority.get('overall_ranking', 0)
            if overall_ranking < 0.5:
                recommendations.append("Consider diversifying across different asset classes or investment styles")
            
            # Specific tactical recommendations
            if superiority.get('superiority_score', 0) < 0.6:
                recommendations.append("Review and optimize stock screening criteria")
                recommendations.append("Consider implementing momentum or mean reversion overlays")
                recommendations.append("Evaluate position sizing and rebalancing frequency")
            
            # If already superior, maintenance recommendations
            if superiority.get('is_superior', False):
                recommendations.append("Maintain current strategy while monitoring for regime changes")
                recommendations.append("Consider gradual position size increases for best performing strategies")
            
            if not recommendations:
                recommendations.append("Portfolio shows reasonable performance - continue current approach with minor optimizations")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations

