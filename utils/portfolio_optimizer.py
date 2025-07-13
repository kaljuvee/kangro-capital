"""
Advanced Portfolio Optimizer with Modern Portfolio Theory and ML-Enhanced Optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.optimize as sco
from scipy import stats

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Advanced portfolio optimizer with multiple optimization strategies
    """
    
    def __init__(self):
        """Initialize the portfolio optimizer"""
        self.optimization_results = {}
        self.factor_models = {}
        self.risk_models = {}
        
        # Optimization constraints
        self.default_constraints = {
            'max_weight': 0.15,  # Maximum 15% per position
            'min_weight': 0.01,  # Minimum 1% per position
            'max_sector_weight': 0.30,  # Maximum 30% per sector
            'max_turnover': 0.50,  # Maximum 50% turnover
            'target_volatility': None,  # Target volatility (if specified)
            'min_expected_return': None,  # Minimum expected return
            'max_tracking_error': None  # Maximum tracking error vs benchmark
        }
        
        # Risk factors for factor-based optimization
        self.risk_factors = [
            'market_beta', 'size_factor', 'value_factor', 'momentum_factor',
            'quality_factor', 'volatility_factor', 'growth_factor'
        ]
    
    def optimize_portfolio(self, stocks: List[str], 
                          price_data: Dict[str, pd.DataFrame],
                          fundamental_data: Dict[str, Dict],
                          optimization_method: str = 'mean_variance',
                          constraints: Dict = None,
                          lookback_days: int = 252) -> Dict:
        """
        Optimize portfolio using specified method
        
        Args:
            stocks: List of stock symbols
            price_data: Price data for stocks
            fundamental_data: Fundamental data for stocks
            optimization_method: 'mean_variance', 'risk_parity', 'black_litterman', 
                                'factor_based', 'ml_enhanced', 'hierarchical_risk_parity'
            constraints: Portfolio constraints
            lookback_days: Historical data lookback period
        
        Returns:
            Optimization results with weights and metrics
        """
        optimization_results = {
            'method': optimization_method,
            'stocks': stocks,
            'constraints': constraints or self.default_constraints,
            'optimal_weights': {},
            'expected_metrics': {},
            'optimization_details': {},
            'risk_analysis': {}
        }
        
        try:
            logger.info(f"Optimizing portfolio using {optimization_method} method")
            
            # Prepare data
            returns_data, price_matrix = self._prepare_optimization_data(
                stocks, price_data, lookback_days
            )
            
            if returns_data.empty:
                optimization_results['error'] = "Insufficient data for optimization"
                return optimization_results
            
            # Apply optimization method
            if optimization_method == 'mean_variance':
                weights = self._mean_variance_optimization(returns_data, constraints)
            elif optimization_method == 'risk_parity':
                weights = self._risk_parity_optimization(returns_data, constraints)
            elif optimization_method == 'black_litterman':
                weights = self._black_litterman_optimization(returns_data, fundamental_data, constraints)
            elif optimization_method == 'factor_based':
                weights = self._factor_based_optimization(returns_data, fundamental_data, constraints)
            elif optimization_method == 'ml_enhanced':
                weights = self._ml_enhanced_optimization(returns_data, fundamental_data, price_data, constraints)
            elif optimization_method == 'hierarchical_risk_parity':
                weights = self._hierarchical_risk_parity(returns_data, constraints)
            else:
                # Default to equal weight
                weights = {stock: 1.0/len(stocks) for stock in stocks}
            
            optimization_results['optimal_weights'] = weights
            
            # Calculate expected metrics
            expected_metrics = self._calculate_expected_portfolio_metrics(weights, returns_data)
            optimization_results['expected_metrics'] = expected_metrics
            
            # Risk analysis
            risk_analysis = self._analyze_portfolio_risk(weights, returns_data, fundamental_data)
            optimization_results['risk_analysis'] = risk_analysis
            
            # Optimization details
            optimization_details = self._generate_optimization_details(
                weights, returns_data, optimization_method
            )
            optimization_results['optimization_details'] = optimization_details
            
            self.optimization_results[optimization_method] = optimization_results
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _prepare_optimization_data(self, stocks: List[str], 
                                 price_data: Dict[str, pd.DataFrame],
                                 lookback_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare returns and price data for optimization"""
        returns_dict = {}
        price_dict = {}
        
        # Calculate end date (most recent common date)
        common_dates = None
        for stock in stocks:
            if stock in price_data and not price_data[stock].empty:
                stock_dates = set(price_data[stock].index)
                if common_dates is None:
                    common_dates = stock_dates
                else:
                    common_dates = common_dates.intersection(stock_dates)
        
        if not common_dates:
            return pd.DataFrame(), pd.DataFrame()
        
        end_date = max(common_dates)
        start_date = end_date - timedelta(days=lookback_days)
        
        # Extract returns and prices for each stock
        for stock in stocks:
            if stock not in price_data:
                continue
            
            stock_data = price_data[stock]
            period_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
            
            if len(period_data) > 20:  # Need minimum data points
                returns = period_data['close'].pct_change().dropna()
                prices = period_data['close']
                
                returns_dict[stock] = returns
                price_dict[stock] = prices
        
        # Create aligned DataFrames
        returns_df = pd.DataFrame(returns_dict) if returns_dict else pd.DataFrame()
        price_df = pd.DataFrame(price_dict) if price_dict else pd.DataFrame()
        
        # Remove rows with any NaN values
        if not returns_df.empty:
            returns_df = returns_df.dropna()
        if not price_df.empty:
            price_df = price_df.dropna()
        
        return returns_df, price_df
    
    def _mean_variance_optimization(self, returns_data: pd.DataFrame, 
                                  constraints: Dict) -> Dict[str, float]:
        """Mean-variance optimization (Markowitz)"""
        try:
            n_assets = len(returns_data.columns)
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            # Objective function: minimize portfolio variance
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Bounds for individual weights
            max_weight = constraints.get('max_weight', 0.15)
            min_weight = constraints.get('min_weight', 0.01)
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            x0 = np.array([1.0/n_assets] * n_assets)
            
            # Optimize
            result = sco.minimize(portfolio_variance, x0, method='SLSQP',
                                bounds=bounds, constraints=constraints_list)
            
            if result.success:
                weights = dict(zip(returns_data.columns, result.x))
            else:
                # Fallback to equal weights
                weights = {stock: 1.0/n_assets for stock in returns_data.columns}
            
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {str(e)}")
            n_assets = len(returns_data.columns)
            weights = {stock: 1.0/n_assets for stock in returns_data.columns}
        
        return weights
    
    def _risk_parity_optimization(self, returns_data: pd.DataFrame, 
                                constraints: Dict) -> Dict[str, float]:
        """Risk parity optimization (equal risk contribution)"""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            # Risk parity objective: minimize sum of squared risk contribution differences
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / len(weights)  # Equal risk contribution
                return np.sum((contrib - target_contrib) ** 2)
            
            n_assets = len(returns_data.columns)
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Bounds
            max_weight = constraints.get('max_weight', 0.15)
            min_weight = constraints.get('min_weight', 0.01)
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            
            # Initial guess (inverse volatility)
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol = 1.0 / volatilities
            x0 = inv_vol / np.sum(inv_vol)
            
            # Optimize
            result = sco.minimize(risk_parity_objective, x0, method='SLSQP',
                                bounds=bounds, constraints=constraints_list)
            
            if result.success:
                weights = dict(zip(returns_data.columns, result.x))
            else:
                # Fallback to inverse volatility
                weights = dict(zip(returns_data.columns, x0))
            
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            # Fallback to equal weights
            n_assets = len(returns_data.columns)
            weights = {stock: 1.0/n_assets for stock in returns_data.columns}
        
        return weights
    
    def _black_litterman_optimization(self, returns_data: pd.DataFrame,
                                    fundamental_data: Dict[str, Dict],
                                    constraints: Dict) -> Dict[str, float]:
        """Black-Litterman optimization with fundamental views"""
        try:
            # Calculate market-implied returns (simplified)
            cov_matrix = returns_data.cov() * 252
            
            # Market capitalization weights (proxy using equal weights)
            n_assets = len(returns_data.columns)
            market_weights = np.array([1.0/n_assets] * n_assets)
            
            # Risk aversion parameter (typical value)
            risk_aversion = 3.0
            
            # Market-implied returns
            implied_returns = risk_aversion * np.dot(cov_matrix, market_weights)
            
            # Generate views based on fundamental data
            views, view_uncertainty = self._generate_fundamental_views(
                returns_data.columns, fundamental_data
            )
            
            if views is not None and len(views) > 0:
                # Black-Litterman formula
                tau = 0.025  # Scaling factor
                
                # Uncertainty in prior
                omega = np.diag(view_uncertainty)
                
                # Picking matrix (which assets the views relate to)
                P = np.eye(len(views), n_assets)
                
                # Black-Litterman expected returns
                M1 = np.linalg.inv(tau * cov_matrix)
                M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
                M3 = np.dot(np.linalg.inv(tau * cov_matrix), implied_returns)
                M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), views))
                
                bl_returns = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
                bl_cov = np.linalg.inv(M1 + M2)
            else:
                # No views, use market-implied returns
                bl_returns = implied_returns
                bl_cov = cov_matrix
            
            # Optimize with Black-Litterman inputs
            def portfolio_utility(weights):
                portfolio_return = np.dot(weights, bl_returns)
                portfolio_variance = np.dot(weights.T, np.dot(bl_cov, weights))
                return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
            
            # Constraints and bounds
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            max_weight = constraints.get('max_weight', 0.15)
            min_weight = constraints.get('min_weight', 0.01)
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            
            x0 = market_weights
            
            result = sco.minimize(portfolio_utility, x0, method='SLSQP',
                                bounds=bounds, constraints=constraints_list)
            
            if result.success:
                weights = dict(zip(returns_data.columns, result.x))
            else:
                weights = dict(zip(returns_data.columns, market_weights))
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            n_assets = len(returns_data.columns)
            weights = {stock: 1.0/n_assets for stock in returns_data.columns}
        
        return weights
    
    def _generate_fundamental_views(self, stocks: List[str], 
                                  fundamental_data: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate views based on fundamental analysis"""
        views = []
        uncertainties = []
        
        try:
            for stock in stocks:
                if stock not in fundamental_data:
                    views.append(0.0)
                    uncertainties.append(0.1)
                    continue
                
                fund_data = fundamental_data[stock]
                
                # Generate view based on fundamental score
                view_score = 0
                
                # ROE component
                roe = fund_data.get('roe', 0)
                if roe > 0.15:  # Strong ROE
                    view_score += 0.02
                elif roe < 0.05:  # Weak ROE
                    view_score -= 0.02
                
                # Growth component
                revenue_growth = fund_data.get('revenue_growth_5y', 0)
                if revenue_growth > 0.1:  # Strong growth
                    view_score += 0.015
                elif revenue_growth < 0:  # Negative growth
                    view_score -= 0.015
                
                # Profitability component
                net_margin = fund_data.get('net_margin', 0)
                if net_margin > 0.1:  # High margin
                    view_score += 0.01
                elif net_margin < 0:  # Negative margin
                    view_score -= 0.01
                
                # Financial health component
                current_ratio = fund_data.get('current_ratio', 1)
                debt_ratio = fund_data.get('debt_to_ebitda', 0)
                
                if current_ratio > 2 and debt_ratio < 2:  # Strong balance sheet
                    view_score += 0.005
                elif current_ratio < 1 or debt_ratio > 5:  # Weak balance sheet
                    view_score -= 0.005
                
                views.append(view_score)
                
                # Uncertainty based on data quality
                uncertainty = 0.05 + abs(view_score) * 0.5  # Higher uncertainty for stronger views
                uncertainties.append(uncertainty)
            
            return np.array(views), np.array(uncertainties)
            
        except Exception as e:
            logger.error(f"Error generating fundamental views: {str(e)}")
            return None, None
    
    def _factor_based_optimization(self, returns_data: pd.DataFrame,
                                 fundamental_data: Dict[str, Dict],
                                 constraints: Dict) -> Dict[str, float]:
        """Factor-based portfolio optimization"""
        try:
            # Calculate factor exposures
            factor_exposures = self._calculate_factor_exposures(
                returns_data.columns, fundamental_data, returns_data
            )
            
            if factor_exposures.empty:
                # Fallback to mean-variance
                return self._mean_variance_optimization(returns_data, constraints)
            
            # Factor risk model
            factor_returns = self._estimate_factor_returns(returns_data, factor_exposures)
            factor_cov = factor_returns.cov() * 252  # Annualized
            
            # Specific risk (idiosyncratic)
            specific_risk = self._estimate_specific_risk(returns_data, factor_exposures, factor_returns)
            
            # Portfolio optimization with factor model
            def factor_portfolio_risk(weights):
                # Factor risk
                portfolio_exposures = np.dot(weights, factor_exposures.values)
                factor_risk = np.dot(portfolio_exposures.T, np.dot(factor_cov, portfolio_exposures))
                
                # Specific risk
                specific_risk_contrib = np.sum(weights**2 * specific_risk**2)
                
                return factor_risk + specific_risk_contrib
            
            n_assets = len(returns_data.columns)
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # Factor exposure constraints (optional)
            # Limit exposure to any single factor
            max_factor_exposure = 0.5
            for i in range(len(self.risk_factors)):
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda x, factor_idx=i: max_factor_exposure - abs(np.dot(x, factor_exposures.iloc[:, factor_idx]))
                })
            
            # Bounds
            max_weight = constraints.get('max_weight', 0.15)
            min_weight = constraints.get('min_weight', 0.01)
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            
            # Initial guess
            x0 = np.array([1.0/n_assets] * n_assets)
            
            # Optimize
            result = sco.minimize(factor_portfolio_risk, x0, method='SLSQP',
                                bounds=bounds, constraints=constraints_list)
            
            if result.success:
                weights = dict(zip(returns_data.columns, result.x))
            else:
                weights = {stock: 1.0/n_assets for stock in returns_data.columns}
            
        except Exception as e:
            logger.error(f"Error in factor-based optimization: {str(e)}")
            n_assets = len(returns_data.columns)
            weights = {stock: 1.0/n_assets for stock in returns_data.columns}
        
        return weights
    
    def _calculate_factor_exposures(self, stocks: List[str],
                                  fundamental_data: Dict[str, Dict],
                                  returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor exposures for each stock"""
        exposures = []
        
        for stock in stocks:
            stock_exposures = {}
            
            # Market beta (from returns)
            if len(returns_data) > 30:
                stock_returns = returns_data[stock] if stock in returns_data.columns else pd.Series()
                market_returns = returns_data.mean(axis=1)  # Equal-weighted market proxy
                
                if len(stock_returns) > 30 and len(market_returns) > 30:
                    beta = np.cov(stock_returns, market_returns)[0, 1] / np.var(market_returns)
                    stock_exposures['market_beta'] = beta
                else:
                    stock_exposures['market_beta'] = 1.0
            else:
                stock_exposures['market_beta'] = 1.0
            
            # Fundamental factors
            if stock in fundamental_data:
                fund_data = fundamental_data[stock]
                
                # Size factor (negative of log market cap, approximated)
                market_cap = fund_data.get('market_cap', 1000000000)  # Default 1B
                stock_exposures['size_factor'] = -np.log(market_cap / 1000000000)  # Normalized
                
                # Value factor (inverse P/E ratio)
                pe_ratio = fund_data.get('pe_ratio', 15)
                stock_exposures['value_factor'] = 1.0 / max(pe_ratio, 1) if pe_ratio > 0 else 0
                
                # Quality factor (ROE)
                roe = fund_data.get('roe', 0.1)
                stock_exposures['quality_factor'] = roe
                
                # Growth factor (revenue growth)
                revenue_growth = fund_data.get('revenue_growth_5y', 0.05)
                stock_exposures['growth_factor'] = revenue_growth
            else:
                # Default exposures
                stock_exposures['size_factor'] = 0
                stock_exposures['value_factor'] = 0.067  # 1/15 (average P/E)
                stock_exposures['quality_factor'] = 0.1
                stock_exposures['growth_factor'] = 0.05
            
            # Momentum factor (from returns)
            if stock in returns_data.columns and len(returns_data) > 60:
                recent_returns = returns_data[stock].tail(60)  # 3-month momentum
                momentum = recent_returns.mean() * 252  # Annualized
                stock_exposures['momentum_factor'] = momentum
            else:
                stock_exposures['momentum_factor'] = 0
            
            # Volatility factor
            if stock in returns_data.columns and len(returns_data) > 30:
                volatility = returns_data[stock].std() * np.sqrt(252)
                stock_exposures['volatility_factor'] = volatility
            else:
                stock_exposures['volatility_factor'] = 0.2  # Default volatility
            
            exposures.append(stock_exposures)
        
        exposures_df = pd.DataFrame(exposures, index=stocks)
        
        # Standardize exposures (z-score)
        for factor in exposures_df.columns:
            mean_exposure = exposures_df[factor].mean()
            std_exposure = exposures_df[factor].std()
            if std_exposure > 0:
                exposures_df[factor] = (exposures_df[factor] - mean_exposure) / std_exposure
            else:
                exposures_df[factor] = 0
        
        return exposures_df
    
    def _estimate_factor_returns(self, returns_data: pd.DataFrame, 
                               factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """Estimate factor returns using cross-sectional regression"""
        factor_returns = []
        
        try:
            for date in returns_data.index:
                daily_returns = returns_data.loc[date]
                
                # Cross-sectional regression: returns = alpha + beta * factors + error
                X = factor_exposures.values
                y = daily_returns.values
                
                # Add intercept
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                
                # Regression
                try:
                    coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                    factor_returns.append(coefficients[1:])  # Exclude intercept
                except:
                    # Fallback to zeros
                    factor_returns.append(np.zeros(len(self.risk_factors)))
            
            factor_returns_df = pd.DataFrame(
                factor_returns, 
                index=returns_data.index, 
                columns=factor_exposures.columns
            )
            
        except Exception as e:
            logger.error(f"Error estimating factor returns: {str(e)}")
            # Return empty DataFrame
            factor_returns_df = pd.DataFrame()
        
        return factor_returns_df
    
    def _estimate_specific_risk(self, returns_data: pd.DataFrame,
                              factor_exposures: pd.DataFrame,
                              factor_returns: pd.DataFrame) -> np.ndarray:
        """Estimate specific (idiosyncratic) risk for each stock"""
        specific_risks = []
        
        try:
            for stock in returns_data.columns:
                stock_returns = returns_data[stock]
                stock_exposures = factor_exposures.loc[stock]
                
                # Calculate factor-explained returns
                factor_explained = np.dot(factor_returns.values, stock_exposures.values)
                factor_explained_series = pd.Series(factor_explained, index=returns_data.index)
                
                # Specific returns (residuals)
                specific_returns = stock_returns - factor_explained_series
                
                # Specific risk (volatility of residuals)
                specific_risk = specific_returns.std() * np.sqrt(252)  # Annualized
                specific_risks.append(specific_risk)
            
        except Exception as e:
            logger.error(f"Error estimating specific risk: {str(e)}")
            # Default specific risk
            specific_risks = [0.3] * len(returns_data.columns)  # 30% annual specific risk
        
        return np.array(specific_risks)
    
    def _ml_enhanced_optimization(self, returns_data: pd.DataFrame,
                                fundamental_data: Dict[str, Dict],
                                price_data: Dict[str, pd.DataFrame],
                                constraints: Dict) -> Dict[str, float]:
        """Machine learning enhanced portfolio optimization"""
        try:
            # Prepare ML features
            ml_features = self._prepare_ml_features(
                returns_data.columns, fundamental_data, price_data, returns_data
            )
            
            if ml_features.empty:
                return self._mean_variance_optimization(returns_data, constraints)
            
            # Train ML model to predict future returns
            ml_expected_returns = self._train_return_prediction_model(
                ml_features, returns_data
            )
            
            # Use ML predictions in optimization
            cov_matrix = returns_data.cov() * 252
            
            def ml_portfolio_utility(weights):
                portfolio_return = np.dot(weights, ml_expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                # Sharpe ratio maximization
                return -(portfolio_return / np.sqrt(portfolio_variance))
            
            n_assets = len(returns_data.columns)
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # Bounds
            max_weight = constraints.get('max_weight', 0.15)
            min_weight = constraints.get('min_weight', 0.01)
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            
            # Initial guess
            x0 = np.array([1.0/n_assets] * n_assets)
            
            # Optimize
            result = sco.minimize(ml_portfolio_utility, x0, method='SLSQP',
                                bounds=bounds, constraints=constraints_list)
            
            if result.success:
                weights = dict(zip(returns_data.columns, result.x))
            else:
                weights = {stock: 1.0/n_assets for stock in returns_data.columns}
            
        except Exception as e:
            logger.error(f"Error in ML-enhanced optimization: {str(e)}")
            n_assets = len(returns_data.columns)
            weights = {stock: 1.0/n_assets for stock in returns_data.columns}
        
        return weights
    
    def _prepare_ml_features(self, stocks: List[str],
                           fundamental_data: Dict[str, Dict],
                           price_data: Dict[str, pd.DataFrame],
                           returns_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        features = []
        
        for stock in stocks:
            stock_features = {}
            
            # Technical features
            if stock in returns_data.columns:
                stock_returns = returns_data[stock]
                
                # Momentum features
                stock_features['momentum_1m'] = stock_returns.tail(21).mean() * 252
                stock_features['momentum_3m'] = stock_returns.tail(63).mean() * 252
                stock_features['momentum_6m'] = stock_returns.tail(126).mean() * 252
                
                # Volatility features
                stock_features['volatility_1m'] = stock_returns.tail(21).std() * np.sqrt(252)
                stock_features['volatility_3m'] = stock_returns.tail(63).std() * np.sqrt(252)
                
                # Trend features
                if len(stock_returns) > 50:
                    x = np.arange(len(stock_returns.tail(50)))
                    y = stock_returns.tail(50).values
                    slope, _, r_value, _, _ = stats.linregress(x, y)
                    stock_features['trend_slope'] = slope * 252
                    stock_features['trend_r_squared'] = r_value ** 2
                else:
                    stock_features['trend_slope'] = 0
                    stock_features['trend_r_squared'] = 0
            else:
                # Default values
                stock_features.update({
                    'momentum_1m': 0, 'momentum_3m': 0, 'momentum_6m': 0,
                    'volatility_1m': 0.2, 'volatility_3m': 0.2,
                    'trend_slope': 0, 'trend_r_squared': 0
                })
            
            # Fundamental features
            if stock in fundamental_data:
                fund_data = fundamental_data[stock]
                
                stock_features['roe'] = fund_data.get('roe', 0.1)
                stock_features['current_ratio'] = fund_data.get('current_ratio', 1.5)
                stock_features['gross_margin'] = fund_data.get('gross_margin', 0.3)
                stock_features['net_margin'] = fund_data.get('net_margin', 0.05)
                stock_features['revenue_growth'] = fund_data.get('revenue_growth_5y', 0.05)
                stock_features['debt_to_ebitda'] = fund_data.get('debt_to_ebitda', 2.0)
                stock_features['pe_ratio'] = fund_data.get('pe_ratio', 15)
            else:
                # Default fundamental values
                stock_features.update({
                    'roe': 0.1, 'current_ratio': 1.5, 'gross_margin': 0.3,
                    'net_margin': 0.05, 'revenue_growth': 0.05,
                    'debt_to_ebitda': 2.0, 'pe_ratio': 15
                })
            
            features.append(stock_features)
        
        features_df = pd.DataFrame(features, index=stocks)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        features_df = pd.DataFrame(features_scaled, index=stocks, columns=features_df.columns)
        
        return features_df
    
    def _train_return_prediction_model(self, features: pd.DataFrame, 
                                     returns_data: pd.DataFrame) -> np.ndarray:
        """Train ML model to predict expected returns"""
        try:
            # Prepare training data
            X = features.values
            
            # Target: future 1-month returns (simplified)
            future_returns = []
            for stock in features.index:
                if stock in returns_data.columns:
                    stock_returns = returns_data[stock]
                    # Use last month's average return as proxy for expected return
                    future_return = stock_returns.tail(21).mean() * 252  # Annualized
                    future_returns.append(future_return)
                else:
                    future_returns.append(0.08)  # Default 8% expected return
            
            y = np.array(future_returns)
            
            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Predict expected returns
            predicted_returns = model.predict(X)
            
            return predicted_returns
            
        except Exception as e:
            logger.error(f"Error training return prediction model: {str(e)}")
            # Return historical mean returns
            expected_returns = []
            for stock in features.index:
                if stock in returns_data.columns:
                    expected_returns.append(returns_data[stock].mean() * 252)
                else:
                    expected_returns.append(0.08)
            return np.array(expected_returns)
    
    def _hierarchical_risk_parity(self, returns_data: pd.DataFrame, 
                                constraints: Dict) -> Dict[str, float]:
        """Hierarchical Risk Parity optimization"""
        try:
            # Calculate correlation matrix
            corr_matrix = returns_data.corr()
            
            # Hierarchical clustering
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            # Simple hierarchical clustering (agglomerative)
            n_assets = len(returns_data.columns)
            clusters = self._simple_hierarchical_clustering(distance_matrix.values, n_assets)
            
            # Allocate weights hierarchically
            weights = self._allocate_hrp_weights(returns_data, clusters)
            
            return dict(zip(returns_data.columns, weights))
            
        except Exception as e:
            logger.error(f"Error in hierarchical risk parity: {str(e)}")
            n_assets = len(returns_data.columns)
            weights = {stock: 1.0/n_assets for stock in returns_data.columns}
            return weights
    
    def _simple_hierarchical_clustering(self, distance_matrix: np.ndarray, 
                                      n_assets: int) -> List[List[int]]:
        """Simple hierarchical clustering implementation"""
        # Start with each asset as its own cluster
        clusters = [[i] for i in range(n_assets)]
        
        # Merge clusters based on minimum distance
        while len(clusters) > 1:
            min_distance = float('inf')
            merge_i, merge_j = 0, 1
            
            # Find closest clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average linkage distance
                    total_distance = 0
                    count = 0
                    for asset_i in clusters[i]:
                        for asset_j in clusters[j]:
                            total_distance += distance_matrix[asset_i, asset_j]
                            count += 1
                    
                    avg_distance = total_distance / count if count > 0 else float('inf')
                    
                    if avg_distance < min_distance:
                        min_distance = avg_distance
                        merge_i, merge_j = i, j
            
            # Merge clusters
            new_cluster = clusters[merge_i] + clusters[merge_j]
            clusters = [clusters[k] for k in range(len(clusters)) if k not in [merge_i, merge_j]]
            clusters.append(new_cluster)
        
        return clusters[0] if clusters else list(range(n_assets))
    
    def _allocate_hrp_weights(self, returns_data: pd.DataFrame, 
                            cluster_order: List[int]) -> np.ndarray:
        """Allocate weights using HRP methodology"""
        n_assets = len(returns_data.columns)
        weights = np.ones(n_assets)
        
        # Calculate inverse variance weights within the ordered structure
        cov_matrix = returns_data.cov().values
        
        for i in range(n_assets):
            asset_idx = cluster_order[i]
            variance = cov_matrix[asset_idx, asset_idx]
            weights[asset_idx] = 1.0 / variance if variance > 0 else 1.0
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def _calculate_expected_portfolio_metrics(self, weights: Dict[str, float], 
                                            returns_data: pd.DataFrame) -> Dict:
        """Calculate expected portfolio metrics"""
        metrics = {}
        
        try:
            # Convert weights to array
            weight_array = np.array([weights.get(stock, 0) for stock in returns_data.columns])
            
            # Expected return
            expected_returns = returns_data.mean() * 252  # Annualized
            portfolio_return = np.dot(weight_array, expected_returns)
            metrics['expected_return'] = portfolio_return
            
            # Expected volatility
            cov_matrix = returns_data.cov() * 252  # Annualized
            portfolio_variance = np.dot(weight_array.T, np.dot(cov_matrix, weight_array))
            portfolio_volatility = np.sqrt(portfolio_variance)
            metrics['expected_volatility'] = portfolio_volatility
            
            # Expected Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            metrics['expected_sharpe'] = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Diversification metrics
            metrics['effective_number_of_stocks'] = 1.0 / np.sum(weight_array ** 2)
            metrics['max_weight'] = np.max(weight_array)
            metrics['weight_concentration'] = np.sum(weight_array ** 2)  # Herfindahl index
            
        except Exception as e:
            logger.error(f"Error calculating expected portfolio metrics: {str(e)}")
        
        return metrics
    
    def _analyze_portfolio_risk(self, weights: Dict[str, float], 
                              returns_data: pd.DataFrame,
                              fundamental_data: Dict[str, Dict]) -> Dict:
        """Analyze portfolio risk characteristics"""
        risk_analysis = {}
        
        try:
            weight_array = np.array([weights.get(stock, 0) for stock in returns_data.columns])
            
            # Concentration risk
            risk_analysis['concentration_risk'] = {
                'herfindahl_index': np.sum(weight_array ** 2),
                'top_5_concentration': np.sum(np.sort(weight_array)[-5:]),
                'effective_stocks': 1.0 / np.sum(weight_array ** 2)
            }
            
            # Factor risk exposure
            if fundamental_data:
                factor_exposures = self._calculate_factor_exposures(
                    list(weights.keys()), fundamental_data, returns_data
                )
                
                if not factor_exposures.empty:
                    portfolio_exposures = np.dot(weight_array, factor_exposures.values)
                    risk_analysis['factor_exposures'] = dict(zip(factor_exposures.columns, portfolio_exposures))
            
            # Tail risk metrics
            portfolio_returns = np.dot(returns_data.values, weight_array)
            risk_analysis['tail_risk'] = {
                'var_95': np.percentile(portfolio_returns, 5),
                'var_99': np.percentile(portfolio_returns, 1),
                'expected_shortfall_95': np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)])
            }
            
            # Correlation analysis
            corr_matrix = returns_data.corr()
            weighted_correlations = []
            
            for i, stock_i in enumerate(returns_data.columns):
                for j, stock_j in enumerate(returns_data.columns):
                    if i != j:
                        correlation = corr_matrix.iloc[i, j]
                        weight_product = weights.get(stock_i, 0) * weights.get(stock_j, 0)
                        weighted_correlations.append(correlation * weight_product)
            
            risk_analysis['correlation_analysis'] = {
                'average_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                'weighted_average_correlation': np.sum(weighted_correlations) if weighted_correlations else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio risk: {str(e)}")
        
        return risk_analysis
    
    def _generate_optimization_details(self, weights: Dict[str, float], 
                                     returns_data: pd.DataFrame,
                                     method: str) -> Dict:
        """Generate detailed optimization information"""
        details = {
            'optimization_method': method,
            'number_of_assets': len(weights),
            'weight_statistics': {},
            'diversification_metrics': {},
            'optimization_constraints_met': {}
        }
        
        try:
            weight_values = list(weights.values())
            
            # Weight statistics
            details['weight_statistics'] = {
                'mean_weight': np.mean(weight_values),
                'median_weight': np.median(weight_values),
                'std_weight': np.std(weight_values),
                'min_weight': np.min(weight_values),
                'max_weight': np.max(weight_values)
            }
            
            # Diversification metrics
            details['diversification_metrics'] = {
                'effective_number_of_stocks': 1.0 / np.sum(np.array(weight_values) ** 2),
                'concentration_ratio': np.sum(np.sort(weight_values)[-5:]),  # Top 5 concentration
                'gini_coefficient': self._calculate_gini_coefficient(weight_values)
            }
            
            # Check constraint compliance
            max_weight_constraint = 0.15  # Default
            min_weight_constraint = 0.01  # Default
            
            details['optimization_constraints_met'] = {
                'max_weight_constraint': np.max(weight_values) <= max_weight_constraint,
                'min_weight_constraint': np.min(weight_values) >= min_weight_constraint,
                'weights_sum_to_one': abs(np.sum(weight_values) - 1.0) < 0.001
            }
            
        except Exception as e:
            logger.error(f"Error generating optimization details: {str(e)}")
        
        return details
    
    def _calculate_gini_coefficient(self, weights: List[float]) -> float:
        """Calculate Gini coefficient for weight distribution"""
        try:
            weights = np.array(sorted(weights))
            n = len(weights)
            cumsum = np.cumsum(weights)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        except:
            return 0
    
    def compare_optimization_methods(self, stocks: List[str],
                                   price_data: Dict[str, pd.DataFrame],
                                   fundamental_data: Dict[str, Dict],
                                   methods: List[str] = None) -> Dict:
        """Compare multiple optimization methods"""
        if methods is None:
            methods = ['mean_variance', 'risk_parity', 'black_litterman', 'factor_based', 'ml_enhanced']
        
        comparison_results = {
            'methods_compared': methods,
            'optimization_results': {},
            'performance_comparison': {},
            'risk_comparison': {},
            'best_method': None
        }
        
        try:
            # Run each optimization method
            for method in methods:
                logger.info(f"Running {method} optimization")
                result = self.optimize_portfolio(stocks, price_data, fundamental_data, method)
                comparison_results['optimization_results'][method] = result
            
            # Compare performance metrics
            performance_comparison = {}
            risk_comparison = {}
            
            for method, result in comparison_results['optimization_results'].items():
                if 'expected_metrics' in result:
                    metrics = result['expected_metrics']
                    performance_comparison[method] = {
                        'expected_return': metrics.get('expected_return', 0),
                        'expected_sharpe': metrics.get('expected_sharpe', 0),
                        'expected_volatility': metrics.get('expected_volatility', 0)
                    }
                
                if 'risk_analysis' in result:
                    risk_analysis = result['risk_analysis']
                    risk_comparison[method] = {
                        'concentration_risk': risk_analysis.get('concentration_risk', {}),
                        'tail_risk': risk_analysis.get('tail_risk', {})
                    }
            
            comparison_results['performance_comparison'] = performance_comparison
            comparison_results['risk_comparison'] = risk_comparison
            
            # Identify best method (by Sharpe ratio)
            if performance_comparison:
                best_method = max(performance_comparison.items(), 
                                key=lambda x: x[1].get('expected_sharpe', 0))
                comparison_results['best_method'] = {
                    'method': best_method[0],
                    'expected_sharpe': best_method[1].get('expected_sharpe', 0)
                }
            
        except Exception as e:
            logger.error(f"Error comparing optimization methods: {str(e)}")
            comparison_results['error'] = str(e)
        
        return comparison_results
    
    def generate_optimization_report(self, optimization_results: Dict) -> Dict:
        """Generate comprehensive optimization report"""
        report = {
            'executive_summary': {},
            'optimal_allocation': {},
            'expected_performance': {},
            'risk_analysis': {},
            'recommendations': []
        }
        
        try:
            # Executive summary
            weights = optimization_results.get('optimal_weights', {})
            expected_metrics = optimization_results.get('expected_metrics', {})
            
            report['executive_summary'] = {
                'optimization_method': optimization_results.get('method', 'Unknown'),
                'number_of_stocks': len(weights),
                'expected_annual_return': f"{expected_metrics.get('expected_return', 0):.2%}",
                'expected_volatility': f"{expected_metrics.get('expected_volatility', 0):.2%}",
                'expected_sharpe_ratio': f"{expected_metrics.get('expected_sharpe', 0):.2f}"
            }
            
            # Optimal allocation (top holdings)
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            report['optimal_allocation'] = {
                'top_10_holdings': sorted_weights[:10],
                'weight_distribution': {
                    'max_weight': max(weights.values()) if weights else 0,
                    'min_weight': min(weights.values()) if weights else 0,
                    'average_weight': np.mean(list(weights.values())) if weights else 0
                }
            }
            
            # Expected performance
            report['expected_performance'] = expected_metrics
            
            # Risk analysis
            report['risk_analysis'] = optimization_results.get('risk_analysis', {})
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(optimization_results)
            report['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {str(e)}")
            report['error'] = str(e)
        
        return report
    
    def _generate_optimization_recommendations(self, optimization_results: Dict) -> List[str]:
        """Generate recommendations based on optimization results"""
        recommendations = []
        
        try:
            expected_metrics = optimization_results.get('expected_metrics', {})
            risk_analysis = optimization_results.get('risk_analysis', {})
            weights = optimization_results.get('optimal_weights', {})
            
            # Performance recommendations
            expected_sharpe = expected_metrics.get('expected_sharpe', 0)
            if expected_sharpe > 1.5:
                recommendations.append("Excellent risk-adjusted return profile. Consider implementing this allocation.")
            elif expected_sharpe > 1.0:
                recommendations.append("Good risk-adjusted returns. Portfolio shows promise for implementation.")
            elif expected_sharpe > 0.5:
                recommendations.append("Moderate risk-adjusted returns. Consider refinements to improve Sharpe ratio.")
            else:
                recommendations.append("Low risk-adjusted returns. Significant optimization needed.")
            
            # Concentration recommendations
            concentration = risk_analysis.get('concentration_risk', {})
            effective_stocks = concentration.get('effective_stocks', 0)
            
            if effective_stocks < 5:
                recommendations.append("High concentration risk. Consider increasing diversification.")
            elif effective_stocks > 15:
                recommendations.append("Well-diversified portfolio. Good risk distribution.")
            
            # Weight distribution recommendations
            if weights:
                max_weight = max(weights.values())
                if max_weight > 0.2:
                    recommendations.append("Consider reducing maximum position size to limit single-stock risk.")
                
                weight_std = np.std(list(weights.values()))
                if weight_std > 0.05:
                    recommendations.append("High weight dispersion. Consider more balanced allocation.")
            
            # Method-specific recommendations
            method = optimization_results.get('method', '')
            if method == 'mean_variance':
                recommendations.append("Mean-variance optimization used. Consider factor-based methods for enhanced risk control.")
            elif method == 'ml_enhanced':
                recommendations.append("ML-enhanced optimization provides forward-looking insights. Monitor model performance.")
            
            if not recommendations:
                recommendations.append("Portfolio optimization completed successfully. Monitor performance and rebalance periodically.")
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {str(e)}")
        
        return recommendations

