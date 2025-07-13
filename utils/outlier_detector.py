"""
Outlier and breakout stock detection module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OutlierDetector:
    """
    Detects outlier and breakout stocks based on performance metrics
    """
    
    def __init__(self):
        """Initialize the outlier detector"""
        self.scaler = StandardScaler()
        self.performance_weights = {
            'total_return': 0.25,
            'sharpe_ratio': 0.20,
            'volatility_adjusted_return': 0.15,
            'momentum_1m': 0.10,
            'momentum_3m': 0.10,
            'relative_strength': 0.10,
            'volume_trend': 0.10
        }
    
    def calculate_performance_metrics(self, price_data: Dict[str, pd.DataFrame], 
                                    lookback_days: int = 252) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics for outlier detection
        
        Args:
            price_data: Dictionary of price DataFrames by symbol
            lookback_days: Number of days to look back for calculations
        
        Returns:
            DataFrame with performance metrics
        """
        metrics_list = []
        
        for symbol, df in price_data.items():
            if df.empty or len(df) < lookback_days:
                continue
            
            try:
                # Get recent data
                recent_data = df.tail(lookback_days).copy()
                
                if len(recent_data) < 50:  # Need minimum data
                    continue
                
                # Calculate returns
                recent_data['daily_return'] = recent_data['close'].pct_change()
                recent_data = recent_data.dropna()
                
                if recent_data.empty:
                    continue
                
                metrics = {'symbol': symbol}
                
                # Basic performance metrics
                metrics['total_return'] = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
                metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(recent_data)) - 1
                
                # Risk metrics
                daily_returns = recent_data['daily_return']
                metrics['volatility'] = daily_returns.std() * np.sqrt(252)
                metrics['sharpe_ratio'] = (daily_returns.mean() * 252) / metrics['volatility'] if metrics['volatility'] > 0 else 0
                
                # Volatility-adjusted return
                metrics['volatility_adjusted_return'] = metrics['total_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
                
                # Maximum drawdown
                cumulative = (1 + daily_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                metrics['max_drawdown'] = drawdown.min()
                
                # Momentum metrics
                if len(recent_data) >= 21:
                    metrics['momentum_1m'] = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[-21]) - 1
                else:
                    metrics['momentum_1m'] = 0
                
                if len(recent_data) >= 63:
                    metrics['momentum_3m'] = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[-63]) - 1
                else:
                    metrics['momentum_3m'] = 0
                
                if len(recent_data) >= 126:
                    metrics['momentum_6m'] = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[-126]) - 1
                else:
                    metrics['momentum_6m'] = 0
                
                # Technical indicators
                # Moving averages
                recent_data['sma_20'] = recent_data['close'].rolling(20).mean()
                recent_data['sma_50'] = recent_data['close'].rolling(50).mean()
                recent_data['sma_200'] = recent_data['close'].rolling(200).mean()
                
                # Price relative to moving averages
                current_price = recent_data['close'].iloc[-1]
                if not recent_data['sma_20'].isna().iloc[-1]:
                    metrics['price_vs_sma20'] = (current_price / recent_data['sma_20'].iloc[-1]) - 1
                else:
                    metrics['price_vs_sma20'] = 0
                
                if not recent_data['sma_50'].isna().iloc[-1]:
                    metrics['price_vs_sma50'] = (current_price / recent_data['sma_50'].iloc[-1]) - 1
                else:
                    metrics['price_vs_sma50'] = 0
                
                # RSI (Relative Strength Index)
                metrics['rsi'] = self._calculate_rsi(recent_data['close'])
                
                # Volume analysis
                if 'volume' in recent_data.columns:
                    recent_volume = recent_data['volume'].tail(20).mean()
                    historical_volume = recent_data['volume'].mean()
                    metrics['volume_ratio'] = recent_volume / historical_volume if historical_volume > 0 else 1
                    
                    # Volume trend
                    volume_trend = np.polyfit(range(len(recent_data['volume'].tail(20))), 
                                            recent_data['volume'].tail(20), 1)[0]
                    metrics['volume_trend'] = volume_trend / historical_volume if historical_volume > 0 else 0
                else:
                    metrics['volume_ratio'] = 1
                    metrics['volume_trend'] = 0
                
                # Breakout indicators
                # New highs/lows
                recent_high = recent_data['high'].tail(20).max()
                historical_high = recent_data['high'].max()
                metrics['near_high'] = recent_high / historical_high
                
                # Price acceleration
                if len(recent_data) >= 10:
                    recent_prices = recent_data['close'].tail(10)
                    price_acceleration = np.polyfit(range(len(recent_prices)), recent_prices, 2)[0]
                    metrics['price_acceleration'] = price_acceleration / current_price
                else:
                    metrics['price_acceleration'] = 0
                
                # Volatility breakout
                recent_vol = daily_returns.tail(20).std()
                historical_vol = daily_returns.std()
                metrics['volatility_breakout'] = recent_vol / historical_vol if historical_vol > 0 else 1
                
                metrics_list.append(metrics)
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(metrics_list)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50
        except:
            return 50
    
    def detect_statistical_outliers(self, metrics_df: pd.DataFrame, 
                                  method: str = 'zscore', threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect statistical outliers using various methods
        
        Args:
            metrics_df: DataFrame with performance metrics
            method: Method to use ('zscore', 'iqr', 'isolation')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outlier flags
        """
        if metrics_df.empty:
            return metrics_df
        
        result_df = metrics_df.copy()
        
        # Select numeric columns for outlier detection
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'symbol']
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(metrics_df[numeric_cols], nan_policy='omit'))
            outlier_mask = (z_scores > threshold).any(axis=1)
            
        elif method == 'iqr':
            # Interquartile Range method
            Q1 = metrics_df[numeric_cols].quantile(0.25)
            Q3 = metrics_df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ((metrics_df[numeric_cols] < (Q1 - threshold * IQR)) | 
                           (metrics_df[numeric_cols] > (Q3 + threshold * IQR))).any(axis=1)
        
        else:  # Default to percentile-based
            # Top/bottom percentile method
            percentile_threshold = (100 - threshold * 10) / 100  # Convert threshold to percentile
            outlier_mask = pd.Series(False, index=metrics_df.index)
            
            for col in numeric_cols:
                if col in ['total_return', 'sharpe_ratio', 'momentum_1m', 'momentum_3m']:
                    # For these metrics, we want high values
                    threshold_val = metrics_df[col].quantile(percentile_threshold)
                    outlier_mask |= metrics_df[col] > threshold_val
                elif col in ['volatility', 'max_drawdown']:
                    # For these metrics, we want low values (outliers are extreme values)
                    threshold_val = metrics_df[col].quantile(1 - percentile_threshold)
                    outlier_mask |= metrics_df[col] < threshold_val
        
        result_df['is_outlier'] = outlier_mask
        return result_df
    
    def detect_breakout_stocks(self, metrics_df: pd.DataFrame, 
                             min_score: float = 0.7) -> pd.DataFrame:
        """
        Detect breakout stocks using composite scoring
        
        Args:
            metrics_df: DataFrame with performance metrics
            min_score: Minimum composite score for breakout classification
        
        Returns:
            DataFrame with breakout scores and classifications
        """
        if metrics_df.empty:
            return metrics_df
        
        result_df = metrics_df.copy()
        
        # Calculate composite breakout score
        breakout_score = 0
        
        # Performance component
        if 'total_return' in result_df.columns:
            return_percentile = result_df['total_return'].rank(pct=True)
            breakout_score += return_percentile * self.performance_weights.get('total_return', 0.25)
        
        # Risk-adjusted performance
        if 'sharpe_ratio' in result_df.columns:
            sharpe_percentile = result_df['sharpe_ratio'].rank(pct=True)
            breakout_score += sharpe_percentile * self.performance_weights.get('sharpe_ratio', 0.20)
        
        # Momentum components
        if 'momentum_1m' in result_df.columns:
            momentum_1m_percentile = result_df['momentum_1m'].rank(pct=True)
            breakout_score += momentum_1m_percentile * self.performance_weights.get('momentum_1m', 0.10)
        
        if 'momentum_3m' in result_df.columns:
            momentum_3m_percentile = result_df['momentum_3m'].rank(pct=True)
            breakout_score += momentum_3m_percentile * self.performance_weights.get('momentum_3m', 0.10)
        
        # Technical indicators
        if 'price_vs_sma20' in result_df.columns:
            sma_percentile = result_df['price_vs_sma20'].rank(pct=True)
            breakout_score += sma_percentile * self.performance_weights.get('relative_strength', 0.10)
        
        # Volume analysis
        if 'volume_trend' in result_df.columns:
            volume_percentile = result_df['volume_trend'].rank(pct=True)
            breakout_score += volume_percentile * self.performance_weights.get('volume_trend', 0.10)
        
        # Volatility-adjusted return
        if 'volatility_adjusted_return' in result_df.columns:
            vol_adj_percentile = result_df['volatility_adjusted_return'].rank(pct=True)
            breakout_score += vol_adj_percentile * self.performance_weights.get('volatility_adjusted_return', 0.15)
        
        result_df['breakout_score'] = breakout_score
        result_df['is_breakout'] = breakout_score >= min_score
        
        # Add breakout classification
        result_df['breakout_type'] = 'NORMAL'
        result_df.loc[result_df['breakout_score'] >= 0.9, 'breakout_type'] = 'STRONG_BREAKOUT'
        result_df.loc[(result_df['breakout_score'] >= 0.7) & (result_df['breakout_score'] < 0.9), 'breakout_type'] = 'MODERATE_BREAKOUT'
        result_df.loc[(result_df['breakout_score'] >= 0.5) & (result_df['breakout_score'] < 0.7), 'breakout_type'] = 'WEAK_BREAKOUT'
        
        return result_df
    
    def cluster_similar_performers(self, metrics_df: pd.DataFrame, 
                                 eps: float = 0.5, min_samples: int = 2) -> pd.DataFrame:
        """
        Cluster stocks with similar performance characteristics
        
        Args:
            metrics_df: DataFrame with performance metrics
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
        
        Returns:
            DataFrame with cluster assignments
        """
        if metrics_df.empty:
            return metrics_df
        
        result_df = metrics_df.copy()
        
        # Select features for clustering
        feature_cols = ['total_return', 'volatility', 'sharpe_ratio', 'momentum_1m', 'momentum_3m']
        available_cols = [col for col in feature_cols if col in metrics_df.columns]
        
        if len(available_cols) < 2:
            result_df['cluster'] = 0
            return result_df
        
        # Prepare data for clustering
        cluster_data = metrics_df[available_cols].fillna(0)
        
        # Standardize features
        scaled_data = self.scaler.fit_transform(cluster_data)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_data)
        
        result_df['cluster'] = clusters
        
        # Add cluster statistics
        cluster_stats = []
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_mask = clusters == cluster_id
            cluster_data_subset = metrics_df[cluster_mask]
            
            stats = {
                'cluster_id': cluster_id,
                'size': cluster_mask.sum(),
                'avg_return': cluster_data_subset['total_return'].mean() if 'total_return' in cluster_data_subset.columns else 0,
                'avg_volatility': cluster_data_subset['volatility'].mean() if 'volatility' in cluster_data_subset.columns else 0,
                'avg_sharpe': cluster_data_subset['sharpe_ratio'].mean() if 'sharpe_ratio' in cluster_data_subset.columns else 0
            }
            cluster_stats.append(stats)
        
        result_df.cluster_stats = pd.DataFrame(cluster_stats)
        
        return result_df
    
    def get_top_outliers(self, metrics_df: pd.DataFrame, top_n: int = 10, 
                        criteria: str = 'breakout_score') -> pd.DataFrame:
        """
        Get top N outlier stocks based on specified criteria
        
        Args:
            metrics_df: DataFrame with performance metrics and outlier flags
            top_n: Number of top stocks to return
            criteria: Column to sort by
        
        Returns:
            DataFrame with top outlier stocks
        """
        if metrics_df.empty or criteria not in metrics_df.columns:
            return pd.DataFrame()
        
        # Sort by criteria and get top N
        top_outliers = metrics_df.sort_values(criteria, ascending=False).head(top_n)
        
        # Add rank
        top_outliers = top_outliers.copy()
        top_outliers['rank'] = range(1, len(top_outliers) + 1)
        
        return top_outliers

