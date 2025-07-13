"""
Data fetching utilities that combine multiple data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import os
import json

from .polygon_client import PolygonClient
from .tavily_client import TavilySearchClient

logger = logging.getLogger(__name__)

class DataFetcher:
    """Unified data fetching interface"""
    
    def __init__(self):
        """Initialize data fetcher with API clients"""
        self.polygon_client = PolygonClient()
        self.tavily_client = TavilySearchClient()
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_stock_universe(self, market_cap_min: float = 1e9) -> List[str]:
        """
        Get a universe of stocks to screen
        
        Args:
            market_cap_min: Minimum market cap in USD
        
        Returns:
            List of stock symbols
        """
        # For now, return a curated list of major stocks
        # In production, this would query for stocks meeting market cap criteria
        major_stocks = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'CRM',
            'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
            
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY', 'MRK',
            'GILD', 'AMGN', 'ISRG', 'VRTX', 'REGN', 'BIIB', 'ILMN', 'MRNA', 'ZTS', 'CVS',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
            'PNC', 'TFC', 'COF', 'BK', 'STT', 'NTRS', 'RF', 'FITB', 'HBAN', 'KEY',
            
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
            'COST', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'NFLX', 'ROKU', 'SPOT',
            
            # Industrial
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD',
            'FDX', 'UNP', 'CSX', 'NSC', 'KSU', 'CP', 'CNI', 'ODFL', 'XPO', 'CHRW',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'BKR',
            'HAL', 'DVN', 'FANG', 'APA', 'EQT', 'COG', 'MRO', 'OVV', 'SM', 'NOV'
        ]
        
        return major_stocks
    
    def get_historical_data(self, symbols: List[str], years_back: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            years_back: Number of years of historical data
        
        Returns:
            Dictionary mapping symbols to price DataFrames
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_str} to {end_str}")
        
        return self.polygon_client.get_multiple_stocks_data(symbols, start_str, end_str)
    
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate various return metrics for stocks
        
        Args:
            price_data: Dictionary of price DataFrames
        
        Returns:
            Dictionary with return calculations
        """
        returns_data = {}
        
        for symbol, df in price_data.items():
            if df.empty:
                continue
            
            returns_df = pd.DataFrame(index=df.index)
            
            # Daily returns
            returns_df['daily_return'] = df['close'].pct_change()
            
            # Cumulative returns
            returns_df['cumulative_return'] = (1 + returns_df['daily_return']).cumprod() - 1
            
            # Rolling returns
            returns_df['return_1m'] = df['close'].pct_change(periods=21)  # ~1 month
            returns_df['return_3m'] = df['close'].pct_change(periods=63)  # ~3 months
            returns_df['return_6m'] = df['close'].pct_change(periods=126) # ~6 months
            returns_df['return_1y'] = df['close'].pct_change(periods=252) # ~1 year
            
            # Volatility measures
            returns_df['volatility_30d'] = returns_df['daily_return'].rolling(30).std() * np.sqrt(252)
            returns_df['volatility_90d'] = returns_df['daily_return'].rolling(90).std() * np.sqrt(252)
            
            # Moving averages
            returns_df['sma_20'] = df['close'].rolling(20).mean()
            returns_df['sma_50'] = df['close'].rolling(50).mean()
            returns_df['sma_200'] = df['close'].rolling(200).mean()
            
            # Price relative to moving averages
            returns_df['price_vs_sma20'] = df['close'] / returns_df['sma_20'] - 1
            returns_df['price_vs_sma50'] = df['close'] / returns_df['sma_50'] - 1
            returns_df['price_vs_sma200'] = df['close'] / returns_df['sma_200'] - 1
            
            # Add price data
            returns_df['close'] = df['close']
            returns_df['volume'] = df['volume']
            
            returns_data[symbol] = returns_df
        
        return returns_data
    
    def identify_outliers(self, returns_data: Dict[str, pd.DataFrame], 
                         lookback_days: int = 252) -> pd.DataFrame:
        """
        Identify outlier performers over a specified period
        
        Args:
            returns_data: Dictionary of return DataFrames
            lookback_days: Number of days to look back for performance
        
        Returns:
            DataFrame with outlier analysis
        """
        outlier_metrics = []
        
        for symbol, df in returns_data.items():
            if df.empty or len(df) < lookback_days:
                continue
            
            # Get recent data
            recent_data = df.tail(lookback_days)
            
            if recent_data.empty:
                continue
            
            # Calculate performance metrics
            total_return = recent_data['cumulative_return'].iloc[-1] if not recent_data['cumulative_return'].isna().all() else 0
            volatility = recent_data['daily_return'].std() * np.sqrt(252) if not recent_data['daily_return'].isna().all() else 0
            sharpe_ratio = (recent_data['daily_return'].mean() * 252) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + recent_data['daily_return'].fillna(0)).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Recent momentum
            recent_1m = recent_data['return_1m'].iloc[-1] if not recent_data['return_1m'].isna().all() else 0
            recent_3m = recent_data['return_3m'].iloc[-1] if not recent_data['return_3m'].isna().all() else 0
            
            # Technical indicators
            price_vs_sma20 = recent_data['price_vs_sma20'].iloc[-1] if not recent_data['price_vs_sma20'].isna().all() else 0
            price_vs_sma50 = recent_data['price_vs_sma50'].iloc[-1] if not recent_data['price_vs_sma50'].isna().all() else 0
            
            outlier_metrics.append({
                'symbol': symbol,
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'return_1m': recent_1m,
                'return_3m': recent_3m,
                'price_vs_sma20': price_vs_sma20,
                'price_vs_sma50': price_vs_sma50,
                'current_price': recent_data['close'].iloc[-1] if not recent_data['close'].empty else 0
            })
        
        outlier_df = pd.DataFrame(outlier_metrics)
        
        if outlier_df.empty:
            return outlier_df
        
        # Calculate percentile ranks for outlier identification
        for col in ['total_return', 'sharpe_ratio', 'return_1m', 'return_3m']:
            if col in outlier_df.columns:
                outlier_df[f'{col}_percentile'] = outlier_df[col].rank(pct=True)
        
        # Create composite outlier score
        score_columns = [col for col in outlier_df.columns if col.endswith('_percentile')]
        if score_columns:
            outlier_df['outlier_score'] = outlier_df[score_columns].mean(axis=1)
        
        return outlier_df.sort_values('outlier_score', ascending=False)
    
    def get_market_research(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get market research and news for symbols
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dictionary with research data for each symbol
        """
        research_data = {}
        
        for symbol in symbols[:5]:  # Limit to avoid rate limits
            try:
                news = self.tavily_client.search_stock_news(symbol, days=30, max_results=5)
                analysis = self.tavily_client.search_company_analysis(symbol, symbol, max_results=3)
                
                research_data[symbol] = {
                    'news': news,
                    'analysis': analysis,
                    'insights': self.tavily_client.extract_key_insights(news + analysis)
                }
                
            except Exception as e:
                logger.error(f"Error fetching research for {symbol}: {str(e)}")
                research_data[symbol] = {'news': [], 'analysis': [], 'insights': {}}
        
        return research_data
    
    def save_data_to_csv(self, data: Dict, filename: str):
        """Save data to CSV file"""
        filepath = os.path.join(self.cache_dir, filename)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath)
        elif isinstance(data, dict):
            # Convert dict to DataFrame if possible
            try:
                df = pd.DataFrame(data)
                df.to_csv(filepath)
            except:
                # Save as JSON if can't convert to DataFrame
                with open(filepath.replace('.csv', '.json'), 'w') as f:
                    json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Data saved to {filepath}")
    
    def load_data_from_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file"""
        filepath = os.path.join(self.cache_dir, filename)
        
        try:
            if os.path.exists(filepath):
                return pd.read_csv(filepath, index_col=0)
            else:
                logger.warning(f"File not found: {filepath}")
                return None
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            return None

