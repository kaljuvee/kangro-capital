"""
Polygon.io API Client for stock data retrieval
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

class PolygonClient:
    """Client for interacting with Polygon.io API"""
    
    def __init__(self, api_key: str = None):
        """Initialize the Polygon client"""
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key is required")
        
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # 100ms between requests for free tier
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to the Polygon API with rate limiting"""
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{self.base_url}{endpoint}"
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise
    
    def get_stock_price_data(self, symbol: str, start_date: str, end_date: str, 
                           timespan: str = "day", multiplier: int = 1) -> pd.DataFrame:
        """
        Get historical stock price data
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timespan: Time span (day, week, month, quarter, year)
            multiplier: Multiplier for timespan
        
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        
        try:
            data = self._make_request(endpoint)
            
            if 'results' not in data or not data['results']:
                logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data['results'])
            
            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            
            # Rename columns to standard format
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'volume_weighted_price',
                'n': 'transactions'
            })
            
            # Set date as index and sort
            df = df.set_index('date').sort_index()
            
            # Select relevant columns
            columns = ['open', 'high', 'low', 'close', 'volume']
            df = df[columns]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_financials(self, symbol: str, limit: int = 5) -> Dict:
        """
        Get financial data for a stock
        
        Args:
            symbol: Stock symbol
            limit: Number of financial reports to retrieve
        
        Returns:
            Dictionary with financial data
        """
        endpoint = f"/vX/reference/financials"
        params = {
            'ticker': symbol,
            'limit': limit,
            'sort': 'filing_date',
            'order': 'desc'
        }
        
        try:
            data = self._make_request(endpoint, params)
            return data
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {str(e)}")
            return {}
    
    def get_ticker_details(self, symbol: str) -> Dict:
        """
        Get detailed information about a ticker
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with ticker details
        """
        endpoint = f"/v3/reference/tickers/{symbol}"
        
        try:
            data = self._make_request(endpoint)
            return data.get('results', {})
        except Exception as e:
            logger.error(f"Error fetching ticker details for {symbol}: {str(e)}")
            return {}
    
    def search_tickers(self, query: str, market: str = "stocks", 
                      active: bool = True, limit: int = 100) -> List[Dict]:
        """
        Search for tickers
        
        Args:
            query: Search query
            market: Market type (stocks, crypto, fx)
            active: Only active tickers
            limit: Maximum number of results
        
        Returns:
            List of ticker information
        """
        endpoint = "/v3/reference/tickers"
        params = {
            'search': query,
            'market': market,
            'active': active,
            'limit': limit
        }
        
        try:
            data = self._make_request(endpoint, params)
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Error searching tickers with query '{query}': {str(e)}")
            return []
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        endpoint = "/v1/marketstatus/now"
        
        try:
            data = self._make_request(endpoint)
            return data
        except Exception as e:
            logger.error(f"Error fetching market status: {str(e)}")
            return {}
    
    def get_multiple_stocks_data(self, symbols: List[str], start_date: str, 
                               end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Get price data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Dictionary mapping symbols to their price DataFrames
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.get_stock_price_data(symbol, start_date, end_date)
            if not df.empty:
                results[symbol] = df
            else:
                logger.warning(f"No data retrieved for {symbol}")
        
        return results

