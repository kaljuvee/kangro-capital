"""
Tavily Search Integration for market research and analysis
"""

import os
from tavily import TavilyClient
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TavilySearchClient:
    """Client for Tavily search API integration"""
    
    def __init__(self, api_key: str = None):
        """Initialize Tavily client"""
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("Tavily API key is required")
        
        self.client = TavilyClient(api_key=self.api_key)
    
    def search_stock_news(self, symbol: str, days: int = 30, max_results: int = 10) -> List[Dict]:
        """
        Search for recent news about a specific stock
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            max_results: Maximum number of results
        
        Returns:
            List of news articles
        """
        query = f"{symbol} stock news earnings financial performance"
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                days=days
            )
            
            return response.get('results', [])
            
        except Exception as e:
            logger.error(f"Error searching news for {symbol}: {str(e)}")
            return []
    
    def search_company_analysis(self, company_name: str, symbol: str, max_results: int = 5) -> List[Dict]:
        """
        Search for company analysis and research reports
        
        Args:
            company_name: Full company name
            symbol: Stock symbol
            max_results: Maximum number of results
        
        Returns:
            List of analysis articles
        """
        query = f"{company_name} {symbol} financial analysis research report valuation"
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            
            return response.get('results', [])
            
        except Exception as e:
            logger.error(f"Error searching analysis for {company_name}: {str(e)}")
            return []
    
    def search_sector_trends(self, sector: str, max_results: int = 10) -> List[Dict]:
        """
        Search for sector trends and analysis
        
        Args:
            sector: Sector name (e.g., "technology", "healthcare")
            max_results: Maximum number of results
        
        Returns:
            List of sector analysis articles
        """
        query = f"{sector} sector trends analysis outlook investment"
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            
            return response.get('results', [])
            
        except Exception as e:
            logger.error(f"Error searching sector trends for {sector}: {str(e)}")
            return []
    
    def search_market_conditions(self, max_results: int = 15) -> List[Dict]:
        """
        Search for general market conditions and outlook
        
        Args:
            max_results: Maximum number of results
        
        Returns:
            List of market analysis articles
        """
        query = "stock market outlook economic conditions investment trends 2024"
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                days=7  # Recent market conditions
            )
            
            return response.get('results', [])
            
        except Exception as e:
            logger.error(f"Error searching market conditions: {str(e)}")
            return []
    
    def search_similar_stocks(self, symbol: str, sector: str, max_results: int = 10) -> List[Dict]:
        """
        Search for information about similar stocks in the same sector
        
        Args:
            symbol: Reference stock symbol
            sector: Sector name
            max_results: Maximum number of results
        
        Returns:
            List of articles about similar stocks
        """
        query = f"stocks similar to {symbol} {sector} sector comparison alternatives"
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            
            return response.get('results', [])
            
        except Exception as e:
            logger.error(f"Error searching similar stocks to {symbol}: {str(e)}")
            return []
    
    def search_financial_metrics(self, symbol: str, metrics: List[str], max_results: int = 8) -> List[Dict]:
        """
        Search for specific financial metrics information
        
        Args:
            symbol: Stock symbol
            metrics: List of financial metrics to search for
            max_results: Maximum number of results
        
        Returns:
            List of articles about financial metrics
        """
        metrics_str = " ".join(metrics)
        query = f"{symbol} {metrics_str} financial metrics ratios analysis"
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            
            return response.get('results', [])
            
        except Exception as e:
            logger.error(f"Error searching financial metrics for {symbol}: {str(e)}")
            return []
    
    def extract_key_insights(self, search_results: List[Dict]) -> Dict:
        """
        Extract key insights from search results
        
        Args:
            search_results: List of search results from Tavily
        
        Returns:
            Dictionary with extracted insights
        """
        insights = {
            'total_articles': len(search_results),
            'sources': [],
            'key_topics': [],
            'sentiment_indicators': [],
            'recent_developments': []
        }
        
        for result in search_results:
            # Extract source information
            if 'url' in result:
                domain = result['url'].split('/')[2] if '/' in result['url'] else result['url']
                insights['sources'].append(domain)
            
            # Extract content snippets for analysis
            if 'content' in result:
                content = result['content'].lower()
                
                # Simple sentiment indicators
                positive_words = ['growth', 'increase', 'strong', 'positive', 'bullish', 'outperform']
                negative_words = ['decline', 'decrease', 'weak', 'negative', 'bearish', 'underperform']
                
                positive_count = sum(1 for word in positive_words if word in content)
                negative_count = sum(1 for word in negative_words if word in content)
                
                if positive_count > negative_count:
                    insights['sentiment_indicators'].append('positive')
                elif negative_count > positive_count:
                    insights['sentiment_indicators'].append('negative')
                else:
                    insights['sentiment_indicators'].append('neutral')
        
        # Remove duplicates and get unique sources
        insights['sources'] = list(set(insights['sources']))
        
        return insights

