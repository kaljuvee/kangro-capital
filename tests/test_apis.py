"""
Test script to verify API connections and basic functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from utils.polygon_client import PolygonClient
from utils.tavily_client import TavilySearchClient
from utils.data_fetcher import DataFetcher

# Load environment variables
load_dotenv()

def test_polygon_api():
    """Test Polygon API connection"""
    print("Testing Polygon API...")
    
    try:
        client = PolygonClient()
        
        # Test market status
        status = client.get_market_status()
        print(f"Market status retrieved: {bool(status)}")
        
        # Test ticker search
        tickers = client.search_tickers("Apple", limit=5)
        print(f"Found {len(tickers)} tickers for 'Apple'")
        
        # Test price data for a single stock
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        price_data = client.get_stock_price_data(
            'AAPL', 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        print(f"Retrieved {len(price_data)} days of AAPL price data")
        
        print("✓ Polygon API test passed")
        return True
        
    except Exception as e:
        print(f"✗ Polygon API test failed: {str(e)}")
        return False

def test_tavily_api():
    """Test Tavily API connection"""
    print("\nTesting Tavily API...")
    
    try:
        client = TavilySearchClient()
        
        # Test stock news search
        news = client.search_stock_news('AAPL', days=7, max_results=3)
        print(f"Found {len(news)} news articles for AAPL")
        
        # Test market conditions search
        market_news = client.search_market_conditions(max_results=3)
        print(f"Found {len(market_news)} market condition articles")
        
        print("✓ Tavily API test passed")
        return True
        
    except Exception as e:
        print(f"✗ Tavily API test failed: {str(e)}")
        return False

def test_data_fetcher():
    """Test unified data fetcher"""
    print("\nTesting Data Fetcher...")
    
    try:
        fetcher = DataFetcher()
        
        # Test stock universe
        universe = fetcher.get_stock_universe()
        print(f"Stock universe contains {len(universe)} symbols")
        
        # Test historical data for a few stocks
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        historical_data = fetcher.get_historical_data(test_symbols, years_back=1)
        print(f"Retrieved historical data for {len(historical_data)} symbols")
        
        # Test returns calculation
        if historical_data:
            returns_data = fetcher.calculate_returns(historical_data)
            print(f"Calculated returns for {len(returns_data)} symbols")
            
            # Test outlier identification
            outliers = fetcher.identify_outliers(returns_data, lookback_days=100)
            print(f"Identified outliers: {len(outliers)} stocks analyzed")
        
        print("✓ Data Fetcher test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data Fetcher test failed: {str(e)}")
        return False

def main():
    """Run all API tests"""
    print("Kangro Capital - API Connection Tests")
    print("=" * 40)
    
    results = []
    
    # Test individual APIs
    results.append(test_polygon_api())
    results.append(test_tavily_api())
    results.append(test_data_fetcher())
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"Polygon API: {'✓ PASS' if results[0] else '✗ FAIL'}")
    print(f"Tavily API: {'✓ PASS' if results[1] else '✗ FAIL'}")
    print(f"Data Fetcher: {'✓ PASS' if results[2] else '✗ FAIL'}")
    
    if all(results):
        print("\n🎉 All tests passed! APIs are ready for use.")
    else:
        print("\n⚠️  Some tests failed. Please check API keys and connections.")
    
    return all(results)

if __name__ == "__main__":
    main()

