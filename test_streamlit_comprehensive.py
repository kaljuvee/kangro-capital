"""
Comprehensive test script for Streamlit app
Tests all functions and identifies errors before deployment
"""

import sys
import traceback
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_function(func_name, func, *args, **kwargs):
    """Test a function and report results"""
    try:
        result = func(*args, **kwargs)
        print(f"✅ {func_name}: SUCCESS")
        return True, result
    except Exception as e:
        print(f"❌ {func_name}: ERROR - {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False, None

def get_stock_data(symbol, period="1y"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def calculate_basic_metrics(data, info, symbol="TEST"):
    """Calculate basic stock metrics - FIXED VERSION"""
    if data is None or data.empty:
        return {}
    
    current_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
    price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
    
    # Calculate moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Basic metrics
    metrics = {
        'current_price': current_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'volume': data['Volume'].iloc[-1],
        'high_52w': data['High'].max(),
        'low_52w': data['Low'].min(),
        'ma20': data['MA20'].iloc[-1] if not pd.isna(data['MA20'].iloc[-1]) else None,
        'ma50': data['MA50'].iloc[-1] if not pd.isna(data['MA50'].iloc[-1]) else None,
    }
    
    # Add info metrics if available
    if info:
        metrics.update({
            'pe_ratio': info.get('trailingPE'),
            'market_cap': info.get('marketCap'),
            'dividend_yield': info.get('dividendYield'),
            'beta': info.get('beta'),
            'company_name': info.get('longName', symbol),  # Fixed: added symbol parameter
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'roe': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'revenue_growth': info.get('revenueGrowth')
        })
    
    return metrics

def get_ai_analysis(prompt, max_tokens=500):
    """Get AI analysis using OpenAI API directly"""
    try:
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            return "AI analysis unavailable - OpenAI API key not configured"
        
        import requests
        
        headers = {
            'Authorization': f'Bearer {openai_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4o-mini',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a professional financial analyst. Provide concise insights.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': max_tokens,
            'temperature': 0.3
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"AI analysis error: {response.status_code}"
            
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

def search_market_news(query, max_results=3):
    """Search for market news using Tavily"""
    try:
        tavily_key = os.getenv('TAVILY_API_KEY')
        if not tavily_key:
            return []
        
        import requests
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": tavily_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": max_results
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            return []
            
    except Exception as e:
        print(f"News search error: {str(e)}")
        return []

def screen_stocks(symbols, criteria):
    """Stock screening with AI-enhanced analysis"""
    results = []
    
    for symbol in symbols[:3]:  # Test with first 3 symbols only
        print(f"Testing screening for {symbol}...")
        data, info = get_stock_data(symbol, period="1y")
        if data is not None and not data.empty:
            metrics = calculate_basic_metrics(data, info, symbol)  # Fixed: pass symbol
            
            # Apply screening criteria
            score = 0
            reasons = []
            
            # PE Ratio check
            if metrics.get('pe_ratio') and criteria.get('max_pe'):
                if metrics['pe_ratio'] <= criteria['max_pe']:
                    score += 1
                    reasons.append(f"PE ratio {metrics['pe_ratio']:.1f} ≤ {criteria['max_pe']}")
            
            # ROE check
            if metrics.get('roe') and criteria.get('min_roe'):
                if metrics['roe'] >= criteria['min_roe']:
                    score += 1
                    reasons.append(f"ROE {metrics['roe']*100:.1f}% ≥ {criteria['min_roe']*100:.1f}%")
            
            results.append({
                'symbol': symbol,
                'company_name': metrics.get('company_name', symbol),
                'current_price': metrics['current_price'],
                'score': score,
                'reasons': '; '.join(reasons) if reasons else 'No criteria met'
            })
    
    return results

def analyze_portfolio_performance(symbols, weights=None):
    """Analyze portfolio performance"""
    if weights is None:
        weights = [1/len(symbols)] * len(symbols)
    
    portfolio_data = []
    
    for symbol, weight in zip(symbols[:2], weights[:2]):  # Test with first 2 symbols
        data, _ = get_stock_data(symbol, period="1y")
        if data is not None and not data.empty:
            returns = data['Close'].pct_change().dropna()
            portfolio_data.append(returns * weight)
    
    if portfolio_data:
        portfolio_returns = pd.concat(portfolio_data, axis=1).sum(axis=1)
        
        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
        volatility = portfolio_returns.std() * (252 ** 0.5)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'returns': portfolio_returns
        }
    
    return None

def main():
    """Run comprehensive tests"""
    print("🧪 COMPREHENSIVE STREAMLIT APP TESTING")
    print("=" * 50)
    
    # Test 1: Data fetching
    print("\n📊 Testing Data Fetching...")
    success, (data, info) = test_function("get_stock_data", get_stock_data, "AAPL", "1y")
    
    if success and data is not None:
        # Test 2: Metrics calculation
        print("\n📈 Testing Metrics Calculation...")
        success, metrics = test_function("calculate_basic_metrics", calculate_basic_metrics, data, info, "AAPL")
        
        if success:
            print(f"   Sample metrics: Price=${metrics.get('current_price', 0):.2f}, PE={metrics.get('pe_ratio', 'N/A')}")
    
    # Test 3: AI Analysis
    print("\n🤖 Testing AI Analysis...")
    test_prompt = "Analyze AAPL stock briefly in 50 words."
    success, ai_result = test_function("get_ai_analysis", get_ai_analysis, test_prompt, 100)
    if success:
        print(f"   AI Response: {ai_result[:100]}...")
    
    # Test 4: News Search
    print("\n📰 Testing News Search...")
    success, news = test_function("search_market_news", search_market_news, "AAPL stock", 2)
    if success:
        print(f"   Found {len(news)} news items")
    
    # Test 5: Stock Screening
    print("\n🔍 Testing Stock Screening...")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    test_criteria = {'max_pe': 25.0, 'min_roe': 0.15}
    success, screening_results = test_function("screen_stocks", screen_stocks, test_symbols, test_criteria)
    if success:
        print(f"   Screened {len(screening_results)} stocks")
    
    # Test 6: Portfolio Analysis
    print("\n💼 Testing Portfolio Analysis...")
    test_portfolio_symbols = ['AAPL', 'MSFT']
    test_weights = [0.6, 0.4]
    success, portfolio_results = test_function("analyze_portfolio_performance", analyze_portfolio_performance, test_portfolio_symbols, test_weights)
    if success and portfolio_results:
        print(f"   Portfolio return: {portfolio_results['annual_return']*100:.1f}%")
    
    # Test 7: Error handling
    print("\n⚠️ Testing Error Handling...")
    success, _ = test_function("get_stock_data_invalid", get_stock_data, "INVALID_SYMBOL", "1y")
    success, _ = test_function("calculate_basic_metrics_empty", calculate_basic_metrics, pd.DataFrame(), {}, "TEST")
    
    print("\n" + "=" * 50)
    print("🎯 TESTING COMPLETE")
    print("Review any ❌ errors above and fix before deployment")

if __name__ == "__main__":
    main()

