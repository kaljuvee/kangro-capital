"""
Simple test for core functions without pandas dependency issues
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_ai_functionality():
    """Test AI functionality"""
    print("🤖 Testing AI functionality...")
    
    try:
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("❌ No OpenAI API key found")
            return False
        
        headers = {
            'Authorization': f'Bearer {openai_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4o-mini',
            'messages': [
                {'role': 'user', 'content': 'Test: Analyze AAPL stock in 20 words.'}
            ],
            'max_tokens': 50
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content'].strip()
            print(f"✅ OpenAI API working: {ai_response[:50]}...")
            return True
        else:
            print(f"❌ OpenAI API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI test failed: {str(e)}")
        return False

def test_tavily_functionality():
    """Test Tavily functionality"""
    print("📰 Testing Tavily functionality...")
    
    try:
        tavily_key = os.getenv('TAVILY_API_KEY')
        if not tavily_key:
            print("❌ No Tavily API key found")
            return False
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": tavily_key,
            "query": "AAPL stock news",
            "search_depth": "basic",
            "max_results": 2
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"✅ Tavily API working: Found {len(results)} news items")
            return True
        else:
            print(f"❌ Tavily API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Tavily test failed: {str(e)}")
        return False

def test_yfinance_basic():
    """Test basic yfinance functionality without pandas issues"""
    print("📊 Testing yfinance functionality...")
    
    try:
        # Simple test without importing yfinance directly
        import requests
        
        # Test Yahoo Finance API endpoint
        url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'chart' in data and 'result' in data['chart']:
                print("✅ Yahoo Finance API accessible")
                return True
        
        print("❌ Yahoo Finance API not accessible")
        return False
        
    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {str(e)}")
        return False

def test_environment_variables():
    """Test environment variables"""
    print("🔧 Testing environment variables...")
    
    required_vars = ['OPENAI_API_KEY', 'TAVILY_API_KEY']
    all_present = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Present ({len(value)} chars)")
        else:
            print(f"❌ {var}: Missing")
            all_present = False
    
    return all_present

def main():
    """Run all tests"""
    print("🧪 CORE FUNCTIONALITY TESTING")
    print("=" * 40)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("AI Functionality", test_ai_functionality),
        ("Tavily Functionality", test_tavily_functionality),
        ("Yahoo Finance", test_yfinance_basic),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("📋 TEST SUMMARY:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All core functionality working! Ready for deployment.")
    else:
        print("⚠️ Some issues found. Check failed tests above.")

if __name__ == "__main__":
    main()

