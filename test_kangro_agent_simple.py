"""
Simple test for the updated Kangro agent
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_kangro_agent_simple():
    """Test the updated Kangro agent with supported model"""
    print("🧠 Testing Updated Kangro Agent...")
    
    try:
        # Import the agent
        sys.path.append('.')
        from utils.kangro_agent import KangroAgent
        
        # Initialize the agent
        agent = KangroAgent()
        print(f"✅ Agent initialized with model: {agent.model}")
        
        # Test a simple analysis
        test_data = {
            'symbol': 'AAPL',
            'pe_ratio': 25.5,
            'roe': 0.15,
            'debt_to_equity': 0.3,
            'revenue_growth': 0.08
        }
        
        # Test fundamental analysis
        result = agent.analyze_stock_fundamentals(test_data)
        
        if result and 'analysis' in result:
            print(f"✅ Fundamental analysis successful: {result['analysis'][:100]}...")
            return True
        else:
            print("❌ Fundamental analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Kangro agent test failed: {e}")
        return False

def test_direct_api_call():
    """Test direct API call with supported model"""
    print("\n🔧 Testing Direct API Call...")
    
    try:
        import requests
        
        openai_key = os.getenv('OPENAI_API_KEY')
        
        headers = {
            'Authorization': f'Bearer {openai_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4.1-mini',
            'messages': [
                {
                    'role': 'user',
                    'content': 'Analyze Apple stock in one sentence.'
                }
            ],
            'max_tokens': 100
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ Direct API call successful: {content}")
            return True
        else:
            print(f"❌ Direct API call failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Kangro Agent with Supported Model")
    print("=" * 50)
    
    # Test direct API call first
    api_success = test_direct_api_call()
    
    # Test Kangro agent
    agent_success = test_kangro_agent_simple()
    
    print("\n" + "=" * 50)
    print("📋 SIMPLE TEST SUMMARY")
    print("=" * 50)
    print(f"Direct API Call:      {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"Kangro Agent:         {'✅ PASS' if agent_success else '❌ FAIL'}")
    
    if api_success and agent_success:
        print("\n🎉 All tests passed! Kangro agent is working with supported model.")
    elif api_success:
        print("\n⚠️ API works but agent has issues. Check agent implementation.")
    else:
        print("\n🚨 API connection failed. Check API key and model support.")

