"""
Comprehensive LangChain/LangGraph Integration Test
Tests the AI agent functionality with the provided OpenAI API key
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_setup():
    """Test that environment variables are properly configured"""
    print("🔧 Testing Environment Setup...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    if not tavily_key:
        print("❌ TAVILY_API_KEY not found in environment")
        return False
    
    print(f"✅ OPENAI_API_KEY: {openai_key[:20]}...")
    print(f"✅ TAVILY_API_KEY: {tavily_key[:20]}...")
    return True

def test_basic_langchain_import():
    """Test basic LangChain imports"""
    print("\n📦 Testing LangChain Imports...")
    
    try:
        from langchain.llms import OpenAI
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        from langchain.agents import initialize_agent, Tool
        from langchain.agents import AgentType
        print("✅ Basic LangChain imports successful")
        return True
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n🤖 Testing OpenAI API Connection...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Initialize ChatOpenAI with the API key
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Test a simple query
        response = llm.invoke("Hello! Please respond with 'OpenAI connection successful'")
        print(f"✅ OpenAI Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def test_tavily_integration():
    """Test Tavily search integration"""
    print("\n🔍 Testing Tavily Search Integration...")
    
    try:
        import requests
        
        tavily_key = os.getenv('TAVILY_API_KEY')
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": tavily_key,
            "query": "Apple stock price today",
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 3
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Tavily search successful: {len(data.get('results', []))} results")
            return True
        else:
            print(f"❌ Tavily search failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Tavily integration failed: {e}")
        return False

def test_kangro_agent():
    """Test the Kangro AI agent functionality"""
    print("\n🧠 Testing Kangro AI Agent...")
    
    try:
        # Import our custom agent
        sys.path.append('.')
        from utils.kangro_agent import KangroAgent
        
        # Initialize the agent
        agent = KangroAgent()
        
        # Test agent initialization
        if hasattr(agent, 'llm') and hasattr(agent, 'search_tool'):
            print("✅ Kangro Agent initialized successfully")
        else:
            print("❌ Kangro Agent missing required components")
            return False
        
        # Test a simple query
        test_query = "What are the key factors to consider when screening stocks?"
        response = agent.analyze_screening_results(
            screening_results={'test': 'data'},
            user_query=test_query
        )
        
        if response and len(response) > 50:
            print(f"✅ Agent response generated: {response[:100]}...")
            return True
        else:
            print("❌ Agent response too short or empty")
            return False
            
    except Exception as e:
        print(f"❌ Kangro Agent test failed: {e}")
        return False

def test_langchain_tools():
    """Test LangChain tools and agents"""
    print("\n🛠️ Testing LangChain Tools and Agents...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import initialize_agent, Tool, AgentType
        from langchain.schema import SystemMessage
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Define a simple tool
        def stock_analysis_tool(query: str) -> str:
            """A simple stock analysis tool"""
            return f"Stock analysis for: {query}. This is a test response."
        
        tools = [
            Tool(
                name="StockAnalysis",
                func=stock_analysis_tool,
                description="Analyze stocks and provide investment insights"
            )
        ]
        
        # Initialize agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )
        
        # Test agent
        response = agent.run("Analyze Apple stock")
        print(f"✅ LangChain agent response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ LangChain tools test failed: {e}")
        return False

def test_investment_strategy_generation():
    """Test AI-powered investment strategy generation"""
    print("\n💡 Testing Investment Strategy Generation...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Test investment strategy generation
        messages = [
            SystemMessage(content="You are an expert investment advisor. Provide concise, actionable investment strategies."),
            HumanMessage(content="Generate a conservative investment strategy for a 30-year-old with $50,000 to invest.")
        ]
        
        response = llm.invoke(messages)
        strategy = response.content
        
        if strategy and len(strategy) > 100:
            print(f"✅ Investment strategy generated: {strategy[:150]}...")
            return True
        else:
            print("❌ Investment strategy generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Investment strategy test failed: {e}")
        return False

def test_market_sentiment_analysis():
    """Test market sentiment analysis using AI"""
    print("\n📊 Testing Market Sentiment Analysis...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Test sentiment analysis
        news_text = "Apple reported strong quarterly earnings, beating analyst expectations. The company's revenue grew 15% year-over-year."
        
        prompt = f"""
        Analyze the market sentiment of the following news:
        "{news_text}"
        
        Provide:
        1. Sentiment (Positive/Negative/Neutral)
        2. Confidence level (1-10)
        3. Key factors
        
        Format as JSON.
        """
        
        response = llm.invoke(prompt)
        sentiment_analysis = response.content
        
        if sentiment_analysis and "sentiment" in sentiment_analysis.lower():
            print(f"✅ Sentiment analysis: {sentiment_analysis[:100]}...")
            return True
        else:
            print("❌ Sentiment analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Market sentiment test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all LangChain integration tests"""
    print("🚀 Starting Comprehensive LangChain Integration Test")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Environment Setup", test_environment_setup),
        ("LangChain Imports", test_basic_langchain_import),
        ("OpenAI Connection", test_openai_connection),
        ("Tavily Integration", test_tavily_integration),
        ("Kangro Agent", test_kangro_agent),
        ("LangChain Tools", test_langchain_tools),
        ("Investment Strategy", test_investment_strategy_generation),
        ("Market Sentiment", test_market_sentiment_analysis)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate >= 75:
        print("🎉 LangChain integration is working well!")
    elif success_rate >= 50:
        print("⚠️ LangChain integration has some issues but core functionality works")
    else:
        print("🚨 LangChain integration needs attention")
    
    # Save test results
    test_report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": success_rate,
        "test_results": test_results
    }
    
    with open("langchain_test_report.json", "w") as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n📄 Test report saved to: langchain_test_report.json")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)

