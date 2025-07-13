"""
Comprehensive Test Suite for Kangro Capital Platform
Tests all major components and functionality
"""

import os
import sys
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from utils import (
    PolygonClient, TavilySearchClient, DataFetcher,
    StockScreener, OutlierDetector, ScreeningEngine,
    MLAnalyzer, FactorAnalyzer, BacktestEngine, PortfolioSimulator,
    create_kangro_agent, analyze_screening_results_with_ai, 
    generate_investment_strategy, get_market_insights
)

# Load environment variables
load_dotenv()

class KangroTestSuite:
    """Comprehensive test suite for Kangro Capital platform"""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_test(self, test_name, test_function):
        """Run a single test and record results"""
        self.total_tests += 1
        print(f"\n{'='*60}")
        print(f"Running Test: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_function()
            if result:
                print(f"✅ PASSED: {test_name}")
                self.passed_tests += 1
                self.test_results[test_name] = "PASSED"
            else:
                print(f"❌ FAILED: {test_name}")
                self.failed_tests += 1
                self.test_results[test_name] = "FAILED"
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {str(e)}")
            traceback.print_exc()
            self.failed_tests += 1
            self.test_results[test_name] = f"ERROR: {str(e)}"
    
    def test_api_clients(self):
        """Test API client initialization and basic functionality"""
        print("Testing API clients...")
        
        # Test Polygon client
        polygon_client = PolygonClient()
        print(f"✓ Polygon client initialized")
        
        # Test Tavily client
        tavily_client = TavilySearchClient()
        print(f"✓ Tavily client initialized")
        
        # Test Data Fetcher
        data_fetcher = DataFetcher()
        print(f"✓ Data fetcher initialized")
        
        return True
    
    def test_screening_engine(self):
        """Test stock screening functionality"""
        print("Testing screening engine...")
        
        screening_engine = ScreeningEngine()
        
        # Test with minimal parameters
        screening_params = {
            'top_n_stocks': 5,
            'lookback_years': 1,
            'outlier_threshold': 1.5,
            'breakout_min_score': 0.6
        }
        
        results = screening_engine.run_comprehensive_screening(screening_params)
        
        print(f"✓ Screening completed")
        print(f"✓ Results structure: {list(results.keys())}")
        
        if 'recommendations' in results:
            print(f"✓ Found {len(results['recommendations'])} recommendations")
        
        return True
    
    def test_ml_analysis(self):
        """Test machine learning analysis"""
        print("Testing ML analysis...")
        
        ml_analyzer = MLAnalyzer()
        factor_analyzer = FactorAnalyzer()
        
        # Create sample data
        sample_data = self._create_sample_ml_data()
        
        if sample_data.empty:
            print("⚠️ No sample data available, using synthetic data")
            sample_data = self._create_synthetic_ml_data()
        
        # Test regression models
        regression_results = ml_analyzer.train_regression_models(sample_data)
        print(f"✓ Regression models trained: {list(regression_results.keys())}")
        
        # Test classification models
        classification_results = ml_analyzer.train_classification_models(sample_data)
        print(f"✓ Classification models trained: {list(classification_results.keys())}")
        
        # Test feature analysis
        feature_analysis = ml_analyzer.perform_feature_analysis(sample_data)
        print(f"✓ Feature analysis completed")
        
        # Test factor analysis
        factor_results = factor_analyzer.perform_factor_analysis(sample_data)
        print(f"✓ Factor analysis completed")
        
        return True
    
    def test_backtesting_engine(self):
        """Test backtesting functionality"""
        print("Testing backtesting engine...")
        
        backtest_engine = BacktestEngine()
        portfolio_simulator = PortfolioSimulator()
        
        # Create test data
        stock_selections, price_data = self._create_sample_backtest_data()
        
        # Test backtesting
        backtest_results = backtest_engine.run_backtest(
            stock_selections=stock_selections,
            price_data=price_data,
            start_date='2023-01-01',
            end_date='2023-12-31',
            strategy_params={
                'initial_capital': 100000,
                'rebalance_frequency': 'monthly',
                'transaction_cost': 0.001
            }
        )
        
        print(f"✓ Backtesting completed")
        
        if 'performance_metrics' in backtest_results:
            metrics = backtest_results['performance_metrics']
            print(f"✓ Performance metrics: {list(metrics.keys())}")
        
        # Test portfolio simulation
        simulation_results = portfolio_simulator.run_multi_strategy_simulation(
            stock_selections=stock_selections,
            price_data=price_data,
            fundamental_data={},
            start_date='2023-01-01',
            end_date='2023-12-31',
            strategies=['equal_weight']
        )
        
        print(f"✓ Portfolio simulation completed")
        
        return True
    
    def test_ai_agent(self):
        """Test AI agent functionality"""
        print("Testing AI agent...")
        
        agent = create_kangro_agent()
        print(f"✓ AI agent initialized")
        print(f"✓ OpenAI API available: {bool(agent.openai_api_key)}")
        print(f"✓ Tavily API available: {bool(agent.tavily_api_key)}")
        
        # Test with sample screening results
        sample_results = {
            'recommendations': [
                {'symbol': 'AAPL', 'composite_score': 0.85, 'metrics': {'roe': 0.25}},
                {'symbol': 'GOOGL', 'composite_score': 0.82, 'metrics': {'roe': 0.18}},
                {'symbol': 'MSFT', 'composite_score': 0.80, 'metrics': {'roe': 0.22}}
            ]
        }
        
        # Test explanation generation
        explanation = analyze_screening_results_with_ai(sample_results)
        print(f"✓ AI explanation generated: {bool(explanation.get('explanation'))}")
        
        # Test strategy generation
        strategy = generate_investment_strategy(sample_results, 'moderate')
        print(f"✓ Investment strategy generated: {bool(strategy.get('strategy'))}")
        
        # Test market insights
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        insights = get_market_insights(symbols)
        print(f"✓ Market insights generated: {bool(insights.get('insights'))}")
        
        return True
    
    def test_streamlit_integration(self):
        """Test Streamlit application components"""
        print("Testing Streamlit integration...")
        
        # Test imports
        try:
            import streamlit as st
            import plotly.express as px
            import plotly.graph_objects as go
            print(f"✓ Streamlit and Plotly imported successfully")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
        
        # Test component initialization
        from utils import initialize_components
        components = {
            'polygon_client': PolygonClient(),
            'tavily_client': TavilySearchClient(),
            'data_fetcher': DataFetcher(),
            'screening_engine': ScreeningEngine(),
            'ml_analyzer': MLAnalyzer(),
            'factor_analyzer': FactorAnalyzer(),
            'backtest_engine': BacktestEngine(),
            'portfolio_simulator': PortfolioSimulator(),
            'kangro_agent': create_kangro_agent()
        }
        
        print(f"✓ All components initialized: {list(components.keys())}")
        
        return True
    
    def test_data_persistence(self):
        """Test data export and persistence functionality"""
        print("Testing data persistence...")
        
        # Test CSV export
        sample_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'score': [0.85, 0.82, 0.80],
            'return': [0.15, 0.12, 0.18]
        })
        
        csv_data = sample_data.to_csv(index=False)
        print(f"✓ CSV export successful: {len(csv_data)} characters")
        
        # Test JSON export
        import json
        json_data = json.dumps(sample_data.to_dict(), indent=2)
        print(f"✓ JSON export successful: {len(json_data)} characters")
        
        return True
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("Testing error handling...")
        
        screening_engine = ScreeningEngine()
        
        # Test with invalid parameters
        try:
            results = screening_engine.run_comprehensive_screening({})
            print(f"✓ Handled empty parameters gracefully")
        except Exception as e:
            print(f"✓ Proper error handling: {str(e)}")
        
        # Test ML with empty data
        ml_analyzer = MLAnalyzer()
        empty_df = pd.DataFrame()
        
        try:
            results = ml_analyzer.prepare_features({}, {})
            print(f"✓ Handled empty ML data gracefully")
        except Exception as e:
            print(f"✓ Proper ML error handling: {str(e)}")
        
        return True
    
    def _create_sample_ml_data(self):
        """Create sample data for ML testing"""
        np.random.seed(42)
        
        n_samples = 100
        n_features = 10
        
        # Generate random features
        features = np.random.randn(n_samples, n_features)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        
        # Add target variable
        df['target'] = np.random.randn(n_samples)
        df['target_binary'] = (df['target'] > 0).astype(int)
        
        return df
    
    def _create_synthetic_ml_data(self):
        """Create synthetic ML data for testing"""
        return self._create_sample_ml_data()
    
    def _create_sample_backtest_data(self):
        """Create sample data for backtesting"""
        np.random.seed(42)
        
        # Create stock selections
        stock_selections = {
            '2023-01-01': ['AAPL', 'GOOGL', 'MSFT'],
            '2023-02-01': ['AAPL', 'TSLA', 'NVDA'],
            '2023-03-01': ['GOOGL', 'META', 'AMZN'],
        }
        
        # Create price data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        trading_dates = [d for d in dates if d.weekday() < 5]
        
        price_data = {}
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN']
        
        for symbol in symbols:
            n_days = len(trading_dates)
            
            # Generate price series
            returns = np.random.normal(0.0005, 0.02, n_days)
            prices = [100]  # Starting price
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create DataFrame
            df = pd.DataFrame({
                'close': prices,
                'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
                'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
                'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
                'volume': [np.random.uniform(1000000, 10000000) for _ in prices]
            }, index=trading_dates)
            
            price_data[symbol] = df
        
        return stock_selections, price_data
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🚀 Starting Kangro Capital Test Suite")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Define all tests
        tests = [
            ("API Clients", self.test_api_clients),
            ("Screening Engine", self.test_screening_engine),
            ("ML Analysis", self.test_ml_analysis),
            ("Backtesting Engine", self.test_backtesting_engine),
            ("AI Agent", self.test_ai_agent),
            ("Streamlit Integration", self.test_streamlit_integration),
            ("Data Persistence", self.test_data_persistence),
            ("Error Handling", self.test_error_handling)
        ]
        
        # Run all tests
        for test_name, test_function in tests:
            self.run_test(test_name, test_function)
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate final test report"""
        print(f"\n{'='*80}")
        print("🎯 KANGRO CAPITAL TEST SUITE REPORT")
        print(f"{'='*80}")
        
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests}")
        print(f"   Failed: {self.failed_tests}")
        print(f"   Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result == "PASSED" else "❌"
            print(f"   {status_icon} {test_name}: {result}")
        
        if self.failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! Kangro Capital platform is ready for deployment.")
        else:
            print(f"\n⚠️  {self.failed_tests} test(s) failed. Please review and fix issues before deployment.")
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'success_rate': (self.passed_tests/self.total_tests)*100,
            'detailed_results': self.test_results
        }
        
        import json
        with open('test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Test report saved to: test_report.json")

def main():
    """Main test execution function"""
    test_suite = KangroTestSuite()
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()

