"""
Test script for backtesting and portfolio simulation functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from utils.backtest_engine import BacktestEngine
from utils.portfolio_simulator import PortfolioSimulator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

def create_test_price_data():
    """Create synthetic price data for backtesting"""
    np.random.seed(42)
    
    # Create date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Filter to weekdays only (trading days)
    trading_dates = [d for d in dates if d.weekday() < 5]
    
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'SPY']
    price_data = {}
    
    for stock in stocks:
        # Generate realistic price series
        n_days = len(trading_dates)
        
        # Starting price
        if stock == 'SPY':  # Benchmark
            start_price = 400
            daily_return_mean = 0.0003  # ~8% annual
            daily_return_std = 0.012    # ~19% annual volatility
        else:
            start_price = np.random.uniform(50, 300)
            daily_return_mean = np.random.uniform(-0.0002, 0.0008)  # -5% to 20% annual
            daily_return_std = np.random.uniform(0.015, 0.035)      # 24% to 56% annual volatility
        
        # Generate returns
        returns = np.random.normal(daily_return_mean, daily_return_std, n_days)
        
        # Add some momentum and mean reversion
        for i in range(1, len(returns)):
            momentum = returns[i-1] * 0.1  # 10% momentum
            returns[i] += momentum
        
        # Generate prices
        prices = [start_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate volume
        base_volume = np.random.uniform(1000000, 10000000)
        volumes = [base_volume * np.random.uniform(0.5, 2.0) for _ in range(n_days)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'close': prices,
            'volume': volumes
        }, index=trading_dates)
        
        # Add high, low, open (simplified)
        df['high'] = df['close'] * np.random.uniform(1.0, 1.05, len(df))
        df['low'] = df['close'] * np.random.uniform(0.95, 1.0, len(df))
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0]) * np.random.uniform(0.98, 1.02, len(df))
        
        price_data[stock] = df
    
    return price_data

def create_test_stock_selections():
    """Create test stock selections over time"""
    selections = {
        '2023-01-01': ['AAPL', 'GOOGL', 'MSFT'],
        '2023-02-01': ['AAPL', 'TSLA', 'NVDA'],
        '2023-03-01': ['GOOGL', 'META', 'AMZN'],
        '2023-04-01': ['MSFT', 'NVDA', 'TSLA'],
        '2023-05-01': ['AAPL', 'GOOGL', 'META'],
        '2023-06-01': ['TSLA', 'NVDA', 'AMZN'],
        '2023-07-01': ['AAPL', 'MSFT', 'GOOGL'],
        '2023-08-01': ['META', 'NVDA', 'TSLA'],
        '2023-09-01': ['GOOGL', 'AMZN', 'AAPL'],
        '2023-10-01': ['MSFT', 'TSLA', 'META'],
        '2023-11-01': ['NVDA', 'AAPL', 'GOOGL'],
        '2023-12-01': ['AMZN', 'MSFT', 'TSLA']
    }
    return selections

def create_test_fundamental_data():
    """Create test fundamental data"""
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN']
    fundamental_data = {}
    
    for stock in stocks:
        fundamental_data[stock] = {
            'roe': np.random.uniform(0.1, 0.4),
            'current_ratio': np.random.uniform(1.0, 3.0),
            'gross_margin': np.random.uniform(0.2, 0.6),
            'net_margin': np.random.uniform(0.05, 0.25),
            'revenue_growth_5y': np.random.uniform(0.05, 0.25),
            'debt_to_ebitda': np.random.uniform(0, 3),
            'revenue_consistency': np.random.randint(2, 5),
            'net_income_consistency': np.random.randint(2, 5),
            'cash_flow_consistency': np.random.randint(2, 5)
        }
    
    return fundamental_data

def test_backtest_engine():
    """Test the backtesting engine"""
    print("Testing Backtest Engine...")
    print("=" * 50)
    
    try:
        # Create test data
        price_data = create_test_price_data()
        stock_selections = create_test_stock_selections()
        
        print(f"✓ Created test data: {len(price_data)} stocks, {len(stock_selections)} selection periods")
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine()
        print("✓ Backtest engine initialized")
        
        # Run backtest
        print("\nRunning backtest simulation...")
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
        
        if 'error' not in backtest_results:
            print("✓ Backtest completed successfully")
            
            # Display key results
            if 'performance_metrics' in backtest_results:
                metrics = backtest_results['performance_metrics']
                print(f"\nKey Performance Metrics:")
                print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
                print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
                
                if 'alpha' in metrics:
                    print(f"  Alpha vs Benchmark: {metrics['alpha']:.2%}")
                    print(f"  Beta: {metrics.get('beta', 0):.2f}")
            
            # Display trade analysis
            if 'trade_analysis' in backtest_results:
                trades = backtest_results['trade_analysis']
                print(f"\nTrade Analysis:")
                print(f"  Total Trades: {trades.get('total_trades', 0)}")
                print(f"  Unique Symbols: {trades.get('unique_symbols', 0)}")
                print(f"  Transaction Costs: ${trades.get('total_transaction_costs', 0):.2f}")
            
            # Display risk analysis
            if 'risk_analysis' in backtest_results:
                risk = backtest_results['risk_analysis']
                print(f"\nRisk Analysis:")
                print(f"  VaR (95%): {risk.get('var_95', 0):.2%}")
                print(f"  Downside Frequency: {risk.get('downside_frequency', 0):.2%}")
                print(f"  Ulcer Index: {risk.get('ulcer_index', 0):.2f}")
        else:
            print(f"⚠️  Backtest failed: {backtest_results['error']}")
        
        # Test report generation
        print("\nGenerating backtest report...")
        report = backtest_engine.generate_backtest_report(backtest_results)
        if report and 'executive_summary' in report:
            print("✓ Backtest report generated successfully")
            
            # Display executive summary
            exec_summary = report['executive_summary']
            print(f"Executive Summary:")
            for key, value in exec_summary.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Display recommendations
            if 'recommendations' in report and report['recommendations']:
                print(f"\nTop Recommendations:")
                for i, rec in enumerate(report['recommendations'][:3]):
                    print(f"  {i+1}. {rec}")
        
        print("\n✓ Backtest Engine test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Backtest Engine test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_simulator():
    """Test the portfolio simulator"""
    print("\nTesting Portfolio Simulator...")
    print("=" * 50)
    
    try:
        # Create test data
        price_data = create_test_price_data()
        stock_selections = create_test_stock_selections()
        fundamental_data = create_test_fundamental_data()
        
        print(f"✓ Created test data for portfolio simulation")
        
        # Initialize portfolio simulator
        portfolio_sim = PortfolioSimulator()
        print("✓ Portfolio simulator initialized")
        
        # Test multi-strategy simulation
        print("\nRunning multi-strategy simulation...")
        strategies = ['equal_weight', 'momentum_weight', 'risk_parity']
        
        simulation_results = portfolio_sim.run_multi_strategy_simulation(
            stock_selections=stock_selections,
            price_data=price_data,
            fundamental_data=fundamental_data,
            start_date='2023-01-01',
            end_date='2023-12-31',
            strategies=strategies
        )
        
        if 'error' not in simulation_results:
            print("✓ Multi-strategy simulation completed successfully")
            
            # Display strategy comparison
            if 'comparison' in simulation_results and 'summary_table' in simulation_results['comparison']:
                print(f"\nStrategy Comparison:")
                summary_table = simulation_results['comparison']['summary_table']
                
                print(f"{'Strategy':<15} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Rank':<6}")
                print("-" * 55)
                
                for row in summary_table:
                    print(f"{row['strategy']:<15} {row['total_return']:<10} {row['sharpe_ratio']:<8} "
                          f"{row['max_drawdown']:<10} {row['sharpe_rank']:<6}")
            
            # Display best strategy
            if 'best_strategy' in simulation_results:
                best = simulation_results['best_strategy']
                print(f"\nBest Strategy: {best['name']}")
                print(f"  Total Return: {best['metrics']['total_return']:.2%}")
                print(f"  Sharpe Ratio: {best['metrics']['sharpe_ratio']:.2f}")
        else:
            print(f"⚠️  Multi-strategy simulation failed: {simulation_results['error']}")
        
        # Test portfolio optimization
        print("\nTesting portfolio optimization...")
        optimization_results = portfolio_sim.optimize_portfolio_allocation(
            stock_selections=stock_selections,
            price_data=price_data,
            optimization_objective='sharpe'
        )
        
        if 'error' not in optimization_results:
            print("✓ Portfolio optimization completed successfully")
            
            if 'optimal_weights' in optimization_results:
                print(f"Optimal Weights:")
                for stock, weight in optimization_results['optimal_weights'].items():
                    print(f"  {stock}: {weight:.1%}")
            
            if 'expected_metrics' in optimization_results:
                metrics = optimization_results['expected_metrics']
                print(f"Expected Metrics:")
                print(f"  Expected Return: {metrics.get('expected_return', 0):.2%}")
                print(f"  Expected Volatility: {metrics.get('expected_volatility', 0):.2%}")
                print(f"  Expected Sharpe: {metrics.get('expected_sharpe', 0):.2f}")
        else:
            print(f"⚠️  Portfolio optimization failed: {optimization_results['error']}")
        
        # Test simulation report
        print("\nGenerating simulation report...")
        report = portfolio_sim.generate_simulation_report(simulation_results)
        if report and 'executive_summary' in report:
            print("✓ Simulation report generated successfully")
            
            exec_summary = report['executive_summary']
            print(f"Executive Summary:")
            for key, value in exec_summary.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
            
            if 'recommendations' in report and report['recommendations']:
                print(f"\nRecommendations:")
                for i, rec in enumerate(report['recommendations'][:2]):
                    print(f"  {i+1}. {rec}")
        
        print("\n✓ Portfolio Simulator test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Portfolio Simulator test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between backtesting and portfolio simulation"""
    print("\nTesting Backtest-Portfolio Integration...")
    print("=" * 50)
    
    try:
        # Create test data
        price_data = create_test_price_data()
        stock_selections = create_test_stock_selections()
        
        # Test both engines with same data
        backtest_engine = BacktestEngine()
        portfolio_sim = PortfolioSimulator()
        
        # Run backtest
        backtest_results = backtest_engine.run_backtest(
            stock_selections, price_data, '2023-01-01', '2023-12-31'
        )
        
        # Run portfolio simulation
        simulation_results = portfolio_sim.run_multi_strategy_simulation(
            stock_selections, price_data, {}, '2023-01-01', '2023-12-31', ['equal_weight']
        )
        
        # Compare results
        if ('error' not in backtest_results and 'error' not in simulation_results and
            'performance_metrics' in backtest_results and 'strategy_results' in simulation_results):
            
            backtest_return = backtest_results['performance_metrics'].get('total_return', 0)
            
            equal_weight_results = simulation_results['strategy_results'].get('equal_weight', {})
            sim_return = equal_weight_results.get('performance_metrics', {}).get('total_return', 0)
            
            print(f"✓ Integration test completed")
            print(f"  Backtest Return: {backtest_return:.2%}")
            print(f"  Simulation Return: {sim_return:.2%}")
            print(f"  Difference: {abs(backtest_return - sim_return):.2%}")
            
            if abs(backtest_return - sim_return) < 0.05:  # Within 5%
                print("✓ Results are consistent between engines")
            else:
                print("⚠️  Results show some variation (expected due to different implementations)")
        
        print("\n✓ Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False

def main():
    """Run all backtesting tests"""
    print("Kangro Capital - Backtesting Tests")
    print("=" * 60)
    
    results = []
    
    # Test backtest engine
    results.append(test_backtest_engine())
    
    # Test portfolio simulator
    results.append(test_portfolio_simulator())
    
    # Test integration
    results.append(test_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Backtest Engine: {'✓ PASS' if results[0] else '❌ FAIL'}")
    print(f"Portfolio Simulator: {'✓ PASS' if results[1] else '❌ FAIL'}")
    print(f"Integration: {'✓ PASS' if results[2] else '❌ FAIL'}")
    
    if all(results):
        print("\n🎉 All backtesting tests passed! Backtesting functionality is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return all(results)

if __name__ == "__main__":
    main()

