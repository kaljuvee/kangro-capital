"""
Test script for the screening engine functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from utils.screening_engine import ScreeningEngine
import pandas as pd

# Load environment variables
load_dotenv()

def test_screening_engine():
    """Test the comprehensive screening engine"""
    print("Testing Screening Engine...")
    print("=" * 50)
    
    try:
        # Initialize screening engine
        engine = ScreeningEngine()
        print("✓ Screening engine initialized")
        
        # Test with limited parameters for quick testing
        test_params = {
            'top_n_stocks': 5,
            'lookback_years': 1,  # Shorter period for testing
            'outlier_threshold': 1.5,
            'breakout_min_score': 0.6
        }
        
        print(f"Running screening with parameters: {test_params}")
        
        # Run comprehensive screening
        results = engine.run_comprehensive_screening(test_params)
        
        print("\nScreening Results Summary:")
        print("-" * 30)
        
        if 'error' in results:
            print(f"❌ Error occurred: {results['error']}")
            return False
        
        # Display summary
        summary = results.get('summary', {})
        print(f"Total stocks analyzed: {summary.get('total_universe_size', 0)}")
        print(f"Outliers identified: {summary.get('outliers_identified', 0)}")
        print(f"Breakouts identified: {summary.get('breakouts_identified', 0)}")
        print(f"Final recommendations: {summary.get('final_recommendations', 0)}")
        
        # Display top recommendations
        combined_results = results.get('combined_results', {})
        top_recs = combined_results.get('top_recommendations', [])
        
        if top_recs:
            print(f"\nTop {len(top_recs)} Recommendations:")
            print("-" * 30)
            for rec in top_recs:
                print(f"{rec['rank']}. {rec['symbol']} - Combined Score: {rec['combined_score']:.3f}")
                print(f"   Technical: {rec['technical_score']:.3f}, Fundamental: {rec['fundamental_score']:.3f}")
                print(f"   Return: {rec['key_metrics']['total_return']:.2%}")
        
        # Display outlier analysis
        outlier_analysis = results.get('outlier_analysis', {})
        top_performers = outlier_analysis.get('top_performers', [])
        
        if top_performers:
            print(f"\nTop Technical Performers:")
            print("-" * 30)
            for i, performer in enumerate(top_performers[:5]):
                print(f"{i+1}. {performer['symbol']} - Return: {performer.get('total_return', 0):.2%}")
        
        print("\n✓ Screening engine test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Screening engine test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_similar_stocks():
    """Test similar stocks functionality"""
    print("\nTesting Similar Stocks Feature...")
    print("=" * 50)
    
    try:
        engine = ScreeningEngine()
        
        # Test finding stocks similar to AAPL
        reference_symbol = 'AAPL'
        similar_stocks = engine.get_similar_stocks(reference_symbol, top_n=3)
        
        if similar_stocks:
            print(f"Stocks similar to {reference_symbol}:")
            for i, stock in enumerate(similar_stocks):
                print(f"{i+1}. {stock['symbol']} - Similarity: {stock['similarity_score']:.3f}")
                print(f"   Return: {stock['total_return']:.2%}, Volatility: {stock['volatility']:.2%}")
        else:
            print(f"No similar stocks found for {reference_symbol}")
        
        print("✓ Similar stocks test completed")
        return True
        
    except Exception as e:
        print(f"❌ Similar stocks test failed: {str(e)}")
        return False

def test_individual_components():
    """Test individual screening components"""
    print("\nTesting Individual Components...")
    print("=" * 50)
    
    try:
        from utils.stock_screener import StockScreener
        from utils.outlier_detector import OutlierDetector
        from utils.data_fetcher import DataFetcher
        
        # Test data fetcher
        fetcher = DataFetcher()
        universe = fetcher.get_stock_universe()
        print(f"✓ Stock universe: {len(universe)} stocks")
        
        # Test with a small sample
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        price_data = fetcher.get_historical_data(test_symbols, years_back=1)
        print(f"✓ Price data fetched for {len(price_data)} stocks")
        
        # Test outlier detector
        detector = OutlierDetector()
        if price_data:
            metrics = detector.calculate_performance_metrics(price_data)
            print(f"✓ Performance metrics calculated for {len(metrics)} stocks")
            
            if not metrics.empty:
                outliers = detector.detect_statistical_outliers(metrics)
                breakouts = detector.detect_breakout_stocks(outliers)
                print(f"✓ Outlier analysis completed")
        
        # Test stock screener
        screener = StockScreener()
        print("✓ Stock screener initialized")
        
        print("✓ All individual components working")
        return True
        
    except Exception as e:
        print(f"❌ Individual components test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all screening tests"""
    print("Kangro Capital - Screening Engine Tests")
    print("=" * 60)
    
    results = []
    
    # Test individual components first
    results.append(test_individual_components())
    
    # Test main screening engine
    results.append(test_screening_engine())
    
    # Test similar stocks feature
    results.append(test_similar_stocks())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Individual Components: {'✓ PASS' if results[0] else '❌ FAIL'}")
    print(f"Screening Engine: {'✓ PASS' if results[1] else '❌ FAIL'}")
    print(f"Similar Stocks: {'✓ PASS' if results[2] else '❌ FAIL'}")
    
    if all(results):
        print("\n🎉 All screening tests passed! Core functionality is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return all(results)

if __name__ == "__main__":
    main()

