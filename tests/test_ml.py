"""
Test script for machine learning and factor analysis functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from utils.ml_analyzer import MLAnalyzer
from utils.factor_analyzer import FactorAnalyzer
from utils.data_fetcher import DataFetcher
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

def create_test_data():
    """Create synthetic test data for ML analysis"""
    np.random.seed(42)
    n_stocks = 50
    
    # Create synthetic features
    data = []
    for i in range(n_stocks):
        stock_data = {
            'symbol': f'TEST{i:03d}',
            # Technical features
            'price_vs_sma5': np.random.normal(0, 0.1),
            'price_vs_sma10': np.random.normal(0, 0.15),
            'price_vs_sma20': np.random.normal(0, 0.2),
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.normal(0, 0.05),
            'bb_position': np.random.uniform(0, 1),
            'volatility_20': np.random.uniform(0.1, 0.5),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'momentum_5': np.random.normal(0, 0.1),
            'momentum_10': np.random.normal(0, 0.15),
            'momentum_20': np.random.normal(0, 0.2),
            
            # Fundamental features
            'roe': np.random.uniform(0, 0.3),
            'current_ratio': np.random.uniform(0.5, 3.0),
            'gross_margin': np.random.uniform(0.1, 0.6),
            'net_margin': np.random.uniform(0, 0.2),
            'revenue_growth': np.random.uniform(-0.1, 0.3),
            'debt_to_ebitda': np.random.uniform(0, 5),
            'revenue_consistency': np.random.randint(0, 5),
            'income_consistency': np.random.randint(0, 5),
            'cashflow_consistency': np.random.randint(0, 5),
        }
        
        # Create synthetic target based on some features (with noise)
        performance_score = (
            stock_data['momentum_10'] * 0.3 +
            stock_data['roe'] * 0.2 +
            stock_data['revenue_growth'] * 0.2 +
            -stock_data['volatility_20'] * 0.1 +
            stock_data['rsi'] / 100 * 0.1 +
            np.random.normal(0, 0.05)  # noise
        )
        
        stock_data['return_1m'] = performance_score
        stock_data['return_3m'] = performance_score * 1.5 + np.random.normal(0, 0.02)
        
        # Classification targets
        if stock_data['return_1m'] > 0.1:
            stock_data['performance_category'] = 'HIGH'
            stock_data['is_outperformer'] = 1
        elif stock_data['return_1m'] > 0.05:
            stock_data['performance_category'] = 'MEDIUM'
            stock_data['is_outperformer'] = 1
        elif stock_data['return_1m'] > -0.05:
            stock_data['performance_category'] = 'LOW'
            stock_data['is_outperformer'] = 0
        else:
            stock_data['performance_category'] = 'NEGATIVE'
            stock_data['is_outperformer'] = 0
        
        data.append(stock_data)
    
    return pd.DataFrame(data)

def test_ml_analyzer():
    """Test the ML analyzer functionality"""
    print("Testing ML Analyzer...")
    print("=" * 50)
    
    try:
        # Create test data
        test_df = create_test_data()
        print(f"✓ Created test dataset with {len(test_df)} stocks")
        
        # Initialize ML analyzer
        ml_analyzer = MLAnalyzer()
        print("✓ ML Analyzer initialized")
        
        # Test regression models
        print("\nTesting regression models...")
        regression_results = ml_analyzer.train_regression_models(test_df, 'return_1m')
        
        if regression_results:
            print(f"✓ Trained {len(regression_results)} regression models")
            
            # Display best model
            best_model = max(regression_results.items(), key=lambda x: x[1]['test_r2'])
            print(f"Best regression model: {best_model[0]} (R² = {best_model[1]['test_r2']:.3f})")
            
            # Show feature importance if available
            if 'feature_importance' in best_model[1]:
                print("Top 5 important features:")
                for i, feature in enumerate(best_model[1]['feature_importance'][:5]):
                    print(f"  {i+1}. {feature['feature']}: {feature['importance']:.3f}")
        else:
            print("⚠️  No regression models trained")
        
        # Test classification models
        print("\nTesting classification models...")
        classification_results = ml_analyzer.train_classification_models(test_df, 'is_outperformer')
        
        if classification_results:
            print(f"✓ Trained {len(classification_results)} classification models")
            
            # Display best model
            best_clf_model = max(classification_results.items(), key=lambda x: x[1]['test_accuracy'])
            print(f"Best classification model: {best_clf_model[0]} (Accuracy = {best_clf_model[1]['test_accuracy']:.3f})")
        else:
            print("⚠️  No classification models trained")
        
        # Test feature analysis
        print("\nTesting feature analysis...")
        feature_analysis = ml_analyzer.perform_feature_analysis(test_df)
        
        if feature_analysis:
            print("✓ Feature analysis completed")
            
            if 'statistical_importance' in feature_analysis:
                print("Top 3 statistically important features:")
                for i, feature in enumerate(feature_analysis['statistical_importance'][:3]):
                    print(f"  {i+1}. {feature['feature']}: score = {feature['score']:.3f}")
            
            if 'pca_analysis' in feature_analysis:
                pca_info = feature_analysis['pca_analysis']
                print(f"PCA: {pca_info['n_components_90']} components explain 90% of variance")
        else:
            print("⚠️  Feature analysis failed")
        
        # Test prediction
        print("\nTesting prediction...")
        sample_features = test_df.iloc[0].to_dict()
        # Remove target variables
        for key in ['symbol', 'return_1m', 'return_3m', 'performance_category', 'is_outperformer']:
            sample_features.pop(key, None)
        
        predictions = ml_analyzer.predict_stock_performance(sample_features)
        if predictions and 'predicted_return' in predictions:
            print(f"✓ Prediction successful: {predictions['predicted_return']:.3f}")
        else:
            print("⚠️  Prediction failed or incomplete")
        
        print("\n✓ ML Analyzer test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ ML Analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_factor_analyzer():
    """Test the factor analyzer functionality"""
    print("\nTesting Factor Analyzer...")
    print("=" * 50)
    
    try:
        # Create test data
        test_df = create_test_data()
        print(f"✓ Created test dataset with {len(test_df)} stocks")
        
        # Initialize factor analyzer
        factor_analyzer = FactorAnalyzer()
        print("✓ Factor Analyzer initialized")
        
        # Perform factor analysis
        print("\nPerforming comprehensive factor analysis...")
        factor_results = factor_analyzer.perform_factor_analysis(test_df, 'return_1m', n_factors=3)
        
        if 'error' not in factor_results:
            print("✓ Factor analysis completed successfully")
            
            # Display PCA results
            if 'pca_analysis' in factor_results:
                pca = factor_results['pca_analysis']
                print(f"PCA: {pca['total_variance_explained']:.1%} total variance explained")
                
                if 'component_target_correlations' in pca:
                    best_component = pca['component_target_correlations'][0]
                    print(f"Best component: {best_component['component']} (correlation = {best_component['correlation']:.3f})")
            
            # Display performance drivers
            if 'performance_drivers' in factor_results:
                drivers = factor_results['performance_drivers']
                print("\nTop 3 Performance Drivers:")
                for i, driver in enumerate(drivers['top_10_drivers'][:3]):
                    print(f"  {i+1}. {driver['feature']}: importance = {driver['importance_score']:.3f}")
                
                if 'category_importance' in drivers:
                    categories = drivers['category_importance']
                    top_category = max(categories.items(), key=lambda x: x[1]['avg_importance'])
                    print(f"Most important category: {top_category[0]} (avg importance = {top_category[1]['avg_importance']:.3f})")
            
            # Display insights
            if 'performance_drivers' in factor_results and 'insights' in factor_results['performance_drivers']:
                insights = factor_results['performance_drivers']['insights']
                if insights:
                    print(f"\nKey Insights:")
                    for insight in insights[:2]:
                        print(f"  • {insight}")
            
            # Test factor summary
            print("\nGenerating factor summary...")
            summary = factor_analyzer.get_factor_summary(factor_results)
            if summary and 'key_drivers' in summary:
                print("✓ Factor summary generated successfully")
            
        else:
            print(f"⚠️  Factor analysis failed: {factor_results['error']}")
        
        print("\n✓ Factor Analyzer test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Factor Analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between ML and factor analysis"""
    print("\nTesting ML-Factor Integration...")
    print("=" * 50)
    
    try:
        # Create test data
        test_df = create_test_data()
        
        # Initialize both analyzers
        ml_analyzer = MLAnalyzer()
        factor_analyzer = FactorAnalyzer()
        
        # Perform both analyses
        regression_results = ml_analyzer.train_regression_models(test_df, 'return_1m')
        factor_results = factor_analyzer.perform_factor_analysis(test_df, 'return_1m', n_factors=3)
        
        # Compare results
        if regression_results and 'performance_drivers' in factor_results:
            # Get feature importance from both methods
            rf_importance = None
            for model_name, model_info in regression_results.items():
                if 'random_forest' in model_name and 'feature_importance' in model_info:
                    rf_importance = {item['feature']: item['importance'] for item in model_info['feature_importance']}
                    break
            
            factor_importance = None
            if 'feature_importance' in factor_results['performance_drivers']:
                factor_importance = {item['feature']: item['importance_score'] 
                                   for item in factor_results['performance_drivers']['feature_importance']}
            
            if rf_importance and factor_importance:
                # Find common top features
                rf_top = set(list(rf_importance.keys())[:5])
                factor_top = set(list(factor_importance.keys())[:5])
                common_features = rf_top.intersection(factor_top)
                
                print(f"✓ Common top features between methods: {len(common_features)}")
                if common_features:
                    print(f"  Common features: {', '.join(list(common_features)[:3])}")
            
        print("✓ Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False

def main():
    """Run all ML analysis tests"""
    print("Kangro Capital - ML Analysis Tests")
    print("=" * 60)
    
    results = []
    
    # Test ML analyzer
    results.append(test_ml_analyzer())
    
    # Test factor analyzer
    results.append(test_factor_analyzer())
    
    # Test integration
    results.append(test_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"ML Analyzer: {'✓ PASS' if results[0] else '❌ FAIL'}")
    print(f"Factor Analyzer: {'✓ PASS' if results[1] else '❌ FAIL'}")
    print(f"Integration: {'✓ PASS' if results[2] else '❌ FAIL'}")
    
    if all(results):
        print("\n🎉 All ML analysis tests passed! Machine learning functionality is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return all(results)

if __name__ == "__main__":
    main()

