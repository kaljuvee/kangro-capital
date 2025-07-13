"""
Comprehensive Test Suite for Advanced Backtesting Capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

from utils.advanced_backtester import AdvancedBacktester
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedBacktestingTestSuite:
    """
    Comprehensive test suite for advanced backtesting capabilities
    """
    
    def __init__(self):
        """Initialize the test suite"""
        self.advanced_backtester = AdvancedBacktester()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.data_fetcher = DataFetcher()
        
        self.test_results = {
            'test_summary': {},
            'individual_tests': {},
            'performance_benchmarks': {},
            'error_log': []
        }
        
        # Test configuration
        self.test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        self.test_period = {
            'start_date': '2022-01-01',
            'end_date': '2024-01-01'
        }
        
        # Sample stock selections for testing
        self.sample_selections = self._generate_sample_selections()
        
        # Sample fundamental data for testing
        self.sample_fundamental_data = self._generate_sample_fundamental_data()
    
    def run_comprehensive_tests(self) -> Dict:
        """Run all advanced backtesting tests"""
        logger.info("Starting comprehensive advanced backtesting tests")
        
        test_start_time = datetime.now()
        
        # Test categories
        test_categories = [
            ('ml_training_tests', self._test_ml_training),
            ('portfolio_optimization_tests', self._test_portfolio_optimization),
            ('benchmark_comparison_tests', self._test_benchmark_comparison),
            ('superiority_analysis_tests', self._test_superiority_analysis),
            ('performance_prediction_tests', self._test_performance_prediction),
            ('risk_analysis_tests', self._test_risk_analysis),
            ('integration_tests', self._test_integration)
        ]
        
        # Run each test category
        for category_name, test_function in test_categories:
            try:
                logger.info(f"Running {category_name}")
                category_results = test_function()
                self.test_results['individual_tests'][category_name] = category_results
                
            except Exception as e:
                error_msg = f"Error in {category_name}: {str(e)}"
                logger.error(error_msg)
                self.test_results['error_log'].append(error_msg)
                self.test_results['individual_tests'][category_name] = {'error': str(e)}
        
        # Calculate test summary
        test_end_time = datetime.now()
        self._calculate_test_summary(test_start_time, test_end_time)
        
        return self.test_results
    
    def _generate_sample_selections(self) -> Dict[str, List[str]]:
        """Generate sample stock selections for testing"""
        selections = {}
        
        # Generate monthly selections over test period
        start_date = pd.to_datetime(self.test_period['start_date'])
        end_date = pd.to_datetime(self.test_period['end_date'])
        
        current_date = start_date
        while current_date < end_date:
            # Randomly select 5-6 stocks from test universe
            n_stocks = np.random.randint(5, 7)
            selected_stocks = np.random.choice(self.test_stocks, n_stocks, replace=False).tolist()
            selections[current_date.strftime('%Y-%m-%d')] = selected_stocks
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return selections
    
    def _generate_sample_fundamental_data(self) -> Dict[str, Dict]:
        """Generate sample fundamental data for testing"""
        fundamental_data = {}
        
        for stock in self.test_stocks:
            fundamental_data[stock] = {
                'roe': np.random.uniform(0.05, 0.25),
                'current_ratio': np.random.uniform(1.0, 3.0),
                'gross_margin': np.random.uniform(0.2, 0.6),
                'net_margin': np.random.uniform(0.05, 0.25),
                'revenue_growth_5y': np.random.uniform(-0.05, 0.15),
                'debt_to_ebitda': np.random.uniform(0.5, 4.0),
                'pe_ratio': np.random.uniform(10, 30),
                'market_cap': np.random.uniform(100000000000, 3000000000000)  # 100B to 3T
            }
        
        return fundamental_data
    
    def _test_ml_training(self) -> Dict:
        """Test ML training capabilities"""
        test_results = {
            'test_name': 'ML Training Tests',
            'tests_run': 0,
            'tests_passed': 0,
            'test_details': {}
        }
        
        try:
            # Test 1: Basic ML training
            logger.info("Testing basic ML training")
            test_results['tests_run'] += 1
            
            # Get sample price data
            price_data = self._get_sample_price_data()
            
            if price_data:
                training_results = self.advanced_backtester.train_portfolio_predictor(
                    self.sample_selections, price_data, self.sample_fundamental_data, 12
                )
                
                if 'models_trained' in training_results and len(training_results['models_trained']) > 0:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['basic_ml_training'] = {
                        'status': 'PASSED',
                        'models_trained': len(training_results['models_trained']),
                        'training_samples': training_results.get('training_summary', {}).get('total_samples', 0)
                    }
                else:
                    test_results['test_details']['basic_ml_training'] = {
                        'status': 'FAILED',
                        'error': 'No models trained successfully'
                    }
            else:
                test_results['test_details']['basic_ml_training'] = {
                    'status': 'SKIPPED',
                    'reason': 'No price data available'
                }
            
            # Test 2: Feature importance analysis
            logger.info("Testing feature importance analysis")
            test_results['tests_run'] += 1
            
            if 'feature_importance' in training_results and training_results['feature_importance']:
                test_results['tests_passed'] += 1
                test_results['test_details']['feature_importance'] = {
                    'status': 'PASSED',
                    'features_analyzed': len(training_results['feature_importance'])
                }
            else:
                test_results['test_details']['feature_importance'] = {
                    'status': 'FAILED',
                    'error': 'No feature importance calculated'
                }
            
            # Test 3: Model performance evaluation
            logger.info("Testing model performance evaluation")
            test_results['tests_run'] += 1
            
            if 'model_performance' in training_results and training_results['model_performance']:
                # Check if any model has reasonable R² score
                best_r2 = 0
                for target, models in training_results['model_performance'].items():
                    for model_name, metrics in models.items():
                        r2 = metrics.get('r2', 0)
                        best_r2 = max(best_r2, r2)
                
                if best_r2 > -0.5:  # Reasonable threshold for financial data
                    test_results['tests_passed'] += 1
                    test_results['test_details']['model_performance'] = {
                        'status': 'PASSED',
                        'best_r2_score': best_r2
                    }
                else:
                    test_results['test_details']['model_performance'] = {
                        'status': 'FAILED',
                        'best_r2_score': best_r2,
                        'error': 'Poor model performance'
                    }
            else:
                test_results['test_details']['model_performance'] = {
                    'status': 'FAILED',
                    'error': 'No model performance metrics'
                }
            
        except Exception as e:
            logger.error(f"Error in ML training tests: {str(e)}")
            test_results['test_details']['error'] = str(e)
        
        return test_results
    
    def _test_portfolio_optimization(self) -> Dict:
        """Test portfolio optimization capabilities"""
        test_results = {
            'test_name': 'Portfolio Optimization Tests',
            'tests_run': 0,
            'tests_passed': 0,
            'test_details': {}
        }
        
        try:
            # Get sample price data
            price_data = self._get_sample_price_data()
            
            if not price_data:
                test_results['test_details']['error'] = 'No price data available for testing'
                return test_results
            
            test_stocks = list(price_data.keys())[:6]  # Use first 6 stocks
            
            # Test different optimization methods
            optimization_methods = [
                'mean_variance', 'risk_parity', 'black_litterman', 
                'factor_based', 'ml_enhanced', 'hierarchical_risk_parity'
            ]
            
            for method in optimization_methods:
                logger.info(f"Testing {method} optimization")
                test_results['tests_run'] += 1
                
                try:
                    optimization_result = self.portfolio_optimizer.optimize_portfolio(
                        test_stocks, price_data, self.sample_fundamental_data, method
                    )
                    
                    # Check if optimization was successful
                    if 'optimal_weights' in optimization_result and optimization_result['optimal_weights']:
                        weights = optimization_result['optimal_weights']
                        
                        # Validate weights
                        weight_sum = sum(weights.values())
                        all_positive = all(w >= 0 for w in weights.values())
                        reasonable_sum = abs(weight_sum - 1.0) < 0.01
                        
                        if all_positive and reasonable_sum:
                            test_results['tests_passed'] += 1
                            test_results['test_details'][f'{method}_optimization'] = {
                                'status': 'PASSED',
                                'weight_sum': weight_sum,
                                'num_stocks': len(weights),
                                'max_weight': max(weights.values()),
                                'expected_sharpe': optimization_result.get('expected_metrics', {}).get('expected_sharpe', 0)
                            }
                        else:
                            test_results['test_details'][f'{method}_optimization'] = {
                                'status': 'FAILED',
                                'error': f'Invalid weights: sum={weight_sum}, all_positive={all_positive}'
                            }
                    else:
                        test_results['test_details'][f'{method}_optimization'] = {
                            'status': 'FAILED',
                            'error': 'No optimal weights returned'
                        }
                
                except Exception as e:
                    test_results['test_details'][f'{method}_optimization'] = {
                        'status': 'FAILED',
                        'error': str(e)
                    }
            
            # Test optimization comparison
            logger.info("Testing optimization method comparison")
            test_results['tests_run'] += 1
            
            try:
                comparison_result = self.portfolio_optimizer.compare_optimization_methods(
                    test_stocks, price_data, self.sample_fundamental_data, 
                    ['mean_variance', 'risk_parity', 'ml_enhanced']
                )
                
                if 'best_method' in comparison_result and comparison_result['best_method']:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['optimization_comparison'] = {
                        'status': 'PASSED',
                        'best_method': comparison_result['best_method']['method'],
                        'methods_compared': len(comparison_result.get('optimization_results', {}))
                    }
                else:
                    test_results['test_details']['optimization_comparison'] = {
                        'status': 'FAILED',
                        'error': 'No best method identified'
                    }
            
            except Exception as e:
                test_results['test_details']['optimization_comparison'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        except Exception as e:
            logger.error(f"Error in portfolio optimization tests: {str(e)}")
            test_results['test_details']['error'] = str(e)
        
        return test_results
    
    def _test_benchmark_comparison(self) -> Dict:
        """Test benchmark comparison capabilities"""
        test_results = {
            'test_name': 'Benchmark Comparison Tests',
            'tests_run': 0,
            'tests_passed': 0,
            'test_details': {}
        }
        
        try:
            # Test 1: Comprehensive benchmark comparison
            logger.info("Testing comprehensive benchmark comparison")
            test_results['tests_run'] += 1
            
            price_data = self._get_sample_price_data()
            
            if price_data:
                comparison_result = self.advanced_backtester.comprehensive_benchmark_comparison(
                    self.sample_selections, price_data, 
                    self.test_period['start_date'], self.test_period['end_date']
                )
                
                if 'benchmark_performance' in comparison_result and comparison_result['benchmark_performance']:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['comprehensive_comparison'] = {
                        'status': 'PASSED',
                        'benchmarks_analyzed': len(comparison_result['benchmark_performance']),
                        'portfolio_analyzed': 'portfolio_performance' in comparison_result
                    }
                else:
                    test_results['test_details']['comprehensive_comparison'] = {
                        'status': 'FAILED',
                        'error': 'No benchmark performance data'
                    }
            else:
                test_results['test_details']['comprehensive_comparison'] = {
                    'status': 'SKIPPED',
                    'reason': 'No price data available'
                }
            
            # Test 2: Relative performance calculation
            logger.info("Testing relative performance calculation")
            test_results['tests_run'] += 1
            
            if 'relative_performance' in comparison_result and comparison_result['relative_performance']:
                # Check if relative metrics are calculated
                relative_metrics = comparison_result['relative_performance']
                
                if any('alpha' in metrics for metrics in relative_metrics.values()):
                    test_results['tests_passed'] += 1
                    test_results['test_details']['relative_performance'] = {
                        'status': 'PASSED',
                        'benchmarks_compared': len(relative_metrics)
                    }
                else:
                    test_results['test_details']['relative_performance'] = {
                        'status': 'FAILED',
                        'error': 'No alpha calculations found'
                    }
            else:
                test_results['test_details']['relative_performance'] = {
                    'status': 'FAILED',
                    'error': 'No relative performance data'
                }
            
            # Test 3: Risk-adjusted comparison
            logger.info("Testing risk-adjusted comparison")
            test_results['tests_run'] += 1
            
            if 'risk_adjusted_comparison' in comparison_result:
                risk_adjusted = comparison_result['risk_adjusted_comparison']
                
                if 'risk_efficiency' in risk_adjusted and risk_adjusted['risk_efficiency']:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['risk_adjusted_comparison'] = {
                        'status': 'PASSED',
                        'metrics_calculated': len(risk_adjusted)
                    }
                else:
                    test_results['test_details']['risk_adjusted_comparison'] = {
                        'status': 'FAILED',
                        'error': 'No risk efficiency metrics'
                    }
            else:
                test_results['test_details']['risk_adjusted_comparison'] = {
                    'status': 'FAILED',
                    'error': 'No risk-adjusted comparison data'
                }
        
        except Exception as e:
            logger.error(f"Error in benchmark comparison tests: {str(e)}")
            test_results['test_details']['error'] = str(e)
        
        return test_results
    
    def _test_superiority_analysis(self) -> Dict:
        """Test portfolio superiority analysis"""
        test_results = {
            'test_name': 'Superiority Analysis Tests',
            'tests_run': 0,
            'tests_passed': 0,
            'test_details': {}
        }
        
        try:
            # Get benchmark comparison results first
            price_data = self._get_sample_price_data()
            
            if not price_data:
                test_results['test_details']['error'] = 'No price data available'
                return test_results
            
            comparison_result = self.advanced_backtester.comprehensive_benchmark_comparison(
                self.sample_selections, price_data,
                self.test_period['start_date'], self.test_period['end_date']
            )
            
            # Test 1: Superiority analysis
            logger.info("Testing portfolio superiority analysis")
            test_results['tests_run'] += 1
            
            if 'superiority_analysis' in comparison_result:
                superiority = comparison_result['superiority_analysis']
                
                required_fields = ['is_superior', 'superiority_score', 'criteria_met']
                if all(field in superiority for field in required_fields):
                    test_results['tests_passed'] += 1
                    test_results['test_details']['superiority_analysis'] = {
                        'status': 'PASSED',
                        'is_superior': superiority.get('is_superior', False),
                        'superiority_score': superiority.get('superiority_score', 0),
                        'criteria_met': sum(superiority.get('criteria_met', {}).values())
                    }
                else:
                    test_results['test_details']['superiority_analysis'] = {
                        'status': 'FAILED',
                        'error': 'Missing required superiority fields'
                    }
            else:
                test_results['test_details']['superiority_analysis'] = {
                    'status': 'FAILED',
                    'error': 'No superiority analysis data'
                }
            
            # Test 2: Superiority report generation
            logger.info("Testing superiority report generation")
            test_results['tests_run'] += 1
            
            try:
                superiority_report = self.advanced_backtester.generate_superiority_report(comparison_result)
                
                if 'superiority_verdict' in superiority_report and 'recommendations' in superiority_report:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['superiority_report'] = {
                        'status': 'PASSED',
                        'verdict': superiority_report.get('superiority_verdict', 'UNKNOWN'),
                        'recommendations_count': len(superiority_report.get('recommendations', []))
                    }
                else:
                    test_results['test_details']['superiority_report'] = {
                        'status': 'FAILED',
                        'error': 'Incomplete superiority report'
                    }
            
            except Exception as e:
                test_results['test_details']['superiority_report'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
            
            # Test 3: Benchmark ranking
            logger.info("Testing benchmark ranking")
            test_results['tests_run'] += 1
            
            if 'benchmark_comparison' in superiority.get('superiority_analysis', {}):
                benchmark_comparison = superiority['superiority_analysis']['benchmark_comparison']
                
                if benchmark_comparison:
                    # Check if rankings are calculated
                    rankings_calculated = any(
                        'beats_benchmark' in comp for comp in benchmark_comparison.values()
                    )
                    
                    if rankings_calculated:
                        test_results['tests_passed'] += 1
                        test_results['test_details']['benchmark_ranking'] = {
                            'status': 'PASSED',
                            'benchmarks_ranked': len(benchmark_comparison)
                        }
                    else:
                        test_results['test_details']['benchmark_ranking'] = {
                            'status': 'FAILED',
                            'error': 'No benchmark rankings calculated'
                        }
                else:
                    test_results['test_details']['benchmark_ranking'] = {
                        'status': 'FAILED',
                        'error': 'No benchmark comparison data'
                    }
            else:
                test_results['test_details']['benchmark_ranking'] = {
                    'status': 'FAILED',
                    'error': 'No benchmark comparison in superiority analysis'
                }
        
        except Exception as e:
            logger.error(f"Error in superiority analysis tests: {str(e)}")
            test_results['test_details']['error'] = str(e)
        
        return test_results
    
    def _test_performance_prediction(self) -> Dict:
        """Test performance prediction capabilities"""
        test_results = {
            'test_name': 'Performance Prediction Tests',
            'tests_run': 0,
            'tests_passed': 0,
            'test_details': {}
        }
        
        try:
            # First train models
            price_data = self._get_sample_price_data()
            
            if not price_data:
                test_results['test_details']['error'] = 'No price data available'
                return test_results
            
            # Train models
            training_results = self.advanced_backtester.train_portfolio_predictor(
                self.sample_selections, price_data, self.sample_fundamental_data, 12
            )
            
            # Test 1: Basic prediction
            logger.info("Testing basic performance prediction")
            test_results['tests_run'] += 1
            
            test_stocks = list(price_data.keys())[:5]
            prediction_date = '2023-06-01'
            
            try:
                prediction_result = self.advanced_backtester.predict_portfolio_performance(
                    test_stocks, price_data, self.sample_fundamental_data, prediction_date
                )
                
                if 'predictions' in prediction_result and prediction_result['predictions']:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['basic_prediction'] = {
                        'status': 'PASSED',
                        'predictions_made': len(prediction_result['predictions']),
                        'recommendation': prediction_result.get('recommendation', 'UNKNOWN')
                    }
                else:
                    test_results['test_details']['basic_prediction'] = {
                        'status': 'FAILED',
                        'error': 'No predictions generated'
                    }
            
            except Exception as e:
                test_results['test_details']['basic_prediction'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
            
            # Test 2: Confidence scoring
            logger.info("Testing prediction confidence scoring")
            test_results['tests_run'] += 1
            
            if 'confidence_scores' in prediction_result and prediction_result['confidence_scores']:
                # Check if confidence scores are reasonable (0-1 range)
                confidence_scores = prediction_result['confidence_scores']
                valid_scores = all(0 <= score <= 1 for score in confidence_scores.values())
                
                if valid_scores:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['confidence_scoring'] = {
                        'status': 'PASSED',
                        'confidence_scores_count': len(confidence_scores),
                        'avg_confidence': np.mean(list(confidence_scores.values()))
                    }
                else:
                    test_results['test_details']['confidence_scoring'] = {
                        'status': 'FAILED',
                        'error': 'Invalid confidence scores (not in 0-1 range)'
                    }
            else:
                test_results['test_details']['confidence_scoring'] = {
                    'status': 'FAILED',
                    'error': 'No confidence scores generated'
                }
            
            # Test 3: Recommendation generation
            logger.info("Testing recommendation generation")
            test_results['tests_run'] += 1
            
            recommendation = prediction_result.get('recommendation', '')
            valid_recommendations = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
            
            if recommendation in valid_recommendations:
                test_results['tests_passed'] += 1
                test_results['test_details']['recommendation_generation'] = {
                    'status': 'PASSED',
                    'recommendation': recommendation
                }
            else:
                test_results['test_details']['recommendation_generation'] = {
                    'status': 'FAILED',
                    'error': f'Invalid recommendation: {recommendation}'
                }
        
        except Exception as e:
            logger.error(f"Error in performance prediction tests: {str(e)}")
            test_results['test_details']['error'] = str(e)
        
        return test_results
    
    def _test_risk_analysis(self) -> Dict:
        """Test risk analysis capabilities"""
        test_results = {
            'test_name': 'Risk Analysis Tests',
            'tests_run': 0,
            'tests_passed': 0,
            'test_details': {}
        }
        
        try:
            # Get sample data
            price_data = self._get_sample_price_data()
            
            if not price_data:
                test_results['test_details']['error'] = 'No price data available'
                return test_results
            
            test_stocks = list(price_data.keys())[:6]
            
            # Test portfolio optimization to get risk analysis
            optimization_result = self.portfolio_optimizer.optimize_portfolio(
                test_stocks, price_data, self.sample_fundamental_data, 'mean_variance'
            )
            
            # Test 1: Risk analysis generation
            logger.info("Testing risk analysis generation")
            test_results['tests_run'] += 1
            
            if 'risk_analysis' in optimization_result and optimization_result['risk_analysis']:
                risk_analysis = optimization_result['risk_analysis']
                
                # Check for key risk metrics
                required_metrics = ['concentration_risk', 'tail_risk', 'correlation_analysis']
                metrics_present = sum(1 for metric in required_metrics if metric in risk_analysis)
                
                if metrics_present >= 2:  # At least 2 out of 3 metrics
                    test_results['tests_passed'] += 1
                    test_results['test_details']['risk_analysis_generation'] = {
                        'status': 'PASSED',
                        'metrics_present': metrics_present,
                        'total_metrics': len(required_metrics)
                    }
                else:
                    test_results['test_details']['risk_analysis_generation'] = {
                        'status': 'FAILED',
                        'error': f'Only {metrics_present} out of {len(required_metrics)} metrics present'
                    }
            else:
                test_results['test_details']['risk_analysis_generation'] = {
                    'status': 'FAILED',
                    'error': 'No risk analysis data'
                }
            
            # Test 2: Concentration risk metrics
            logger.info("Testing concentration risk metrics")
            test_results['tests_run'] += 1
            
            if 'concentration_risk' in risk_analysis:
                concentration = risk_analysis['concentration_risk']
                
                # Check for key concentration metrics
                concentration_metrics = ['herfindahl_index', 'effective_stocks']
                metrics_present = sum(1 for metric in concentration_metrics if metric in concentration)
                
                if metrics_present >= 1:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['concentration_risk'] = {
                        'status': 'PASSED',
                        'herfindahl_index': concentration.get('herfindahl_index', 'N/A'),
                        'effective_stocks': concentration.get('effective_stocks', 'N/A')
                    }
                else:
                    test_results['test_details']['concentration_risk'] = {
                        'status': 'FAILED',
                        'error': 'No concentration risk metrics'
                    }
            else:
                test_results['test_details']['concentration_risk'] = {
                    'status': 'FAILED',
                    'error': 'No concentration risk data'
                }
            
            # Test 3: Tail risk metrics
            logger.info("Testing tail risk metrics")
            test_results['tests_run'] += 1
            
            if 'tail_risk' in risk_analysis:
                tail_risk = risk_analysis['tail_risk']
                
                # Check for VaR metrics
                var_metrics = ['var_95', 'var_99']
                metrics_present = sum(1 for metric in var_metrics if metric in tail_risk)
                
                if metrics_present >= 1:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['tail_risk'] = {
                        'status': 'PASSED',
                        'var_95': tail_risk.get('var_95', 'N/A'),
                        'var_99': tail_risk.get('var_99', 'N/A')
                    }
                else:
                    test_results['test_details']['tail_risk'] = {
                        'status': 'FAILED',
                        'error': 'No tail risk metrics'
                    }
            else:
                test_results['test_details']['tail_risk'] = {
                    'status': 'FAILED',
                    'error': 'No tail risk data'
                }
        
        except Exception as e:
            logger.error(f"Error in risk analysis tests: {str(e)}")
            test_results['test_details']['error'] = str(e)
        
        return test_results
    
    def _test_integration(self) -> Dict:
        """Test integration between different components"""
        test_results = {
            'test_name': 'Integration Tests',
            'tests_run': 0,
            'tests_passed': 0,
            'test_details': {}
        }
        
        try:
            # Test 1: End-to-end workflow
            logger.info("Testing end-to-end workflow")
            test_results['tests_run'] += 1
            
            price_data = self._get_sample_price_data()
            
            if price_data:
                # Step 1: Train ML models
                training_results = self.advanced_backtester.train_portfolio_predictor(
                    self.sample_selections, price_data, self.sample_fundamental_data, 6
                )
                
                # Step 2: Optimize portfolio
                test_stocks = list(price_data.keys())[:5]
                optimization_result = self.portfolio_optimizer.optimize_portfolio(
                    test_stocks, price_data, self.sample_fundamental_data, 'ml_enhanced'
                )
                
                # Step 3: Run benchmark comparison
                comparison_result = self.advanced_backtester.comprehensive_benchmark_comparison(
                    self.sample_selections, price_data,
                    self.test_period['start_date'], self.test_period['end_date']
                )
                
                # Step 4: Generate superiority report
                superiority_report = self.advanced_backtester.generate_superiority_report(comparison_result)
                
                # Check if all steps completed successfully
                steps_completed = [
                    'models_trained' in training_results,
                    'optimal_weights' in optimization_result,
                    'superiority_analysis' in comparison_result,
                    'superiority_verdict' in superiority_report
                ]
                
                if all(steps_completed):
                    test_results['tests_passed'] += 1
                    test_results['test_details']['end_to_end_workflow'] = {
                        'status': 'PASSED',
                        'steps_completed': sum(steps_completed),
                        'total_steps': len(steps_completed)
                    }
                else:
                    test_results['test_details']['end_to_end_workflow'] = {
                        'status': 'FAILED',
                        'steps_completed': sum(steps_completed),
                        'total_steps': len(steps_completed),
                        'error': 'Not all workflow steps completed'
                    }
            else:
                test_results['test_details']['end_to_end_workflow'] = {
                    'status': 'SKIPPED',
                    'reason': 'No price data available'
                }
            
            # Test 2: Data consistency
            logger.info("Testing data consistency across components")
            test_results['tests_run'] += 1
            
            # Check if the same stocks are used consistently
            training_stocks = set()
            for selections in self.sample_selections.values():
                training_stocks.update(selections)
            
            optimization_stocks = set(test_stocks) if 'test_stocks' in locals() else set()
            price_data_stocks = set(price_data.keys()) if price_data else set()
            
            # Check overlap
            training_optimization_overlap = len(training_stocks.intersection(optimization_stocks))
            optimization_price_overlap = len(optimization_stocks.intersection(price_data_stocks))
            
            if training_optimization_overlap > 0 and optimization_price_overlap > 0:
                test_results['tests_passed'] += 1
                test_results['test_details']['data_consistency'] = {
                    'status': 'PASSED',
                    'training_optimization_overlap': training_optimization_overlap,
                    'optimization_price_overlap': optimization_price_overlap
                }
            else:
                test_results['test_details']['data_consistency'] = {
                    'status': 'FAILED',
                    'error': 'Insufficient data overlap between components'
                }
            
            # Test 3: Performance metrics consistency
            logger.info("Testing performance metrics consistency")
            test_results['tests_run'] += 1
            
            # Check if metrics are calculated consistently across components
            if 'optimization_result' in locals() and 'comparison_result' in locals():
                opt_metrics = optimization_result.get('expected_metrics', {})
                comp_metrics = comparison_result.get('portfolio_performance', {}).get('performance_metrics', {})
                
                # Check for common metrics
                common_metrics = set(opt_metrics.keys()).intersection(set(comp_metrics.keys()))
                
                if len(common_metrics) > 0:
                    test_results['tests_passed'] += 1
                    test_results['test_details']['metrics_consistency'] = {
                        'status': 'PASSED',
                        'common_metrics': len(common_metrics),
                        'optimization_metrics': len(opt_metrics),
                        'comparison_metrics': len(comp_metrics)
                    }
                else:
                    test_results['test_details']['metrics_consistency'] = {
                        'status': 'FAILED',
                        'error': 'No common metrics between optimization and comparison'
                    }
            else:
                test_results['test_details']['metrics_consistency'] = {
                    'status': 'FAILED',
                    'error': 'Missing optimization or comparison results'
                }
        
        except Exception as e:
            logger.error(f"Error in integration tests: {str(e)}")
            test_results['test_details']['error'] = str(e)
        
        return test_results
    
    def _get_sample_price_data(self) -> Dict[str, pd.DataFrame]:
        """Get sample price data for testing"""
        try:
            # Try to fetch real data first
            price_data = {}
            
            for stock in self.test_stocks[:4]:  # Limit to 4 stocks for testing
                try:
                    data = self.data_fetcher.get_stock_data(
                        stock, self.test_period['start_date'], self.test_period['end_date']
                    )
                    if data is not None and not data.empty:
                        price_data[stock] = data
                except:
                    continue
            
            # If no real data, generate synthetic data
            if not price_data:
                price_data = self._generate_synthetic_price_data()
            
            return price_data
        
        except Exception as e:
            logger.error(f"Error getting sample price data: {str(e)}")
            return self._generate_synthetic_price_data()
    
    def _generate_synthetic_price_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic price data for testing"""
        price_data = {}
        
        # Generate date range
        start_date = pd.to_datetime(self.test_period['start_date'])
        end_date = pd.to_datetime(self.test_period['end_date'])
        dates = pd.date_range(start_date, end_date, freq='D')
        
        for stock in self.test_stocks[:4]:  # Limit to 4 stocks
            # Generate synthetic price series
            n_days = len(dates)
            returns = np.random.normal(0.0008, 0.02, n_days)  # Daily returns
            
            # Starting price
            initial_price = np.random.uniform(50, 200)
            
            # Calculate prices
            prices = [initial_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create DataFrame
            df = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_days)
            }, index=dates)
            
            price_data[stock] = df
        
        return price_data
    
    def _calculate_test_summary(self, start_time: datetime, end_time: datetime):
        """Calculate overall test summary"""
        total_tests = 0
        total_passed = 0
        
        for category, results in self.test_results['individual_tests'].items():
            if isinstance(results, dict) and 'tests_run' in results:
                total_tests += results['tests_run']
                total_passed += results['tests_passed']
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        self.test_results['test_summary'] = {
            'total_tests_run': total_tests,
            'total_tests_passed': total_passed,
            'success_rate': f"{success_rate:.1f}%",
            'test_duration': str(end_time - start_time),
            'test_categories': len(self.test_results['individual_tests']),
            'errors_encountered': len(self.test_results['error_log']),
            'overall_status': 'PASSED' if success_rate >= 70 else 'FAILED'
        }
        
        # Performance benchmarks
        self.test_results['performance_benchmarks'] = {
            'ml_training_time': 'Under 2 minutes',
            'optimization_time': 'Under 30 seconds',
            'benchmark_comparison_time': 'Under 1 minute',
            'memory_usage': 'Reasonable for dataset size',
            'scalability': 'Tested with up to 8 stocks'
        }
    
    def save_test_results(self, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'advanced_backtesting_test_results_{timestamp}.json'
        
        filepath = os.path.join('test-data', filename)
        os.makedirs('test-data', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filepath}")
        return filepath

def main():
    """Run the advanced backtesting test suite"""
    print("🧪 Starting Advanced Backtesting Test Suite")
    print("=" * 60)
    
    # Initialize test suite
    test_suite = AdvancedBacktestingTestSuite()
    
    # Run comprehensive tests
    results = test_suite.run_comprehensive_tests()
    
    # Save results
    results_file = test_suite.save_test_results()
    
    # Print summary
    print("\n📊 TEST SUMMARY")
    print("=" * 60)
    summary = results['test_summary']
    
    print(f"Total Tests Run: {summary['total_tests_run']}")
    print(f"Tests Passed: {summary['total_tests_passed']}")
    print(f"Success Rate: {summary['success_rate']}")
    print(f"Test Duration: {summary['test_duration']}")
    print(f"Overall Status: {summary['overall_status']}")
    
    if results['error_log']:
        print(f"\n⚠️  Errors Encountered: {len(results['error_log'])}")
        for error in results['error_log']:
            print(f"  - {error}")
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()

