"""
Comprehensive screening engine that combines fundamental analysis and outlier detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json
import os

from .data_fetcher import DataFetcher
from .stock_screener import StockScreener
from .outlier_detector import OutlierDetector

logger = logging.getLogger(__name__)

class ScreeningEngine:
    """
    Main screening engine that orchestrates the entire screening process
    """
    
    def __init__(self):
        """Initialize the screening engine"""
        self.data_fetcher = DataFetcher()
        self.stock_screener = StockScreener()
        self.outlier_detector = OutlierDetector()
        
        # Default screening parameters
        self.default_params = {
            'top_n_stocks': 10,
            'lookback_years': 3,
            'min_market_cap': 1e9,
            'outlier_threshold': 2.0,
            'breakout_min_score': 0.7,
            'combine_scores': True
        }
    
    def run_comprehensive_screening(self, params: Dict = None) -> Dict:
        """
        Run comprehensive stock screening combining fundamental and technical analysis
        
        Args:
            params: Screening parameters dictionary
        
        Returns:
            Dictionary with comprehensive screening results
        """
        # Merge with default parameters
        screening_params = {**self.default_params, **(params or {})}
        
        logger.info("Starting comprehensive stock screening...")
        
        results = {
            'parameters': screening_params,
            'timestamp': datetime.now().isoformat(),
            'fundamental_screening': {},
            'outlier_analysis': {},
            'combined_results': {},
            'summary': {}
        }
        
        try:
            # Step 1: Get stock universe
            logger.info("Getting stock universe...")
            stock_universe = self.data_fetcher.get_stock_universe(
                market_cap_min=screening_params['min_market_cap']
            )
            
            # Limit to manageable number for testing
            stock_universe = stock_universe[:50]  # Test with first 50 stocks
            
            logger.info(f"Screening {len(stock_universe)} stocks")
            
            # Step 2: Fetch historical price data
            logger.info("Fetching historical price data...")
            price_data = self.data_fetcher.get_historical_data(
                stock_universe, 
                years_back=screening_params['lookback_years']
            )
            
            logger.info(f"Retrieved price data for {len(price_data)} stocks")
            
            # Step 3: Calculate returns and performance metrics
            logger.info("Calculating performance metrics...")
            returns_data = self.data_fetcher.calculate_returns(price_data)
            
            # Step 4: Outlier and breakout analysis
            logger.info("Performing outlier analysis...")
            outlier_metrics = self.outlier_detector.calculate_performance_metrics(
                price_data, 
                lookback_days=screening_params['lookback_years'] * 252
            )
            
            if not outlier_metrics.empty:
                # Detect statistical outliers
                outlier_results = self.outlier_detector.detect_statistical_outliers(
                    outlier_metrics, 
                    threshold=screening_params['outlier_threshold']
                )
                
                # Detect breakout stocks
                breakout_results = self.outlier_detector.detect_breakout_stocks(
                    outlier_results, 
                    min_score=screening_params['breakout_min_score']
                )
                
                # Get top outliers
                top_outliers = self.outlier_detector.get_top_outliers(
                    breakout_results, 
                    top_n=screening_params['top_n_stocks']
                )
                
                results['outlier_analysis'] = {
                    'total_analyzed': len(outlier_metrics),
                    'outliers_detected': len(outlier_results[outlier_results['is_outlier']]),
                    'breakouts_detected': len(breakout_results[breakout_results['is_breakout']]),
                    'top_performers': top_outliers.to_dict('records') if not top_outliers.empty else []
                }
            
            # Step 5: Fundamental screening (for top performers)
            logger.info("Performing fundamental screening...")
            top_symbols = []
            if not outlier_metrics.empty:
                top_symbols = outlier_metrics.nlargest(20, 'total_return')['symbol'].tolist()
            else:
                top_symbols = list(price_data.keys())[:20]
            
            # Get financial data for top performers
            fundamental_results = []
            for symbol in top_symbols[:10]:  # Limit to avoid API rate limits
                try:
                    financials = self.data_fetcher.polygon_client.get_financials(symbol)
                    price_df = price_data.get(symbol, pd.DataFrame())
                    
                    screening_result = self.stock_screener.screen_stock(
                        symbol, financials, price_df
                    )
                    
                    fundamental_results.append(screening_result)
                    
                except Exception as e:
                    logger.error(f"Error in fundamental screening for {symbol}: {str(e)}")
                    continue
            
            results['fundamental_screening'] = {
                'stocks_analyzed': len(fundamental_results),
                'results': fundamental_results
            }
            
            # Step 6: Combine results
            logger.info("Combining screening results...")
            combined_results = self._combine_screening_results(
                outlier_metrics if not outlier_metrics.empty else pd.DataFrame(),
                fundamental_results,
                screening_params
            )
            
            results['combined_results'] = combined_results
            
            # Step 7: Generate summary
            results['summary'] = self._generate_screening_summary(results)
            
            # Step 8: Save results
            self._save_screening_results(results)
            
            logger.info("Comprehensive screening completed successfully")
            
        except Exception as e:
            logger.error(f"Error in comprehensive screening: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _combine_screening_results(self, outlier_df: pd.DataFrame, 
                                 fundamental_results: List[Dict], 
                                 params: Dict) -> Dict:
        """Combine outlier analysis and fundamental screening results"""
        
        combined = {
            'top_recommendations': [],
            'methodology': 'Combined fundamental and technical analysis',
            'weights': {
                'technical_score': 0.6,
                'fundamental_score': 0.4
            }
        }
        
        try:
            # Create lookup for outlier scores
            outlier_scores = {}
            if not outlier_df.empty:
                for _, row in outlier_df.iterrows():
                    outlier_scores[row['symbol']] = {
                        'breakout_score': row.get('breakout_score', 0),
                        'total_return': row.get('total_return', 0),
                        'sharpe_ratio': row.get('sharpe_ratio', 0),
                        'volatility': row.get('volatility', 0)
                    }
            
            # Combine scores for stocks that have both analyses
            for fund_result in fundamental_results:
                symbol = fund_result['symbol']
                
                if symbol in outlier_scores:
                    technical_score = outlier_scores[symbol]['breakout_score']
                    fundamental_score = fund_result['overall_score']
                    
                    # Calculate combined score
                    combined_score = (
                        technical_score * combined['weights']['technical_score'] +
                        fundamental_score * combined['weights']['fundamental_score']
                    )
                    
                    recommendation = {
                        'symbol': symbol,
                        'combined_score': combined_score,
                        'technical_score': technical_score,
                        'fundamental_score': fundamental_score,
                        'recommendation': fund_result['recommendation'],
                        'key_metrics': {
                            'total_return': outlier_scores[symbol]['total_return'],
                            'sharpe_ratio': outlier_scores[symbol]['sharpe_ratio'],
                            'volatility': outlier_scores[symbol]['volatility'],
                            'fundamental_criteria_passed': sum(1 for v in fund_result['pass_criteria'].values() if v)
                        }
                    }
                    
                    combined['top_recommendations'].append(recommendation)
            
            # Sort by combined score
            combined['top_recommendations'].sort(
                key=lambda x: x['combined_score'], 
                reverse=True
            )
            
            # Add ranks
            for i, rec in enumerate(combined['top_recommendations']):
                rec['rank'] = i + 1
            
            # Limit to top N
            combined['top_recommendations'] = combined['top_recommendations'][:params['top_n_stocks']]
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            combined['error'] = str(e)
        
        return combined
    
    def _generate_screening_summary(self, results: Dict) -> Dict:
        """Generate summary statistics for the screening results"""
        
        summary = {
            'screening_date': results['timestamp'],
            'total_universe_size': 0,
            'stocks_with_price_data': 0,
            'outliers_identified': 0,
            'breakouts_identified': 0,
            'fundamental_analysis_completed': 0,
            'final_recommendations': 0,
            'top_sectors': [],
            'performance_distribution': {}
        }
        
        try:
            # Extract statistics from results
            if 'outlier_analysis' in results:
                summary['total_universe_size'] = results['outlier_analysis'].get('total_analyzed', 0)
                summary['outliers_identified'] = results['outlier_analysis'].get('outliers_detected', 0)
                summary['breakouts_identified'] = results['outlier_analysis'].get('breakouts_detected', 0)
            
            if 'fundamental_screening' in results:
                summary['fundamental_analysis_completed'] = results['fundamental_screening'].get('stocks_analyzed', 0)
            
            if 'combined_results' in results:
                summary['final_recommendations'] = len(results['combined_results'].get('top_recommendations', []))
            
            # Performance distribution
            if 'outlier_analysis' in results and results['outlier_analysis'].get('top_performers'):
                returns = [p.get('total_return', 0) for p in results['outlier_analysis']['top_performers']]
                if returns:
                    summary['performance_distribution'] = {
                        'min_return': min(returns),
                        'max_return': max(returns),
                        'avg_return': sum(returns) / len(returns),
                        'median_return': sorted(returns)[len(returns)//2] if returns else 0
                    }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            summary['error'] = str(e)
        
        return summary
    
    def _save_screening_results(self, results: Dict):
        """Save screening results to CSV and JSON files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save full results as JSON
            json_filename = f'screening_results_{timestamp}.json'
            self.data_fetcher.save_data_to_csv(results, json_filename)
            
            # Save top recommendations as CSV
            if 'combined_results' in results and results['combined_results'].get('top_recommendations'):
                recommendations_df = pd.DataFrame(results['combined_results']['top_recommendations'])
                csv_filename = f'top_recommendations_{timestamp}.csv'
                self.data_fetcher.save_data_to_csv(recommendations_df, csv_filename)
            
            # Save outlier analysis as CSV
            if 'outlier_analysis' in results and results['outlier_analysis'].get('top_performers'):
                outliers_df = pd.DataFrame(results['outlier_analysis']['top_performers'])
                outliers_filename = f'outlier_analysis_{timestamp}.csv'
                self.data_fetcher.save_data_to_csv(outliers_df, outliers_filename)
            
            logger.info(f"Screening results saved with timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def get_similar_stocks(self, reference_symbol: str, top_n: int = 5) -> List[Dict]:
        """
        Find stocks similar to a reference stock based on performance characteristics
        
        Args:
            reference_symbol: Symbol of the reference stock
            top_n: Number of similar stocks to return
        
        Returns:
            List of similar stocks with similarity scores
        """
        try:
            # Get recent screening results or run a quick analysis
            stock_universe = self.data_fetcher.get_stock_universe()[:30]  # Smaller universe for similarity
            
            if reference_symbol not in stock_universe:
                stock_universe.append(reference_symbol)
            
            price_data = self.data_fetcher.get_historical_data(stock_universe, years_back=1)
            
            if reference_symbol not in price_data:
                return []
            
            # Calculate performance metrics
            outlier_metrics = self.outlier_detector.calculate_performance_metrics(price_data)
            
            if outlier_metrics.empty:
                return []
            
            # Find reference stock metrics
            ref_metrics = outlier_metrics[outlier_metrics['symbol'] == reference_symbol]
            if ref_metrics.empty:
                return []
            
            ref_row = ref_metrics.iloc[0]
            
            # Calculate similarity scores
            similarity_scores = []
            feature_cols = ['total_return', 'volatility', 'sharpe_ratio', 'momentum_1m', 'momentum_3m']
            
            for _, row in outlier_metrics.iterrows():
                if row['symbol'] == reference_symbol:
                    continue
                
                # Calculate Euclidean distance
                distances = []
                for col in feature_cols:
                    if col in ref_row and col in row:
                        ref_val = ref_row[col] if not pd.isna(ref_row[col]) else 0
                        comp_val = row[col] if not pd.isna(row[col]) else 0
                        distances.append((ref_val - comp_val) ** 2)
                
                if distances:
                    similarity_score = 1 / (1 + np.sqrt(sum(distances)))  # Convert distance to similarity
                    
                    similarity_scores.append({
                        'symbol': row['symbol'],
                        'similarity_score': similarity_score,
                        'total_return': row.get('total_return', 0),
                        'volatility': row.get('volatility', 0),
                        'sharpe_ratio': row.get('sharpe_ratio', 0)
                    })
            
            # Sort by similarity score and return top N
            similarity_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarity_scores[:top_n]
            
        except Exception as e:
            logger.error(f"Error finding similar stocks to {reference_symbol}: {str(e)}")
            return []
    
    def update_screening_criteria(self, new_criteria: Dict):
        """Update screening criteria"""
        try:
            if 'fundamental' in new_criteria:
                self.stock_screener.screening_criteria.update(new_criteria['fundamental'])
            
            if 'performance_weights' in new_criteria:
                self.outlier_detector.performance_weights.update(new_criteria['performance_weights'])
            
            if 'default_params' in new_criteria:
                self.default_params.update(new_criteria['default_params'])
            
            logger.info("Screening criteria updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating screening criteria: {str(e)}")
            raise

