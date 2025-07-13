"""
Core stock screening logic implementing the 7-step criteria for identifying great businesses
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StockScreener:
    """
    Implements the 7-step screening criteria for identifying great businesses
    """
    
    def __init__(self):
        """Initialize the stock screener"""
        self.screening_criteria = {
            'sales_growth_5y': 0.1,  # 10% minimum 5-year sales growth
            'net_income_consistency': 3,  # At least 3 years of consistent growth
            'cash_flow_consistency': 3,  # At least 3 years of consistent growth
            'roe_min': 0.12,  # Minimum 12% ROE
            'roic_min': 0.12,  # Minimum 12% ROIC
            'current_ratio_min': 1.0,  # Current ratio > 1
            'debt_to_ebitda_max': 3.0,  # Debt/EBITDA < 3
            'debt_service_ratio_max': 0.30  # Debt service ratio < 30%
        }
    
    def calculate_financial_metrics(self, financials: Dict) -> Dict:
        """
        Calculate key financial metrics from financial data
        
        Args:
            financials: Financial data dictionary
        
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}
        
        try:
            if 'results' not in financials or not financials['results']:
                return metrics
            
            # Sort financial data by filing date (most recent first)
            financial_data = sorted(financials['results'], 
                                  key=lambda x: x.get('filing_date', ''), 
                                  reverse=True)
            
            # Extract key metrics for the last 5 years
            years_data = []
            for report in financial_data[:5]:  # Last 5 reports
                if 'financials' not in report:
                    continue
                
                year_metrics = {}
                financials_data = report['financials']
                
                # Income Statement metrics
                if 'income_statement' in financials_data:
                    income = financials_data['income_statement']
                    year_metrics['revenue'] = self._get_financial_value(income, 'revenues')
                    year_metrics['net_income'] = self._get_financial_value(income, 'net_income_loss')
                    year_metrics['operating_income'] = self._get_financial_value(income, 'operating_income_loss')
                    year_metrics['gross_profit'] = self._get_financial_value(income, 'gross_profit')
                
                # Balance Sheet metrics
                if 'balance_sheet' in financials_data:
                    balance = financials_data['balance_sheet']
                    year_metrics['total_assets'] = self._get_financial_value(balance, 'assets')
                    year_metrics['current_assets'] = self._get_financial_value(balance, 'current_assets')
                    year_metrics['current_liabilities'] = self._get_financial_value(balance, 'current_liabilities')
                    year_metrics['total_debt'] = self._get_financial_value(balance, 'liabilities')
                    year_metrics['shareholders_equity'] = self._get_financial_value(balance, 'equity')
                
                # Cash Flow metrics
                if 'cash_flow_statement' in financials_data:
                    cash_flow = financials_data['cash_flow_statement']
                    year_metrics['operating_cash_flow'] = self._get_financial_value(cash_flow, 'net_cash_flow_from_operating_activities')
                
                year_metrics['filing_date'] = report.get('filing_date', '')
                years_data.append(year_metrics)
            
            # Calculate derived metrics
            metrics = self._calculate_derived_metrics(years_data)
            
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {str(e)}")
        
        return metrics
    
    def _get_financial_value(self, financial_section: Dict, key: str) -> float:
        """Extract financial value from nested structure"""
        try:
            if key in financial_section:
                value = financial_section[key].get('value', 0)
                return float(value) if value is not None else 0.0
            return 0.0
        except (KeyError, ValueError, AttributeError):
            return 0.0
    
    def _calculate_derived_metrics(self, years_data: List[Dict]) -> Dict:
        """Calculate derived financial metrics"""
        metrics = {}
        
        if not years_data:
            return metrics
        
        try:
            # Revenue growth analysis
            revenues = [year.get('revenue', 0) for year in years_data if year.get('revenue', 0) > 0]
            if len(revenues) >= 2:
                metrics['revenue_growth_5y'] = self._calculate_cagr(revenues)
                metrics['revenue_consistency'] = self._check_consistency(revenues)
            
            # Net income analysis
            net_incomes = [year.get('net_income', 0) for year in years_data if year.get('net_income') is not None]
            if len(net_incomes) >= 2:
                metrics['net_income_growth'] = self._calculate_cagr(net_incomes)
                metrics['net_income_consistency'] = self._check_consistency(net_incomes)
            
            # Operating cash flow analysis
            cash_flows = [year.get('operating_cash_flow', 0) for year in years_data if year.get('operating_cash_flow') is not None]
            if len(cash_flows) >= 2:
                metrics['cash_flow_growth'] = self._calculate_cagr(cash_flows)
                metrics['cash_flow_consistency'] = self._check_consistency(cash_flows)
            
            # Most recent year metrics
            latest = years_data[0] if years_data else {}
            
            # Return on Equity (ROE)
            if latest.get('net_income', 0) and latest.get('shareholders_equity', 0):
                metrics['roe'] = latest['net_income'] / latest['shareholders_equity']
            
            # Current Ratio
            if latest.get('current_assets', 0) and latest.get('current_liabilities', 0):
                metrics['current_ratio'] = latest['current_assets'] / latest['current_liabilities']
            
            # Debt to EBITDA (approximated)
            if latest.get('total_debt', 0) and latest.get('operating_income', 0):
                # Approximate EBITDA as operating income (simplified)
                metrics['debt_to_ebitda'] = latest['total_debt'] / abs(latest['operating_income']) if latest['operating_income'] != 0 else float('inf')
            
            # Profit margins
            if latest.get('revenue', 0) > 0:
                if latest.get('gross_profit', 0):
                    metrics['gross_margin'] = latest['gross_profit'] / latest['revenue']
                if latest.get('net_income', 0):
                    metrics['net_margin'] = latest['net_income'] / latest['revenue']
            
        except Exception as e:
            logger.error(f"Error in derived metrics calculation: {str(e)}")
        
        return metrics
    
    def _calculate_cagr(self, values: List[float]) -> float:
        """Calculate Compound Annual Growth Rate"""
        try:
            if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
                return 0.0
            
            years = len(values) - 1
            cagr = (values[0] / values[-1]) ** (1/years) - 1
            return cagr
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    def _check_consistency(self, values: List[float]) -> int:
        """Check how many years show positive growth"""
        if len(values) < 2:
            return 0
        
        consistent_years = 0
        for i in range(1, len(values)):
            if values[i-1] > values[i]:  # Growth from previous year
                consistent_years += 1
        
        return consistent_years
    
    def screen_stock(self, symbol: str, financials: Dict, price_data: pd.DataFrame) -> Dict:
        """
        Screen a single stock against the 7-step criteria
        
        Args:
            symbol: Stock symbol
            financials: Financial data from API
            price_data: Historical price data
        
        Returns:
            Dictionary with screening results
        """
        result = {
            'symbol': symbol,
            'pass_criteria': {},
            'metrics': {},
            'overall_score': 0,
            'recommendation': 'HOLD'
        }
        
        try:
            # Calculate financial metrics
            metrics = self.calculate_financial_metrics(financials)
            result['metrics'] = metrics
            
            # Apply 7-step screening criteria
            criteria_results = {}
            
            # Step 1: Consistently Increasing Sales, Net Income and Cash Flow
            criteria_results['sales_growth'] = metrics.get('revenue_growth_5y', 0) >= self.screening_criteria['sales_growth_5y']
            criteria_results['sales_consistency'] = metrics.get('revenue_consistency', 0) >= self.screening_criteria['net_income_consistency']
            criteria_results['income_consistency'] = metrics.get('net_income_consistency', 0) >= self.screening_criteria['net_income_consistency']
            criteria_results['cashflow_consistency'] = metrics.get('cash_flow_consistency', 0) >= self.screening_criteria['cash_flow_consistency']
            
            # Step 2: Positive Growth Rate (covered in step 1)
            
            # Step 3: Sustainable Competitive Advantage (qualitative - placeholder)
            criteria_results['competitive_advantage'] = True  # Would need qualitative analysis
            
            # Step 4: Profitable and Operationally Efficient
            criteria_results['roe_sufficient'] = metrics.get('roe', 0) >= self.screening_criteria['roe_min']
            # ROIC calculation would need more detailed data
            criteria_results['roic_sufficient'] = True  # Placeholder
            
            # Step 5: Conservative Debt
            criteria_results['current_ratio_good'] = metrics.get('current_ratio', 0) >= self.screening_criteria['current_ratio_min']
            criteria_results['debt_ratio_good'] = metrics.get('debt_to_ebitda', float('inf')) <= self.screening_criteria['debt_to_ebitda_max']
            
            # Step 6: Management Quality (qualitative - placeholder)
            criteria_results['management_quality'] = True  # Would need qualitative analysis
            
            # Step 7: Price Analysis (would need valuation models)
            criteria_results['fair_price'] = True  # Placeholder for valuation analysis
            
            result['pass_criteria'] = criteria_results
            
            # Calculate overall score
            passed_criteria = sum(1 for passed in criteria_results.values() if passed)
            total_criteria = len(criteria_results)
            result['overall_score'] = passed_criteria / total_criteria if total_criteria > 0 else 0
            
            # Generate recommendation
            if result['overall_score'] >= 0.8:
                result['recommendation'] = 'BUY'
            elif result['overall_score'] >= 0.6:
                result['recommendation'] = 'HOLD'
            else:
                result['recommendation'] = 'SELL'
            
        except Exception as e:
            logger.error(f"Error screening stock {symbol}: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def screen_multiple_stocks(self, stock_data: Dict) -> pd.DataFrame:
        """
        Screen multiple stocks and return results as DataFrame
        
        Args:
            stock_data: Dictionary with stock symbols as keys and data as values
        
        Returns:
            DataFrame with screening results
        """
        results = []
        
        for symbol, data in stock_data.items():
            try:
                financials = data.get('financials', {})
                price_data = data.get('price_data', pd.DataFrame())
                
                screening_result = self.screen_stock(symbol, financials, price_data)
                
                # Flatten the result for DataFrame
                flat_result = {
                    'symbol': symbol,
                    'overall_score': screening_result['overall_score'],
                    'recommendation': screening_result['recommendation']
                }
                
                # Add metrics
                for key, value in screening_result['metrics'].items():
                    flat_result[f'metric_{key}'] = value
                
                # Add criteria results
                for key, value in screening_result['pass_criteria'].items():
                    flat_result[f'pass_{key}'] = value
                
                results.append(flat_result)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def rank_stocks(self, screening_results: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Rank stocks based on screening results
        
        Args:
            screening_results: DataFrame with screening results
            top_n: Number of top stocks to return
        
        Returns:
            DataFrame with top-ranked stocks
        """
        if screening_results.empty:
            return pd.DataFrame()
        
        # Sort by overall score
        ranked = screening_results.sort_values('overall_score', ascending=False)
        
        # Add rank column
        ranked['rank'] = range(1, len(ranked) + 1)
        
        return ranked.head(top_n)
    
    def get_screening_summary(self, screening_results: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for screening results
        
        Args:
            screening_results: DataFrame with screening results
        
        Returns:
            Dictionary with summary statistics
        """
        if screening_results.empty:
            return {}
        
        summary = {
            'total_stocks': len(screening_results),
            'buy_recommendations': len(screening_results[screening_results['recommendation'] == 'BUY']),
            'hold_recommendations': len(screening_results[screening_results['recommendation'] == 'HOLD']),
            'sell_recommendations': len(screening_results[screening_results['recommendation'] == 'SELL']),
            'average_score': screening_results['overall_score'].mean(),
            'median_score': screening_results['overall_score'].median(),
            'top_score': screening_results['overall_score'].max(),
            'bottom_score': screening_results['overall_score'].min()
        }
        
        return summary

