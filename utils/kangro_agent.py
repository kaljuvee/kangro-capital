"""
LangChain Agent Integration for Kangro Capital
Intelligent stock recommendation and analysis agent
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class KangroAgent:
    """
    Intelligent agent for stock analysis and recommendations
    Uses OpenAI API directly for LLM capabilities and Tavily for search
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.openai_base_url = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. Agent functionality will be limited.")
        
        if not self.tavily_api_key:
            logger.warning("Tavily API key not found. Search functionality will be limited.")
    
    def analyze_stock_fundamentals(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze stock fundamentals using AI
        """
        try:
            if not self.openai_api_key:
                return self._fallback_fundamental_analysis(stock_data)
            
            # Prepare the prompt
            prompt = self._create_fundamental_analysis_prompt(stock_data)
            
            # Call OpenAI API
            response = self._call_openai_api(prompt, max_tokens=500)
            
            if response:
                return {
                    'analysis': response,
                    'confidence_score': self._calculate_confidence_score(stock_data),
                    'key_insights': self._extract_key_insights(response),
                    'recommendation': self._extract_recommendation(response),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._fallback_fundamental_analysis(stock_data)
                
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {str(e)}")
            return self._fallback_fundamental_analysis(stock_data)
    
    def generate_market_insights(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Generate market insights for given symbols
        """
        try:
            insights = {}
            
            for symbol in symbols[:5]:  # Limit to 5 symbols
                # Search for recent news
                news_data = self._search_stock_news(symbol)
                
                # Analyze with AI if available
                if self.openai_api_key and news_data:
                    analysis = self._analyze_news_sentiment(symbol, news_data)
                    insights[symbol] = analysis
                else:
                    insights[symbol] = self._fallback_news_analysis(symbol, news_data)
            
            return {
                'insights': insights,
                'market_summary': self._generate_market_summary(insights),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating market insights: {str(e)}")
            return {'error': str(e)}
    
    def create_investment_strategy(self, screening_results: Dict[str, Any], 
                                 risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Create personalized investment strategy
        """
        try:
            if not screening_results.get('recommendations'):
                return {'error': 'No screening results available'}
            
            # Prepare strategy prompt
            prompt = self._create_strategy_prompt(screening_results, risk_tolerance)
            
            if self.openai_api_key:
                strategy_text = self._call_openai_api(prompt, max_tokens=800)
                
                if strategy_text:
                    return {
                        'strategy': strategy_text,
                        'risk_level': risk_tolerance,
                        'recommended_allocation': self._extract_allocation(strategy_text),
                        'key_principles': self._extract_principles(strategy_text),
                        'timeline': self._extract_timeline(strategy_text),
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Fallback strategy
            return self._create_fallback_strategy(screening_results, risk_tolerance)
            
        except Exception as e:
            logger.error(f"Error creating investment strategy: {str(e)}")
            return {'error': str(e)}
    
    def explain_screening_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain screening results in plain English
        """
        try:
            if not results.get('recommendations'):
                return {'explanation': 'No stocks were found that meet the screening criteria.'}
            
            # Create explanation prompt
            prompt = self._create_explanation_prompt(results)
            
            if self.openai_api_key:
                explanation = self._call_openai_api(prompt, max_tokens=600)
                
                if explanation:
                    return {
                        'explanation': explanation,
                        'key_findings': self._extract_key_findings(results),
                        'actionable_insights': self._extract_actionable_insights(explanation),
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Fallback explanation
            return self._create_fallback_explanation(results)
            
        except Exception as e:
            logger.error(f"Error explaining results: {str(e)}")
            return {'error': str(e)}
    
    def _call_openai_api(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """
        Call OpenAI API directly
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a financial analyst expert specializing in stock analysis and investment strategies.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': max_tokens,
                'temperature': 0.7
            }
            
            response = requests.post(
                f'{self.openai_base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return None
    
    def _search_stock_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Search for stock news using Tavily
        """
        try:
            if not self.tavily_api_key:
                return []
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                'api_key': self.tavily_api_key,
                'query': f'{symbol} stock news earnings financial performance',
                'search_depth': 'basic',
                'max_results': 5
            }
            
            response = requests.post(
                'https://api.tavily.com/search',
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('results', [])
            else:
                logger.error(f"Tavily API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching news: {str(e)}")
            return []
    
    def _create_fundamental_analysis_prompt(self, stock_data: Dict[str, Any]) -> str:
        """Create prompt for fundamental analysis"""
        return f"""
        Analyze the following stock fundamental data and provide insights:
        
        Stock Symbol: {stock_data.get('symbol', 'Unknown')}
        Financial Metrics:
        - ROE: {stock_data.get('roe', 'N/A')}
        - Current Ratio: {stock_data.get('current_ratio', 'N/A')}
        - Gross Margin: {stock_data.get('gross_margin', 'N/A')}
        - Net Margin: {stock_data.get('net_margin', 'N/A')}
        - Revenue Growth: {stock_data.get('revenue_growth_5y', 'N/A')}
        - Debt to EBITDA: {stock_data.get('debt_to_ebitda', 'N/A')}
        
        Please provide:
        1. Overall financial health assessment
        2. Key strengths and weaknesses
        3. Investment recommendation (Buy/Hold/Sell)
        4. Risk factors to consider
        
        Keep the analysis concise and actionable.
        """
    
    def _create_strategy_prompt(self, screening_results: Dict[str, Any], risk_tolerance: str) -> str:
        """Create prompt for investment strategy"""
        top_stocks = screening_results.get('recommendations', [])[:5]
        stock_list = ', '.join([stock.get('symbol', '') for stock in top_stocks])
        
        return f"""
        Create a personalized investment strategy based on these screening results:
        
        Top Recommended Stocks: {stock_list}
        Risk Tolerance: {risk_tolerance}
        Number of Recommendations: {len(top_stocks)}
        
        Please provide:
        1. Portfolio allocation strategy
        2. Risk management approach
        3. Investment timeline recommendations
        4. Key principles to follow
        5. Exit strategy considerations
        
        Tailor the strategy to the {risk_tolerance} risk tolerance level.
        """
    
    def _create_explanation_prompt(self, results: Dict[str, Any]) -> str:
        """Create prompt for explaining screening results"""
        recommendations = results.get('recommendations', [])
        top_3 = recommendations[:3]
        
        stock_info = []
        for stock in top_3:
            stock_info.append(f"- {stock.get('symbol', 'Unknown')}: Score {stock.get('composite_score', 0):.2f}")
        
        return f"""
        Explain these stock screening results in simple terms:
        
        Total stocks screened: {len(recommendations)}
        Top 3 recommendations:
        {chr(10).join(stock_info)}
        
        Please explain:
        1. What these results mean for an average investor
        2. Why these stocks were selected
        3. What the composite scores indicate
        4. Practical next steps for investors
        
        Use clear, non-technical language that anyone can understand.
        """
    
    def _fallback_fundamental_analysis(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when AI is not available"""
        symbol = stock_data.get('symbol', 'Unknown')
        roe = stock_data.get('roe', 0)
        current_ratio = stock_data.get('current_ratio', 0)
        
        # Simple rule-based analysis
        strengths = []
        weaknesses = []
        
        if roe > 0.15:
            strengths.append("Strong return on equity")
        elif roe < 0.05:
            weaknesses.append("Low return on equity")
        
        if current_ratio > 1.5:
            strengths.append("Good liquidity position")
        elif current_ratio < 1.0:
            weaknesses.append("Potential liquidity concerns")
        
        recommendation = "Hold"
        if len(strengths) > len(weaknesses):
            recommendation = "Buy"
        elif len(weaknesses) > len(strengths):
            recommendation = "Sell"
        
        return {
            'analysis': f"Basic analysis for {symbol}: {len(strengths)} strengths, {len(weaknesses)} weaknesses identified.",
            'confidence_score': 0.6,
            'key_insights': strengths + weaknesses,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
    
    def _fallback_news_analysis(self, symbol: str, news_data: List[Dict]) -> Dict[str, Any]:
        """Fallback news analysis"""
        return {
            'sentiment': 'Neutral',
            'news_count': len(news_data),
            'summary': f"Found {len(news_data)} recent news articles for {symbol}",
            'confidence': 0.5
        }
    
    def _create_fallback_strategy(self, screening_results: Dict[str, Any], risk_tolerance: str) -> Dict[str, Any]:
        """Create fallback investment strategy"""
        recommendations = screening_results.get('recommendations', [])
        
        if risk_tolerance == 'conservative':
            allocation = {'stocks': 60, 'bonds': 30, 'cash': 10}
        elif risk_tolerance == 'aggressive':
            allocation = {'stocks': 90, 'bonds': 5, 'cash': 5}
        else:  # moderate
            allocation = {'stocks': 70, 'bonds': 20, 'cash': 10}
        
        return {
            'strategy': f"Based on {len(recommendations)} screened stocks, implement a {risk_tolerance} portfolio strategy.",
            'risk_level': risk_tolerance,
            'recommended_allocation': allocation,
            'key_principles': [
                'Diversify across sectors',
                'Regular portfolio rebalancing',
                'Long-term investment horizon'
            ],
            'timeline': '3-5 years',
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_fallback_explanation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback explanation"""
        recommendations = results.get('recommendations', [])
        
        return {
            'explanation': f"The screening process identified {len(recommendations)} stocks that meet the specified criteria. These stocks scored well on fundamental metrics like profitability, growth, and financial stability.",
            'key_findings': [
                f"Total stocks analyzed: {len(recommendations)}",
                "Stocks ranked by composite score",
                "Focus on fundamental strength"
            ],
            'actionable_insights': [
                "Review top-ranked stocks individually",
                "Consider portfolio diversification",
                "Monitor performance regularly"
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    # Helper methods for extracting information from AI responses
    def _calculate_confidence_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on data completeness"""
        total_fields = 7
        available_fields = sum(1 for key in ['roe', 'current_ratio', 'gross_margin', 'net_margin', 
                                           'revenue_growth_5y', 'debt_to_ebitda'] 
                             if stock_data.get(key) is not None)
        return available_fields / total_fields
    
    def _extract_key_insights(self, text: str) -> List[str]:
        """Extract key insights from analysis text"""
        # Simple extraction - look for numbered points or bullet points
        insights = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                insights.append(line)
        return insights[:5]  # Limit to 5 insights
    
    def _extract_recommendation(self, text: str) -> str:
        """Extract investment recommendation from text"""
        text_lower = text.lower()
        if 'buy' in text_lower and 'sell' not in text_lower:
            return 'Buy'
        elif 'sell' in text_lower:
            return 'Sell'
        else:
            return 'Hold'
    
    def _analyze_news_sentiment(self, symbol: str, news_data: List[Dict]) -> Dict[str, Any]:
        """Analyze news sentiment using AI"""
        if not news_data or not self.openai_api_key:
            return self._fallback_news_analysis(symbol, news_data)
        
        # Create news summary
        news_text = ""
        for article in news_data[:3]:  # Limit to 3 articles
            news_text += f"- {article.get('title', '')}\n"
        
        prompt = f"""
        Analyze the sentiment of these recent news headlines for {symbol}:
        
        {news_text}
        
        Provide:
        1. Overall sentiment (Positive/Negative/Neutral)
        2. Key themes
        3. Impact on stock price (likely direction)
        
        Keep response concise.
        """
        
        response = self._call_openai_api(prompt, max_tokens=200)
        
        if response:
            sentiment = 'Neutral'
            if 'positive' in response.lower():
                sentiment = 'Positive'
            elif 'negative' in response.lower():
                sentiment = 'Negative'
            
            return {
                'sentiment': sentiment,
                'analysis': response,
                'news_count': len(news_data),
                'confidence': 0.8
            }
        
        return self._fallback_news_analysis(symbol, news_data)
    
    def _generate_market_summary(self, insights: Dict[str, Any]) -> str:
        """Generate overall market summary"""
        if not insights:
            return "No market data available for analysis."
        
        positive_count = sum(1 for insight in insights.values() 
                           if insight.get('sentiment') == 'Positive')
        negative_count = sum(1 for insight in insights.values() 
                           if insight.get('sentiment') == 'Negative')
        
        total = len(insights)
        
        if positive_count > negative_count:
            return f"Market sentiment appears positive for {positive_count}/{total} analyzed stocks."
        elif negative_count > positive_count:
            return f"Market sentiment appears negative for {negative_count}/{total} analyzed stocks."
        else:
            return f"Mixed market sentiment across {total} analyzed stocks."
    
    def _extract_allocation(self, text: str) -> Dict[str, float]:
        """Extract allocation percentages from strategy text"""
        # Simple pattern matching for percentages
        import re
        percentages = re.findall(r'(\d+)%', text)
        
        if len(percentages) >= 2:
            return {
                'stocks': float(percentages[0]),
                'bonds': float(percentages[1]) if len(percentages) > 1 else 20,
                'cash': float(percentages[2]) if len(percentages) > 2 else 10
            }
        
        return {'stocks': 70, 'bonds': 20, 'cash': 10}  # Default allocation
    
    def _extract_principles(self, text: str) -> List[str]:
        """Extract key principles from strategy text"""
        return self._extract_key_insights(text)
    
    def _extract_timeline(self, text: str) -> str:
        """Extract investment timeline from text"""
        text_lower = text.lower()
        if 'long-term' in text_lower or 'years' in text_lower:
            return '3-5 years'
        elif 'short-term' in text_lower or 'months' in text_lower:
            return '6-12 months'
        else:
            return '1-3 years'
    
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from screening results"""
        recommendations = results.get('recommendations', [])
        
        findings = [
            f"Total recommendations: {len(recommendations)}",
        ]
        
        if recommendations:
            top_score = recommendations[0].get('composite_score', 0)
            findings.append(f"Top score: {top_score:.2f}")
            
            # Count by sector if available
            sectors = {}
            for stock in recommendations:
                sector = stock.get('sector', 'Unknown')
                sectors[sector] = sectors.get(sector, 0) + 1
            
            if sectors:
                top_sector = max(sectors.items(), key=lambda x: x[1])
                findings.append(f"Top sector: {top_sector[0]} ({top_sector[1]} stocks)")
        
        return findings
    
    def _extract_actionable_insights(self, text: str) -> List[str]:
        """Extract actionable insights from explanation text"""
        insights = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['should', 'consider', 'recommend', 'suggest']):
                insights.append(line)
        
        if not insights:
            insights = [
                "Review individual stock fundamentals",
                "Consider portfolio diversification",
                "Monitor market conditions regularly"
            ]
        
        return insights[:5]  # Limit to 5 insights

# Convenience functions for Streamlit integration
def create_kangro_agent() -> KangroAgent:
    """Create and return a KangroAgent instance"""
    return KangroAgent()

def analyze_screening_results_with_ai(screening_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze screening results using AI agent"""
    agent = create_kangro_agent()
    return agent.explain_screening_results(screening_results)

def generate_investment_strategy(screening_results: Dict[str, Any], 
                               risk_tolerance: str = 'moderate') -> Dict[str, Any]:
    """Generate investment strategy using AI agent"""
    agent = create_kangro_agent()
    return agent.create_investment_strategy(screening_results, risk_tolerance)

def get_market_insights(symbols: List[str]) -> Dict[str, Any]:
    """Get market insights for symbols using AI agent"""
    agent = create_kangro_agent()
    return agent.generate_market_insights(symbols)

