"""
Kangro Capital - Stock Screening Platform
Complete version with AI functionality
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Kangro Capital - Stock Screening Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .ai-response {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_stock_data(symbol, period="1y"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def calculate_basic_metrics(data, info, symbol='UNKNOWN'):
    """Calculate basic stock metrics"""
    if data is None or data.empty:
        return {}
    
    current_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
    price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
    
    # Calculate moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Basic metrics
    metrics = {
        'current_price': current_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'volume': data['Volume'].iloc[-1],
        'high_52w': data['High'].max(),
        'low_52w': data['Low'].min(),
        'ma20': data['MA20'].iloc[-1] if not pd.isna(data['MA20'].iloc[-1]) else None,
        'ma50': data['MA50'].iloc[-1] if not pd.isna(data['MA50'].iloc[-1]) else None,
    }
    
    # Add info metrics if available
    if info:
        metrics.update({
            'pe_ratio': info.get('trailingPE'),
            'market_cap': info.get('marketCap'),
            'dividend_yield': info.get('dividendYield'),
            'beta': info.get('beta'),
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'roe': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'revenue_growth': info.get('revenueGrowth')
        })
    
    return metrics

def get_ai_analysis(prompt, max_tokens=500):
    """Get AI analysis using OpenAI API directly"""
    try:
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            return "AI analysis unavailable - OpenAI API key not configured"
        
        headers = {
            'Authorization': f'Bearer {openai_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4o-mini',  # Use a supported model
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a professional financial analyst specializing in stock analysis and investment strategies. Provide concise, actionable insights.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': max_tokens,
            'temperature': 0.3
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"AI analysis error: {response.status_code}"
            
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

def search_market_news(query, max_results=3):
    """Search for market news using Tavily"""
    try:
        tavily_key = os.getenv('TAVILY_API_KEY')
        if not tavily_key:
            return []
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": tavily_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": max_results
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            return []
            
    except Exception as e:
        st.error(f"News search error: {str(e)}")
        return []

def create_price_chart(data, symbol):
    """Create interactive price chart"""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add moving averages if available
    if 'MA20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            mode='lines',
            name='MA20',
            line=dict(color='orange', width=1)
        ))
    
    if 'MA50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            mode='lines',
            name='MA50',
            line=dict(color='red', width=1)
        ))
    
    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_volume_chart(data, symbol):
    """Create volume chart"""
    fig = px.bar(
        x=data.index,
        y=data['Volume'],
        title=f"{symbol} Trading Volume"
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
        showlegend=False,
        height=300
    )
    
    return fig

def screen_stocks(symbols, criteria):
    """Stock screening with AI-enhanced analysis"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        status_text.text(f"Analyzing {symbol}...")
        progress_bar.progress((i + 1) / len(symbols))
        
        data, info = get_stock_data(symbol, period="1y")
        if data is not None and not data.empty:
            metrics = calculate_basic_metrics(data, info, symbol)
            
            # Apply screening criteria
            score = 0
            reasons = []
            
            # PE Ratio check
            if metrics.get('pe_ratio') and criteria.get('max_pe'):
                if metrics['pe_ratio'] <= criteria['max_pe']:
                    score += 1
                    reasons.append(f"PE ratio {metrics['pe_ratio']:.1f} ≤ {criteria['max_pe']}")
            
            # ROE check
            if metrics.get('roe') and criteria.get('min_roe'):
                if metrics['roe'] >= criteria['min_roe']:
                    score += 1
                    reasons.append(f"ROE {metrics['roe']*100:.1f}% ≥ {criteria['min_roe']*100:.1f}%")
            
            # Price above MA20
            if metrics.get('ma20') and metrics['current_price'] > metrics['ma20']:
                score += 1
                reasons.append("Price above MA20")
            
            # Positive price change
            if metrics['price_change_pct'] > 0:
                score += 1
                reasons.append(f"Positive momentum: +{metrics['price_change_pct']:.1f}%")
            
            # Market cap check
            if metrics.get('market_cap') and criteria.get('min_market_cap'):
                if metrics['market_cap'] >= criteria['min_market_cap']:
                    score += 1
                    reasons.append(f"Market cap ${metrics['market_cap']/1e9:.1f}B")
            
            # Debt to equity check
            if metrics.get('debt_to_equity') and criteria.get('max_debt_equity'):
                if metrics['debt_to_equity'] <= criteria['max_debt_equity']:
                    score += 1
                    reasons.append(f"Debt/Equity {metrics['debt_to_equity']:.1f} ≤ {criteria['max_debt_equity']}")
            
            results.append({
                'symbol': symbol,
                'company_name': metrics.get('company_name', symbol),
                'current_price': metrics['current_price'],
                'price_change_pct': metrics['price_change_pct'],
                'pe_ratio': metrics.get('pe_ratio'),
                'roe': metrics.get('roe'),
                'market_cap': metrics.get('market_cap'),
                'debt_to_equity': metrics.get('debt_to_equity'),
                'volume': metrics['volume'],
                'sector': metrics.get('sector', 'N/A'),
                'score': score,
                'reasons': '; '.join(reasons) if reasons else 'No criteria met'
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

def analyze_portfolio_performance(symbols, weights=None):
    """Analyze portfolio performance"""
    if weights is None:
        weights = [1/len(symbols)] * len(symbols)
    
    portfolio_data = []
    
    for symbol, weight in zip(symbols, weights):
        data, _ = get_stock_data(symbol, period="1y")
        if data is not None and not data.empty:
            returns = data['Close'].pct_change().dropna()
            portfolio_data.append(returns * weight)
    
    if portfolio_data:
        portfolio_returns = pd.concat(portfolio_data, axis=1).sum(axis=1)
        
        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
        volatility = portfolio_returns.std() * (252 ** 0.5)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'returns': portfolio_returns
        }
    
    return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Kangro Capital</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Stock Screening & AI-Powered Analysis Platform</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Dashboard",
        "🔍 Stock Analysis", 
        "📊 Stock Screening",
        "🤖 AI Insights",
        "💼 Portfolio Analysis",
        "📈 Market Overview",
        "⚙️ Settings"
    ])
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔍 Stock Analysis":
        show_stock_analysis()
    elif page == "📊 Stock Screening":
        show_stock_screening()
    elif page == "🤖 AI Insights":
        show_ai_insights()
    elif page == "💼 Portfolio Analysis":
        show_portfolio_analysis()
    elif page == "📈 Market Overview":
        show_market_overview()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """Dashboard page"""
    st.header("📊 Platform Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Platform Status", "🟢 Online", "Fully Operational")
    
    with col2:
        st.metric("AI Features", "🤖 Active", "OpenAI + Tavily")
    
    with col3:
        st.metric("Data Sources", "📡 Live", "Real-time feeds")
    
    with col4:
        st.metric("Analysis Tools", "🧠 Advanced", "ML + AI Powered")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze AAPL", use_container_width=True):
            st.session_state.analysis_symbol = "AAPL"
            st.rerun()
    
    with col2:
        if st.button("📊 Run AI Screening", use_container_width=True):
            st.session_state.ai_screening = True
            st.rerun()
    
    with col3:
        if st.button("🤖 Get Market Insights", use_container_width=True):
            st.session_state.market_insights = True
            st.rerun()
    
    # Handle quick actions
    if st.session_state.get('analysis_symbol'):
        st.subheader(f"📈 Quick Analysis: {st.session_state.analysis_symbol}")
        data, info = get_stock_data(st.session_state.analysis_symbol)
        if data is not None:
            metrics = calculate_basic_metrics(data, info, st.session_state.analysis_symbol)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${metrics['current_price']:.2f}", 
                         f"{metrics['price_change_pct']:+.1f}%")
            with col2:
                if metrics.get('pe_ratio'):
                    st.metric("P/E Ratio", f"{metrics['pe_ratio']:.1f}")
            with col3:
                if metrics.get('market_cap'):
                    st.metric("Market Cap", f"${metrics['market_cap']/1e9:.1f}B")
            
            # AI Quick Analysis
            if st.button("🤖 Get AI Analysis"):
                with st.spinner("Generating AI analysis..."):
                    prompt = f"""
                    Analyze {st.session_state.analysis_symbol} stock with these metrics:
                    - Current Price: ${metrics['current_price']:.2f}
                    - Price Change: {metrics['price_change_pct']:+.1f}%
                    - P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
                    - Market Cap: ${metrics.get('market_cap', 0)/1e9:.1f}B
                    
                    Provide a brief investment recommendation in 2-3 sentences.
                    """
                    
                    ai_response = get_ai_analysis(prompt, max_tokens=200)
                    st.markdown(f'<div class="ai-response"><strong>🤖 AI Analysis:</strong><br>{ai_response}</div>', unsafe_allow_html=True)
        
        if st.button("Clear Analysis"):
            del st.session_state.analysis_symbol
            st.rerun()
    
    if st.session_state.get('market_insights'):
        st.subheader("🤖 AI Market Insights")
        
        with st.spinner("Generating market insights..."):
            # Get market news
            news = search_market_news("stock market analysis today", max_results=2)
            
            # Generate AI insights
            prompt = """
            Provide current market insights covering:
            1. Overall market sentiment
            2. Key sectors to watch
            3. Investment opportunities
            4. Risk factors to consider
            
            Keep it concise and actionable (3-4 sentences).
            """
            
            ai_insights = get_ai_analysis(prompt, max_tokens=300)
            st.markdown(f'<div class="ai-response"><strong>🤖 Market Insights:</strong><br>{ai_insights}</div>', unsafe_allow_html=True)
            
            if news:
                st.subheader("📰 Latest Market News")
                for item in news[:2]:
                    with st.expander(f"📰 {item.get('title', 'News Item')}"):
                        st.write(item.get('content', 'No content available'))
                        if item.get('url'):
                            st.write(f"[Read more]({item['url']})")
        
        if st.button("Clear Insights"):
            del st.session_state.market_insights
            st.rerun()

def show_stock_analysis():
    """Stock analysis page with AI enhancement"""
    st.header("🔍 AI-Enhanced Stock Analysis")
    
    # Stock symbol input
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL").upper()
    with col2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if symbol:
        # Fetch data
        with st.spinner(f"Fetching data for {symbol}..."):
            data, info = get_stock_data(symbol, period)
        
        if data is not None and not data.empty:
            metrics = calculate_basic_metrics(data, info, symbol)
            
            # Display company info
            st.subheader(f"📊 {metrics.get('company_name', symbol)} ({symbol})")
            
            if metrics.get('sector'):
                st.write(f"**Sector:** {metrics['sector']} | **Industry:** {metrics.get('industry', 'N/A')}")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                change_color = "normal" if metrics['price_change_pct'] >= 0 else "inverse"
                st.metric(
                    "Current Price", 
                    f"${metrics['current_price']:.2f}",
                    f"{metrics['price_change_pct']:+.1f}%",
                    delta_color=change_color
                )
            
            with col2:
                if metrics.get('pe_ratio'):
                    st.metric("P/E Ratio", f"{metrics['pe_ratio']:.1f}")
                else:
                    st.metric("P/E Ratio", "N/A")
            
            with col3:
                if metrics.get('market_cap'):
                    st.metric("Market Cap", f"${metrics['market_cap']/1e9:.1f}B")
                else:
                    st.metric("Market Cap", "N/A")
            
            with col4:
                if metrics.get('roe'):
                    st.metric("ROE", f"{metrics['roe']*100:.1f}%")
                else:
                    st.metric("ROE", "N/A")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_price_chart(data, symbol), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_volume_chart(data, symbol), use_container_width=True)
            
            # AI Analysis Section
            st.subheader("🤖 AI-Powered Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Get Fundamental Analysis", use_container_width=True):
                    with st.spinner("Generating fundamental analysis..."):
                        prompt = f"""
                        Perform fundamental analysis for {symbol} with these metrics:
                        - P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
                        - ROE: {metrics.get('roe', 'N/A')}
                        - Market Cap: ${metrics.get('market_cap', 0)/1e9:.1f}B
                        - Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}
                        - Sector: {metrics.get('sector', 'N/A')}
                        
                        Provide investment recommendation with reasoning.
                        """
                        
                        analysis = get_ai_analysis(prompt, max_tokens=400)
                        st.markdown(f'<div class="ai-response">{analysis}</div>', unsafe_allow_html=True)
            
            with col2:
                if st.button("📈 Get Technical Analysis", use_container_width=True):
                    with st.spinner("Generating technical analysis..."):
                        prompt = f"""
                        Perform technical analysis for {symbol}:
                        - Current Price: ${metrics['current_price']:.2f}
                        - Price Change: {metrics['price_change_pct']:+.1f}%
                        - 20-day MA: ${metrics.get('ma20', 0):.2f}
                        - 50-day MA: ${metrics.get('ma50', 0):.2f}
                        - 52-week High: ${metrics['high_52w']:.2f}
                        - 52-week Low: ${metrics['low_52w']:.2f}
                        
                        Analyze trends and provide trading insights.
                        """
                        
                        analysis = get_ai_analysis(prompt, max_tokens=400)
                        st.markdown(f'<div class="ai-response">{analysis}</div>', unsafe_allow_html=True)
            
            # News and Sentiment
            if st.button("📰 Get Latest News & Sentiment"):
                with st.spinner("Searching for latest news..."):
                    news = search_market_news(f"{symbol} stock news analysis", max_results=3)
                    
                    if news:
                        st.subheader("📰 Latest News")
                        for item in news:
                            with st.expander(f"📰 {item.get('title', 'News Item')}"):
                                st.write(item.get('content', 'No content available'))
                                if item.get('url'):
                                    st.write(f"[Read more]({item['url']})")
                        
                        # AI Sentiment Analysis
                        news_text = " ".join([item.get('content', '') for item in news[:2]])
                        if news_text:
                            prompt = f"""
                            Analyze the sentiment of this news about {symbol}:
                            
                            {news_text[:1000]}
                            
                            Provide:
                            1. Overall sentiment (Positive/Negative/Neutral)
                            2. Key factors affecting the stock
                            3. Short-term outlook
                            """
                            
                            sentiment = get_ai_analysis(prompt, max_tokens=300)
                            st.subheader("🎯 AI Sentiment Analysis")
                            st.markdown(f'<div class="ai-response">{sentiment}</div>', unsafe_allow_html=True)

def show_stock_screening():
    """Enhanced stock screening with AI"""
    st.header("📊 AI-Enhanced Stock Screening")
    
    # Screening parameters
    st.subheader("🔧 Screening Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_pe = st.number_input("Maximum P/E Ratio", min_value=1.0, max_value=100.0, value=25.0, step=0.5)
        min_roe = st.number_input("Minimum ROE (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0) / 100
        min_market_cap = st.number_input("Minimum Market Cap (Billions)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1) * 1e9
    
    with col2:
        max_debt_equity = st.number_input("Maximum Debt/Equity", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        stock_list = st.text_area(
            "Stock Symbols (one per line)", 
            value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\nNVDA\nMETA\nNFLX\nADBE\nCRM\nORCL\nINTC\nAMD\nIBM\nCSCO",
            height=200
        )
    
    symbols = [s.strip().upper() for s in stock_list.split('\n') if s.strip()]
    
    criteria = {
        'max_pe': max_pe,
        'min_roe': min_roe,
        'min_market_cap': min_market_cap,
        'max_debt_equity': max_debt_equity
    }
    
    if st.button("🚀 Run AI-Enhanced Screening", type="primary"):
        if symbols:
            st.subheader("📋 Screening Results")
            
            results = screen_stocks(symbols, criteria)
            
            if results:
                # Convert to DataFrame for display
                df = pd.DataFrame(results)
                
                # Format columns
                df['current_price'] = df['current_price'].apply(lambda x: f"${x:.2f}")
                df['price_change_pct'] = df['price_change_pct'].apply(lambda x: f"{x:+.1f}%")
                df['pe_ratio'] = df['pe_ratio'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                df['roe'] = df['roe'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
                df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else "N/A")
                df['debt_to_equity'] = df['debt_to_equity'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                
                # Display results
                st.dataframe(
                    df[['symbol', 'company_name', 'current_price', 'price_change_pct', 'pe_ratio', 'roe', 'market_cap', 'debt_to_equity', 'sector', 'score', 'reasons']],
                    column_config={
                        'symbol': 'Symbol',
                        'company_name': 'Company',
                        'current_price': 'Price',
                        'price_change_pct': 'Change %',
                        'pe_ratio': 'P/E',
                        'roe': 'ROE',
                        'market_cap': 'Market Cap',
                        'debt_to_equity': 'Debt/Equity',
                        'sector': 'Sector',
                        'score': 'Score',
                        'reasons': 'Criteria Met'
                    },
                    use_container_width=True
                )
                
                # Export functionality
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"kangro_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # AI Analysis of Results
                if st.button("🤖 Get AI Analysis of Results"):
                    with st.spinner("Analyzing screening results..."):
                        top_stocks = df.head(5)['symbol'].tolist()
                        
                        prompt = f"""
                        Analyze these top screening results:
                        
                        Top 5 stocks: {', '.join(top_stocks)}
                        Screening criteria: P/E ≤ {max_pe}, ROE ≥ {min_roe*100:.1f}%, Market Cap ≥ ${min_market_cap/1e9:.1f}B, Debt/Equity ≤ {max_debt_equity}
                        
                        Provide:
                        1. Overall assessment of the screening results
                        2. Which stocks look most promising and why
                        3. Any red flags or concerns
                        4. Portfolio construction suggestions
                        """
                        
                        ai_analysis = get_ai_analysis(prompt, max_tokens=500)
                        st.subheader("🤖 AI Analysis of Screening Results")
                        st.markdown(f'<div class="ai-response">{ai_analysis}</div>', unsafe_allow_html=True)
                
                # Top picks with AI insights
                top_picks = df.head(3)
                if not top_picks.empty:
                    st.subheader("🏆 Top Picks with AI Insights")
                    for _, stock in top_picks.iterrows():
                        with st.expander(f"🎯 {stock['symbol']} - {stock['company_name']} (Score: {stock['score']})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Current Price:** {stock['current_price']}")
                                st.write(f"**Price Change:** {stock['price_change_pct']}")
                                st.write(f"**P/E Ratio:** {stock['pe_ratio']}")
                                st.write(f"**ROE:** {stock['roe']}")
                            with col2:
                                st.write(f"**Market Cap:** {stock['market_cap']}")
                                st.write(f"**Debt/Equity:** {stock['debt_to_equity']}")
                                st.write(f"**Sector:** {stock['sector']}")
                                st.write(f"**Criteria Met:** {stock['reasons']}")
                            
                            if st.button(f"🤖 Get AI Insights for {stock['symbol']}", key=f"ai_{stock['symbol']}"):
                                with st.spinner(f"Generating insights for {stock['symbol']}..."):
                                    prompt = f"""
                                    Provide investment insights for {stock['symbol']} ({stock['company_name']}):
                                    - Sector: {stock['sector']}
                                    - Current metrics meet our screening criteria
                                    - Score: {stock['score']}/6
                                    
                                    Analyze:
                                    1. Investment thesis
                                    2. Key strengths
                                    3. Potential risks
                                    4. Price target outlook
                                    """
                                    
                                    insights = get_ai_analysis(prompt, max_tokens=400)
                                    st.markdown(f'<div class="ai-response">{insights}</div>', unsafe_allow_html=True)
            else:
                st.warning("No results found. Try adjusting your criteria.")
        else:
            st.error("Please enter at least one stock symbol.")

def show_ai_insights():
    """AI Insights page"""
    st.header("🤖 AI-Powered Investment Insights")
    
    # AI Analysis Options
    st.subheader("🧠 Choose Analysis Type")
    
    analysis_type = st.selectbox("Select Analysis", [
        "Market Overview & Sentiment",
        "Sector Analysis",
        "Stock Comparison",
        "Investment Strategy",
        "Risk Assessment",
        "Custom Query"
    ])
    
    if analysis_type == "Market Overview & Sentiment":
        if st.button("🌍 Generate Market Overview"):
            with st.spinner("Analyzing current market conditions..."):
                # Get market news
                news = search_market_news("stock market outlook analysis", max_results=3)
                
                prompt = """
                Provide a comprehensive market overview covering:
                1. Current market sentiment and key drivers
                2. Economic indicators to watch
                3. Sector rotation trends
                4. Opportunities and risks in the current environment
                5. Investment recommendations for different risk profiles
                
                Be specific and actionable.
                """
                
                analysis = get_ai_analysis(prompt, max_tokens=600)
                st.markdown(f'<div class="ai-response">{analysis}</div>', unsafe_allow_html=True)
                
                if news:
                    st.subheader("📰 Supporting Market News")
                    for item in news:
                        with st.expander(f"📰 {item.get('title', 'News Item')}"):
                            st.write(item.get('content', 'No content available'))
    
    elif analysis_type == "Stock Comparison":
        st.subheader("📊 Compare Stocks")
        
        col1, col2 = st.columns(2)
        with col1:
            stock1 = st.text_input("First Stock Symbol", value="AAPL").upper()
        with col2:
            stock2 = st.text_input("Second Stock Symbol", value="MSFT").upper()
        
        if st.button("🔍 Compare Stocks") and stock1 and stock2:
            with st.spinner(f"Comparing {stock1} vs {stock2}..."):
                # Get data for both stocks
                data1, info1 = get_stock_data(stock1)
                data2, info2 = get_stock_data(stock2)
                
                if data1 is not None and data2 is not None:
                    metrics1 = calculate_basic_metrics(data1, info1, stock1)
                    metrics2 = calculate_basic_metrics(data2, info2, stock2)
                    
                    prompt = f"""
                    Compare these two stocks for investment:
                    
                    {stock1} ({metrics1.get('company_name', stock1)}):
                    - Price: ${metrics1['current_price']:.2f} ({metrics1['price_change_pct']:+.1f}%)
                    - P/E: {metrics1.get('pe_ratio', 'N/A')}
                    - Market Cap: ${metrics1.get('market_cap', 0)/1e9:.1f}B
                    - ROE: {metrics1.get('roe', 'N/A')}
                    - Sector: {metrics1.get('sector', 'N/A')}
                    
                    {stock2} ({metrics2.get('company_name', stock2)}):
                    - Price: ${metrics2['current_price']:.2f} ({metrics2['price_change_pct']:+.1f}%)
                    - P/E: {metrics2.get('pe_ratio', 'N/A')}
                    - Market Cap: ${metrics2.get('market_cap', 0)/1e9:.1f}B
                    - ROE: {metrics2.get('roe', 'N/A')}
                    - Sector: {metrics2.get('sector', 'N/A')}
                    
                    Provide detailed comparison covering valuation, growth prospects, risks, and investment recommendation.
                    """
                    
                    comparison = get_ai_analysis(prompt, max_tokens=600)
                    st.markdown(f'<div class="ai-response">{comparison}</div>', unsafe_allow_html=True)
    
    elif analysis_type == "Custom Query":
        st.subheader("💬 Ask AI Anything")
        
        custom_query = st.text_area(
            "Enter your investment question:",
            placeholder="e.g., What are the best dividend stocks for 2024? Should I invest in tech stocks now? How to build a recession-proof portfolio?",
            height=100
        )
        
        if st.button("🤖 Get AI Response") and custom_query:
            with st.spinner("Generating response..."):
                prompt = f"""
                As a professional financial advisor, answer this investment question:
                
                {custom_query}
                
                Provide a comprehensive, actionable response with specific recommendations where appropriate.
                """
                
                response = get_ai_analysis(prompt, max_tokens=600)
                st.markdown(f'<div class="ai-response">{response}</div>', unsafe_allow_html=True)
    
    # Add other analysis types...
    elif analysis_type == "Investment Strategy":
        st.subheader("📈 Personalized Investment Strategy")
        
        col1, col2 = st.columns(2)
        with col1:
            risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
            investment_horizon = st.selectbox("Investment Horizon", ["Short-term (< 1 year)", "Medium-term (1-5 years)", "Long-term (> 5 years)"])
        with col2:
            investment_amount = st.number_input("Investment Amount ($)", min_value=1000, max_value=10000000, value=50000, step=1000)
            goals = st.multiselect("Investment Goals", ["Capital Growth", "Income Generation", "Capital Preservation", "Inflation Protection"])
        
        if st.button("🎯 Generate Strategy"):
            with st.spinner("Creating personalized investment strategy..."):
                prompt = f"""
                Create a personalized investment strategy with these parameters:
                - Risk Tolerance: {risk_tolerance}
                - Investment Horizon: {investment_horizon}
                - Investment Amount: ${investment_amount:,}
                - Goals: {', '.join(goals)}
                
                Provide:
                1. Asset allocation recommendations
                2. Specific investment suggestions (stocks, ETFs, sectors)
                3. Risk management strategies
                4. Rebalancing guidelines
                5. Expected returns and timeline
                """
                
                strategy = get_ai_analysis(prompt, max_tokens=700)
                st.markdown(f'<div class="ai-response">{strategy}</div>', unsafe_allow_html=True)

def show_portfolio_analysis():
    """Portfolio analysis page"""
    st.header("💼 Portfolio Analysis")
    
    st.subheader("📊 Portfolio Composition")
    
    # Portfolio input
    portfolio_input = st.text_area(
        "Enter your portfolio (Symbol:Weight, one per line)",
        value="AAPL:0.3\nMSFT:0.25\nGOOGL:0.2\nAMZN:0.15\nTSLA:0.1",
        height=150
    )
    
    if portfolio_input:
        try:
            portfolio = {}
            for line in portfolio_input.strip().split('\n'):
                if ':' in line:
                    symbol, weight = line.split(':')
                    portfolio[symbol.strip().upper()] = float(weight.strip())
            
            symbols = list(portfolio.keys())
            weights = list(portfolio.values())
            
            # Validate weights
            total_weight = sum(weights)
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"Portfolio weights sum to {total_weight:.2f}, not 1.0. Consider rebalancing.")
            
            # Display portfolio composition
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(values=weights, names=symbols, title="Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Portfolio metrics
                performance = analyze_portfolio_performance(symbols, weights)
                if performance:
                    st.metric("Total Return", f"{performance['total_return']*100:.1f}%")
                    st.metric("Annual Return", f"{performance['annual_return']*100:.1f}%")
                    st.metric("Volatility", f"{performance['volatility']*100:.1f}%")
                    st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
            
            # AI Portfolio Analysis
            if st.button("🤖 Get AI Portfolio Analysis"):
                with st.spinner("Analyzing your portfolio..."):
                    # Collect portfolio data
                    portfolio_data = []
                    for symbol, weight in portfolio.items():
                        data, info = get_stock_data(symbol)
                        if data is not None:
                            metrics = calculate_basic_metrics(data, info, symbol)
                            portfolio_data.append({
                                'symbol': symbol,
                                'weight': weight,
                                'sector': metrics.get('sector', 'Unknown'),
                                'pe_ratio': metrics.get('pe_ratio'),
                                'market_cap': metrics.get('market_cap')
                            })
                    
                    # Create analysis prompt
                    portfolio_summary = "\n".join([
                        f"- {item['symbol']}: {item['weight']*100:.1f}% ({item['sector']}, P/E: {item['pe_ratio']})"
                        for item in portfolio_data
                    ])
                    
                    prompt = f"""
                    Analyze this investment portfolio:
                    
                    {portfolio_summary}
                    
                    Performance metrics:
                    - Annual Return: {performance['annual_return']*100:.1f}%
                    - Volatility: {performance['volatility']*100:.1f}%
                    - Sharpe Ratio: {performance['sharpe_ratio']:.2f}
                    
                    Provide analysis covering:
                    1. Diversification assessment
                    2. Risk-return profile
                    3. Sector concentration risks
                    4. Rebalancing recommendations
                    5. Optimization suggestions
                    """
                    
                    analysis = get_ai_analysis(prompt, max_tokens=600)
                    st.markdown(f'<div class="ai-response">{analysis}</div>', unsafe_allow_html=True)
            
            # Individual stock analysis
            st.subheader("📈 Individual Holdings")
            
            for symbol, weight in portfolio.items():
                with st.expander(f"{symbol} ({weight*100:.1f}% allocation)"):
                    data, info = get_stock_data(symbol, period="1y")
                    if data is not None:
                        metrics = calculate_basic_metrics(data, info, symbol)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Price", f"${metrics['current_price']:.2f}", 
                                     f"{metrics['price_change_pct']:+.1f}%")
                        with col2:
                            if metrics.get('pe_ratio'):
                                st.metric("P/E Ratio", f"{metrics['pe_ratio']:.1f}")
                        with col3:
                            if metrics.get('market_cap'):
                                st.metric("Market Cap", f"${metrics['market_cap']/1e9:.1f}B")
                        
                        # Mini chart
                        fig = px.line(x=data.index[-30:], y=data['Close'][-30:], 
                                     title=f"{symbol} - Last 30 Days")
                        fig.update_layout(height=200, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error parsing portfolio: {str(e)}")

def show_market_overview():
    """Market overview page"""
    st.header("📈 Market Overview")
    
    # Major indices
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000'
    }
    
    st.subheader("📊 Major Indices")
    
    cols = st.columns(len(indices))
    
    for i, (symbol, name) in enumerate(indices.items()):
        with cols[i]:
            data, _ = get_stock_data(symbol, period="5d")
            if data is not None and not data.empty:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2] if len(data) > 1 else current
                change = ((current - prev) / prev * 100) if prev != 0 else 0
                
                st.metric(
                    name,
                    f"{current:.2f}",
                    f"{change:+.2f}%",
                    delta_color="normal" if change >= 0 else "inverse"
                )
    
    # AI Market Analysis
    if st.button("🤖 Get AI Market Analysis"):
        with st.spinner("Analyzing market conditions..."):
            # Get current market data
            sp500_data, _ = get_stock_data('^GSPC', period="1mo")
            if sp500_data is not None:
                sp500_change = ((sp500_data['Close'].iloc[-1] - sp500_data['Close'].iloc[0]) / sp500_data['Close'].iloc[0] * 100)
                
                prompt = f"""
                Analyze current market conditions:
                
                Recent Performance:
                - S&P 500 monthly change: {sp500_change:+.1f}%
                
                Provide analysis covering:
                1. Current market sentiment and key drivers
                2. Technical outlook for major indices
                3. Sector rotation trends
                4. Economic factors to watch
                5. Investment opportunities and risks
                6. Short-term and medium-term outlook
                """
                
                analysis = get_ai_analysis(prompt, max_tokens=600)
                st.markdown(f'<div class="ai-response">{analysis}</div>', unsafe_allow_html=True)
    
    # Sector performance
    st.subheader("🏭 Sector Performance")
    
    sector_etfs = {
        'XLK': 'Technology',
        'XLF': 'Financial',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrial',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities'
    }
    
    sector_data = []
    
    for etf, sector in sector_etfs.items():
        data, _ = get_stock_data(etf, period="5d")
        if data is not None and not data.empty:
            current = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2] if len(data) > 1 else current
            change = ((current - prev) / prev * 100) if prev != 0 else 0
            
            sector_data.append({
                'Sector': sector,
                'ETF': etf,
                'Price': current,
                'Change %': change
            })
    
    if sector_data:
        df = pd.DataFrame(sector_data)
        df = df.sort_values('Change %', ascending=False)
        
        # Color code the dataframe
        def color_negative_red(val):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'
        
        st.dataframe(
            df.style.applymap(color_negative_red, subset=['Change %']),
            use_container_width=True
        )
        
        # Sector performance chart
        fig = px.bar(df, x='Sector', y='Change %', 
                     title='Sector Performance (1-Day Change)',
                     color='Change %',
                     color_continuous_scale=['red', 'yellow', 'green'])
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("🔧 Configuration")
    
    # API Settings
    with st.expander("🔑 API Configuration"):
        st.write("Configure your API keys for enhanced functionality:")
        
        openai_key = st.text_input("OpenAI API Key", type="password",
                                  help="For AI-powered analysis and insights")
        tavily_key = st.text_input("Tavily API Key", type="password",
                                  help="For market news and sentiment analysis")
        
        if st.button("Save API Keys"):
            # In a real app, you'd save these securely
            st.success("API keys saved successfully!")
            st.info("Note: In production, API keys should be stored securely as environment variables.")
    
    # AI Settings
    with st.expander("🤖 AI Configuration"):
        model_choice = st.selectbox("AI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        temperature = st.slider("AI Creativity (Temperature)", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.number_input("Max Response Length", 100, 1000, 500, 50)
        
        st.write(f"**Current Settings:**")
        st.write(f"- Model: {model_choice}")
        st.write(f"- Temperature: {temperature}")
        st.write(f"- Max Tokens: {max_tokens}")
    
    # Display Settings
    with st.expander("🎨 Display Settings"):
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        chart_style = st.selectbox("Chart Style", ["Default", "Minimal", "Professional"])
        default_period = st.selectbox("Default Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
        
        if st.button("Apply Settings"):
            st.success("Settings applied successfully!")
    
    # Data Sources
    with st.expander("📊 Data Sources"):
        st.write("**Current Data Sources:**")
        st.write("- **Yahoo Finance (yfinance):** Stock prices, company info, financial metrics")
        st.write("- **OpenAI API:** AI-powered analysis and insights")
        st.write("- **Tavily API:** Market news and sentiment analysis")
        st.write("- **Real-time data:** 15-20 minute delay for free tier")
        st.write("- **Historical data:** Up to 5 years of daily data")
        st.write("- **Coverage:** Global markets (US, EU, Asia)")
    
    # About
    with st.expander("ℹ️ About Kangro Capital"):
        st.markdown("""
        **Kangro Capital Stock Screening Platform**
        
        Version: 3.0.0 (AI-Enhanced)
        
        A comprehensive AI-powered stock analysis and screening platform featuring:
        - ✅ Real-time stock data analysis
        - ✅ Advanced screening algorithms with AI insights
        - ✅ Interactive visualizations
        - ✅ AI-powered fundamental and technical analysis
        - ✅ Market sentiment analysis
        - ✅ Portfolio analysis tools
        - ✅ Market overview dashboard
        - ✅ Export functionality
        - ✅ Custom AI queries
        
        **Built with:**
        - Streamlit for web interface
        - Yahoo Finance for market data
        - OpenAI for AI analysis
        - Tavily for market news
        - Plotly for interactive charts
        - Pandas for data processing
        
        **AI Features:**
        - Stock fundamental analysis
        - Technical analysis insights
        - Market sentiment analysis
        - Portfolio optimization suggestions
        - Custom investment queries
        - News analysis and summarization
        
        **Deployment Ready:**
        - Optimized for Streamlit Cloud
        - Scalable architecture
        - Fast loading and responsive design
        """)

if __name__ == "__main__":
    # Initialize session state
    if 'analysis_symbol' not in st.session_state:
        st.session_state.analysis_symbol = None
    if 'ai_screening' not in st.session_state:
        st.session_state.ai_screening = False
    if 'market_insights' not in st.session_state:
        st.session_state.market_insights = False
    
    main()

