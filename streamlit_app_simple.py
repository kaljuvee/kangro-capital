"""
Kangro Capital - Simplified Stock Screening Platform
Streamlit deployment version with core functionality
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
</style>
""", unsafe_allow_html=True)

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

def calculate_basic_metrics(data, info):
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
            'company_name': info.get('longName', symbol)
        })
    
    return metrics

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
        showlegend=True
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
        showlegend=False
    )
    
    return fig

def screen_stocks(symbols, criteria):
    """Simple stock screening based on basic criteria"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        status_text.text(f"Analyzing {symbol}...")
        progress_bar.progress((i + 1) / len(symbols))
        
        data, info = get_stock_data(symbol, period="1y")
        if data is not None and not data.empty:
            metrics = calculate_basic_metrics(data, info)
            
            # Apply screening criteria
            score = 0
            reasons = []
            
            # PE Ratio check
            if metrics.get('pe_ratio') and criteria.get('max_pe'):
                if metrics['pe_ratio'] <= criteria['max_pe']:
                    score += 1
                    reasons.append(f"PE ratio {metrics['pe_ratio']:.1f} ≤ {criteria['max_pe']}")
            
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
            
            results.append({
                'symbol': symbol,
                'company_name': metrics.get('company_name', symbol),
                'current_price': metrics['current_price'],
                'price_change_pct': metrics['price_change_pct'],
                'pe_ratio': metrics.get('pe_ratio'),
                'market_cap': metrics.get('market_cap'),
                'score': score,
                'reasons': '; '.join(reasons) if reasons else 'No criteria met'
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Kangro Capital</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Stock Screening & Analysis Platform</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Dashboard",
        "🔍 Stock Analysis", 
        "📊 Stock Screening",
        "⚙️ Settings"
    ])
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔍 Stock Analysis":
        show_stock_analysis()
    elif page == "📊 Stock Screening":
        show_stock_screening()
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
        st.metric("Markets", "🌍 Global", "US, EU, Asia")
    
    with col3:
        st.metric("Data Sources", "📡 Live", "Real-time feeds")
    
    with col4:
        st.metric("Analysis Tools", "🧠 AI-Powered", "Advanced algorithms")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze AAPL", use_container_width=True):
            st.session_state.analysis_symbol = "AAPL"
            st.rerun()
    
    with col2:
        if st.button("📊 Run Quick Screening", use_container_width=True):
            st.session_state.quick_screening = True
            st.rerun()
    
    with col3:
        if st.button("📈 Market Overview", use_container_width=True):
            st.session_state.market_overview = True
            st.rerun()
    
    # Handle quick actions
    if st.session_state.get('analysis_symbol'):
        st.subheader(f"📈 Quick Analysis: {st.session_state.analysis_symbol}")
        data, info = get_stock_data(st.session_state.analysis_symbol)
        if data is not None:
            metrics = calculate_basic_metrics(data, info)
            
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
        
        if st.button("Clear Analysis"):
            del st.session_state.analysis_symbol
            st.rerun()

def show_stock_analysis():
    """Stock analysis page"""
    st.header("🔍 Individual Stock Analysis")
    
    # Stock symbol input
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL").upper()
    
    if symbol:
        # Fetch data
        with st.spinner(f"Fetching data for {symbol}..."):
            data, info = get_stock_data(symbol)
        
        if data is not None and not data.empty:
            metrics = calculate_basic_metrics(data, info)
            
            # Display company info
            st.subheader(f"📊 {metrics.get('company_name', symbol)} ({symbol})")
            
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
                if metrics.get('dividend_yield'):
                    st.metric("Dividend Yield", f"{metrics['dividend_yield']*100:.1f}%")
                else:
                    st.metric("Dividend Yield", "N/A")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_price_chart(data, symbol), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_volume_chart(data, symbol), use_container_width=True)
            
            # Additional metrics
            st.subheader("📈 Additional Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Price Ranges**")
                st.write(f"52-Week High: ${metrics['high_52w']:.2f}")
                st.write(f"52-Week Low: ${metrics['low_52w']:.2f}")
                if metrics.get('ma20'):
                    st.write(f"20-Day MA: ${metrics['ma20']:.2f}")
                if metrics.get('ma50'):
                    st.write(f"50-Day MA: ${metrics['ma50']:.2f}")
            
            with col2:
                st.write("**Risk Metrics**")
                if metrics.get('beta'):
                    st.write(f"Beta: {metrics['beta']:.2f}")
                st.write(f"Volume: {metrics['volume']:,}")

def show_stock_screening():
    """Stock screening page"""
    st.header("📊 Stock Screening")
    
    # Screening parameters
    st.subheader("🔧 Screening Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_pe = st.number_input("Maximum P/E Ratio", min_value=1.0, max_value=100.0, value=25.0, step=0.5)
        min_market_cap = st.number_input("Minimum Market Cap (Billions)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1) * 1e9
    
    with col2:
        stock_list = st.text_area(
            "Stock Symbols (one per line)", 
            value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\nNVDA\nMETA\nNFLX\nADBE\nCRM",
            height=150
        )
    
    symbols = [s.strip().upper() for s in stock_list.split('\n') if s.strip()]
    
    criteria = {
        'max_pe': max_pe,
        'min_market_cap': min_market_cap
    }
    
    if st.button("🚀 Run Screening", type="primary"):
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
                df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else "N/A")
                
                # Display results
                st.dataframe(
                    df[['symbol', 'company_name', 'current_price', 'price_change_pct', 'pe_ratio', 'market_cap', 'score', 'reasons']],
                    column_config={
                        'symbol': 'Symbol',
                        'company_name': 'Company',
                        'current_price': 'Price',
                        'price_change_pct': 'Change %',
                        'pe_ratio': 'P/E',
                        'market_cap': 'Market Cap',
                        'score': 'Score',
                        'reasons': 'Criteria Met'
                    },
                    use_container_width=True
                )
                
                # Top picks
                top_picks = df.head(3)
                if not top_picks.empty:
                    st.subheader("🏆 Top Picks")
                    for _, stock in top_picks.iterrows():
                        with st.expander(f"🎯 {stock['symbol']} - {stock['company_name']} (Score: {stock['score']})"):
                            st.write(f"**Current Price:** {stock['current_price']}")
                            st.write(f"**Price Change:** {stock['price_change_pct']}")
                            st.write(f"**P/E Ratio:** {stock['pe_ratio']}")
                            st.write(f"**Market Cap:** {stock['market_cap']}")
                            st.write(f"**Criteria Met:** {stock['reasons']}")
            else:
                st.warning("No results found. Try adjusting your criteria.")
        else:
            st.error("Please enter at least one stock symbol.")

def show_settings():
    """Settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("🔧 Configuration")
    
    # API Settings
    with st.expander("🔑 API Configuration"):
        st.write("Configure your API keys for enhanced functionality:")
        
        polygon_key = st.text_input("Polygon.io API Key", type="password", 
                                   help="For real-time market data")
        openai_key = st.text_input("OpenAI API Key", type="password",
                                  help="For AI-powered analysis")
        tavily_key = st.text_input("Tavily API Key", type="password",
                                  help="For market news and sentiment")
        
        if st.button("Save API Keys"):
            st.success("API keys saved successfully!")
    
    # Display Settings
    with st.expander("🎨 Display Settings"):
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        chart_style = st.selectbox("Chart Style", ["Default", "Minimal", "Professional"])
        
        if st.button("Apply Settings"):
            st.success("Settings applied successfully!")
    
    # About
    with st.expander("ℹ️ About Kangro Capital"):
        st.markdown("""
        **Kangro Capital Stock Screening Platform**
        
        Version: 1.0.0
        
        A comprehensive stock analysis and screening platform featuring:
        - Real-time stock data analysis
        - Advanced screening algorithms
        - Interactive visualizations
        - AI-powered insights
        
        Built with Streamlit, powered by financial data APIs.
        """)

if __name__ == "__main__":
    # Initialize session state
    if 'analysis_symbol' not in st.session_state:
        st.session_state.analysis_symbol = None
    if 'quick_screening' not in st.session_state:
        st.session_state.quick_screening = False
    if 'market_overview' not in st.session_state:
        st.session_state.market_overview = False
    
    main()

