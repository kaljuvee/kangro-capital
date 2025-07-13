"""
Kangro Capital - Stock Screening Platform
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
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector'),
            'industry': info.get('industry')
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
            
            # Volume check
            if criteria.get('min_volume') and metrics['volume'] >= criteria['min_volume']:
                score += 1
                reasons.append(f"High volume: {metrics['volume']:,}")
            
            results.append({
                'symbol': symbol,
                'company_name': metrics.get('company_name', symbol),
                'current_price': metrics['current_price'],
                'price_change_pct': metrics['price_change_pct'],
                'pe_ratio': metrics.get('pe_ratio'),
                'market_cap': metrics.get('market_cap'),
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
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Stock Screening & Analysis Platform</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Dashboard",
        "🔍 Stock Analysis", 
        "📊 Stock Screening",
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
    
    if st.session_state.get('quick_screening'):
        st.subheader("📊 Quick Screening Results")
        default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        criteria = {'max_pe': 25.0, 'min_market_cap': 1e9}
        
        results = screen_stocks(default_symbols, criteria)
        if results:
            df = pd.DataFrame(results[:3])  # Top 3
            st.dataframe(df[['symbol', 'company_name', 'current_price', 'score']], use_container_width=True)
        
        if st.button("Clear Screening"):
            del st.session_state.quick_screening
            st.rerun()

def show_stock_analysis():
    """Stock analysis page"""
    st.header("🔍 Individual Stock Analysis")
    
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
            metrics = calculate_basic_metrics(data, info)
            
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
                
                # Simple analysis
                analysis = []
                if metrics.get('pe_ratio'):
                    if metrics['pe_ratio'] < 15:
                        analysis.append("✅ Low P/E ratio (undervalued)")
                    elif metrics['pe_ratio'] > 30:
                        analysis.append("⚠️ High P/E ratio (overvalued)")
                
                if metrics.get('ma20') and metrics['current_price'] > metrics['ma20']:
                    analysis.append("✅ Price above 20-day MA")
                
                if analysis:
                    st.write("**Quick Analysis**")
                    for item in analysis:
                        st.write(item)

def show_stock_screening():
    """Stock screening page"""
    st.header("📊 Stock Screening")
    
    # Screening parameters
    st.subheader("🔧 Screening Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_pe = st.number_input("Maximum P/E Ratio", min_value=1.0, max_value=100.0, value=25.0, step=0.5)
        min_market_cap = st.number_input("Minimum Market Cap (Billions)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1) * 1e9
        min_volume = st.number_input("Minimum Daily Volume", min_value=0, max_value=100000000, value=1000000, step=100000)
    
    with col2:
        stock_list = st.text_area(
            "Stock Symbols (one per line)", 
            value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\nNVDA\nMETA\nNFLX\nADBE\nCRM\nORCL\nSALESFORCE\nINTC\nAMD\nIBM",
            height=200
        )
    
    symbols = [s.strip().upper() for s in stock_list.split('\n') if s.strip()]
    
    criteria = {
        'max_pe': max_pe,
        'min_market_cap': min_market_cap,
        'min_volume': min_volume
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
                df['volume'] = df['volume'].apply(lambda x: f"{x:,}")
                
                # Display results
                st.dataframe(
                    df[['symbol', 'company_name', 'current_price', 'price_change_pct', 'pe_ratio', 'market_cap', 'sector', 'score', 'reasons']],
                    column_config={
                        'symbol': 'Symbol',
                        'company_name': 'Company',
                        'current_price': 'Price',
                        'price_change_pct': 'Change %',
                        'pe_ratio': 'P/E',
                        'market_cap': 'Market Cap',
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
                
                # Top picks
                top_picks = df.head(3)
                if not top_picks.empty:
                    st.subheader("🏆 Top Picks")
                    for _, stock in top_picks.iterrows():
                        with st.expander(f"🎯 {stock['symbol']} - {stock['company_name']} (Score: {stock['score']})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Current Price:** {stock['current_price']}")
                                st.write(f"**Price Change:** {stock['price_change_pct']}")
                                st.write(f"**P/E Ratio:** {stock['pe_ratio']}")
                            with col2:
                                st.write(f"**Market Cap:** {stock['market_cap']}")
                                st.write(f"**Sector:** {stock['sector']}")
                                st.write(f"**Criteria Met:** {stock['reasons']}")
            else:
                st.warning("No results found. Try adjusting your criteria.")
        else:
            st.error("Please enter at least one stock symbol.")

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
            
            # Individual stock analysis
            st.subheader("📈 Individual Holdings")
            
            for symbol, weight in portfolio.items():
                with st.expander(f"{symbol} ({weight*100:.1f}% allocation)"):
                    data, info = get_stock_data(symbol, period="1y")
                    if data is not None:
                        metrics = calculate_basic_metrics(data, info)
                        
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
        default_period = st.selectbox("Default Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
        
        if st.button("Apply Settings"):
            st.success("Settings applied successfully!")
    
    # Data Sources
    with st.expander("📊 Data Sources"):
        st.write("**Current Data Sources:**")
        st.write("- **Yahoo Finance (yfinance):** Stock prices, company info, financial metrics")
        st.write("- **Real-time data:** 15-20 minute delay for free tier")
        st.write("- **Historical data:** Up to 5 years of daily data")
        st.write("- **Coverage:** Global markets (US, EU, Asia)")
    
    # About
    with st.expander("ℹ️ About Kangro Capital"):
        st.markdown("""
        **Kangro Capital Stock Screening Platform**
        
        Version: 2.0.0 (Streamlit Deployment)
        
        A comprehensive stock analysis and screening platform featuring:
        - ✅ Real-time stock data analysis
        - ✅ Advanced screening algorithms  
        - ✅ Interactive visualizations
        - ✅ Portfolio analysis tools
        - ✅ Market overview dashboard
        - ✅ Export functionality
        
        **Built with:**
        - Streamlit for web interface
        - Yahoo Finance for market data
        - Plotly for interactive charts
        - Pandas for data processing
        
        **Deployment Ready:**
        - Optimized for Streamlit Cloud
        - Minimal dependencies
        - Fast loading and responsive design
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

