"""
Stock Screening Page - Enhanced with Realistic Parameters and Better Explanations
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from utils.screening_engine import ScreeningEngine
from utils.data_fetcher import DataFetcher

def show_stock_screening():
    st.markdown("## 🔍 Stock Screening Engine")
    st.markdown("""
    **What we're screening for:** High-quality companies with strong fundamentals, sustainable growth, 
    and reasonable valuations. Our 7-step process identifies stocks that have historically outperformed the market.
    """)
    
    # Screening Parameters with realistic defaults
    with st.expander("⚙️ Screening Parameters", expanded=True):
        st.markdown("### 📊 Fundamental Analysis Criteria")
        st.markdown("*These parameters are based on proven value investing principles*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📈 Profitability Metrics**")
            min_roe = st.slider("Min ROE (%)", 
                               min_value=5.0, max_value=30.0, value=12.0, step=0.5,
                               help="Return on Equity - measures how efficiently a company uses shareholders' equity. 12%+ indicates strong management.")
            
            min_gross_margin = st.slider("Min Gross Margin (%)", 
                                        min_value=15.0, max_value=80.0, value=25.0, step=1.0,
                                        help="Gross profit margin - indicates pricing power and operational efficiency. 25%+ shows competitive advantage.")
            
            min_net_margin = st.slider("Min Net Margin (%)", 
                                      min_value=2.0, max_value=25.0, value=5.0, step=0.5,
                                      help="Net profit margin - shows bottom-line profitability. 5%+ indicates efficient operations.")
        
        with col2:
            st.markdown("**💰 Financial Health**")
            min_current_ratio = st.slider("Min Current Ratio", 
                                         min_value=0.8, max_value=3.0, value=1.2, step=0.1,
                                         help="Current assets / Current liabilities. 1.2+ ensures company can pay short-term debts.")
            
            max_debt_to_equity = st.slider("Max Debt-to-Equity", 
                                          min_value=0.1, max_value=2.0, value=0.6, step=0.1,
                                          help="Total debt / Total equity. <0.6 indicates conservative debt management.")
            
            min_interest_coverage = st.slider("Min Interest Coverage", 
                                             min_value=2.0, max_value=20.0, value=5.0, step=0.5,
                                             help="EBIT / Interest expense. 5x+ shows ability to service debt comfortably.")
        
        with col3:
            st.markdown("**📊 Growth & Valuation**")
            min_revenue_growth = st.slider("Min Revenue Growth (%)", 
                                          min_value=0.0, max_value=30.0, value=5.0, step=1.0,
                                          help="Annual revenue growth rate. 5%+ indicates expanding business.")
            
            max_pe_ratio = st.slider("Max P/E Ratio", 
                                    min_value=5.0, max_value=50.0, value=25.0, step=1.0,
                                    help="Price-to-Earnings ratio. <25 suggests reasonable valuation.")
            
            min_dividend_yield = st.slider("Min Dividend Yield (%)", 
                                          min_value=0.0, max_value=8.0, value=1.0, step=0.5,
                                          help="Annual dividend / Stock price. 1%+ provides income component.")
        
        # Additional parameters
        col1, col2 = st.columns(2)
        with col1:
            top_n_stocks = st.number_input("Top N Stocks to Return", 
                                          min_value=5, max_value=50, value=15,
                                          help="Number of best-scoring stocks to display")
            
            lookback_years = st.selectbox("Analysis Period", 
                                         options=[1, 2, 3, 5], 
                                         index=1,
                                         help="Years of historical data to analyze")
        
        with col2:
            market_cap_filter = st.selectbox("Market Cap Filter", 
                                           options=["All", "Large Cap (>$10B)", "Mid Cap ($2B-$10B)", "Small Cap (<$2B)"],
                                           index=1,
                                           help="Filter by company size")
            
            sector_filter = st.multiselect("Sector Filter (Optional)", 
                                         options=["Technology", "Healthcare", "Finance", "Consumer Discretionary", 
                                                "Consumer Staples", "Energy", "Industrials", "Materials", "Utilities"],
                                         help="Leave empty to screen all sectors")
    
    # Run Screening Button
    if st.button("🔍 Run Comprehensive Screening", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing thousands of stocks... This may take a few minutes"):
            try:
                # Prepare screening parameters
                screening_params = {
                    'min_roe': min_roe,
                    'min_current_ratio': min_current_ratio,
                    'min_gross_margin': min_gross_margin,
                    'min_net_margin': min_net_margin,
                    'min_revenue_growth': min_revenue_growth,
                    'max_debt_to_equity': max_debt_to_equity,
                    'min_interest_coverage': min_interest_coverage,
                    'max_pe_ratio': max_pe_ratio,
                    'min_dividend_yield': min_dividend_yield,
                    'top_n_stocks': top_n_stocks,
                    'lookback_years': lookback_years,
                    'market_cap_filter': market_cap_filter,
                    'sector_filter': sector_filter
                }
                
                # Initialize screening engine
                screening_engine = ScreeningEngine()
                
                # Run screening
                results = screening_engine.run_comprehensive_screening(screening_params)
                
                # Store results in session state
                st.session_state.screening_results = results
                st.session_state.screening_params = screening_params
                
                st.success(f"✅ Screening completed! Found {len(results.get('qualified_stocks', []))} qualifying stocks")
                
            except Exception as e:
                st.error(f"❌ Screening failed: {str(e)}")
                # Create sample data for demonstration
                st.warning("📊 Showing sample data for demonstration purposes")
                create_sample_screening_results()
    
    # Display Results
    if 'screening_results' in st.session_state and st.session_state.screening_results:
        display_screening_results()

def create_sample_screening_results():
    """Create sample screening results for demonstration"""
    sample_stocks = [
        {"symbol": "AAPL", "company": "Apple Inc.", "sector": "Technology", "market_cap": 2800, 
         "roe": 26.4, "current_ratio": 1.1, "gross_margin": 38.3, "net_margin": 25.3, 
         "revenue_growth": 8.2, "debt_to_equity": 0.31, "pe_ratio": 28.9, "dividend_yield": 0.5, "composite_score": 8.7},
        {"symbol": "MSFT", "company": "Microsoft Corp.", "sector": "Technology", "market_cap": 2400,
         "roe": 36.1, "current_ratio": 1.9, "gross_margin": 68.4, "net_margin": 36.7,
         "revenue_growth": 12.1, "debt_to_equity": 0.35, "pe_ratio": 32.1, "dividend_yield": 0.7, "composite_score": 9.2},
        {"symbol": "JNJ", "company": "Johnson & Johnson", "sector": "Healthcare", "market_cap": 450,
         "roe": 23.8, "current_ratio": 1.3, "gross_margin": 66.8, "net_margin": 20.4,
         "revenue_growth": 6.3, "debt_to_equity": 0.42, "pe_ratio": 15.7, "dividend_yield": 2.9, "composite_score": 8.9},
        {"symbol": "PG", "company": "Procter & Gamble", "sector": "Consumer Staples", "market_cap": 380,
         "roe": 24.6, "current_ratio": 0.9, "gross_margin": 49.2, "net_margin": 19.5,
         "revenue_growth": 4.8, "debt_to_equity": 0.54, "pe_ratio": 24.3, "dividend_yield": 2.4, "composite_score": 8.1},
        {"symbol": "HD", "company": "Home Depot Inc.", "sector": "Consumer Discretionary", "market_cap": 320,
         "roe": 45.2, "current_ratio": 1.1, "gross_margin": 33.5, "net_margin": 10.9,
         "revenue_growth": 8.9, "debt_to_equity": 0.89, "pe_ratio": 19.8, "dividend_yield": 2.7, "composite_score": 8.5}
    ]
    
    results = {
        'qualified_stocks': sample_stocks,
        'total_screened': 3247,
        'outliers_detected': 12,
        'avg_composite_score': 8.5,
        'screening_summary': {
            'passed_roe': 89,
            'passed_current_ratio': 156,
            'passed_margins': 234,
            'passed_growth': 178,
            'passed_all_criteria': len(sample_stocks)
        }
    }
    
    st.session_state.screening_results = results

def display_screening_results():
    """Display comprehensive screening results with Plotly visualizations"""
    results = st.session_state.screening_results
    
    st.markdown("## 📊 Screening Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stocks Screened", f"{results.get('total_screened', 0):,}")
    
    with col2:
        st.metric("Qualifying Stocks", len(results.get('qualified_stocks', [])))
    
    with col3:
        st.metric("Outliers Detected", results.get('outliers_detected', 0))
    
    with col4:
        avg_score = results.get('avg_composite_score', 0)
        st.metric("Avg Composite Score", f"{avg_score:.1f}/10")
    
    # Qualified stocks table
    if results.get('qualified_stocks'):
        st.markdown("### 🏆 Top Qualifying Stocks")
        
        df = pd.DataFrame(results['qualified_stocks'])
        
        # Create interactive table with color coding
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Symbol', 'Company', 'Sector', 'ROE (%)', 'Current Ratio', 
                       'Gross Margin (%)', 'Net Margin (%)', 'Revenue Growth (%)', 'P/E Ratio', 'Score'],
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='white')
            ),
            cells=dict(
                values=[df['symbol'], df['company'], df['sector'], 
                       df['roe'].round(1), df['current_ratio'].round(2),
                       df['gross_margin'].round(1), df['net_margin'].round(1),
                       df['revenue_growth'].round(1), df['pe_ratio'].round(1),
                       df['composite_score'].round(1)],
                fill_color=[['white' if i % 2 == 0 else 'lightgray' for i in range(len(df))] for _ in range(10)],
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(title="Qualified Stocks Summary", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Composite score distribution
            fig = px.bar(
                df.sort_values('composite_score', ascending=True),
                x='composite_score',
                y='symbol',
                orientation='h',
                title="Composite Scores by Stock",
                color='composite_score',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sector distribution
            sector_counts = df['sector'].value_counts()
            fig = px.pie(
                values=sector_counts.values,
                names=sector_counts.index,
                title="Qualifying Stocks by Sector"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Financial metrics comparison
        st.markdown("### 📈 Financial Metrics Comparison")
        
        # Create subplot with multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROE vs P/E Ratio', 'Gross Margin vs Net Margin', 
                          'Revenue Growth vs Debt-to-Equity', 'Current Ratio vs Dividend Yield'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROE vs P/E
        fig.add_trace(
            go.Scatter(x=df['roe'], y=df['pe_ratio'], mode='markers+text',
                      text=df['symbol'], textposition="top center",
                      marker=dict(size=df['composite_score']*3, color=df['composite_score'], 
                                colorscale='RdYlGn', showscale=True),
                      name='ROE vs P/E'),
            row=1, col=1
        )
        
        # Gross vs Net Margin
        fig.add_trace(
            go.Scatter(x=df['gross_margin'], y=df['net_margin'], mode='markers+text',
                      text=df['symbol'], textposition="top center",
                      marker=dict(size=10, color='blue'),
                      name='Margins'),
            row=1, col=2
        )
        
        # Revenue Growth vs Debt-to-Equity
        fig.add_trace(
            go.Scatter(x=df['revenue_growth'], y=df['debt_to_equity'], mode='markers+text',
                      text=df['symbol'], textposition="top center",
                      marker=dict(size=10, color='green'),
                      name='Growth vs Debt'),
            row=2, col=1
        )
        
        # Current Ratio vs Dividend Yield
        fig.add_trace(
            go.Scatter(x=df['current_ratio'], y=df['dividend_yield'], mode='markers+text',
                      text=df['symbol'], textposition="top center",
                      marker=dict(size=10, color='red'),
                      name='Liquidity vs Yield'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("💾 Save to Data Folder"):
                df.to_csv(f"data/screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
                st.success("✅ Results saved to data folder!")
        
        with col3:
            if st.button("📊 Generate Report"):
                st.info("📋 Detailed PDF report generation coming soon!")
    
    # Screening funnel analysis
    if results.get('screening_summary'):
        st.markdown("### 🔍 Screening Funnel Analysis")
        
        summary = results['screening_summary']
        funnel_data = {
            'Stage': ['Total Stocks', 'ROE ≥ Threshold', 'Current Ratio ≥ Threshold', 
                     'Margin Requirements', 'Growth Requirements', 'All Criteria'],
            'Count': [results.get('total_screened', 0), summary.get('passed_roe', 0),
                     summary.get('passed_current_ratio', 0), summary.get('passed_margins', 0),
                     summary.get('passed_growth', 0), summary.get('passed_all_criteria', 0)]
        }
        
        funnel_df = pd.DataFrame(funnel_data)
        
        fig = go.Figure(go.Funnel(
            y=funnel_df['Stage'],
            x=funnel_df['Count'],
            textinfo="value+percent initial",
            marker=dict(color=["deepskyblue", "lightsalmon", "tan", "teal", "silver", "gold"])
        ))
        
        fig.update_layout(title="Stock Screening Funnel", height=400)
        st.plotly_chart(fig, use_container_width=True)

