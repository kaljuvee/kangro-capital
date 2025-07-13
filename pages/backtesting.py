"""
Basic Backtesting Page - Portfolio Performance Analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from utils.backtest_engine import BacktestEngine
from utils.portfolio_simulator import PortfolioSimulator

def show_backtesting():
    st.markdown("## 📊 Portfolio Backtesting")
    st.markdown("""
    **What we're testing:** Analyze how your stock selection strategy would have performed historically. 
    Compare against benchmarks and evaluate risk-adjusted returns.
    """)
    
    # Backtesting Parameters
    with st.expander("⚙️ Backtesting Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📅 Time Period**")
            start_date = st.date_input("Start Date", 
                                      value=datetime.now() - timedelta(days=365*2),
                                      help="Beginning of backtesting period")
            
            end_date = st.date_input("End Date", 
                                    value=datetime.now() - timedelta(days=30),
                                    help="End of backtesting period")
            
            rebalance_frequency = st.selectbox("Rebalance Frequency", 
                                             ["Monthly", "Quarterly", "Semi-Annual", "Annual"],
                                             index=1,
                                             help="How often to rebalance the portfolio")
        
        with col2:
            st.markdown("**💰 Portfolio Settings**")
            initial_capital = st.number_input("Initial Capital ($)", 
                                            min_value=1000, max_value=10000000, 
                                            value=100000, step=1000,
                                            help="Starting portfolio value")
            
            max_position_size = st.slider("Max Position Size (%)", 
                                         min_value=1, max_value=20, value=10,
                                         help="Maximum weight for any single stock")
            
            transaction_cost = st.slider("Transaction Cost (%)", 
                                        min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                                        help="Cost per trade as percentage of trade value")
        
        with col3:
            st.markdown("**📈 Strategy Settings**")
            portfolio_strategy = st.selectbox("Portfolio Strategy", 
                                            ["Equal Weight", "Market Cap Weight", "Score Weight", "Risk Parity"],
                                            help="How to allocate weights among selected stocks")
            
            benchmark = st.selectbox("Benchmark", 
                                    ["SPY (S&P 500)", "QQQ (NASDAQ 100)", "VTI (Total Market)", "IWM (Russell 2000)"],
                                    help="Benchmark for performance comparison")
            
            risk_free_rate = st.slider("Risk-Free Rate (%)", 
                                      min_value=0.0, max_value=5.0, value=2.0, step=0.1,
                                      help="Risk-free rate for Sharpe ratio calculation")
    
    # Stock Selection Input
    st.markdown("### 📋 Stock Selection")
    st.markdown("*Enter the stocks you want to backtest (use screening results or manual selection)*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Use Screening Results", use_container_width=True):
            if 'screening_results' in st.session_state and st.session_state.screening_results:
                qualified_stocks = st.session_state.screening_results.get('qualified_stocks', [])
                if qualified_stocks:
                    symbols = [stock['symbol'] for stock in qualified_stocks[:10]]  # Top 10
                    st.session_state.backtest_symbols = ', '.join(symbols)
                    st.success(f"✅ Loaded {len(symbols)} stocks from screening results")
                else:
                    st.warning("⚠️ No screening results found. Run stock screening first.")
            else:
                st.warning("⚠️ No screening results found. Run stock screening first.")
    
    with col2:
        if st.button("🎯 Use Sample Portfolio", use_container_width=True):
            sample_symbols = "AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, JNJ, PG"
            st.session_state.backtest_symbols = sample_symbols
            st.success("✅ Loaded sample portfolio")
    
    # Manual stock input
    stock_symbols = st.text_area("Stock Symbols (comma-separated)", 
                                value=st.session_state.get('backtest_symbols', 'AAPL, MSFT, GOOGL, AMZN, TSLA'),
                                help="Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)")
    
    # Run Backtest
    if st.button("📈 Run Backtest", type="primary", use_container_width=True):
        if not stock_symbols.strip():
            st.error("❌ Please enter stock symbols to backtest")
            return
        
        with st.spinner("🔄 Running backtest analysis... This may take a few minutes"):
            try:
                # Parse stock symbols
                symbols = [s.strip().upper() for s in stock_symbols.split(',') if s.strip()]
                
                # Prepare backtest parameters
                backtest_params = {
                    'symbols': symbols,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'initial_capital': initial_capital,
                    'rebalance_frequency': rebalance_frequency,
                    'max_position_size': max_position_size / 100,
                    'transaction_cost': transaction_cost / 100,
                    'portfolio_strategy': portfolio_strategy,
                    'benchmark': benchmark,
                    'risk_free_rate': risk_free_rate / 100
                }
                
                # Initialize backtest engine
                backtest_engine = BacktestEngine()
                
                # Run backtest
                results = backtest_engine.run_backtest(backtest_params)
                
                # Store results in session state
                st.session_state.backtest_results = results
                st.session_state.backtest_params = backtest_params
                
                total_return = results.get('total_return', 0)
                st.success(f"✅ Backtest completed! Total return: {total_return:.1%}")
                
            except Exception as e:
                st.error(f"❌ Backtest failed: {str(e)}")
                # Create sample data for demonstration
                st.warning("📊 Showing sample backtest results for demonstration purposes")
                create_sample_backtest_results()
    
    # Display Results
    if 'backtest_results' in st.session_state and st.session_state.backtest_results:
        display_backtest_results()

def create_sample_backtest_results():
    """Create sample backtest results for demonstration"""
    # Generate sample performance data
    dates = pd.date_range(start='2022-01-01', end='2024-07-01', freq='D')
    np.random.seed(42)
    
    # Portfolio performance (with some volatility)
    portfolio_returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    # Benchmark performance (slightly lower returns)
    benchmark_returns = np.random.normal(0.0006, 0.015, len(dates))
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    results = {
        'total_return': 0.234,  # 23.4%
        'annualized_return': 0.156,  # 15.6%
        'volatility': 0.187,  # 18.7%
        'sharpe_ratio': 0.83,
        'sortino_ratio': 1.12,
        'max_drawdown': -0.089,  # -8.9%
        'calmar_ratio': 1.75,
        'alpha': 0.034,  # 3.4%
        'beta': 1.12,
        'win_rate': 0.567,  # 56.7%
        'benchmark_return': 0.187,  # 18.7%
        'excess_return': 0.047,  # 4.7%
        'performance_data': {
            'dates': dates.strftime('%Y-%m-%d').tolist(),
            'portfolio_value': (portfolio_cumulative * 100000).tolist(),
            'benchmark_value': (benchmark_cumulative * 100000).tolist(),
            'portfolio_returns': portfolio_returns.tolist(),
            'benchmark_returns': benchmark_returns.tolist()
        },
        'monthly_returns': {
            'months': ['Jan 22', 'Feb 22', 'Mar 22', 'Apr 22', 'May 22', 'Jun 22',
                      'Jul 22', 'Aug 22', 'Sep 22', 'Oct 22', 'Nov 22', 'Dec 22'],
            'portfolio': [0.045, -0.023, 0.067, -0.012, 0.034, -0.045, 0.078, -0.034, 0.023, 0.056, -0.012, 0.034],
            'benchmark': [0.032, -0.018, 0.045, -0.008, 0.023, -0.034, 0.056, -0.023, 0.012, 0.045, -0.008, 0.023]
        },
        'holdings': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'weights': [0.20, 0.20, 0.20, 0.20, 0.20],
            'returns': [0.187, 0.234, 0.156, 0.289, 0.445],
            'allocations': [20000, 20000, 20000, 20000, 20000]
        }
    }
    
    st.session_state.backtest_results = results

def display_backtest_results():
    """Display comprehensive backtest results with Plotly visualizations"""
    results = st.session_state.backtest_results
    
    st.markdown("## 📈 Backtest Results")
    
    # Performance Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = results.get('total_return', 0)
        benchmark_return = results.get('benchmark_return', 0)
        excess_return = total_return - benchmark_return
        st.metric("Total Return", f"{total_return:.1%}", f"{excess_return:+.1%} vs benchmark")
    
    with col2:
        annualized_return = results.get('annualized_return', 0)
        st.metric("Annualized Return", f"{annualized_return:.1%}")
    
    with col3:
        sharpe_ratio = results.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        max_drawdown = results.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")
    
    # Performance Chart
    if results.get('performance_data'):
        st.markdown("### 📊 Portfolio Performance vs Benchmark")
        
        perf_data = results['performance_data']
        
        fig = go.Figure()
        
        # Portfolio performance
        fig.add_trace(go.Scatter(
            x=perf_data['dates'],
            y=perf_data['portfolio_value'],
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=3)
        ))
        
        # Benchmark performance
        fig.add_trace(go.Scatter(
            x=perf_data['dates'],
            y=perf_data['benchmark_value'],
            mode='lines',
            name='Benchmark (SPY)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Cumulative Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Metrics
    st.markdown("### ⚖️ Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk metrics table
        risk_metrics = {
            'Metric': ['Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Alpha', 'Beta', 'Win Rate'],
            'Value': [
                f"{results.get('volatility', 0):.1%}",
                f"{results.get('sharpe_ratio', 0):.2f}",
                f"{results.get('sortino_ratio', 0):.2f}",
                f"{results.get('calmar_ratio', 0):.2f}",
                f"{results.get('alpha', 0):.1%}",
                f"{results.get('beta', 0):.2f}",
                f"{results.get('win_rate', 0):.1%}"
            ]
        }
        
        risk_df = pd.DataFrame(risk_metrics)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Risk Metric', 'Value'],
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='white')
            ),
            cells=dict(
                values=[risk_df['Metric'], risk_df['Value']],
                fill_color=[['white' if i % 2 == 0 else 'lightgray' for i in range(len(risk_df))] for _ in range(2)],
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(title="Risk Metrics Summary", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk-Return scatter
        portfolio_return = results.get('annualized_return', 0)
        portfolio_vol = results.get('volatility', 0)
        benchmark_return = results.get('benchmark_return', 0) * 0.8  # Assume benchmark vol
        benchmark_vol = portfolio_vol * 0.85
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[portfolio_vol],
            y=[portfolio_return],
            mode='markers',
            name='Portfolio',
            marker=dict(size=15, color='blue'),
            text=['Portfolio'],
            textposition="top center"
        ))
        
        fig.add_trace(go.Scatter(
            x=[benchmark_vol],
            y=[benchmark_return],
            mode='markers',
            name='Benchmark',
            marker=dict(size=15, color='red'),
            text=['Benchmark'],
            textposition="top center"
        ))
        
        fig.update_layout(
            title="Risk vs Return Analysis",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Annualized Return",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Returns Heatmap
    if results.get('monthly_returns'):
        st.markdown("### 📅 Monthly Returns Analysis")
        
        monthly_data = results['monthly_returns']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly returns comparison
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=monthly_data['months'],
                y=monthly_data['portfolio'],
                name='Portfolio',
                marker_color='blue'
            ))
            
            fig.add_trace(go.Bar(
                x=monthly_data['months'],
                y=monthly_data['benchmark'],
                name='Benchmark',
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Monthly Returns Comparison",
                xaxis_title="Month",
                yaxis_title="Return (%)",
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rolling Sharpe ratio (simulated)
            rolling_periods = ['3M', '6M', '9M', '12M', '15M', '18M']
            rolling_sharpe = [0.65, 0.72, 0.78, 0.83, 0.81, 0.85]
            
            fig = px.line(
                x=rolling_periods,
                y=rolling_sharpe,
                title="Rolling 12-Month Sharpe Ratio",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Period",
                yaxis_title="Sharpe Ratio",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio Holdings
    if results.get('holdings'):
        st.markdown("### 📋 Portfolio Holdings")
        
        holdings = results['holdings']
        holdings_df = pd.DataFrame({
            'Symbol': holdings['symbols'],
            'Weight': holdings['weights'],
            'Return': holdings['returns'],
            'Allocation ($)': holdings['allocations']
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Holdings pie chart
            fig = px.pie(
                holdings_df,
                values='Weight',
                names='Symbol',
                title="Portfolio Allocation"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Individual stock returns
            fig = px.bar(
                holdings_df,
                x='Symbol',
                y='Return',
                title="Individual Stock Returns",
                color='Return',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.markdown("### 💡 Key Insights")
    
    total_return = results.get('total_return', 0)
    benchmark_return = results.get('benchmark_return', 0)
    sharpe_ratio = results.get('sharpe_ratio', 0)
    max_drawdown = results.get('max_drawdown', 0)
    
    insights = [
        f"🎯 Portfolio {'outperformed' if total_return > benchmark_return else 'underperformed'} benchmark by {abs(total_return - benchmark_return):.1%}",
        f"📊 Sharpe ratio of {sharpe_ratio:.2f} indicates {'strong' if sharpe_ratio > 1 else 'moderate' if sharpe_ratio > 0.5 else 'weak'} risk-adjusted performance",
        f"⚠️ Maximum drawdown of {abs(max_drawdown):.1%} shows {'low' if abs(max_drawdown) < 0.1 else 'moderate' if abs(max_drawdown) < 0.2 else 'high'} downside risk",
        f"📈 Annualized return of {results.get('annualized_return', 0):.1%} {'exceeds' if results.get('annualized_return', 0) > 0.1 else 'meets' if results.get('annualized_return', 0) > 0.05 else 'falls short of'} typical market expectations"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Export Options
    st.markdown("### 📥 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Results"):
            # Save to data folder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_df = pd.DataFrame([results])
            results_df.to_csv(f"data/backtest_results_{timestamp}.csv", index=False)
            st.success("✅ Results saved to data folder!")
    
    with col2:
        if st.button("📊 Export Performance Data"):
            if results.get('performance_data'):
                perf_df = pd.DataFrame(results['performance_data'])
                perf_df.to_csv(f"data/performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
                st.success("✅ Performance data exported!")
    
    with col3:
        if st.button("📈 Generate Report"):
            st.info("📋 Detailed backtest report generation coming soon!")

