"""
Kangro Capital - Stock Screening and Backtesting Platform
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import os

# Import our utility modules
from utils import (
    PolygonClient, TavilySearchClient, DataFetcher,
    StockScreener, OutlierDetector, ScreeningEngine,
    MLAnalyzer, FactorAnalyzer, BacktestEngine, PortfolioSimulator,
    create_kangro_agent, analyze_screening_results_with_ai, 
    generate_investment_strategy, get_market_insights
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        font-size: 2.5rem;
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
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = {}
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = {}
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}
if 'price_data' not in st.session_state:
    st.session_state.price_data = {}

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components (cached for performance)"""
    return {
        'polygon_client': PolygonClient(),
        'tavily_client': TavilySearchClient(),
        'data_fetcher': DataFetcher(),
        'screening_engine': ScreeningEngine(),
        'ml_analyzer': MLAnalyzer(),
        'factor_analyzer': FactorAnalyzer(),
        'backtest_engine': BacktestEngine(),
        'portfolio_simulator': PortfolioSimulator(),
        'kangro_agent': create_kangro_agent()
    }

components = initialize_components()

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Kangro Capital</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Stock Screening & Backtesting Platform</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🔍 Stock Screening", "🧠 ML Analysis", "📊 Backtesting", "🤖 AI Insights", "⚙️ Settings"]
    )
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔍 Stock Screening":
        show_screening_page()
    elif page == "🧠 ML Analysis":
        show_ml_analysis_page()
    elif page == "📊 Backtesting":
        show_backtesting_page()
    elif page == "🤖 AI Insights":
        show_ai_insights_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_dashboard():
    """Show the main dashboard"""
    st.header("Dashboard Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        screening_count = len(st.session_state.screening_results.get('recommendations', []))
        st.metric("Screened Stocks", screening_count)
    
    with col2:
        has_ml = bool(st.session_state.ml_results)
        st.metric("ML Models Trained", "Yes" if has_ml else "No")
    
    with col3:
        has_backtest = bool(st.session_state.backtest_results)
        st.metric("Backtests Run", "Yes" if has_backtest else "No")
    
    with col4:
        last_update = st.session_state.get('last_update', 'Never')
        st.metric("Last Update", last_update)
    
    st.divider()
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Run Quick Screening", use_container_width=True):
            run_quick_screening()
    
    with col2:
        if st.button("🧠 Run ML Analysis", use_container_width=True, disabled=not bool(st.session_state.screening_results)):
            run_quick_ml_analysis()
    
    with col3:
        if st.button("📊 Run Backtest", use_container_width=True, disabled=not bool(st.session_state.screening_results)):
            run_quick_backtest()
    
    # Recent results summary
    if st.session_state.screening_results:
        st.subheader("Recent Screening Results")
        show_screening_summary()
    
    if st.session_state.ml_results:
        st.subheader("ML Analysis Summary")
        show_ml_summary()
    
    if st.session_state.backtest_results:
        st.subheader("Backtesting Summary")
        show_backtest_summary()

def show_screening_page():
    """Show the stock screening page"""
    st.header("🔍 Stock Screening")
    
    # Screening parameters
    st.subheader("Screening Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_n_stocks = st.slider("Number of stocks to screen", 5, 50, 10)
        lookback_years = st.slider("Lookback period (years)", 1, 5, 1)
    
    with col2:
        outlier_threshold = st.slider("Outlier detection threshold", 1.0, 3.0, 1.5)
        breakout_min_score = st.slider("Breakout minimum score", 0.1, 1.0, 0.6)
    
    # Run screening button
    if st.button("🚀 Run Comprehensive Screening", use_container_width=True):
        run_screening(top_n_stocks, lookback_years, outlier_threshold, breakout_min_score)
    
    # Display results
    if st.session_state.screening_results:
        st.divider()
        display_screening_results()

def show_ml_analysis_page():
    """Show the ML analysis page"""
    st.header("🧠 Machine Learning Analysis")
    
    if not st.session_state.screening_results:
        st.warning("Please run stock screening first to generate data for ML analysis.")
        return
    
    # ML Analysis options
    st.subheader("Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        run_regression = st.checkbox("Regression Analysis", value=True)
        run_classification = st.checkbox("Classification Analysis", value=True)
    
    with col2:
        run_factor_analysis = st.checkbox("Factor Analysis", value=True)
        run_feature_importance = st.checkbox("Feature Importance", value=True)
    
    # Run ML analysis button
    if st.button("🧠 Run ML Analysis", use_container_width=True):
        run_ml_analysis(run_regression, run_classification, run_factor_analysis, run_feature_importance)
    
    # Display results
    if st.session_state.ml_results:
        st.divider()
        display_ml_results()

def show_backtesting_page():
    """Show the backtesting page"""
    st.header("📊 Backtesting & Portfolio Simulation")
    
    if not st.session_state.screening_results:
        st.warning("Please run stock screening first to generate data for backtesting.")
        return
    
    # Backtesting parameters
    st.subheader("Backtesting Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000)
        rebalance_frequency = st.selectbox("Rebalancing Frequency", 
                                         ["daily", "weekly", "monthly", "quarterly"])
    
    with col2:
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
        max_position_size = st.slider("Max Position Size (%)", 5, 50, 10) / 100
    
    # Strategy selection
    st.subheader("Portfolio Strategies")
    strategies = st.multiselect(
        "Select strategies to test:",
        ["equal_weight", "momentum_weight", "risk_parity", "fundamental_weight"],
        default=["equal_weight", "momentum_weight"]
    )
    
    # Run backtesting button
    if st.button("📊 Run Backtesting", use_container_width=True):
        run_backtesting(initial_capital, rebalance_frequency, transaction_cost, max_position_size, strategies)
    
    # Display results
    if st.session_state.backtest_results:
        st.divider()
        display_backtest_results()

def show_ai_insights_page():
    """Show the AI insights page"""
    st.header("🤖 AI-Powered Insights")
    
    if not st.session_state.screening_results:
        st.warning("Please run stock screening first to generate AI insights.")
        return
    
    # AI Analysis Options
    st.subheader("AI Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Explain Screening Results", use_container_width=True):
            with st.spinner("Generating AI explanation..."):
                explanation = analyze_screening_results_with_ai(st.session_state.screening_results)
                st.session_state.ai_explanation = explanation
    
    with col2:
        risk_tolerance = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"])
        if st.button("💡 Generate Investment Strategy", use_container_width=True):
            with st.spinner("Creating personalized strategy..."):
                strategy = generate_investment_strategy(st.session_state.screening_results, risk_tolerance)
                st.session_state.ai_strategy = strategy
    
    # Market Insights
    st.subheader("Market Insights")
    if st.session_state.screening_results.get('recommendations'):
        symbols = [stock['symbol'] for stock in st.session_state.screening_results['recommendations'][:5]]
        
        if st.button("📰 Get Market Insights", use_container_width=True):
            with st.spinner("Analyzing market sentiment..."):
                insights = get_market_insights(symbols)
                st.session_state.market_insights = insights
    
    st.divider()
    
    # Display AI Results
    if hasattr(st.session_state, 'ai_explanation') and st.session_state.ai_explanation:
        st.subheader("📝 AI Explanation of Results")
        
        explanation = st.session_state.ai_explanation
        
        if 'explanation' in explanation:
            st.write(explanation['explanation'])
        
        if 'key_findings' in explanation:
            st.subheader("🔍 Key Findings")
            for finding in explanation['key_findings']:
                st.info(f"• {finding}")
        
        if 'actionable_insights' in explanation:
            st.subheader("💡 Actionable Insights")
            for insight in explanation['actionable_insights']:
                st.success(f"• {insight}")
    
    if hasattr(st.session_state, 'ai_strategy') and st.session_state.ai_strategy:
        st.subheader("💡 AI Investment Strategy")
        
        strategy = st.session_state.ai_strategy
        
        if 'strategy' in strategy:
            st.write(strategy['strategy'])
        
        if 'recommended_allocation' in strategy:
            st.subheader("📊 Recommended Allocation")
            allocation = strategy['recommended_allocation']
            
            # Create pie chart for allocation
            if isinstance(allocation, dict):
                fig = px.pie(
                    values=list(allocation.values()),
                    names=list(allocation.keys()),
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if 'key_principles' in strategy:
            st.subheader("📋 Key Investment Principles")
            for principle in strategy['key_principles']:
                st.info(f"• {principle}")
        
        if 'timeline' in strategy:
            st.metric("⏰ Recommended Timeline", strategy['timeline'])
    
    if hasattr(st.session_state, 'market_insights') and st.session_state.market_insights:
        st.subheader("📰 Market Sentiment Analysis")
        
        insights = st.session_state.market_insights
        
        if 'market_summary' in insights:
            st.info(insights['market_summary'])
        
        if 'insights' in insights and insights['insights']:
            st.subheader("📈 Individual Stock Insights")
            
            for symbol, insight in insights['insights'].items():
                with st.expander(f"{symbol} Analysis"):
                    if 'sentiment' in insight:
                        sentiment_color = {
                            'Positive': '🟢',
                            'Negative': '🔴',
                            'Neutral': '🟡'
                        }.get(insight['sentiment'], '⚪')
                        
                        st.write(f"**Sentiment:** {sentiment_color} {insight['sentiment']}")
                    
                    if 'analysis' in insight:
                        st.write(f"**Analysis:** {insight['analysis']}")
                    
                    if 'news_count' in insight:
                        st.write(f"**News Articles Analyzed:** {insight['news_count']}")
    
    # Export AI Results
    st.divider()
    st.subheader("📥 Export AI Results")
    
    if st.button("Download AI Analysis Report", use_container_width=True):
        ai_report = {
            'explanation': getattr(st.session_state, 'ai_explanation', {}),
            'strategy': getattr(st.session_state, 'ai_strategy', {}),
            'market_insights': getattr(st.session_state, 'market_insights', {}),
            'generated_at': datetime.now().isoformat()
        }
        
        json_data = json.dumps(ai_report, indent=2, default=str)
        
        st.download_button(
            label="📥 Download AI Report (JSON)",
            data=json_data,
            file_name=f"kangro_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def show_settings_page():
    """Show the settings page"""
    st.header("⚙️ Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    
    polygon_key = st.text_input("Polygon.io API Key", 
                               value=os.getenv('POLYGON_API_KEY', ''), 
                               type="password")
    tavily_key = st.text_input("Tavily API Key", 
                              value=os.getenv('TAVILY_API_KEY', ''), 
                              type="password")
    openai_key = st.text_input("OpenAI API Key", 
                              value=os.getenv('OPENAI_API_KEY', ''), 
                              type="password")
    
    # Data Management
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All Data", use_container_width=True):
            clear_all_data()
    
    with col2:
        if st.button("Export Results", use_container_width=True):
            export_results()
    
    # System Information
    st.subheader("System Information")
    
    st.info(f"""
    **Application Version:** 1.0.0
    **Python Version:** {os.sys.version.split()[0]}
    **Streamlit Version:** {st.__version__}
    **Data Status:** {len(st.session_state.screening_results)} screening results, {len(st.session_state.ml_results)} ML results
    """)

# Helper functions for quick actions
def run_quick_screening():
    """Run quick screening with default parameters"""
    run_screening(10, 1, 1.5, 0.6)

def run_quick_ml_analysis():
    """Run quick ML analysis"""
    run_ml_analysis(True, True, True, True)

def run_quick_backtest():
    """Run quick backtest"""
    run_backtesting(100000, "monthly", 0.001, 0.1, ["equal_weight"])

# Main screening function
def run_screening(top_n_stocks, lookback_years, outlier_threshold, breakout_min_score):
    """Run comprehensive stock screening"""
    with st.spinner("Running comprehensive screening..."):
        try:
            screening_params = {
                'top_n_stocks': top_n_stocks,
                'lookback_years': lookback_years,
                'outlier_threshold': outlier_threshold,
                'breakout_min_score': breakout_min_score
            }
            
            results = components['screening_engine'].run_comprehensive_screening(screening_params)
            
            # Store results
            st.session_state.screening_results = results
            st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Fetch price data for top stocks
            if 'recommendations' in results and results['recommendations']:
                symbols = [stock['symbol'] for stock in results['recommendations'][:5]]
                price_data = components['data_fetcher'].fetch_batch_data(symbols, days=252)
                st.session_state.price_data.update(price_data)
            
            st.success(f"✅ Screening completed! Found {len(results.get('recommendations', []))} recommendations.")
            
        except Exception as e:
            st.error(f"❌ Screening failed: {str(e)}")
            logger.error(f"Screening error: {str(e)}")

def run_ml_analysis(run_regression, run_classification, run_factor_analysis, run_feature_importance):
    """Run ML analysis"""
    with st.spinner("Running ML analysis..."):
        try:
            # Prepare data
            price_data = st.session_state.price_data
            screening_results = st.session_state.screening_results
            
            # Extract financial metrics
            financial_metrics = {}
            if 'recommendations' in screening_results:
                for stock in screening_results['recommendations']:
                    symbol = stock['symbol']
                    financial_metrics[symbol] = stock.get('metrics', {})
            
            # Prepare features
            features_df = components['ml_analyzer'].prepare_features(price_data, financial_metrics)
            
            if features_df.empty:
                st.error("❌ Insufficient data for ML analysis.")
                return
            
            ml_results = {'timestamp': datetime.now().isoformat()}
            
            # Run selected analyses
            if run_regression:
                ml_results['regression_results'] = components['ml_analyzer'].train_regression_models(features_df)
            
            if run_classification:
                ml_results['classification_results'] = components['ml_analyzer'].train_classification_models(features_df)
            
            if run_feature_importance:
                ml_results['feature_analysis'] = components['ml_analyzer'].perform_feature_analysis(features_df)
            
            if run_factor_analysis:
                ml_results['factor_analysis'] = components['factor_analyzer'].perform_factor_analysis(features_df)
                ml_results['factor_summary'] = components['factor_analyzer'].get_factor_summary(ml_results['factor_analysis'])
            
            # Store results
            st.session_state.ml_results = ml_results
            
            st.success("✅ ML analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ ML analysis failed: {str(e)}")
            logger.error(f"ML analysis error: {str(e)}")

def run_backtesting(initial_capital, rebalance_frequency, transaction_cost, max_position_size, strategies):
    """Run backtesting and portfolio simulation"""
    with st.spinner("Running backtesting..."):
        try:
            # Create stock selections
            screening_results = st.session_state.screening_results
            stock_selections = {}
            
            if 'recommendations' in screening_results and screening_results['recommendations']:
                current_date = datetime.now()
                for i in range(12):  # 12 months of selections
                    selection_date = (current_date - timedelta(days=30*i)).strftime('%Y-%m-%d')
                    selected_stocks = [stock['symbol'] for stock in screening_results['recommendations'][:5]]
                    stock_selections[selection_date] = selected_stocks
            
            # Set date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Run backtest
            strategy_params = {
                'initial_capital': initial_capital,
                'rebalance_frequency': rebalance_frequency,
                'transaction_cost': transaction_cost,
                'max_position_size': max_position_size
            }
            
            backtest_results = components['backtest_engine'].run_backtest(
                stock_selections=stock_selections,
                price_data=st.session_state.price_data,
                start_date=start_date,
                end_date=end_date,
                strategy_params=strategy_params
            )
            
            # Run portfolio simulation
            fundamental_data = {}
            if 'recommendations' in screening_results:
                for stock in screening_results['recommendations']:
                    symbol = stock['symbol']
                    fundamental_data[symbol] = stock.get('metrics', {})
            
            simulation_results = components['portfolio_simulator'].run_multi_strategy_simulation(
                stock_selections=stock_selections,
                price_data=st.session_state.price_data,
                fundamental_data=fundamental_data,
                start_date=start_date,
                end_date=end_date,
                strategies=strategies
            )
            
            # Combine results
            combined_results = {
                'backtest_results': backtest_results,
                'simulation_results': simulation_results,
                'parameters': strategy_params,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store results
            st.session_state.backtest_results = combined_results
            
            st.success("✅ Backtesting completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Backtesting failed: {str(e)}")
            logger.error(f"Backtesting error: {str(e)}")

# Display functions
def show_screening_summary():
    """Show screening results summary"""
    results = st.session_state.screening_results
    if 'recommendations' in results:
        df = pd.DataFrame(results['recommendations'][:5])
        st.dataframe(df[['symbol', 'composite_score']], use_container_width=True)

def show_ml_summary():
    """Show ML results summary"""
    results = st.session_state.ml_results
    if 'regression_results' in results:
        st.write(f"Trained {len(results['regression_results'])} regression models")
    if 'classification_results' in results:
        st.write(f"Trained {len(results['classification_results'])} classification models")

def show_backtest_summary():
    """Show backtest results summary"""
    results = st.session_state.backtest_results
    if 'backtest_results' in results and 'performance_metrics' in results['backtest_results']:
        metrics = results['backtest_results']['performance_metrics']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")

def display_screening_results():
    """Display detailed screening results"""
    results = st.session_state.screening_results
    
    if 'recommendations' in results and results['recommendations']:
        st.subheader("Top Stock Recommendations")
        
        # Create DataFrame
        df = pd.DataFrame(results['recommendations'])
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Create visualization
        fig = px.bar(df.head(10), x='symbol', y='composite_score', 
                    title="Top 10 Stocks by Composite Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"screening_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def display_ml_results():
    """Display ML analysis results"""
    results = st.session_state.ml_results
    
    # Feature importance
    if 'feature_analysis' in results and 'statistical_importance' in results['feature_analysis']:
        st.subheader("Feature Importance Analysis")
        
        importance_data = results['feature_analysis']['statistical_importance'][:10]
        df_importance = pd.DataFrame(importance_data)
        
        fig = px.bar(df_importance, x='score', y='feature', orientation='h',
                    title="Top 10 Most Important Features")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    if 'regression_results' in results:
        st.subheader("Regression Model Performance")
        
        model_data = []
        for model_name, model_info in results['regression_results'].items():
            model_data.append({
                'Model': model_name,
                'R² Score': model_info.get('test_r2', 0),
                'RMSE': model_info.get('test_rmse', 0)
            })
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
    
    # Factor analysis
    if 'factor_summary' in results:
        st.subheader("Factor Analysis Summary")
        factor_summary = results['factor_summary']
        
        if 'key_drivers' in factor_summary and 'insights' in factor_summary['key_drivers']:
            for insight in factor_summary['key_drivers']['insights']:
                st.info(insight)

def display_backtest_results():
    """Display backtesting results"""
    results = st.session_state.backtest_results
    
    # Performance metrics
    if 'backtest_results' in results and 'performance_metrics' in results['backtest_results']:
        st.subheader("Performance Metrics")
        
        metrics = results['backtest_results']['performance_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
        with col2:
            st.metric("Annualized Return", f"{metrics.get('annualized_return', 0):.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    
    # Portfolio performance chart
    if 'backtest_results' in results and 'portfolio_performance' in results['backtest_results']:
        st.subheader("Portfolio Performance Over Time")
        
        portfolio_perf = results['backtest_results']['portfolio_performance']
        
        df_perf = pd.DataFrame({
            'Date': pd.to_datetime(portfolio_perf.get('dates', [])),
            'Portfolio Value': portfolio_perf.get('portfolio_value', []),
            'Cumulative Return': [r * 100 for r in portfolio_perf.get('cumulative_returns', [])]
        })
        
        fig = px.line(df_perf, x='Date', y='Cumulative Return',
                     title="Cumulative Returns Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategy comparison
    if 'simulation_results' in results and 'comparison' in results['simulation_results']:
        st.subheader("Strategy Comparison")
        
        comparison = results['simulation_results']['comparison']
        if 'summary_table' in comparison:
            df_strategies = pd.DataFrame(comparison['summary_table'])
            st.dataframe(df_strategies, use_container_width=True)

def clear_all_data():
    """Clear all stored data"""
    st.session_state.screening_results = {}
    st.session_state.ml_results = {}
    st.session_state.backtest_results = {}
    st.session_state.price_data = {}
    st.success("✅ All data cleared!")

def export_results():
    """Export all results"""
    export_data = {
        'screening_results': st.session_state.screening_results,
        'ml_results': st.session_state.ml_results,
        'backtest_results': st.session_state.backtest_results,
        'export_timestamp': datetime.now().isoformat()
    }
    
    json_data = json.dumps(export_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download All Results (JSON)",
        data=json_data,
        file_name=f"kangro_capital_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()

