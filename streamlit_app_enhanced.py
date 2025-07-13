"""
Kangro Capital - Enhanced Stock Screening Platform
Complete version with Advanced Backtesting, Training, and Testing Capabilities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import requests
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.data_fetcher import DataFetcher
    from utils.stock_screener import StockScreener
    from utils.outlier_detector import OutlierDetector
    from utils.screening_engine import ScreeningEngine
    from utils.ml_analyzer import MLAnalyzer
    from utils.factor_analyzer import FactorAnalyzer
    from utils.backtest_engine import BacktestEngine
    from utils.portfolio_simulator import PortfolioSimulator
    from utils.kangro_agent import KangroAgent
    from utils.advanced_backtester import AdvancedBacktester
    from utils.portfolio_optimizer import PortfolioOptimizer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Kangro Capital - Enhanced Stock Screening Platform",
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
    .danger-metric {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .backtest-section {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .superiority-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.875rem;
    }
    .superior {
        background-color: #28a745;
        color: white;
    }
    .inferior {
        background-color: #dc3545;
        color: white;
    }
    .neutral {
        background-color: #6c757d;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'advanced_backtest_results' not in st.session_state:
    st.session_state.advanced_backtest_results = None
if 'portfolio_optimization_results' not in st.session_state:
    st.session_state.portfolio_optimization_results = None
if 'superiority_analysis' not in st.session_state:
    st.session_state.superiority_analysis = None

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components"""
    try:
        data_fetcher = DataFetcher()
        stock_screener = StockScreener()
        outlier_detector = OutlierDetector()
        screening_engine = ScreeningEngine()
        ml_analyzer = MLAnalyzer()
        factor_analyzer = FactorAnalyzer()
        backtest_engine = BacktestEngine()
        portfolio_simulator = PortfolioSimulator()
        kangro_agent = KangroAgent()
        advanced_backtester = AdvancedBacktester()
        portfolio_optimizer = PortfolioOptimizer()
        
        return {
            'data_fetcher': data_fetcher,
            'stock_screener': stock_screener,
            'outlier_detector': outlier_detector,
            'screening_engine': screening_engine,
            'ml_analyzer': ml_analyzer,
            'factor_analyzer': factor_analyzer,
            'backtest_engine': backtest_engine,
            'portfolio_simulator': portfolio_simulator,
            'kangro_agent': kangro_agent,
            'advanced_backtester': advanced_backtester,
            'portfolio_optimizer': portfolio_optimizer
        }
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None

# Main header
st.markdown('<h1 class="main-header">📈 Kangro Capital</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #666;">Enhanced Stock Screening Platform with Advanced Backtesting</h2>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    [
        "🏠 Home",
        "🔍 Stock Screening",
        "🧠 ML Analysis",
        "📊 Traditional Backtesting",
        "🚀 Advanced Backtesting & Training",
        "⚖️ Portfolio Optimization",
        "🏆 Superiority Analysis",
        "🤖 AI Assistant"
    ]
)

# Initialize components
components = initialize_components()
if not components:
    st.error("Failed to initialize components. Please check your environment setup.")
    st.stop()

# Home Page
if page == "🏠 Home":
    st.header("Welcome to Kangro Capital Enhanced Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 Stock Screening
        - 7-step fundamental analysis
        - Statistical outlier detection
        - Comprehensive scoring system
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 ML Analysis
        - Predictive modeling
        - Feature importance analysis
        - Factor analysis with PCA
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Backtesting
        - Traditional portfolio simulation
        - Advanced ML-enhanced backtesting
        - Benchmark comparison
        """)
    
    st.markdown("---")
    
    # New Enhanced Features
    st.header("🚀 New Enhanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ⚖️ Portfolio Optimization
        - Modern Portfolio Theory
        - Risk Parity Optimization
        - Black-Litterman Model
        - ML-Enhanced Optimization
        - Hierarchical Risk Parity
        """)
    
    with col2:
        st.markdown("""
        ### 🏆 Superiority Analysis
        - Benchmark Comparison
        - Statistical Significance Testing
        - Risk-Adjusted Performance
        - Alpha Generation Analysis
        - Performance Attribution
        """)
    
    st.markdown("---")
    
    # Performance Metrics Overview
    st.header("📈 Platform Capabilities")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Optimization Methods", "6", help="Mean Variance, Risk Parity, Black-Litterman, Factor-Based, ML-Enhanced, HRP")
    
    with metrics_col2:
        st.metric("Benchmark Indices", "10+", help="SPY, QQQ, IWM, VTI, and sector-specific ETFs")
    
    with metrics_col3:
        st.metric("Performance Metrics", "15+", help="Sharpe, Sortino, Alpha, Beta, Max Drawdown, VaR, and more")
    
    with metrics_col4:
        st.metric("ML Models", "5", help="Random Forest, Gradient Boosting, SVM, Neural Networks, Ensemble")

# Stock Screening Page
elif page == "🔍 Stock Screening":
    st.header("Stock Screening")
    
    # Stock selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_input = st.text_input(
            "Enter stock symbols (comma-separated)",
            value="AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NFLX",
            help="Enter stock symbols separated by commas"
        )
    
    with col2:
        run_screening = st.button("🔍 Run Screening", type="primary")
    
    if run_screening and stock_input:
        stocks = [s.strip().upper() for s in stock_input.split(',')]
        
        with st.spinner("Running comprehensive stock screening..."):
            try:
                # Run screening
                screening_results = components['screening_engine'].run_comprehensive_screening(stocks)
                st.session_state.screening_results = screening_results
                
                if screening_results and 'screened_stocks' in screening_results:
                    st.success(f"✅ Screening completed for {len(screening_results['screened_stocks'])} stocks")
                    
                    # Display results
                    st.subheader("📊 Screening Results")
                    
                    # Create results DataFrame
                    results_data = []
                    for stock_data in screening_results['screened_stocks']:
                        results_data.append({
                            'Symbol': stock_data['symbol'],
                            'Composite Score': stock_data.get('composite_score', 0),
                            'ROE': stock_data.get('fundamental_metrics', {}).get('roe', 'N/A'),
                            'Current Ratio': stock_data.get('fundamental_metrics', {}).get('current_ratio', 'N/A'),
                            'Gross Margin': stock_data.get('fundamental_metrics', {}).get('gross_margin', 'N/A'),
                            'Net Margin': stock_data.get('fundamental_metrics', {}).get('net_margin', 'N/A'),
                            'Revenue Growth': stock_data.get('fundamental_metrics', {}).get('revenue_growth_5y', 'N/A'),
                            'Debt/EBITDA': stock_data.get('fundamental_metrics', {}).get('debt_to_ebitda', 'N/A'),
                            'Outlier Score': stock_data.get('outlier_analysis', {}).get('outlier_score', 'N/A')
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Visualization
                    if len(results_df) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Composite Score Chart
                            fig_scores = px.bar(
                                results_df, 
                                x='Symbol', 
                                y='Composite Score',
                                title="Composite Scores by Stock",
                                color='Composite Score',
                                color_continuous_scale='RdYlGn'
                            )
                            st.plotly_chart(fig_scores, use_container_width=True)
                        
                        with col2:
                            # ROE vs Current Ratio Scatter
                            numeric_df = results_df.copy()
                            for col in ['ROE', 'Current Ratio']:
                                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
                            
                            fig_scatter = px.scatter(
                                numeric_df,
                                x='ROE',
                                y='Current Ratio',
                                size='Composite Score',
                                hover_name='Symbol',
                                title="ROE vs Current Ratio"
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                
                else:
                    st.error("❌ No screening results obtained")
                    
            except Exception as e:
                st.error(f"❌ Error during screening: {str(e)}")

# ML Analysis Page
elif page == "🧠 ML Analysis":
    st.header("Machine Learning Analysis")
    
    if st.session_state.screening_results is None:
        st.warning("⚠️ Please run stock screening first to get data for ML analysis.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("📊 Using data from previous screening results")
        
        with col2:
            run_ml = st.button("🧠 Run ML Analysis", type="primary")
        
        if run_ml:
            with st.spinner("Running ML analysis..."):
                try:
                    # Extract stock symbols from screening results
                    stocks = [stock['symbol'] for stock in st.session_state.screening_results['screened_stocks']]
                    
                    # Run ML analysis
                    ml_results = components['ml_analyzer'].run_comprehensive_analysis(stocks)
                    st.session_state.ml_results = ml_results
                    
                    if ml_results:
                        st.success("✅ ML Analysis completed successfully")
                        
                        # Display results
                        st.subheader("🎯 ML Predictions")
                        
                        if 'predictions' in ml_results:
                            predictions_data = []
                            for stock, prediction in ml_results['predictions'].items():
                                predictions_data.append({
                                    'Symbol': stock,
                                    'Predicted Return': f"{prediction.get('predicted_return', 0):.2%}",
                                    'Confidence': f"{prediction.get('confidence', 0):.2f}",
                                    'Recommendation': prediction.get('recommendation', 'HOLD')
                                })
                            
                            predictions_df = pd.DataFrame(predictions_data)
                            st.dataframe(predictions_df, use_container_width=True)
                        
                        # Feature Importance
                        if 'feature_importance' in ml_results:
                            st.subheader("📈 Feature Importance")
                            
                            importance_data = ml_results['feature_importance']
                            if importance_data:
                                features = list(importance_data.keys())
                                importance = list(importance_data.values())
                                
                                fig_importance = px.bar(
                                    x=importance,
                                    y=features,
                                    orientation='h',
                                    title="Feature Importance in ML Model",
                                    labels={'x': 'Importance', 'y': 'Features'}
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Model Performance
                        if 'model_performance' in ml_results:
                            st.subheader("🎯 Model Performance")
                            
                            performance = ml_results['model_performance']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("R² Score", f"{performance.get('r2_score', 0):.3f}")
                            
                            with col2:
                                st.metric("Mean Absolute Error", f"{performance.get('mae', 0):.3f}")
                            
                            with col3:
                                st.metric("Cross-Val Score", f"{performance.get('cv_score', 0):.3f}")
                    
                    else:
                        st.error("❌ ML analysis failed")
                        
                except Exception as e:
                    st.error(f"❌ Error during ML analysis: {str(e)}")

# Traditional Backtesting Page
elif page == "📊 Traditional Backtesting":
    st.header("Traditional Backtesting")
    
    if st.session_state.screening_results is None:
        st.warning("⚠️ Please run stock screening first to get portfolio data.")
    else:
        # Backtesting parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        with col3:
            initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
        
        run_backtest = st.button("📊 Run Traditional Backtest", type="primary")
        
        if run_backtest:
            with st.spinner("Running traditional backtest..."):
                try:
                    # Extract top stocks from screening
                    top_stocks = [stock['symbol'] for stock in st.session_state.screening_results['screened_stocks'][:10]]
                    
                    # Run backtest
                    backtest_results = components['backtest_engine'].run_backtest(
                        top_stocks,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        initial_capital
                    )
                    
                    st.session_state.backtest_results = backtest_results
                    
                    if backtest_results:
                        st.success("✅ Traditional backtest completed")
                        
                        # Performance metrics
                        st.subheader("📈 Performance Metrics")
                        
                        metrics = backtest_results.get('performance_metrics', {})
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
                        
                        with col2:
                            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                        
                        with col3:
                            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
                        
                        with col4:
                            st.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
                        
                        # Portfolio value chart
                        if 'portfolio_values' in backtest_results:
                            st.subheader("💰 Portfolio Value Over Time")
                            
                            portfolio_data = backtest_results['portfolio_values']
                            if portfolio_data:
                                dates = list(portfolio_data.keys())
                                values = list(portfolio_data.values())
                                
                                fig_portfolio = go.Figure()
                                fig_portfolio.add_trace(go.Scatter(
                                    x=dates,
                                    y=values,
                                    mode='lines',
                                    name='Portfolio Value',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                fig_portfolio.update_layout(
                                    title="Portfolio Value Over Time",
                                    xaxis_title="Date",
                                    yaxis_title="Portfolio Value ($)",
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig_portfolio, use_container_width=True)
                    
                    else:
                        st.error("❌ Traditional backtest failed")
                        
                except Exception as e:
                    st.error(f"❌ Error during traditional backtest: {str(e)}")

# Advanced Backtesting & Training Page
elif page == "🚀 Advanced Backtesting & Training":
    st.header("Advanced Backtesting & Training")
    st.markdown('<div class="backtest-section">', unsafe_allow_html=True)
    st.markdown("### 🎯 ML-Enhanced Portfolio Training & Testing")
    st.markdown("Train machine learning models to predict portfolio performance and test against historical data.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Training Parameters
    st.subheader("🔧 Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        training_period = st.selectbox(
            "Training Period (months)",
            [6, 12, 18, 24, 36],
            index=1,
            help="Number of months to use for training the ML models"
        )
    
    with col2:
        prediction_horizon = st.selectbox(
            "Prediction Horizon (months)",
            [1, 3, 6, 12],
            index=2,
            help="How far ahead to predict portfolio performance"
        )
    
    with col3:
        rebalancing_frequency = st.selectbox(
            "Rebalancing Frequency",
            ["Monthly", "Quarterly", "Semi-Annual", "Annual"],
            index=1,
            help="How often to rebalance the portfolio"
        )
    
    # Stock Selection for Training
    st.subheader("📊 Stock Universe")
    
    if st.session_state.screening_results:
        available_stocks = [stock['symbol'] for stock in st.session_state.screening_results['screened_stocks']]
        st.info(f"Using {len(available_stocks)} stocks from previous screening: {', '.join(available_stocks[:10])}{'...' if len(available_stocks) > 10 else ''}")
        selected_stocks = available_stocks
    else:
        stock_universe = st.text_input(
            "Enter stock symbols for training (comma-separated)",
            value="AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NFLX,ADBE,CRM",
            help="Enter stock symbols to use for training the ML models"
        )
        selected_stocks = [s.strip().upper() for s in stock_universe.split(',')]
    
    # Training Controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"🎯 Ready to train on {len(selected_stocks)} stocks with {training_period}-month training period")
    
    with col2:
        run_training = st.button("🚀 Start Training", type="primary")
    
    if run_training and selected_stocks:
        with st.spinner("Training ML models and running advanced backtesting..."):
            try:
                # Generate sample portfolio selections for training
                sample_selections = {}
                start_date = datetime.now() - timedelta(days=training_period * 30)
                
                for i in range(training_period):
                    selection_date = start_date + timedelta(days=i * 30)
                    # Randomly select 5-7 stocks for each month
                    n_stocks = np.random.randint(5, min(8, len(selected_stocks) + 1))
                    monthly_selection = np.random.choice(selected_stocks, n_stocks, replace=False).tolist()
                    sample_selections[selection_date.strftime('%Y-%m-%d')] = monthly_selection
                
                # Get price data
                price_data = {}
                data_fetcher = components['data_fetcher']
                
                for stock in selected_stocks:
                    try:
                        stock_data = data_fetcher.get_stock_data(
                            stock,
                            (datetime.now() - timedelta(days=training_period * 30 + 365)).strftime('%Y-%m-%d'),
                            datetime.now().strftime('%Y-%m-%d')
                        )
                        if stock_data is not None and not stock_data.empty:
                            price_data[stock] = stock_data
                    except:
                        continue
                
                if not price_data:
                    st.error("❌ Could not fetch price data for training")
                    st.stop()
                
                # Generate sample fundamental data
                fundamental_data = {}
                for stock in selected_stocks:
                    fundamental_data[stock] = {
                        'roe': np.random.uniform(0.05, 0.25),
                        'current_ratio': np.random.uniform(1.0, 3.0),
                        'gross_margin': np.random.uniform(0.2, 0.6),
                        'net_margin': np.random.uniform(0.05, 0.25),
                        'revenue_growth_5y': np.random.uniform(-0.05, 0.15),
                        'debt_to_ebitda': np.random.uniform(0.5, 4.0),
                        'pe_ratio': np.random.uniform(10, 30),
                        'market_cap': np.random.uniform(100000000000, 3000000000000)
                    }
                
                # Train portfolio predictor
                st.info("🧠 Training ML models...")
                training_results = components['advanced_backtester'].train_portfolio_predictor(
                    sample_selections, price_data, fundamental_data, training_period
                )
                
                if training_results and 'models_trained' in training_results:
                    st.success(f"✅ Successfully trained {len(training_results['models_trained'])} ML models")
                    
                    # Display training results
                    st.subheader("🎯 Training Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Models Trained", len(training_results['models_trained']))
                    
                    with col2:
                        training_samples = training_results.get('training_summary', {}).get('total_samples', 0)
                        st.metric("Training Samples", training_samples)
                    
                    with col3:
                        best_r2 = 0
                        if 'model_performance' in training_results:
                            for target, models in training_results['model_performance'].items():
                                for model_name, metrics in models.items():
                                    r2 = metrics.get('r2', 0)
                                    best_r2 = max(best_r2, r2)
                        st.metric("Best R² Score", f"{best_r2:.3f}")
                    
                    # Feature importance
                    if 'feature_importance' in training_results and training_results['feature_importance']:
                        st.subheader("📊 Feature Importance")
                        
                        importance_data = training_results['feature_importance']
                        features = list(importance_data.keys())
                        importance = list(importance_data.values())
                        
                        fig_importance = px.bar(
                            x=importance,
                            y=features,
                            orientation='h',
                            title="Feature Importance in Portfolio Prediction",
                            labels={'x': 'Importance', 'y': 'Features'},
                            color=importance,
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Model performance details
                    if 'model_performance' in training_results:
                        st.subheader("🎯 Model Performance Details")
                        
                        performance_data = []
                        for target, models in training_results['model_performance'].items():
                            for model_name, metrics in models.items():
                                performance_data.append({
                                    'Target': target,
                                    'Model': model_name,
                                    'R² Score': f"{metrics.get('r2', 0):.3f}",
                                    'MAE': f"{metrics.get('mae', 0):.3f}",
                                    'RMSE': f"{metrics.get('rmse', 0):.3f}"
                                })
                        
                        if performance_data:
                            performance_df = pd.DataFrame(performance_data)
                            st.dataframe(performance_df, use_container_width=True)
                    
                    # Store results
                    st.session_state.advanced_backtest_results = training_results
                    
                    # Prediction test
                    st.subheader("🔮 Prediction Test")
                    
                    test_stocks = selected_stocks[:5]  # Use first 5 stocks for prediction
                    prediction_date = datetime.now().strftime('%Y-%m-%d')
                    
                    try:
                        prediction_results = components['advanced_backtester'].predict_portfolio_performance(
                            test_stocks, price_data, fundamental_data, prediction_date
                        )
                        
                        if prediction_results and 'predictions' in prediction_results:
                            st.success("✅ Prediction test completed")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📈 Predictions:**")
                                for target, prediction in prediction_results['predictions'].items():
                                    st.write(f"- {target}: {prediction:.3f}")
                            
                            with col2:
                                st.markdown("**🎯 Confidence Scores:**")
                                if 'confidence_scores' in prediction_results:
                                    for target, confidence in prediction_results['confidence_scores'].items():
                                        st.write(f"- {target}: {confidence:.2f}")
                            
                            recommendation = prediction_results.get('recommendation', 'HOLD')
                            st.markdown(f"**🎯 Overall Recommendation:** `{recommendation}`")
                    
                    except Exception as e:
                        st.warning(f"⚠️ Prediction test failed: {str(e)}")
                
                else:
                    st.error("❌ Training failed - no models were successfully trained")
                    
            except Exception as e:
                st.error(f"❌ Error during advanced backtesting: {str(e)}")
                st.exception(e)

# Portfolio Optimization Page
elif page == "⚖️ Portfolio Optimization":
    st.header("Portfolio Optimization")
    st.markdown('<div class="backtest-section">', unsafe_allow_html=True)
    st.markdown("### 🎯 Modern Portfolio Theory & Advanced Optimization")
    st.markdown("Optimize portfolio weights using various methodologies including ML-enhanced approaches.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Optimization Parameters
    st.subheader("🔧 Optimization Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_method = st.selectbox(
            "Optimization Method",
            [
                "mean_variance",
                "risk_parity", 
                "black_litterman",
                "factor_based",
                "ml_enhanced",
                "hierarchical_risk_parity"
            ],
            format_func=lambda x: {
                "mean_variance": "Mean Variance (Markowitz)",
                "risk_parity": "Risk Parity",
                "black_litterman": "Black-Litterman",
                "factor_based": "Factor-Based",
                "ml_enhanced": "ML-Enhanced",
                "hierarchical_risk_parity": "Hierarchical Risk Parity"
            }[x],
            help="Choose the portfolio optimization methodology"
        )
    
    with col2:
        risk_tolerance = st.slider(
            "Risk Tolerance",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Higher values allow for more risk in pursuit of returns"
        )
    
    # Stock Selection
    st.subheader("📊 Portfolio Universe")
    
    if st.session_state.screening_results:
        available_stocks = [stock['symbol'] for stock in st.session_state.screening_results['screened_stocks'][:10]]
        st.info(f"Using top 10 stocks from screening: {', '.join(available_stocks)}")
        portfolio_stocks = available_stocks
    else:
        portfolio_input = st.text_input(
            "Enter stock symbols for optimization (comma-separated)",
            value="AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA",
            help="Enter 4-10 stock symbols for portfolio optimization"
        )
        portfolio_stocks = [s.strip().upper() for s in portfolio_input.split(',')]
    
    # Optimization Controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"🎯 Ready to optimize portfolio with {len(portfolio_stocks)} stocks using {optimization_method.replace('_', ' ').title()}")
    
    with col2:
        run_optimization = st.button("⚖️ Optimize Portfolio", type="primary")
    
    if run_optimization and portfolio_stocks:
        with st.spinner(f"Running {optimization_method.replace('_', ' ').title()} optimization..."):
            try:
                # Get price data
                price_data = {}
                data_fetcher = components['data_fetcher']
                
                for stock in portfolio_stocks:
                    try:
                        stock_data = data_fetcher.get_stock_data(
                            stock,
                            (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
                            datetime.now().strftime('%Y-%m-%d')
                        )
                        if stock_data is not None and not stock_data.empty:
                            price_data[stock] = stock_data
                    except:
                        continue
                
                if not price_data:
                    st.error("❌ Could not fetch price data for optimization")
                    st.stop()
                
                # Generate fundamental data
                fundamental_data = {}
                for stock in portfolio_stocks:
                    fundamental_data[stock] = {
                        'roe': np.random.uniform(0.05, 0.25),
                        'current_ratio': np.random.uniform(1.0, 3.0),
                        'gross_margin': np.random.uniform(0.2, 0.6),
                        'net_margin': np.random.uniform(0.05, 0.25),
                        'revenue_growth_5y': np.random.uniform(-0.05, 0.15),
                        'debt_to_ebitda': np.random.uniform(0.5, 4.0)
                    }
                
                # Run optimization
                optimization_results = components['portfolio_optimizer'].optimize_portfolio(
                    portfolio_stocks, price_data, fundamental_data, optimization_method
                )
                
                if optimization_results and 'optimal_weights' in optimization_results:
                    st.success(f"✅ Portfolio optimization completed using {optimization_method.replace('_', ' ').title()}")
                    
                    # Store results
                    st.session_state.portfolio_optimization_results = optimization_results
                    
                    # Display optimal weights
                    st.subheader("⚖️ Optimal Portfolio Weights")
                    
                    weights = optimization_results['optimal_weights']
                    
                    # Create weights DataFrame
                    weights_data = []
                    for stock, weight in weights.items():
                        weights_data.append({
                            'Stock': stock,
                            'Weight': f"{weight:.1%}",
                            'Weight_Numeric': weight
                        })
                    
                    weights_df = pd.DataFrame(weights_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(weights_df[['Stock', 'Weight']], use_container_width=True)
                    
                    with col2:
                        # Pie chart of weights
                        fig_pie = px.pie(
                            weights_df,
                            values='Weight_Numeric',
                            names='Stock',
                            title="Portfolio Allocation"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Expected metrics
                    if 'expected_metrics' in optimization_results:
                        st.subheader("📈 Expected Portfolio Metrics")
                        
                        metrics = optimization_results['expected_metrics']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Expected Return", f"{metrics.get('expected_return', 0):.2%}")
                        
                        with col2:
                            st.metric("Expected Volatility", f"{metrics.get('expected_volatility', 0):.2%}")
                        
                        with col3:
                            st.metric("Expected Sharpe", f"{metrics.get('expected_sharpe', 0):.2f}")
                        
                        with col4:
                            st.metric("Max Weight", f"{max(weights.values()):.1%}")
                    
                    # Risk analysis
                    if 'risk_analysis' in optimization_results:
                        st.subheader("⚠️ Risk Analysis")
                        
                        risk_analysis = optimization_results['risk_analysis']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'concentration_risk' in risk_analysis:
                                concentration = risk_analysis['concentration_risk']
                                st.markdown("**Concentration Risk:**")
                                st.write(f"- Herfindahl Index: {concentration.get('herfindahl_index', 'N/A'):.3f}")
                                st.write(f"- Effective Stocks: {concentration.get('effective_stocks', 'N/A'):.1f}")
                        
                        with col2:
                            if 'tail_risk' in risk_analysis:
                                tail_risk = risk_analysis['tail_risk']
                                st.markdown("**Tail Risk:**")
                                st.write(f"- VaR (95%): {tail_risk.get('var_95', 'N/A'):.2%}")
                                st.write(f"- VaR (99%): {tail_risk.get('var_99', 'N/A'):.2%}")
                    
                    # Optimization comparison
                    st.subheader("🔄 Method Comparison")
                    
                    if st.button("🔄 Compare All Methods"):
                        with st.spinner("Comparing optimization methods..."):
                            try:
                                comparison_methods = ['mean_variance', 'risk_parity', 'ml_enhanced']
                                comparison_results = components['portfolio_optimizer'].compare_optimization_methods(
                                    portfolio_stocks, price_data, fundamental_data, comparison_methods
                                )
                                
                                if comparison_results and 'optimization_results' in comparison_results:
                                    st.success("✅ Method comparison completed")
                                    
                                    # Create comparison table
                                    comparison_data = []
                                    for method, result in comparison_results['optimization_results'].items():
                                        metrics = result.get('expected_metrics', {})
                                        comparison_data.append({
                                            'Method': method.replace('_', ' ').title(),
                                            'Expected Return': f"{metrics.get('expected_return', 0):.2%}",
                                            'Expected Volatility': f"{metrics.get('expected_volatility', 0):.2%}",
                                            'Expected Sharpe': f"{metrics.get('expected_sharpe', 0):.2f}",
                                            'Max Weight': f"{max(result.get('optimal_weights', {}).values()) if result.get('optimal_weights') else 0:.1%}"
                                        })
                                    
                                    comparison_df = pd.DataFrame(comparison_data)
                                    st.dataframe(comparison_df, use_container_width=True)
                                    
                                    # Best method
                                    if 'best_method' in comparison_results:
                                        best = comparison_results['best_method']
                                        st.success(f"🏆 Best Method: **{best['method'].replace('_', ' ').title()}** (Score: {best['score']:.3f})")
                            
                            except Exception as e:
                                st.error(f"❌ Method comparison failed: {str(e)}")
                
                else:
                    st.error("❌ Portfolio optimization failed")
                    
            except Exception as e:
                st.error(f"❌ Error during portfolio optimization: {str(e)}")
                st.exception(e)

# Superiority Analysis Page
elif page == "🏆 Superiority Analysis":
    st.header("Portfolio Superiority Analysis")
    st.markdown('<div class="backtest-section">', unsafe_allow_html=True)
    st.markdown("### 🎯 Benchmark Comparison & Performance Attribution")
    st.markdown("Determine if your portfolio selection strategy generates superior risk-adjusted returns.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis Parameters
    st.subheader("🔧 Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_period = st.selectbox(
            "Analysis Period",
            ["1 Year", "2 Years", "3 Years", "5 Years"],
            index=1,
            help="Historical period for superiority analysis"
        )
    
    with col2:
        benchmark_selection = st.multiselect(
            "Benchmark Indices",
            ["SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "BND"],
            default=["SPY", "QQQ", "IWM"],
            help="Select benchmark indices for comparison"
        )
    
    with col3:
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Statistical confidence level for superiority testing"
        )
    
    # Portfolio Selection
    st.subheader("📊 Portfolio Configuration")
    
    if st.session_state.screening_results and st.session_state.portfolio_optimization_results:
        st.info("🎯 Using optimized portfolio from previous analysis")
        portfolio_weights = st.session_state.portfolio_optimization_results['optimal_weights']
        portfolio_stocks = list(portfolio_weights.keys())
        
        # Display current portfolio
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Portfolio:**")
            for stock, weight in portfolio_weights.items():
                st.write(f"- {stock}: {weight:.1%}")
        
        with col2:
            optimization_method = st.session_state.portfolio_optimization_results.get('method', 'Unknown')
            expected_sharpe = st.session_state.portfolio_optimization_results.get('expected_metrics', {}).get('expected_sharpe', 0)
            st.markdown(f"**Optimization Method:** {optimization_method.replace('_', ' ').title()}")
            st.markdown(f"**Expected Sharpe Ratio:** {expected_sharpe:.2f}")
    
    elif st.session_state.screening_results:
        st.info("🎯 Using equal-weighted portfolio from screening results")
        portfolio_stocks = [stock['symbol'] for stock in st.session_state.screening_results['screened_stocks'][:8]]
        portfolio_weights = {stock: 1.0/len(portfolio_stocks) for stock in portfolio_stocks}
        
        st.write(f"**Portfolio:** {', '.join(portfolio_stocks)}")
        st.write(f"**Weighting:** Equal weight ({1.0/len(portfolio_stocks):.1%} each)")
    
    else:
        st.warning("⚠️ Please run stock screening and portfolio optimization first.")
        portfolio_input = st.text_input(
            "Or enter portfolio manually (symbol:weight, comma-separated)",
            value="AAPL:0.25,MSFT:0.25,GOOGL:0.25,AMZN:0.25",
            help="Format: AAPL:0.25,MSFT:0.25,GOOGL:0.50"
        )
        
        if portfolio_input:
            try:
                portfolio_weights = {}
                for item in portfolio_input.split(','):
                    stock, weight = item.strip().split(':')
                    portfolio_weights[stock.upper()] = float(weight)
                
                portfolio_stocks = list(portfolio_weights.keys())
                
                # Normalize weights
                total_weight = sum(portfolio_weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"⚠️ Weights sum to {total_weight:.2f}, normalizing to 1.0")
                    portfolio_weights = {k: v/total_weight for k, v in portfolio_weights.items()}
            except:
                st.error("❌ Invalid portfolio format")
                portfolio_weights = {}
                portfolio_stocks = []
    
    # Analysis Controls
    if 'portfolio_weights' in locals() and portfolio_weights and benchmark_selection:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"🎯 Ready to analyze {len(portfolio_stocks)}-stock portfolio against {len(benchmark_selection)} benchmarks")
        
        with col2:
            run_superiority = st.button("🏆 Run Analysis", type="primary")
        
        if run_superiority:
            with st.spinner("Running comprehensive superiority analysis..."):
                try:
                    # Calculate analysis period dates
                    period_days = {
                        "1 Year": 365,
                        "2 Years": 730,
                        "3 Years": 1095,
                        "5 Years": 1825
                    }
                    
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=period_days[analysis_period])
                    
                    # Generate sample portfolio selections for the analysis period
                    sample_selections = {}
                    current_date = start_date
                    
                    while current_date < end_date:
                        # Use current portfolio for each month
                        sample_selections[current_date.strftime('%Y-%m-%d')] = portfolio_stocks
                        
                        # Move to next month
                        if current_date.month == 12:
                            current_date = current_date.replace(year=current_date.year + 1, month=1)
                        else:
                            current_date = current_date.replace(month=current_date.month + 1)
                    
                    # Get price data
                    price_data = {}
                    data_fetcher = components['data_fetcher']
                    
                    # Get portfolio stock data
                    for stock in portfolio_stocks:
                        try:
                            stock_data = data_fetcher.get_stock_data(
                                stock,
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d')
                            )
                            if stock_data is not None and not stock_data.empty:
                                price_data[stock] = stock_data
                        except:
                            continue
                    
                    # Get benchmark data
                    for benchmark in benchmark_selection:
                        try:
                            benchmark_data = data_fetcher.get_stock_data(
                                benchmark,
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d')
                            )
                            if benchmark_data is not None and not benchmark_data.empty:
                                price_data[benchmark] = benchmark_data
                        except:
                            continue
                    
                    if not price_data:
                        st.error("❌ Could not fetch price data for analysis")
                        st.stop()
                    
                    # Run comprehensive benchmark comparison
                    comparison_results = components['advanced_backtester'].comprehensive_benchmark_comparison(
                        sample_selections, price_data,
                        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                    )
                    
                    if comparison_results:
                        st.success("✅ Superiority analysis completed")
                        
                        # Store results
                        st.session_state.superiority_analysis = comparison_results
                        
                        # Portfolio Performance
                        st.subheader("📈 Portfolio Performance")
                        
                        portfolio_perf = comparison_results.get('portfolio_performance', {})
                        portfolio_metrics = portfolio_perf.get('performance_metrics', {})
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_return = portfolio_metrics.get('total_return', 0)
                            st.metric("Total Return", f"{total_return:.2%}")
                        
                        with col2:
                            sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
                            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        
                        with col3:
                            max_drawdown = portfolio_metrics.get('max_drawdown', 0)
                            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                        
                        with col4:
                            volatility = portfolio_metrics.get('volatility', 0)
                            st.metric("Volatility", f"{volatility:.2%}")
                        
                        # Benchmark Comparison
                        st.subheader("🏆 Benchmark Comparison")
                        
                        benchmark_perf = comparison_results.get('benchmark_performance', {})
                        
                        if benchmark_perf:
                            comparison_data = []
                            
                            # Add portfolio row
                            comparison_data.append({
                                'Asset': 'Portfolio',
                                'Total Return': f"{total_return:.2%}",
                                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                                'Max Drawdown': f"{max_drawdown:.2%}",
                                'Volatility': f"{volatility:.2%}",
                                'Type': 'Portfolio'
                            })
                            
                            # Add benchmark rows
                            for benchmark, perf in benchmark_perf.items():
                                metrics = perf.get('performance_metrics', {})
                                comparison_data.append({
                                    'Asset': benchmark,
                                    'Total Return': f"{metrics.get('total_return', 0):.2%}",
                                    'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                                    'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                                    'Volatility': f"{metrics.get('volatility', 0):.2%}",
                                    'Type': 'Benchmark'
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # Style the dataframe
                            def highlight_portfolio(row):
                                if row['Type'] == 'Portfolio':
                                    return ['background-color: #e8f4fd'] * len(row)
                                else:
                                    return [''] * len(row)
                            
                            styled_df = comparison_df.style.apply(highlight_portfolio, axis=1)
                            st.dataframe(styled_df, use_container_width=True)
                        
                        # Superiority Analysis
                        if 'superiority_analysis' in comparison_results:
                            st.subheader("🎯 Superiority Analysis")
                            
                            superiority = comparison_results['superiority_analysis']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                is_superior = superiority.get('is_superior', False)
                                superiority_score = superiority.get('superiority_score', 0)
                                
                                if is_superior:
                                    st.markdown(
                                        f'<span class="superiority-badge superior">SUPERIOR PORTFOLIO</span>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'<span class="superiority-badge inferior">INFERIOR PORTFOLIO</span>',
                                        unsafe_allow_html=True
                                    )
                                
                                st.metric("Superiority Score", f"{superiority_score:.2f}")
                            
                            with col2:
                                criteria_met = superiority.get('criteria_met', {})
                                
                                st.markdown("**Criteria Met:**")
                                for criterion, met in criteria_met.items():
                                    status = "✅" if met else "❌"
                                    st.write(f"{status} {criterion.replace('_', ' ').title()}")
                        
                        # Relative Performance
                        if 'relative_performance' in comparison_results:
                            st.subheader("📊 Relative Performance")
                            
                            relative_perf = comparison_results['relative_performance']
                            
                            relative_data = []
                            for benchmark, metrics in relative_perf.items():
                                relative_data.append({
                                    'Benchmark': benchmark,
                                    'Alpha': f"{metrics.get('alpha', 0):.2%}",
                                    'Beta': f"{metrics.get('beta', 0):.2f}",
                                    'Information Ratio': f"{metrics.get('information_ratio', 0):.2f}",
                                    'Tracking Error': f"{metrics.get('tracking_error', 0):.2%}"
                                })
                            
                            if relative_data:
                                relative_df = pd.DataFrame(relative_data)
                                st.dataframe(relative_df, use_container_width=True)
                        
                        # Generate and display superiority report
                        try:
                            superiority_report = components['advanced_backtester'].generate_superiority_report(comparison_results)
                            
                            if superiority_report:
                                st.subheader("📋 Superiority Report")
                                
                                verdict = superiority_report.get('superiority_verdict', 'UNKNOWN')
                                st.markdown(f"**Overall Verdict:** `{verdict}`")
                                
                                if 'key_findings' in superiority_report:
                                    st.markdown("**Key Findings:**")
                                    for finding in superiority_report['key_findings']:
                                        st.write(f"• {finding}")
                                
                                if 'recommendations' in superiority_report:
                                    st.markdown("**Recommendations:**")
                                    for recommendation in superiority_report['recommendations']:
                                        st.write(f"• {recommendation}")
                        
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate superiority report: {str(e)}")
                    
                    else:
                        st.error("❌ Superiority analysis failed")
                        
                except Exception as e:
                    st.error(f"❌ Error during superiority analysis: {str(e)}")
                    st.exception(e)

# AI Assistant Page
elif page == "🤖 AI Assistant":
    st.header("AI Assistant")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your portfolio analysis..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare context from previous analyses
                    context = {
                        'screening_results': st.session_state.screening_results,
                        'ml_results': st.session_state.ml_results,
                        'backtest_results': st.session_state.backtest_results,
                        'advanced_backtest_results': st.session_state.advanced_backtest_results,
                        'portfolio_optimization_results': st.session_state.portfolio_optimization_results,
                        'superiority_analysis': st.session_state.superiority_analysis
                    }
                    
                    # Get AI response
                    response = components['kangro_agent'].get_investment_insights(prompt, context)
                    
                    if response:
                        st.write(response)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
                        st.write(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Kangro Capital Enhanced Platform v2.0 | Built with Streamlit | 
        <a href='https://github.com/kaljuvee/kangro-capital' target='_blank'>GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

