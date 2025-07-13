"""
Kangro Capital - Enhanced Stock Screening & AI-Powered Investment Platform
Refactored with Pages Structure and Improved UX
"""
import streamlit as st
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import page modules
from pages.dashboard import show_dashboard
from pages.stock_screening import show_stock_screening
from pages.ml_analysis import show_ml_analysis
from pages.backtesting import show_backtesting
from pages.advanced_backtesting import show_advanced_backtesting

# Page configuration
st.set_page_config(
    page_title="Kangro Capital",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .success-metric { border-left: 4px solid #28a745; }
    .info-metric { border-left: 4px solid #17a2b8; }
    .warning-metric { border-left: 4px solid #ffc107; }
    .primary-metric { border-left: 4px solid #007bff; }
    
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page_history' not in st.session_state:
    st.session_state.page_history = []

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Kangro Capital</h1>
    <p>Advanced Stock Screening & AI-Powered Investment Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.markdown("Choose a page")
    
    # Page selection
    page = st.selectbox(
        "Select Page",
        [
            "🏠 Dashboard",
            "🔍 Stock Screening", 
            "🧠 ML Analysis",
            "📊 Backtesting",
            "🚀 Advanced Backtesting & Training",
            "⚖️ Portfolio Optimization",
            "🏆 Superiority Analysis",
            "🤖 AI Insights"
        ],
        index=0
    )
    
    # Add page to history
    if page not in st.session_state.page_history:
        st.session_state.page_history.append(page)
        if len(st.session_state.page_history) > 5:
            st.session_state.page_history.pop(0)
    
    # Quick navigation
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔍 Quick Screen", use_container_width=True):
        st.session_state.quick_screen = True
        
    if st.button("📊 Sample Analysis", use_container_width=True):
        st.session_state.sample_analysis = True
        
    if st.button("🧠 Train Models", use_container_width=True):
        st.session_state.train_models = True
    
    # Recent pages
    if len(st.session_state.page_history) > 1:
        st.markdown("---")
        st.markdown("### 📚 Recent Pages")
        for recent_page in reversed(st.session_state.page_history[-4:]):
            if recent_page != page:
                if st.button(recent_page, key=f"recent_{recent_page}", use_container_width=True):
                    st.session_state.selected_page = recent_page
                    st.rerun()
    
    # System status
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    st.success("🟢 Platform Online")
    st.info("🤖 AI Models Active")
    st.warning("📊 Data Sources Live")

# Main content area
try:
    if page == "🏠 Dashboard":
        show_dashboard()
        
    elif page == "🔍 Stock Screening":
        show_stock_screening()
        
    elif page == "🧠 ML Analysis":
        show_ml_analysis()
        
    elif page == "📊 Backtesting":
        show_backtesting()
        
    elif page == "🚀 Advanced Backtesting & Training":
        show_advanced_backtesting()
        
    elif page == "⚖️ Portfolio Optimization":
        st.markdown("## ⚖️ Portfolio Optimization")
        st.info("🚧 Portfolio Optimization page is being refactored. Coming soon!")
        
    elif page == "🏆 Superiority Analysis":
        st.markdown("## 🏆 Superiority Analysis")
        st.info("🚧 Superiority Analysis page is being refactored. Coming soon!")
        
    elif page == "🤖 AI Insights":
        st.markdown("## 🤖 AI Insights")
        st.info("🚧 AI Insights page is being refactored. Coming soon!")
        
except Exception as e:
    st.error(f"❌ Error loading page: {str(e)}")
    st.markdown("### 🔧 Troubleshooting")
    st.markdown("""
    If you're seeing this error, try:
    1. Refreshing the page
    2. Checking your internet connection
    3. Clearing browser cache
    4. Contacting support if the issue persists
    """)
    
    # Show error details in expander
    with st.expander("🐛 Error Details"):
        st.code(str(e))

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Data Sources**")
    st.markdown("- Yahoo Finance")
    st.markdown("- Polygon.io")
    st.markdown("- Alpha Vantage")

with col2:
    st.markdown("**🤖 AI Models**")
    st.markdown("- OpenAI GPT-4")
    st.markdown("- Scikit-learn")
    st.markdown("- TensorFlow")

with col3:
    st.markdown("**📈 Features**")
    st.markdown("- Stock Screening")
    st.markdown("- ML Analysis")
    st.markdown("- Portfolio Optimization")

st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>© 2024 Kangro Capital. Advanced Stock Screening & AI-Powered Investment Platform.</p>
    <p>Built with Streamlit • Powered by AI • Data-Driven Insights</p>
</div>
""", unsafe_allow_html=True)

