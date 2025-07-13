"""
Dashboard Page - Platform Overview and Status
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

def show_dashboard():
    st.markdown("## 📊 Platform Dashboard")
    
    # Platform status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
        st.markdown("### Platform Status")
        st.markdown("🟢 **Online**")
        st.markdown("✅ Fully Operational")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card info-metric">', unsafe_allow_html=True)
        st.markdown("### AI Features")
        st.markdown("🤖 **Active**")
        st.markdown("✅ OpenAI + Tavily")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card warning-metric">', unsafe_allow_html=True)
        st.markdown("### Data Sources")
        st.markdown("📊 **Live**")
        st.markdown("✅ Real-time feeds")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card primary-metric">', unsafe_allow_html=True)
        st.markdown("### Analysis Tools")
        st.markdown("🧠 **Advanced**")
        st.markdown("✅ ML + AI Powered")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze AAPL", use_container_width=True):
            st.session_state.quick_analysis_symbol = "AAPL"
            st.success("✅ AAPL analysis queued!")
    
    with col2:
        if st.button("📊 Run AI Screening", use_container_width=True):
            st.session_state.run_screening = True
            st.success("✅ AI screening initiated!")
    
    with col3:
        if st.button("🧠 Get Market Insights", use_container_width=True):
            st.session_state.get_insights = True
            st.success("✅ Market insights loading!")
    
    # Recent Activity
    st.markdown("## 📈 Recent Activity")
    
    # Create sample activity data
    activity_data = {
        'Time': ['10:30 AM', '10:15 AM', '10:00 AM', '09:45 AM', '09:30 AM'],
        'Action': ['Stock Screening', 'Portfolio Analysis', 'ML Training', 'Market Research', 'Data Update'],
        'Status': ['Completed', 'Completed', 'In Progress', 'Completed', 'Completed'],
        'Details': ['Found 12 qualifying stocks', 'Portfolio optimized', 'Training Random Forest model', 'Analyzed 50 news articles', 'Updated 500+ stock prices']
    }
    
    activity_df = pd.DataFrame(activity_data)
    
    # Create activity chart
    fig = px.timeline(
        activity_df, 
        x_start='Time', 
        x_end='Time',
        y='Action',
        color='Status',
        title="Platform Activity Timeline",
        color_discrete_map={'Completed': '#28a745', 'In Progress': '#ffc107'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Screening Performance")
        
        # Sample screening metrics
        metrics_data = {
            'Metric': ['Stocks Analyzed', 'Qualifying Stocks', 'Success Rate', 'Avg Processing Time'],
            'Value': [1247, 89, '7.1%', '2.3s'],
            'Change': ['+12%', '+8%', '+0.3%', '-15%']
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create metrics chart
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_df['Metric'],
                y=[1247, 89, 7.1, 2.3],
                text=metrics_df['Value'],
                textposition='auto',
                marker_color=['#007bff', '#28a745', '#ffc107', '#dc3545']
            )
        ])
        fig.update_layout(
            title="Key Performance Indicators",
            yaxis_title="Value",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Portfolio Insights")
        
        # Sample portfolio data
        portfolio_data = {
            'Sector': ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy'],
            'Allocation': [35, 25, 20, 15, 5],
            'Performance': [12.5, 8.3, 6.7, 4.2, -2.1]
        }
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        # Create portfolio pie chart
        fig = px.pie(
            portfolio_df, 
            values='Allocation', 
            names='Sector',
            title="Portfolio Allocation by Sector",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # System Health
    st.markdown("## 🔧 System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Response Time", "145ms", "-23ms")
    
    with col2:
        st.metric("Data Freshness", "Real-time", "0s delay")
    
    with col3:
        st.metric("ML Model Accuracy", "94.2%", "+1.3%")
    
    with col4:
        st.metric("Active Users", "1,247", "+89")
    
    # Recent Alerts
    st.markdown("## 🚨 Recent Alerts")
    
    alerts = [
        {"time": "10:45 AM", "type": "info", "message": "New high-quality stocks detected in screening"},
        {"time": "10:30 AM", "type": "success", "message": "Portfolio optimization completed successfully"},
        {"time": "10:15 AM", "type": "warning", "message": "Market volatility increased - review risk settings"},
        {"time": "10:00 AM", "type": "info", "message": "ML model training completed with 94.2% accuracy"}
    ]
    
    for alert in alerts:
        icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}[alert["type"]]
        color = {"info": "#17a2b8", "success": "#28a745", "warning": "#ffc107", "error": "#dc3545"}[alert["type"]]
        
        st.markdown(f"""
        <div style="padding: 10px; border-left: 4px solid {color}; background-color: rgba(0,0,0,0.05); margin: 5px 0;">
            <strong>{icon} {alert['time']}</strong> - {alert['message']}
        </div>
        """, unsafe_allow_html=True)

