# 📈 Kangro Capital - Advanced Stock Screening & Backtesting Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-78.9%25-brightgreen.svg)](test_comprehensive_framework.py)

A comprehensive stock screening and backtesting platform that identifies high-performing stocks using advanced financial criteria, machine learning analysis, portfolio optimization, and AI-powered insights. The platform now includes advanced backtesting capabilities to determine if your portfolio selection strategy generates superior risk-adjusted returns.

## 🚀 Enhanced Features

### Core Functionality
- **Advanced Stock Screening**: 7-step fundamental analysis based on proven investment criteria
- **Machine Learning Analysis**: Regression, classification, and factor analysis for stock prediction
- **Comprehensive Backtesting**: Portfolio performance analysis with detailed metrics
- **🆕 Advanced Backtesting & Training**: ML-enhanced backtesting with predictive modeling
- **🆕 Portfolio Optimization**: 6 advanced optimization methods including Modern Portfolio Theory
- **🆕 Superiority Analysis**: Statistical validation of portfolio performance vs benchmarks
- **AI-Powered Insights**: Intelligent explanations and investment strategy generation
- **Interactive Web Interface**: Modern Streamlit-based dashboard with real-time updates

### Key Capabilities
- **Multi-Factor Screening**: ROE, Current Ratio, Gross Margin, Revenue Growth, and more
- **Outlier Detection**: Identify breakout stocks using statistical analysis
- **Performance Metrics**: 15+ comprehensive metrics including Sharpe, Sortino, Alpha, Beta, VaR
- **🆕 ML Training**: Train models on historical portfolio selections for future performance prediction
- **🆕 Risk Analysis**: Concentration risk, tail risk, correlation analysis, and stress testing
- **🆕 Benchmark Comparison**: Compare against multiple market indices with statistical significance
- **Data Visualization**: Interactive charts and graphs using Plotly
- **Export Functionality**: CSV, JSON, and PDF report generation
- **API Integration**: Real-time data from Polygon.io and market insights from Tavily

## 📊 Screening Criteria

The platform implements a proven 7-step stock screening process:

1. **Return on Equity (ROE) ≥ 15%**: Measures management effectiveness
2. **Current Ratio ≥ 1.5**: Ensures adequate liquidity
3. **Gross Margin ≥ 40%**: Indicates pricing power and efficiency
4. **Net Margin ≥ 10%**: Shows profitability after all expenses
5. **Revenue Growth ≥ 10% (5-year)**: Demonstrates consistent growth
6. **Debt-to-EBITDA ≤ 3**: Ensures manageable debt levels
7. **Composite Score Ranking**: Weighted scoring system for final selection

## 🆕 Advanced Backtesting & Portfolio Optimization

### ML-Enhanced Backtesting
The platform now includes sophisticated machine learning capabilities to train models on historical portfolio selections and predict future performance:

#### Training Capabilities
- **Historical Analysis**: Train on 6-36 months of portfolio selection data
- **Feature Engineering**: Fundamental metrics, technical indicators, market data, sentiment analysis
- **Model Selection**: Random Forest, Gradient Boosting, SVM, Neural Networks
- **Validation Methods**: Time series split, walk-forward analysis, cross-validation
- **Performance Prediction**: Forecast returns, volatility, and risk metrics

#### Advanced Metrics
- **Return Analysis**: Total return, annualized return, rolling returns, risk-adjusted returns
- **Risk Assessment**: Volatility, Value at Risk (VaR), Conditional VaR, maximum drawdown
- **Performance Ratios**: Sharpe, Sortino, Calmar, Information, Treynor ratios
- **Statistical Measures**: Alpha, beta, correlation, tracking error, information coefficient

### Portfolio Optimization Suite
Choose from 6 advanced optimization methods to construct optimal portfolios:

#### Optimization Methods
1. **Mean Variance Optimization**: Classic Markowitz efficient frontier approach
2. **Risk Parity**: Equal risk contribution from all assets
3. **Black-Litterman**: Bayesian approach incorporating market views
4. **Factor-Based Optimization**: Multi-factor model optimization
5. **ML-Enhanced Optimization**: Machine learning-driven weight allocation
6. **Hierarchical Risk Parity (HRP)**: Graph theory-based diversification

#### Risk Management
- **Concentration Limits**: Maximum position sizes and sector exposure
- **Risk Budgeting**: Allocate risk across different factors and assets
- **Stress Testing**: Portfolio performance under various market scenarios
- **Transaction Costs**: Incorporate realistic trading costs and constraints

### Superiority Analysis Framework
Determine if your portfolio strategy generates superior returns through comprehensive statistical analysis:

#### Benchmark Comparison
- **Market Indices**: SPY, QQQ, IWM, VTI, sector-specific ETFs
- **Custom Benchmarks**: User-defined benchmark portfolios
- **Peer Comparison**: Compare against similar investment strategies

#### Statistical Validation
- **Significance Testing**: T-tests, bootstrap confidence intervals
- **Performance Attribution**: Decompose returns into alpha, beta, and residual components
- **Superiority Scoring**: Composite score based on multiple performance dimensions
- **Confidence Intervals**: Statistical confidence in outperformance claims

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- API keys for Polygon.io, Tavily, and OpenAI (optional)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/kaljuvee/kangro-capital.git
   cd kangro-capital
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_core.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys
POLYGON_API_KEY=your_polygon_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
```

### API Key Setup

1. **Polygon.io**: Sign up at [polygon.io](https://polygon.io) for stock market data
2. **Tavily**: Get your API key at [tavily.com](https://tavily.com) for market insights
3. **OpenAI**: Obtain API access at [openai.com](https://openai.com) for AI features

## 📱 Enhanced Usage

### Web Interface

The Streamlit web interface now provides eight comprehensive sections:

#### 🏠 Dashboard
- Overview of screening results and key metrics
- Quick action buttons for common tasks
- Real-time status updates and platform health monitoring

#### 🔍 Stock Screening
- Configure screening parameters with advanced filters
- Run comprehensive fundamental analysis
- View detailed results with sorting and filtering options

#### 🧠 ML Analysis
- Train machine learning models on stock data
- Analyze feature importance and model performance
- Generate predictive insights with confidence intervals

#### 📊 Backtesting
- Test investment strategies on historical data
- Compare multiple approaches with detailed metrics
- Analyze risk-adjusted returns and drawdown periods

#### 🆕 🚀 Advanced Backtesting & Training
- **ML Model Training**: Train on historical portfolio selections
- **Performance Prediction**: Forecast future portfolio performance
- **Feature Analysis**: Identify key drivers of portfolio success
- **Confidence Scoring**: Quantify prediction reliability

#### 🆕 ⚖️ Portfolio Optimization
- **Multi-Method Optimization**: Choose from 6 optimization approaches
- **Risk Management**: Set constraints and risk limits
- **Expected Performance**: Preview optimized portfolio metrics
- **Method Comparison**: Compare optimization results side-by-side

#### 🆕 🏆 Superiority Analysis
- **Benchmark Comparison**: Test against multiple market indices
- **Statistical Significance**: Validate performance differences
- **Attribution Analysis**: Understand sources of outperformance
- **Superiority Scoring**: Comprehensive performance ranking

#### 🤖 AI Insights
- Get AI-powered explanations of screening and optimization results
- Generate personalized investment strategies based on risk tolerance
- Access market sentiment analysis and news interpretation

### Advanced Command Line Interface

Run enhanced components programmatically:

```python
from utils import ScreeningEngine, MLAnalyzer, BacktestEngine
from utils.advanced_backtester import AdvancedBacktester
from utils.portfolio_optimizer import PortfolioOptimizer

# Advanced backtesting with ML training
advanced_backtester = AdvancedBacktester()
training_results = advanced_backtester.train_ml_models(
    portfolio_data=historical_selections,
    training_period_months=12,
    model_type='random_forest',
    features=['fundamental_metrics', 'market_data']
)

# Portfolio optimization
optimizer = PortfolioOptimizer()
optimized_portfolio = optimizer.optimize_portfolio(
    returns_data=returns,
    method='mean_variance',
    risk_tolerance=0.15,
    constraints={'max_weight': 0.1, 'min_weight': 0.01}
)

# Superiority analysis
superiority_results = advanced_backtester.analyze_superiority(
    portfolio_returns=portfolio_returns,
    benchmark_returns=spy_returns,
    confidence_level=0.95
)
```

## 🧪 Enhanced Testing

Run the comprehensive test suite:

```bash
# Core functionality tests
python test_comprehensive.py

# Advanced backtesting tests
python test_comprehensive_framework.py

# Individual component tests
python test_advanced_backtesting.py
```

The enhanced test suite covers:
- API client functionality and error handling
- Screening engine accuracy and edge cases
- Machine learning model training and validation
- Advanced backtesting calculations and metrics
- Portfolio optimization algorithms
- Superiority analysis statistical methods
- AI agent integration and response quality
- Data persistence and caching
- Performance benchmarking

## 📈 Comprehensive Performance Metrics

The platform now calculates 15+ performance metrics across multiple categories:

### Return Metrics
- **Total Return**: Cumulative portfolio return over the period
- **Annualized Return**: Geometric mean annual return
- **Rolling Returns**: Performance over rolling time windows
- **Alpha**: Excess return vs. benchmark after adjusting for risk
- **Beta**: Portfolio sensitivity to market movements
- **Tracking Error**: Standard deviation of excess returns

### Risk Metrics
- **Volatility**: Standard deviation of returns (annualized)
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return to maximum drawdown ratio
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss at given confidence level
- **Conditional VaR**: Expected loss beyond VaR threshold

### Advanced Metrics
- **Information Ratio**: Active return to tracking error ratio
- **Treynor Ratio**: Return per unit of systematic risk
- **Win Rate**: Percentage of profitable periods
- **Win/Loss Ratio**: Average win to average loss ratio
- **Tail Ratio**: Ratio of 95th to 5th percentile returns
- **Skewness**: Asymmetry of return distribution
- **Kurtosis**: Tail heaviness of return distribution

## 🏗️ Enhanced Architecture

### Core Components

```
kangro-capital/
├── streamlit_app.py              # Enhanced main Streamlit application
├── utils/                        # Core utility modules
│   ├── polygon_client.py         # Polygon.io API integration
│   ├── tavily_client.py          # Tavily search integration
│   ├── data_fetcher.py           # Unified data fetching
│   ├── stock_screener.py         # Fundamental screening logic
│   ├── outlier_detector.py       # Statistical outlier detection
│   ├── screening_engine.py       # Comprehensive screening engine
│   ├── ml_analyzer.py            # Machine learning analysis
│   ├── factor_analyzer.py        # Factor analysis and PCA
│   ├── backtest_engine.py        # Basic backtesting framework
│   ├── portfolio_simulator.py    # Portfolio simulation
│   ├── 🆕 advanced_backtester.py # ML-enhanced backtesting
│   ├── 🆕 portfolio_optimizer.py # Multi-method optimization
│   └── kangro_agent.py           # AI agent integration
├── data/                         # Data storage directory
├── models/                       # Trained ML models
├── test-data/                    # 🆕 Comprehensive test reports
├── static/                       # Static web assets
└── templates/                    # HTML templates
```

### Enhanced Data Flow

1. **Data Ingestion**: Fetch real-time and historical data via APIs
2. **Screening**: Apply fundamental analysis criteria with advanced filters
3. **ML Analysis**: Train models and generate predictions with confidence intervals
4. **🆕 Advanced Training**: Train on historical portfolio selections
5. **🆕 Portfolio Optimization**: Apply multiple optimization methods
6. **Backtesting**: Simulate historical performance with enhanced metrics
7. **🆕 Superiority Analysis**: Statistical validation against benchmarks
8. **AI Insights**: Generate explanations and strategies
9. **Visualization**: Present results through interactive dashboard

## 🤖 Enhanced AI Integration

The platform includes advanced AI capabilities with expanded functionality:

### Features
- **Intelligent Explanations**: AI-powered analysis of screening and optimization results
- **Strategy Generation**: Personalized investment strategies based on risk tolerance and goals
- **Market Sentiment**: Real-time news analysis and sentiment scoring
- **Natural Language Queries**: Ask complex questions about portfolio performance
- **🆕 Performance Attribution**: AI-driven analysis of return sources
- **🆕 Risk Assessment**: Intelligent risk profiling and recommendations

### AI Models
- **OpenAI GPT-4**: For advanced natural language processing and generation
- **Tavily Search**: For real-time market information and news analysis
- **Custom ML Models**: Random Forest, Gradient Boosting, SVM, Neural Networks
- **🆕 Ensemble Methods**: Combining multiple models for improved predictions

## 📊 Enhanced Data Sources

### Market Data
- **Polygon.io**: Real-time and historical stock prices, fundamentals, options data
- **Yahoo Finance**: Backup data source for historical prices and dividends
- **Custom Data**: Support for CSV imports and manual data entry
- **🆕 Alternative Data**: ESG scores, analyst ratings, insider trading data

### News & Sentiment
- **Tavily API**: Real-time news aggregation and analysis
- **Financial News APIs**: Multiple sources for comprehensive coverage
- **🆕 Social Media**: Twitter sentiment and Reddit discussion analysis
- **🆕 Earnings Transcripts**: AI analysis of earnings call sentiment

## 🔒 Security & Privacy

- **API Key Management**: Secure environment variable storage with encryption
- **Data Encryption**: Sensitive data encrypted at rest and in transit
- **No Data Persistence**: User data not stored permanently without consent
- **Local Processing**: All analysis performed locally when possible
- **🆕 Audit Logging**: Comprehensive logging of all system activities
- **🆕 Access Controls**: Role-based access for multi-user environments

## 🚀 Enhanced Deployment

### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Production Deployment
The application can be deployed on various platforms with enhanced configurations:

- **Streamlit Cloud**: Direct deployment from GitHub with secrets management
- **Heroku**: Container-based deployment with auto-scaling
- **AWS/GCP/Azure**: Cloud platform deployment with managed services
- **Docker**: Containerized deployment with multi-stage builds
- **🆕 Kubernetes**: Scalable deployment with load balancing

### Enhanced Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_core.txt .
RUN pip install --no-cache-dir -r requirements_core.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📚 Enhanced Documentation

### API Reference
- [Polygon.io Documentation](https://polygon.io/docs)
- [Tavily API Documentation](https://docs.tavily.com)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [🆕 Advanced Backtesting Guide](docs/advanced_backtesting.md)
- [🆕 Portfolio Optimization Manual](docs/portfolio_optimization.md)

### Financial Concepts
- [Fundamental Analysis Guide](docs/fundamental_analysis.md)
- [Technical Indicators Reference](docs/technical_indicators.md)
- [Risk Management Principles](docs/risk_management.md)
- [🆕 Modern Portfolio Theory](docs/modern_portfolio_theory.md)
- [🆕 Statistical Significance Testing](docs/statistical_testing.md)

### User Guides
- [🆕 Getting Started with Advanced Backtesting](docs/backtesting_quickstart.md)
- [🆕 Portfolio Optimization Best Practices](docs/optimization_guide.md)
- [🆕 Interpreting Superiority Analysis](docs/superiority_guide.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive docstrings to all functions
- Include unit tests for new features with >80% coverage
- Update documentation and README for new features
- 🆕 Include performance benchmarks for optimization algorithms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Benjamin Graham**: Value investing principles and fundamental analysis
- **Warren Buffett**: Investment philosophy inspiration and long-term thinking
- **Joel Greenblatt**: Magic Formula methodology and quantitative screening
- **Harry Markowitz**: Modern Portfolio Theory and optimization foundations
- **🆕 William Sharpe**: CAPM model and performance measurement
- **🆕 Eugene Fama**: Efficient Market Hypothesis and factor models
- **Open Source Community**: Libraries and tools that make this possible

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kaljuvee/kangro-capital/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kaljuvee/kangro-capital/discussions)
- **Documentation**: [Wiki](https://github.com/kaljuvee/kangro-capital/wiki)
- **Email**: support@kangrocapital.com

## 🔄 Changelog

### Version 2.0.0 (2025-07-13) - 🆕 Advanced Backtesting Release
- **🚀 Advanced Backtesting & Training**: ML-enhanced backtesting with predictive modeling
- **⚖️ Portfolio Optimization**: 6 optimization methods including Modern Portfolio Theory
- **🏆 Superiority Analysis**: Statistical validation against benchmarks
- **📊 Enhanced Metrics**: 15+ comprehensive performance metrics
- **🧪 Comprehensive Testing**: 78.9% test coverage with detailed reporting
- **📚 Enhanced Documentation**: Complete guides for new features
- **🔧 Improved Architecture**: Modular design with advanced components

### Version 1.0.0 (2025-01-13)
- Initial release
- Complete stock screening functionality
- Machine learning integration
- AI-powered insights
- Basic backtesting capabilities
- Streamlit web interface

---

**Disclaimer**: This software is for educational and research purposes only. It does not constitute financial advice. Past performance does not guarantee future results. The superiority analysis and ML predictions are based on historical data and may not reflect future market conditions. Always consult with qualified financial professionals before making investment decisions.

**Risk Warning**: All investments carry risk of loss. The advanced backtesting and optimization features are tools to aid analysis but cannot eliminate investment risk. Users should understand the limitations of historical backtesting and the potential for model overfitting.

