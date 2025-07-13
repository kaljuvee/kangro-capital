# 📈 Kangro Capital - Advanced Stock Screening & Backtesting Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-87.5%25-brightgreen.svg)](test_comprehensive.py)

A comprehensive stock screening and backtesting platform that identifies high-performing stocks using advanced financial criteria, machine learning analysis, and AI-powered insights.

## 🚀 Features

### Core Functionality
- **Advanced Stock Screening**: 7-step fundamental analysis based on proven investment criteria
- **Machine Learning Analysis**: Regression, classification, and factor analysis for stock prediction
- **Comprehensive Backtesting**: Portfolio performance analysis with detailed metrics
- **AI-Powered Insights**: Intelligent explanations and investment strategy generation
- **Interactive Web Interface**: Modern Streamlit-based dashboard with real-time updates

### Key Capabilities
- **Multi-Factor Screening**: ROE, Current Ratio, Gross Margin, Revenue Growth, and more
- **Outlier Detection**: Identify breakout stocks using statistical analysis
- **Performance Metrics**: Sharpe ratio, Sortino ratio, Maximum Drawdown, Alpha, Beta
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

## 📱 Usage

### Web Interface

The Streamlit web interface provides five main sections:

#### 🏠 Dashboard
- Overview of screening results and key metrics
- Quick action buttons for common tasks
- Real-time status updates

#### 🔍 Stock Screening
- Configure screening parameters
- Run comprehensive analysis
- View detailed results with filtering options

#### 🧠 ML Analysis
- Train machine learning models on stock data
- Analyze feature importance
- Generate predictive insights

#### 📊 Backtesting
- Test investment strategies on historical data
- Compare multiple approaches
- Analyze risk-adjusted returns

#### 🤖 AI Insights
- Get AI-powered explanations of screening results
- Generate personalized investment strategies
- Access market sentiment analysis

### Command Line Interface

Run individual components programmatically:

```python
from utils import ScreeningEngine, MLAnalyzer, BacktestEngine

# Stock screening
screening_engine = ScreeningEngine()
results = screening_engine.run_comprehensive_screening({
    'top_n_stocks': 10,
    'lookback_years': 2
})

# Machine learning analysis
ml_analyzer = MLAnalyzer()
ml_results = ml_analyzer.train_regression_models(stock_data)

# Backtesting
backtest_engine = BacktestEngine()
backtest_results = backtest_engine.run_backtest(
    stock_selections=selections,
    price_data=prices,
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_comprehensive.py
```

The test suite covers:
- API client functionality
- Screening engine accuracy
- Machine learning model training
- Backtesting calculations
- AI agent integration
- Data persistence
- Error handling

## 📈 Performance Metrics

The platform calculates comprehensive performance metrics:

### Return Metrics
- **Total Return**: Cumulative portfolio return
- **Annualized Return**: Geometric mean annual return
- **Alpha**: Excess return vs. benchmark
- **Beta**: Portfolio sensitivity to market movements

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return to maximum drawdown ratio

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Win/Loss Ratio**: Average win to average loss ratio
- **Information Ratio**: Active return to tracking error

## 🏗️ Architecture

### Core Components

```
kangro-capital/
├── streamlit_app.py          # Main Streamlit application
├── utils/                    # Core utility modules
│   ├── polygon_client.py     # Polygon.io API integration
│   ├── tavily_client.py      # Tavily search integration
│   ├── data_fetcher.py       # Unified data fetching
│   ├── stock_screener.py     # Fundamental screening logic
│   ├── outlier_detector.py   # Statistical outlier detection
│   ├── screening_engine.py   # Comprehensive screening engine
│   ├── ml_analyzer.py        # Machine learning analysis
│   ├── factor_analyzer.py    # Factor analysis and PCA
│   ├── backtest_engine.py    # Backtesting framework
│   ├── portfolio_simulator.py # Portfolio simulation
│   └── kangro_agent.py       # AI agent integration
├── data/                     # Data storage directory
├── models/                   # Trained ML models
├── static/                   # Static web assets
└── templates/                # HTML templates
```

### Data Flow

1. **Data Ingestion**: Fetch real-time and historical data via APIs
2. **Screening**: Apply fundamental analysis criteria
3. **ML Analysis**: Train models and generate predictions
4. **Backtesting**: Simulate historical performance
5. **AI Insights**: Generate explanations and strategies
6. **Visualization**: Present results through interactive dashboard

## 🤖 AI Integration

The platform includes advanced AI capabilities:

### Features
- **Intelligent Explanations**: AI-powered analysis of screening results
- **Strategy Generation**: Personalized investment strategies based on risk tolerance
- **Market Sentiment**: Real-time news analysis and sentiment scoring
- **Natural Language Queries**: Ask questions about your portfolio in plain English

### AI Models
- **OpenAI GPT**: For natural language processing and generation
- **Tavily Search**: For real-time market information and news analysis
- **Custom ML Models**: For stock prediction and factor analysis

## 📊 Data Sources

### Market Data
- **Polygon.io**: Real-time and historical stock prices, fundamentals
- **Yahoo Finance**: Backup data source for historical prices
- **Custom Data**: Support for CSV imports and manual data entry

### News & Sentiment
- **Tavily API**: Real-time news aggregation and analysis
- **Financial News APIs**: Multiple sources for comprehensive coverage

## 🔒 Security & Privacy

- **API Key Management**: Secure environment variable storage
- **Data Encryption**: Sensitive data encrypted at rest
- **No Data Persistence**: User data not stored permanently
- **Local Processing**: All analysis performed locally

## 🚀 Deployment

### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Production Deployment
The application can be deployed on various platforms:

- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Container-based deployment
- **AWS/GCP/Azure**: Cloud platform deployment
- **Docker**: Containerized deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_core.txt .
RUN pip install -r requirements_core.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📚 Documentation

### API Reference
- [Polygon.io Documentation](https://polygon.io/docs)
- [Tavily API Documentation](https://docs.tavily.com)
- [OpenAI API Documentation](https://platform.openai.com/docs)

### Financial Concepts
- [Fundamental Analysis Guide](docs/fundamental_analysis.md)
- [Technical Indicators Reference](docs/technical_indicators.md)
- [Risk Management Principles](docs/risk_management.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Benjamin Graham**: Value investing principles
- **Warren Buffett**: Investment philosophy inspiration
- **Joel Greenblatt**: Magic Formula methodology
- **Open Source Community**: Libraries and tools that make this possible

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kaljuvee/kangro-capital/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kaljuvee/kangro-capital/discussions)
- **Email**: support@kangrocapital.com

## 🔄 Changelog

### Version 1.0.0 (2025-01-13)
- Initial release
- Complete stock screening functionality
- Machine learning integration
- AI-powered insights
- Comprehensive backtesting
- Streamlit web interface

---

**Disclaimer**: This software is for educational and research purposes only. It does not constitute financial advice. Always consult with qualified financial professionals before making investment decisions.

