# Contributing to Kangro Capital

Thank you for your interest in contributing to Kangro Capital! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use the issue template** when creating new issues
3. **Provide detailed information** including:
   - Steps to reproduce the problem
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Screenshots or error messages if applicable

### Suggesting Features

1. **Check the roadmap** to see if the feature is already planned
2. **Open a feature request** using the appropriate template
3. **Describe the use case** and why the feature would be valuable
4. **Consider implementation complexity** and potential alternatives

### Code Contributions

#### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/kangro-capital.git
   cd kangro-capital
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements_core.txt
   pip install -r requirements_dev.txt  # If available
   ```

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following the coding standards below
3. **Add tests** for new functionality
4. **Run the test suite**:
   ```bash
   python test_comprehensive.py
   ```
5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request** on GitHub

## 📝 Coding Standards

### Python Style Guide

- **Follow PEP 8** for Python code style
- **Use type hints** where appropriate
- **Maximum line length**: 88 characters (Black formatter standard)
- **Import organization**:
  ```python
  # Standard library imports
  import os
  import sys
  
  # Third-party imports
  import pandas as pd
  import numpy as np
  
  # Local imports
  from utils import ScreeningEngine
  ```

### Documentation

- **Add docstrings** to all public functions and classes:
  ```python
  def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
      """
      Calculate the Sharpe ratio for a series of returns.
      
      Args:
          returns: Series of portfolio returns
          risk_free_rate: Risk-free rate for calculation (default: 0.02)
      
      Returns:
          Sharpe ratio as a float
      
      Raises:
          ValueError: If returns series is empty
      """
  ```
- **Update README.md** if adding new features
- **Add inline comments** for complex logic

### Testing

- **Write unit tests** for new functions
- **Maintain test coverage** above 80%
- **Test edge cases** and error conditions
- **Use descriptive test names**:
  ```python
  def test_screening_engine_handles_empty_data():
      """Test that screening engine gracefully handles empty input data."""
  ```

### Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring
- `style:` for formatting changes

Examples:
```
feat: Add AI-powered investment strategy generation
fix: Resolve division by zero in Sharpe ratio calculation
docs: Update installation instructions for Windows
test: Add comprehensive tests for backtesting engine
```

## 🏗️ Project Structure

Understanding the codebase structure:

```
kangro-capital/
├── streamlit_app.py          # Main Streamlit application
├── utils/                    # Core utility modules
│   ├── __init__.py          # Package initialization
│   ├── polygon_client.py    # Polygon.io API client
│   ├── screening_engine.py  # Stock screening logic
│   ├── ml_analyzer.py       # Machine learning analysis
│   ├── backtest_engine.py   # Backtesting framework
│   └── kangro_agent.py      # AI agent integration
├── tests/                   # Test files
├── docs/                    # Documentation
├── data/                    # Data storage
└── requirements_core.txt    # Core dependencies
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run comprehensive test suite
python test_comprehensive.py

# Run specific component tests
python -m pytest tests/test_screening.py

# Run with coverage
python -m pytest --cov=utils tests/
```

### Writing Tests

- **Test public interfaces** rather than implementation details
- **Use fixtures** for common test data
- **Mock external API calls** to avoid dependencies
- **Test both success and failure cases**

Example test structure:
```python
import pytest
from utils.screening_engine import ScreeningEngine

class TestScreeningEngine:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.engine = ScreeningEngine()
    
    def test_screening_with_valid_parameters(self):
        """Test screening with valid input parameters."""
        params = {'top_n_stocks': 10, 'lookback_years': 2}
        results = self.engine.run_comprehensive_screening(params)
        
        assert 'recommendations' in results
        assert len(results['recommendations']) <= 10
    
    def test_screening_with_invalid_parameters(self):
        """Test screening handles invalid parameters gracefully."""
        with pytest.raises(ValueError):
            self.engine.run_comprehensive_screening({})
```

## 🚀 Feature Development

### Adding New Screening Criteria

1. **Update `stock_screener.py`** with new criteria logic
2. **Add parameter validation** in screening engine
3. **Update the Streamlit interface** to expose new parameters
4. **Add comprehensive tests** for the new criteria
5. **Update documentation** with criteria explanation

### Adding New ML Models

1. **Extend `ml_analyzer.py`** with new model class
2. **Implement training and prediction methods**
3. **Add model evaluation metrics**
4. **Update the web interface** to display new model results
5. **Add model-specific tests**

### Adding New Visualizations

1. **Create visualization functions** using Plotly
2. **Add to appropriate Streamlit pages**
3. **Ensure responsive design** for mobile devices
4. **Test with various data sizes**
5. **Document visualization parameters**

## 🐛 Bug Fixes

### Debugging Process

1. **Reproduce the issue** locally
2. **Add logging** to understand the problem
3. **Write a failing test** that demonstrates the bug
4. **Fix the issue** while keeping the test passing
5. **Verify the fix** doesn't break existing functionality

### Common Issues

- **API rate limiting**: Implement proper retry logic
- **Data quality**: Add validation and cleaning steps
- **Performance**: Profile code and optimize bottlenecks
- **Memory usage**: Use generators for large datasets

## 📊 Performance Considerations

### Optimization Guidelines

- **Use vectorized operations** with NumPy/Pandas
- **Cache expensive computations** using `@lru_cache`
- **Implement lazy loading** for large datasets
- **Profile code** using `cProfile` for bottlenecks
- **Consider async operations** for I/O-bound tasks

### Memory Management

- **Use generators** for processing large datasets
- **Clear unused variables** in long-running functions
- **Monitor memory usage** during development
- **Implement data chunking** for very large datasets

## 🔒 Security Guidelines

### API Key Management

- **Never commit API keys** to version control
- **Use environment variables** for sensitive data
- **Implement key rotation** procedures
- **Validate API responses** before processing

### Data Privacy

- **Don't store user data** permanently
- **Implement data anonymization** where needed
- **Follow GDPR guidelines** for EU users
- **Secure data transmission** using HTTPS

## 📈 Performance Benchmarks

### Target Performance Metrics

- **Screening time**: < 30 seconds for 1000 stocks
- **ML training time**: < 5 minutes for standard models
- **Backtesting time**: < 2 minutes for 5-year period
- **Web interface response**: < 3 seconds for page loads

### Monitoring

- **Add timing decorators** to critical functions
- **Log performance metrics** in production
- **Set up alerts** for performance degradation
- **Regular performance reviews** with each release

## 🎯 Roadmap

### Short-term Goals (Next Release)

- [ ] Add more technical indicators
- [ ] Implement portfolio optimization
- [ ] Add cryptocurrency screening
- [ ] Improve mobile interface

### Medium-term Goals (Next Quarter)

- [ ] Add options trading analysis
- [ ] Implement sector rotation strategies
- [ ] Add ESG scoring integration
- [ ] Create mobile app

### Long-term Goals (Next Year)

- [ ] Add real-time trading integration
- [ ] Implement social sentiment analysis
- [ ] Add alternative data sources
- [ ] Create API for third-party integration

## 💬 Communication

### Getting Help

- **GitHub Discussions**: For general questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Code Reviews**: For feedback on pull requests
- **Documentation**: Check existing docs before asking

### Community Guidelines

- **Be respectful** and professional in all interactions
- **Help others** when you can
- **Share knowledge** and best practices
- **Give constructive feedback** on contributions

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Hall of Fame** for major features
- **Special badges** for consistent contributors

Thank you for contributing to Kangro Capital! 🚀

