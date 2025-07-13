"""
Kangro Capital - Stock Screening and Backtesting Platform
Main Flask application entry point
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # API Keys
    app.config['POLYGON_API_KEY'] = os.getenv('POLYGON_API_KEY')
    app.config['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    
    # Application Settings
    app.config['DEFAULT_STOCK_COUNT'] = int(os.getenv('DEFAULT_STOCK_COUNT', 10))
    app.config['DEFAULT_LOOKBACK_YEARS'] = int(os.getenv('DEFAULT_LOOKBACK_YEARS', 3))
    app.config['MAX_STOCK_COUNT'] = int(os.getenv('MAX_STOCK_COUNT', 50))
    
    # Routes
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('index.html')
    
    @app.route('/screening')
    def screening():
        """Stock screening page"""
        return render_template('screening.html')
    
    @app.route('/backtesting')
    def backtesting():
        """Backtesting page"""
        return render_template('backtesting.html')
    
    @app.route('/analysis')
    def analysis():
        """Analysis and visualization page"""
        return render_template('analysis.html')
    
    @app.route('/api/screen', methods=['POST'])
    def api_screen():
        """API endpoint for stock screening"""
        try:
            data = request.get_json()
            # TODO: Implement screening logic
            return jsonify({'status': 'success', 'message': 'Screening functionality coming soon'})
        except Exception as e:
            logger.error(f"Error in screening: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/backtest', methods=['POST'])
    def api_backtest():
        """API endpoint for backtesting"""
        try:
            data = request.get_json()
            # TODO: Implement backtesting logic
            return jsonify({'status': 'success', 'message': 'Backtesting functionality coming soon'})
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/health')
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'apis_configured': {
                'polygon': bool(app.config.get('POLYGON_API_KEY')),
                'tavily': bool(app.config.get('TAVILY_API_KEY')),
                'openai': bool(app.config.get('OPENAI_API_KEY'))
            }
        })
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

