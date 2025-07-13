"""
Machine Learning Analysis Module for Stock Prediction and Feature Importance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib
import os

logger = logging.getLogger(__name__)

class MLAnalyzer:
    """
    Machine Learning analyzer for stock prediction and feature importance analysis
    """
    
    def __init__(self):
        """Initialize the ML analyzer"""
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.label_encoder = LabelEncoder()
        
        # Model configurations
        self.regression_models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0)
        }
        
        self.classification_models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest_clf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svc': SVC(kernel='rbf', C=1.0, random_state=42)
        }
    
    def prepare_features(self, price_data: Dict[str, pd.DataFrame], 
                        financial_metrics: Dict[str, Dict]) -> pd.DataFrame:
        """
        Prepare feature matrix for machine learning
        
        Args:
            price_data: Dictionary of price DataFrames
            financial_metrics: Dictionary of financial metrics
        
        Returns:
            DataFrame with features and targets
        """
        features_list = []
        
        for symbol, df in price_data.items():
            if df.empty or len(df) < 100:
                continue
                
            try:
                # Calculate technical indicators
                features = self._calculate_technical_features(df)
                
                # Add fundamental features if available
                if symbol in financial_metrics:
                    fund_features = self._extract_fundamental_features(financial_metrics[symbol])
                    features.update(fund_features)
                
                # Add target variables
                targets = self._calculate_targets(df)
                features.update(targets)
                
                features['symbol'] = symbol
                features_list.append(features)
                
            except Exception as e:
                logger.error(f"Error preparing features for {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(features_list)
    
    def _calculate_technical_features(self, df: pd.DataFrame) -> Dict:
        """Calculate technical analysis features"""
        features = {}
        
        try:
            # Price-based features
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volatility measures
            df['volatility_20'] = df['returns'].rolling(20).std()
            df['volatility_50'] = df['returns'].rolling(50).std()
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                
                # On-Balance Volume
                df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
                df['obv_sma'] = df['obv'].rolling(20).mean()
            
            # Price momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Extract latest values as features
            latest = df.iloc[-1]
            
            # Price relative to moving averages
            features['price_vs_sma5'] = (latest['close'] / latest['sma_5'] - 1) if not pd.isna(latest['sma_5']) else 0
            features['price_vs_sma10'] = (latest['close'] / latest['sma_10'] - 1) if not pd.isna(latest['sma_10']) else 0
            features['price_vs_sma20'] = (latest['close'] / latest['sma_20'] - 1) if not pd.isna(latest['sma_20']) else 0
            features['price_vs_sma50'] = (latest['close'] / latest['sma_50'] - 1) if not pd.isna(latest['sma_50']) else 0
            
            # Technical indicators
            features['rsi'] = latest['rsi'] if not pd.isna(latest['rsi']) else 50
            features['macd'] = latest['macd'] if not pd.isna(latest['macd']) else 0
            features['macd_signal'] = latest['macd_signal'] if not pd.isna(latest['macd_signal']) else 0
            features['bb_position'] = latest['bb_position'] if not pd.isna(latest['bb_position']) else 0.5
            features['bb_width'] = latest['bb_width'] if not pd.isna(latest['bb_width']) else 0
            
            # Volatility
            features['volatility_20'] = latest['volatility_20'] if not pd.isna(latest['volatility_20']) else 0
            features['volatility_50'] = latest['volatility_50'] if not pd.isna(latest['volatility_50']) else 0
            
            # Momentum
            features['momentum_5'] = latest['momentum_5'] if not pd.isna(latest['momentum_5']) else 0
            features['momentum_10'] = latest['momentum_10'] if not pd.isna(latest['momentum_10']) else 0
            features['momentum_20'] = latest['momentum_20'] if not pd.isna(latest['momentum_20']) else 0
            
            # Volume features
            if 'volume' in df.columns:
                features['volume_ratio'] = latest['volume_ratio'] if not pd.isna(latest['volume_ratio']) else 1
                features['obv_trend'] = (latest['obv'] / latest['obv_sma'] - 1) if not pd.isna(latest['obv_sma']) and latest['obv_sma'] != 0 else 0
            else:
                features['volume_ratio'] = 1
                features['obv_trend'] = 0
            
        except Exception as e:
            logger.error(f"Error calculating technical features: {str(e)}")
        
        return features
    
    def _extract_fundamental_features(self, financial_metrics: Dict) -> Dict:
        """Extract fundamental analysis features"""
        features = {}
        
        try:
            # Financial ratios and metrics
            features['roe'] = financial_metrics.get('roe', 0)
            features['current_ratio'] = financial_metrics.get('current_ratio', 1)
            features['debt_to_ebitda'] = financial_metrics.get('debt_to_ebitda', 0)
            features['gross_margin'] = financial_metrics.get('gross_margin', 0)
            features['net_margin'] = financial_metrics.get('net_margin', 0)
            features['revenue_growth'] = financial_metrics.get('revenue_growth_5y', 0)
            features['net_income_growth'] = financial_metrics.get('net_income_growth', 0)
            features['cash_flow_growth'] = financial_metrics.get('cash_flow_growth', 0)
            features['revenue_consistency'] = financial_metrics.get('revenue_consistency', 0)
            features['income_consistency'] = financial_metrics.get('net_income_consistency', 0)
            features['cashflow_consistency'] = financial_metrics.get('cash_flow_consistency', 0)
            
        except Exception as e:
            logger.error(f"Error extracting fundamental features: {str(e)}")
        
        return features
    
    def _calculate_targets(self, df: pd.DataFrame) -> Dict:
        """Calculate target variables for prediction"""
        targets = {}
        
        try:
            # Future returns (targets for regression)
            if len(df) >= 30:
                current_price = df['close'].iloc[-1]
                
                # Future return targets (if we had future data)
                # For now, we'll use historical data as proxy
                targets['return_1w'] = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) if len(df) >= 5 else 0
                targets['return_1m'] = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) if len(df) >= 21 else 0
                targets['return_3m'] = (df['close'].iloc[-1] / df['close'].iloc[-63] - 1) if len(df) >= 63 else 0
                
                # Classification targets
                # Performance categories based on returns
                if targets['return_1m'] > 0.1:  # >10% return
                    targets['performance_category'] = 'HIGH'
                elif targets['return_1m'] > 0.05:  # 5-10% return
                    targets['performance_category'] = 'MEDIUM'
                elif targets['return_1m'] > -0.05:  # -5% to 5% return
                    targets['performance_category'] = 'LOW'
                else:  # <-5% return
                    targets['performance_category'] = 'NEGATIVE'
                
                # Binary classification: outperformer vs underperformer
                targets['is_outperformer'] = 1 if targets['return_1m'] > 0.05 else 0
            
        except Exception as e:
            logger.error(f"Error calculating targets: {str(e)}")
        
        return targets
    
    def train_regression_models(self, features_df: pd.DataFrame, 
                              target_column: str = 'return_1m') -> Dict:
        """
        Train regression models to predict stock returns
        
        Args:
            features_df: DataFrame with features and targets
            target_column: Target column name
        
        Returns:
            Dictionary with model performance results
        """
        results = {}
        
        try:
            if features_df.empty or target_column not in features_df.columns:
                return results
            
            # Prepare features and target
            feature_cols = [col for col in features_df.columns 
                           if col not in ['symbol', 'return_1w', 'return_1m', 'return_3m', 
                                        'performance_category', 'is_outperformer']]
            
            X = features_df[feature_cols].fillna(0)
            y = features_df[target_column].fillna(0)
            
            if len(X) < 10:  # Need minimum samples
                return results
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            for model_name, model in self.regression_models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    
                    results[model_name] = {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'model': model
                    }
                    
                    # Feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        results[model_name]['feature_importance'] = importance_df.to_dict('records')
                    
                    # Store model
                    self.models[f'{model_name}_regression'] = model
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            self.model_performance['regression'] = results
            
        except Exception as e:
            logger.error(f"Error in regression training: {str(e)}")
        
        return results
    
    def train_classification_models(self, features_df: pd.DataFrame, 
                                  target_column: str = 'is_outperformer') -> Dict:
        """
        Train classification models to predict stock performance categories
        
        Args:
            features_df: DataFrame with features and targets
            target_column: Target column name
        
        Returns:
            Dictionary with model performance results
        """
        results = {}
        
        try:
            if features_df.empty or target_column not in features_df.columns:
                return results
            
            # Prepare features and target
            feature_cols = [col for col in features_df.columns 
                           if col not in ['symbol', 'return_1w', 'return_1m', 'return_3m', 
                                        'performance_category', 'is_outperformer']]
            
            X = features_df[feature_cols].fillna(0)
            y = features_df[target_column]
            
            if len(X) < 10:  # Need minimum samples
                return results
            
            # Handle categorical targets
            if target_column == 'performance_category':
                y = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            for model_name, model in self.classification_models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    train_accuracy = accuracy_score(y_train, y_pred_train)
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    
                    results[model_name] = {
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'model': model
                    }
                    
                    # Feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        results[model_name]['feature_importance'] = importance_df.to_dict('records')
                    
                    # Store model
                    self.models[f'{model_name}_classification'] = model
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            self.model_performance['classification'] = results
            
        except Exception as e:
            logger.error(f"Error in classification training: {str(e)}")
        
        return results
    
    def perform_feature_analysis(self, features_df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive feature importance and factor analysis
        
        Args:
            features_df: DataFrame with features
        
        Returns:
            Dictionary with feature analysis results
        """
        analysis_results = {}
        
        try:
            if features_df.empty:
                return analysis_results
            
            # Prepare features
            feature_cols = [col for col in features_df.columns 
                           if col not in ['symbol', 'return_1w', 'return_1m', 'return_3m', 
                                        'performance_category', 'is_outperformer']]
            
            X = features_df[feature_cols].fillna(0)
            
            if 'return_1m' in features_df.columns:
                y = features_df['return_1m'].fillna(0)
                
                # Statistical feature selection
                selector = SelectKBest(score_func=f_regression, k='all')
                selector.fit(X, y)
                
                feature_scores = pd.DataFrame({
                    'feature': feature_cols,
                    'score': selector.scores_,
                    'p_value': selector.pvalues_
                }).sort_values('score', ascending=False)
                
                analysis_results['statistical_importance'] = feature_scores.to_dict('records')
                
                # Mutual information
                mi_scores = mutual_info_regression(X, y)
                mi_df = pd.DataFrame({
                    'feature': feature_cols,
                    'mutual_info': mi_scores
                }).sort_values('mutual_info', ascending=False)
                
                analysis_results['mutual_information'] = mi_df.to_dict('records')
            
            # Principal Component Analysis
            if len(X.columns) > 3:
                pca = PCA()
                pca.fit(X)
                
                # Explained variance
                explained_variance = pd.DataFrame({
                    'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
                })
                
                analysis_results['pca_analysis'] = {
                    'explained_variance': explained_variance.to_dict('records'),
                    'n_components_95': int(np.argmax(explained_variance['cumulative_variance'] >= 0.95)) + 1,
                    'n_components_90': int(np.argmax(explained_variance['cumulative_variance'] >= 0.90)) + 1
                }
                
                # Feature loadings for first few components
                loadings = pd.DataFrame(
                    pca.components_[:5].T,  # First 5 components
                    columns=[f'PC{i+1}' for i in range(5)],
                    index=feature_cols
                )
                
                analysis_results['pca_loadings'] = loadings.to_dict('index')
            
            # Correlation analysis
            correlation_matrix = X.corr()
            
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            analysis_results['high_correlations'] = high_corr_pairs
            
            # Feature statistics
            feature_stats = X.describe().T
            feature_stats['missing_ratio'] = features_df[feature_cols].isnull().mean()
            
            analysis_results['feature_statistics'] = feature_stats.to_dict('index')
            
        except Exception as e:
            logger.error(f"Error in feature analysis: {str(e)}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def predict_stock_performance(self, features: Dict, model_type: str = 'random_forest') -> Dict:
        """
        Predict stock performance using trained models
        
        Args:
            features: Dictionary of feature values
            model_type: Type of model to use
        
        Returns:
            Dictionary with predictions
        """
        predictions = {}
        
        try:
            # Prepare feature vector
            feature_vector = pd.DataFrame([features])
            feature_vector = feature_vector.fillna(0)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Regression prediction
            reg_model_key = f'{model_type}_regression'
            if reg_model_key in self.models:
                reg_prediction = self.models[reg_model_key].predict(feature_vector_scaled)[0]
                predictions['predicted_return'] = reg_prediction
            
            # Classification prediction
            clf_model_key = f'{model_type}_classification'
            if clf_model_key in self.models:
                clf_prediction = self.models[clf_model_key].predict(feature_vector_scaled)[0]
                clf_proba = self.models[clf_model_key].predict_proba(feature_vector_scaled)[0]
                
                predictions['predicted_class'] = clf_prediction
                predictions['class_probabilities'] = clf_proba.tolist()
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            predictions['error'] = str(e)
        
        return predictions
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                
                self.models = model_data.get('models', {})
                self.scaler = model_data.get('scaler', StandardScaler())
                self.label_encoder = model_data.get('label_encoder', LabelEncoder())
                self.feature_importance = model_data.get('feature_importance', {})
                self.model_performance = model_data.get('model_performance', {})
                
                logger.info(f"Models loaded from {filepath}")
            else:
                logger.warning(f"Model file not found: {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

