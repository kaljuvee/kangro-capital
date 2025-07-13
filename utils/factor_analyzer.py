"""
Factor Analysis Module for identifying key performance drivers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FactorAnalyzer:
    """
    Factor analyzer for identifying key performance drivers in stock data
    """
    
    def __init__(self):
        """Initialize the factor analyzer"""
        self.scaler = StandardScaler()
        self.factor_loadings = {}
        self.factor_scores = {}
        self.factor_interpretation = {}
        
        # Define factor categories
        self.factor_categories = {
            'momentum': ['momentum_5', 'momentum_10', 'momentum_20', 'price_vs_sma5', 'price_vs_sma10'],
            'technical': ['rsi', 'macd', 'bb_position', 'volatility_20', 'volume_ratio'],
            'fundamental': ['roe', 'current_ratio', 'gross_margin', 'net_margin', 'revenue_growth'],
            'quality': ['revenue_consistency', 'income_consistency', 'cashflow_consistency'],
            'valuation': ['debt_to_ebitda', 'net_income_growth', 'cash_flow_growth']
        }
    
    def perform_factor_analysis(self, features_df: pd.DataFrame, 
                              target_column: str = 'return_1m',
                              n_factors: int = 5) -> Dict:
        """
        Perform comprehensive factor analysis
        
        Args:
            features_df: DataFrame with features and targets
            target_column: Target variable for analysis
            n_factors: Number of factors to extract
        
        Returns:
            Dictionary with factor analysis results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_factors': n_factors,
            'target_column': target_column
        }
        
        try:
            if features_df.empty or target_column not in features_df.columns:
                return results
            
            # Prepare data
            feature_cols = [col for col in features_df.columns 
                           if col not in ['symbol', 'return_1w', 'return_1m', 'return_3m', 
                                        'performance_category', 'is_outperformer']]
            
            X = features_df[feature_cols].fillna(0)
            y = features_df[target_column].fillna(0)
            
            if len(X) < 10 or len(feature_cols) < 3:
                return results
            
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
            
            # 1. Principal Component Analysis
            pca_results = self._perform_pca_analysis(X_scaled_df, y, n_factors)
            results['pca_analysis'] = pca_results
            
            # 2. Factor Analysis
            fa_results = self._perform_factor_analysis(X_scaled_df, y, n_factors)
            results['factor_analysis'] = fa_results
            
            # 3. Correlation Analysis with Target
            correlation_results = self._analyze_correlations(X_scaled_df, y, feature_cols)
            results['correlation_analysis'] = correlation_results
            
            # 4. Factor Interpretation
            interpretation = self._interpret_factors(pca_results, fa_results, feature_cols)
            results['factor_interpretation'] = interpretation
            
            # 5. Performance Driver Analysis
            driver_analysis = self._analyze_performance_drivers(X_scaled_df, y, feature_cols)
            results['performance_drivers'] = driver_analysis
            
            # 6. Factor Clustering
            cluster_results = self._cluster_factors(X_scaled_df, feature_cols)
            results['factor_clusters'] = cluster_results
            
        except Exception as e:
            logger.error(f"Error in factor analysis: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _perform_pca_analysis(self, X_scaled: pd.DataFrame, y: pd.Series, n_factors: int) -> Dict:
        """Perform Principal Component Analysis"""
        pca_results = {}
        
        try:
            # Fit PCA
            pca = PCA(n_components=n_factors)
            pca_scores = pca.fit_transform(X_scaled)
            
            # Explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            pca_results['explained_variance'] = explained_variance.tolist()
            pca_results['cumulative_variance'] = cumulative_variance.tolist()
            pca_results['total_variance_explained'] = cumulative_variance[-1]
            
            # Component loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_factors)],
                index=X_scaled.columns
            )
            
            pca_results['loadings'] = loadings.to_dict('index')
            
            # Correlation of components with target
            component_correlations = []
            for i in range(n_factors):
                corr, p_value = pearsonr(pca_scores[:, i], y)
                component_correlations.append({
                    'component': f'PC{i+1}',
                    'correlation': corr,
                    'p_value': p_value,
                    'abs_correlation': abs(corr)
                })
            
            # Sort by absolute correlation
            component_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            pca_results['component_target_correlations'] = component_correlations
            
            # Store PCA scores
            self.factor_scores['pca'] = pca_scores
            
        except Exception as e:
            logger.error(f"Error in PCA analysis: {str(e)}")
            pca_results['error'] = str(e)
        
        return pca_results
    
    def _perform_factor_analysis(self, X_scaled: pd.DataFrame, y: pd.Series, n_factors: int) -> Dict:
        """Perform Factor Analysis"""
        fa_results = {}
        
        try:
            # Fit Factor Analysis
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            fa_scores = fa.fit_transform(X_scaled)
            
            # Factor loadings
            loadings = pd.DataFrame(
                fa.components_.T,
                columns=[f'Factor{i+1}' for i in range(n_factors)],
                index=X_scaled.columns
            )
            
            fa_results['loadings'] = loadings.to_dict('index')
            
            # Communalities (proportion of variance explained by factors)
            communalities = np.sum(fa.components_.T ** 2, axis=1)
            communality_df = pd.DataFrame({
                'feature': X_scaled.columns,
                'communality': communalities
            }).sort_values('communality', ascending=False)
            
            fa_results['communalities'] = communality_df.to_dict('records')
            
            # Correlation of factors with target
            factor_correlations = []
            for i in range(n_factors):
                corr, p_value = pearsonr(fa_scores[:, i], y)
                factor_correlations.append({
                    'factor': f'Factor{i+1}',
                    'correlation': corr,
                    'p_value': p_value,
                    'abs_correlation': abs(corr)
                })
            
            factor_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            fa_results['factor_target_correlations'] = factor_correlations
            
            # Store factor scores
            self.factor_scores['fa'] = fa_scores
            
        except Exception as e:
            logger.error(f"Error in Factor Analysis: {str(e)}")
            fa_results['error'] = str(e)
        
        return fa_results
    
    def _analyze_correlations(self, X_scaled: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Dict:
        """Analyze correlations between features and target"""
        correlation_results = {}
        
        try:
            # Pearson correlations
            pearson_correlations = []
            for col in feature_cols:
                corr, p_value = pearsonr(X_scaled[col], y)
                pearson_correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'p_value': p_value,
                    'abs_correlation': abs(corr),
                    'significance': 'significant' if p_value < 0.05 else 'not_significant'
                })
            
            pearson_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            correlation_results['pearson'] = pearson_correlations
            
            # Spearman correlations (rank-based)
            spearman_correlations = []
            for col in feature_cols:
                corr, p_value = spearmanr(X_scaled[col], y)
                spearman_correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'p_value': p_value,
                    'abs_correlation': abs(corr),
                    'significance': 'significant' if p_value < 0.05 else 'not_significant'
                })
            
            spearman_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            correlation_results['spearman'] = spearman_correlations
            
            # Top correlated features
            top_pearson = [item for item in pearson_correlations[:10]]
            top_spearman = [item for item in spearman_correlations[:10]]
            
            correlation_results['top_correlations'] = {
                'pearson_top_10': top_pearson,
                'spearman_top_10': top_spearman
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            correlation_results['error'] = str(e)
        
        return correlation_results
    
    def _interpret_factors(self, pca_results: Dict, fa_results: Dict, feature_cols: List[str]) -> Dict:
        """Interpret factors based on loadings and domain knowledge"""
        interpretation = {}
        
        try:
            # Interpret PCA components
            if 'loadings' in pca_results:
                pca_interpretation = {}
                loadings_df = pd.DataFrame(pca_results['loadings']).T
                
                for i, component in enumerate(loadings_df.columns):
                    # Get top loading features
                    component_loadings = loadings_df[component].abs().sort_values(ascending=False)
                    top_features = component_loadings.head(5)
                    
                    # Categorize based on feature types
                    categories = []
                    for feature in top_features.index:
                        for category, features in self.factor_categories.items():
                            if any(f in feature for f in features):
                                categories.append(category)
                                break
                    
                    # Determine dominant category
                    if categories:
                        dominant_category = max(set(categories), key=categories.count)
                    else:
                        dominant_category = 'mixed'
                    
                    pca_interpretation[component] = {
                        'dominant_category': dominant_category,
                        'top_features': top_features.to_dict(),
                        'interpretation': self._get_factor_interpretation(dominant_category, top_features.index.tolist())
                    }
                
                interpretation['pca'] = pca_interpretation
            
            # Interpret Factor Analysis factors
            if 'loadings' in fa_results:
                fa_interpretation = {}
                loadings_df = pd.DataFrame(fa_results['loadings']).T
                
                for i, factor in enumerate(loadings_df.columns):
                    # Get top loading features
                    factor_loadings = loadings_df[factor].abs().sort_values(ascending=False)
                    top_features = factor_loadings.head(5)
                    
                    # Categorize based on feature types
                    categories = []
                    for feature in top_features.index:
                        for category, features in self.factor_categories.items():
                            if any(f in feature for f in features):
                                categories.append(category)
                                break
                    
                    # Determine dominant category
                    if categories:
                        dominant_category = max(set(categories), key=categories.count)
                    else:
                        dominant_category = 'mixed'
                    
                    fa_interpretation[factor] = {
                        'dominant_category': dominant_category,
                        'top_features': top_features.to_dict(),
                        'interpretation': self._get_factor_interpretation(dominant_category, top_features.index.tolist())
                    }
                
                interpretation['factor_analysis'] = fa_interpretation
            
        except Exception as e:
            logger.error(f"Error in factor interpretation: {str(e)}")
            interpretation['error'] = str(e)
        
        return interpretation
    
    def _get_factor_interpretation(self, category: str, top_features: List[str]) -> str:
        """Get human-readable interpretation of factors"""
        interpretations = {
            'momentum': 'This factor captures momentum and trend-following characteristics. High values indicate stocks with strong recent price momentum.',
            'technical': 'This factor represents technical analysis indicators. It captures overbought/oversold conditions and technical patterns.',
            'fundamental': 'This factor reflects fundamental financial health. It indicates profitability, efficiency, and financial strength.',
            'quality': 'This factor measures business quality and consistency. It captures the reliability and predictability of financial performance.',
            'valuation': 'This factor relates to valuation metrics and growth characteristics. It indicates whether a stock is fairly valued relative to its growth.',
            'mixed': 'This factor combines multiple aspects of stock performance and may represent a complex interaction of various drivers.'
        }
        
        base_interpretation = interpretations.get(category, 'This factor captures various aspects of stock performance.')
        
        # Add specific feature mentions
        feature_mention = f" Key contributing features include: {', '.join(top_features[:3])}."
        
        return base_interpretation + feature_mention
    
    def _analyze_performance_drivers(self, X_scaled: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Dict:
        """Analyze key performance drivers"""
        driver_analysis = {}
        
        try:
            # Calculate feature importance based on correlation and variance
            feature_importance = []
            
            for col in feature_cols:
                # Correlation with target
                corr, p_value = pearsonr(X_scaled[col], y)
                
                # Feature variance (information content)
                variance = X_scaled[col].var()
                
                # Combined importance score
                importance_score = abs(corr) * np.sqrt(variance)
                
                feature_importance.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'variance': variance,
                    'importance_score': importance_score,
                    'p_value': p_value,
                    'category': self._categorize_feature(col)
                })
            
            # Sort by importance score
            feature_importance.sort(key=lambda x: x['importance_score'], reverse=True)
            driver_analysis['feature_importance'] = feature_importance
            
            # Category-wise analysis
            category_importance = {}
            for category in self.factor_categories.keys():
                category_features = [f for f in feature_importance if f['category'] == category]
                if category_features:
                    avg_importance = np.mean([f['importance_score'] for f in category_features])
                    avg_correlation = np.mean([abs(f['correlation']) for f in category_features])
                    
                    category_importance[category] = {
                        'avg_importance': avg_importance,
                        'avg_correlation': avg_correlation,
                        'feature_count': len(category_features),
                        'top_feature': category_features[0]['feature'] if category_features else None
                    }
            
            driver_analysis['category_importance'] = category_importance
            
            # Top drivers summary
            top_drivers = feature_importance[:10]
            driver_analysis['top_10_drivers'] = top_drivers
            
            # Performance driver insights
            insights = self._generate_driver_insights(feature_importance, category_importance)
            driver_analysis['insights'] = insights
            
        except Exception as e:
            logger.error(f"Error in performance driver analysis: {str(e)}")
            driver_analysis['error'] = str(e)
        
        return driver_analysis
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature based on its name"""
        for category, features in self.factor_categories.items():
            if any(f in feature_name for f in features):
                return category
        return 'other'
    
    def _cluster_factors(self, X_scaled: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Cluster features based on similarity"""
        cluster_results = {}
        
        try:
            # Correlation-based clustering
            correlation_matrix = X_scaled.corr()
            
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(correlation_matrix)
            
            # K-means clustering on features
            n_clusters = min(5, len(feature_cols) // 3)  # Reasonable number of clusters
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                feature_clusters = kmeans.fit_predict(X_scaled.T)
                
                # Organize clusters
                clusters = {}
                for i, feature in enumerate(feature_cols):
                    cluster_id = feature_clusters[i]
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(feature)
                
                cluster_results['feature_clusters'] = clusters
                
                # Cluster characteristics
                cluster_characteristics = {}
                for cluster_id, features in clusters.items():
                    if len(features) > 1:
                        cluster_data = X_scaled[features]
                        avg_correlation = cluster_data.corr().values[np.triu_indices_from(cluster_data.corr().values, k=1)].mean()
                        
                        cluster_characteristics[cluster_id] = {
                            'features': features,
                            'size': len(features),
                            'avg_internal_correlation': avg_correlation,
                            'dominant_category': self._get_dominant_category(features)
                        }
                
                cluster_results['cluster_characteristics'] = cluster_characteristics
            
        except Exception as e:
            logger.error(f"Error in factor clustering: {str(e)}")
            cluster_results['error'] = str(e)
        
        return cluster_results
    
    def _get_dominant_category(self, features: List[str]) -> str:
        """Get the dominant category for a list of features"""
        categories = [self._categorize_feature(f) for f in features]
        if categories:
            return max(set(categories), key=categories.count)
        return 'mixed'
    
    def _generate_driver_insights(self, feature_importance: List[Dict], category_importance: Dict) -> List[str]:
        """Generate human-readable insights about performance drivers"""
        insights = []
        
        try:
            # Top driver insight
            if feature_importance:
                top_driver = feature_importance[0]
                insights.append(f"The most important performance driver is '{top_driver['feature']}' with an importance score of {top_driver['importance_score']:.3f}.")
            
            # Category insights
            if category_importance:
                sorted_categories = sorted(category_importance.items(), key=lambda x: x[1]['avg_importance'], reverse=True)
                top_category = sorted_categories[0]
                insights.append(f"The '{top_category[0]}' category shows the highest average importance ({top_category[1]['avg_importance']:.3f}) in driving performance.")
            
            # Correlation insights
            high_corr_features = [f for f in feature_importance if abs(f['correlation']) > 0.3]
            if high_corr_features:
                insights.append(f"{len(high_corr_features)} features show strong correlation (>0.3) with performance.")
            
            # Significance insights
            significant_features = [f for f in feature_importance if f['p_value'] < 0.05]
            if significant_features:
                insights.append(f"{len(significant_features)} features are statistically significant (p < 0.05) predictors of performance.")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
        
        return insights
    
    def get_factor_summary(self, factor_results: Dict) -> Dict:
        """Generate a summary of factor analysis results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'factor_analysis_summary'
        }
        
        try:
            # PCA summary
            if 'pca_analysis' in factor_results:
                pca = factor_results['pca_analysis']
                summary['pca_summary'] = {
                    'total_variance_explained': pca.get('total_variance_explained', 0),
                    'most_important_component': None,
                    'top_features': []
                }
                
                if 'component_target_correlations' in pca and pca['component_target_correlations']:
                    top_component = pca['component_target_correlations'][0]
                    summary['pca_summary']['most_important_component'] = top_component
            
            # Performance drivers summary
            if 'performance_drivers' in factor_results:
                drivers = factor_results['performance_drivers']
                summary['key_drivers'] = {
                    'top_3_features': drivers.get('top_10_drivers', [])[:3],
                    'most_important_category': None,
                    'insights': drivers.get('insights', [])
                }
                
                if 'category_importance' in drivers:
                    categories = drivers['category_importance']
                    if categories:
                        top_category = max(categories.items(), key=lambda x: x[1]['avg_importance'])
                        summary['key_drivers']['most_important_category'] = {
                            'name': top_category[0],
                            'importance': top_category[1]['avg_importance']
                        }
            
            # Factor interpretation summary
            if 'factor_interpretation' in factor_results:
                interpretation = factor_results['factor_interpretation']
                summary['factor_themes'] = []
                
                for method in ['pca', 'factor_analysis']:
                    if method in interpretation:
                        for factor_name, factor_info in interpretation[method].items():
                            summary['factor_themes'].append({
                                'method': method,
                                'factor': factor_name,
                                'theme': factor_info.get('dominant_category', 'unknown'),
                                'interpretation': factor_info.get('interpretation', '')
                            })
            
        except Exception as e:
            logger.error(f"Error generating factor summary: {str(e)}")
            summary['error'] = str(e)
        
        return summary

