from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from scipy.optimize import minimize
import warnings

from .base_processor import BaseProcessor

class EconomicProcessor(BaseProcessor):
    """Advanced economic data processor with macro analysis"""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Initialize models
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Explain 95% of variance
        
        # Cache for computed metrics
        self.regime_cache = {}
        self.correlation_cache = {}
        
    def _process_implementation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Implement economic-specific processing logic.
        
        Args:
            df: Preprocessed DataFrame with economic data
            
        Returns:
            Dictionary containing processed results
        """
        # Process data
        results = {
            'regime_analysis': self._analyze_regimes(df),
            'macro_trends': self._analyze_macro_trends(df),
            'correlation_analysis': self._analyze_correlations(df),
            'leading_indicators': self._analyze_leading_indicators(df),
            'structural_analysis': self._analyze_structural_changes(df),
            'cycle_analysis': self._analyze_cycles(df),
            'risk_analysis': self._analyze_risks(df),
            'impact_analysis': self._analyze_market_impact(df),
            'forecast_analysis': self._generate_forecasts(df),
            'composite_indicators': self._create_composite_indicators(df)
        }
        
        return results
        
    def generate_insights(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate economic-specific insights from processed data.
        
        Args:
            processed_data: Dictionary containing processed economic data
            
        Returns:
            Dictionary containing generated insights
        """
        insights = {
            'economic_state': {
                'regime': processed_data['regime_analysis']['regime_model']['current_regime'],
                'cycle_phase': self._determine_cycle_phase(processed_data['cycle_analysis']),
                'trend_direction': self._determine_trend_direction(processed_data['macro_trends'])
            },
            'risk_assessment': {
                'current_risks': processed_data['risk_analysis']['risk_indicators'],
                'vulnerabilities': processed_data['risk_analysis']['vulnerabilities'],
                'early_warnings': processed_data['risk_analysis']['early_warnings']
            },
            'forward_looking': {
                'forecasts': processed_data['forecast_analysis']['base_forecasts'],
                'leading_indicators': processed_data['leading_indicators'],
                'scenarios': processed_data['forecast_analysis']['scenarios']
            },
            'market_implications': {
                'sector_impacts': processed_data['impact_analysis']['sector_impacts'],
                'asset_correlations': processed_data['impact_analysis']['correlations'],
                'risk_premia': processed_data['impact_analysis']['risk_premia']
            }
        }
        
        return insights
        
    def validate_insights(self, insights: Dict[str, Any]) -> bool:
        """
        Validate economic-specific insights.
        
        Args:
            insights: Dictionary containing generated insights
            
        Returns:
            Boolean indicating if insights are valid
        """
        try:
            # Validate required sections
            required_sections = ['economic_state', 'risk_assessment', 'forward_looking', 'market_implications']
            if not all(section in insights for section in required_sections):
                logger.error("Missing required insight sections")
                return False
                
            # Validate economic state
            state = insights['economic_state']
            if not all(k in state for k in ['regime', 'cycle_phase', 'trend_direction']):
                logger.error("Invalid economic state format")
                return False
                
            # Validate risk assessment
            risk = insights['risk_assessment']
            if not all(k in risk for k in ['current_risks', 'vulnerabilities', 'early_warnings']):
                logger.error("Invalid risk assessment format")
                return False
                
            # Validate forward looking
            forward = insights['forward_looking']
            if not all(k in forward for k in ['forecasts', 'leading_indicators', 'scenarios']):
                logger.error("Invalid forward looking format")
                return False
                
            # Validate market implications
            market = insights['market_implications']
            if not all(k in market for k in ['sector_impacts', 'asset_correlations', 'risk_premia']):
                logger.error("Invalid market implications format")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Insight validation error: {str(e)}")
            return False
    
    def _analyze_regimes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze economic regimes and transitions
        
        Features:
        - Regime identification
        - Transition probabilities
        - Regime characteristics
        - Stability analysis
        """
        results = {}
        
        # Prepare data
        if 'value' not in df.columns or 'indicator_name' not in df.columns:
            return results
            
        # Pivot data for multi-indicator analysis
        pivot_df = df.pivot(columns='indicator_name', values='value')
        
        # Standardize data
        scaled_data = self.scaler.fit_transform(pivot_df)
        
        # Fit Markov regime switching model
        try:
            model = MarkovRegression(scaled_data[:, 0], k_regimes=3, trend='c')
            fit = model.fit()
            
            results['regime_model'] = {
                'current_regime': int(fit.smoothed_marginal_probabilities[-1].argmax()),
                'regime_probs': fit.smoothed_marginal_probabilities[-1].tolist(),
                'transition_matrix': fit.transition_probabilities.tolist(),
                'expected_durations': fit.expected_durations.tolist()
            }
        except:
            # Fallback to simpler regime detection
            results['regime_model'] = self._detect_regimes_kmeans(scaled_data)
        
        # Analyze regime characteristics
        results['characteristics'] = self._analyze_regime_characteristics(df, results['regime_model'])
        
        # Analyze regime stability
        results['stability'] = self._analyze_regime_stability(df, results['regime_model'])
        
        return results
    
    def _analyze_macro_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze macroeconomic trends
        
        Features:
        - Trend decomposition
        - Growth analysis
        - Cyclical components
        - Structural changes
        """
        results = {}
        
        # Group indicators by type
        growth_indicators = self._filter_indicators(df, ['gdp', 'industrial_production'])
        inflation_indicators = self._filter_indicators(df, ['cpi', 'ppi'])
        employment_indicators = self._filter_indicators(df, ['unemployment', 'payrolls'])
        monetary_indicators = self._filter_indicators(df, ['interest_rate', 'money_supply'])
        
        # Analyze each group
        results['growth'] = self._analyze_indicator_group(growth_indicators)
        results['inflation'] = self._analyze_indicator_group(inflation_indicators)
        results['employment'] = self._analyze_indicator_group(employment_indicators)
        results['monetary'] = self._analyze_indicator_group(monetary_indicators)
        
        # Analyze cross-group relationships
        results['relationships'] = self._analyze_macro_relationships(df)
        
        return results
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations and dependencies
        
        Features:
        - Cross-correlations
        - Lead-lag relationships
        - Granger causality
        - Network analysis
        """
        # Prepare correlation matrix
        corr_matrix = df.pivot(columns='indicator_name', values='value').corr()
        
        # Build correlation network
        G = self._build_correlation_network(corr_matrix)
        
        # Analyze lead-lag relationships
        lead_lag = self._analyze_lead_lag_relationships(df)
        
        # Test Granger causality
        causality = self._test_granger_causality(df)
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'network_metrics': self._calculate_network_metrics(G),
            'lead_lag': lead_lag,
            'causality': causality
        }
    
    def _analyze_leading_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze leading economic indicators
        
        Features:
        - Indicator identification
        - Predictive power
        - Composite indicators
        - Signal generation
        """
        # Identify leading indicators
        leaders = self._identify_leading_indicators(df)
        
        # Assess predictive power
        predictive_power = self._assess_predictive_power(df, leaders)
        
        # Create composite index
        composite = self._create_composite_index(df, leaders)
        
        # Generate signals
        signals = self._generate_indicator_signals(df, composite)
        
        return {
            'leading_indicators': leaders,
            'predictive_power': predictive_power,
            'composite_index': composite,
            'signals': signals
        }
    
    def _analyze_structural_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze structural changes in economy
        
        Features:
        - Break point detection
        - Trend changes
        - Structural stability
        - Impact assessment
        """
        # Detect break points
        breaks = self._detect_break_points(df)
        
        # Analyze trend changes
        trends = self._analyze_trend_changes(df, breaks)
        
        # Test structural stability
        stability = self._test_structural_stability(df)
        
        # Assess impact
        impact = self._assess_structural_impact(df, breaks)
        
        return {
            'break_points': breaks,
            'trend_changes': trends,
            'stability': stability,
            'impact': impact
        }
    
    def _analyze_cycles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze economic cycles
        
        Features:
        - Cycle identification
        - Phase analysis
        - Synchronization
        - Turning points
        """
        # Identify cycles
        cycles = self._identify_cycles(df)
        
        # Analyze cycle phases
        phases = self._analyze_cycle_phases(df, cycles)
        
        # Measure synchronization
        sync = self._measure_cycle_synchronization(df)
        
        # Detect turning points
        turning_points = self._detect_turning_points(df)
        
        return {
            'cycles': cycles,
            'phases': phases,
            'synchronization': sync,
            'turning_points': turning_points
        }
    
    def _analyze_risks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze economic risks
        
        Features:
        - Risk indicators
        - Vulnerability assessment
        - Stress testing
        - Early warnings
        """
        # Calculate risk indicators
        indicators = self._calculate_risk_indicators(df)
        
        # Assess vulnerabilities
        vulnerabilities = self._assess_vulnerabilities(df)
        
        # Perform stress tests
        stress_tests = self._perform_stress_tests(df)
        
        # Generate early warnings
        warnings = self._generate_early_warnings(df)
        
        return {
            'risk_indicators': indicators,
            'vulnerabilities': vulnerabilities,
            'stress_tests': stress_tests,
            'early_warnings': warnings
        }
    
    def _analyze_market_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market impact of economic data
        
        Features:
        - Asset correlations
        - Sector impacts
        - Risk premia
        - Market regimes
        """
        # Analyze asset correlations
        correlations = self._analyze_asset_correlations(df)
        
        # Calculate sector impacts
        impacts = self._calculate_sector_impacts(df)
        
        # Estimate risk premia
        premia = self._estimate_risk_premia(df)
        
        # Identify market regimes
        regimes = self._identify_market_regimes(df)
        
        return {
            'correlations': correlations,
            'sector_impacts': impacts,
            'risk_premia': premia,
            'market_regimes': regimes
        }
    
    def _generate_forecasts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate economic forecasts
        
        Features:
        - Time series forecasts
        - Scenario analysis
        - Confidence intervals
        - Model combination
        """
        # Generate base forecasts
        forecasts = self._generate_base_forecasts(df)
        
        # Perform scenario analysis
        scenarios = self._analyze_scenarios(df)
        
        # Calculate confidence intervals
        intervals = self._calculate_confidence_intervals(forecasts)
        
        # Combine model forecasts
        combined = self._combine_forecasts(forecasts)
        
        return {
            'base_forecasts': forecasts,
            'scenarios': scenarios,
            'confidence_intervals': intervals,
            'combined_forecast': combined
        }
    
    def _create_composite_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create composite economic indicators
        
        Features:
        - Component selection
        - Weighting schemes
        - Performance metrics
        - Signal generation
        """
        # Select components
        components = self._select_indicator_components(df)
        
        # Calculate weights
        weights = self._calculate_component_weights(components)
        
        # Create composite
        composite = self._construct_composite_indicator(components, weights)
        
        # Generate signals
        signals = self._generate_composite_signals(composite)
        
        return {
            'components': components,
            'weights': weights,
            'composite_values': composite,
            'signals': signals
        }
