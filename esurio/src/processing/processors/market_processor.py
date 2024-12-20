from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import het_white
from arch import arch_model
import talib
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from datetime import datetime, timedelta

from .base_processor import BaseProcessor

class MarketProcessor(BaseProcessor):
    """Advanced market data processor with microstructure analysis"""
    
    def __init__(self) -> None:
        super().__init__()
        self._load_models()
        
    def _process_implementation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Implement market-specific processing logic.
        
        Args:
            df: Preprocessed DataFrame with market data
            
        Returns:
            Dictionary containing processed results
        """
        # Basic features
        features = self.extract_features(df['price'])
        
        # Advanced analysis
        results = {
            'basic_features': features,
            'microstructure': self.analyze_microstructure(df),
            'liquidity_metrics': self.calculate_liquidity_metrics(df),
            'volatility_analysis': self.analyze_volatility(df),
            'order_flow': self.analyze_order_flow(df),
            'price_impact': self.calculate_price_impact(df),
            'market_efficiency': self.analyze_market_efficiency(df),
            'regime_analysis': self.analyze_market_regime(df),
            'technical_indicators': self.calculate_technical_indicators(df),
            'market_quality': self.analyze_market_quality(df),
            'structural_breaks': self.detect_structural_breaks(df['price']),
            'complexity': self.calculate_complexity_measures(df['price'])
        }
        
        return results
        
    def generate_insights(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate market-specific insights from processed data.
        
        Args:
            processed_data: Dictionary containing processed market data
            
        Returns:
            Dictionary containing generated insights
        """
        insights = {
            'market_state': {
                'regime': processed_data['regime_analysis']['current_regime'],
                'volatility_level': self._classify_volatility(processed_data['volatility_analysis']),
                'liquidity_state': self._assess_liquidity(processed_data['liquidity_metrics'])
            },
            'trading_signals': {
                'technical': self._generate_technical_signals(processed_data['technical_indicators']),
                'microstructure': self._generate_microstructure_signals(processed_data['microstructure']),
                'flow': self._generate_flow_signals(processed_data['order_flow'])
            },
            'risk_metrics': {
                'market_quality': processed_data['market_quality'],
                'stability': self._assess_stability(processed_data)
            },
            'anomalies': self._detect_anomalies(processed_data)
        }
        
        return insights
        
    def validate_insights(self, insights: Dict[str, Any]) -> bool:
        """
        Validate market-specific insights.
        
        Args:
            insights: Dictionary containing generated insights
            
        Returns:
            Boolean indicating if insights are valid
        """
        try:
            # Validate required sections
            required_sections = ['market_state', 'trading_signals', 'risk_metrics', 'anomalies']
            if not all(section in insights for section in required_sections):
                logger.error("Missing required insight sections")
                return False
                
            # Validate market state
            market_state = insights['market_state']
            if not all(k in market_state for k in ['regime', 'volatility_level', 'liquidity_state']):
                logger.error("Invalid market state format")
                return False
                
            # Validate trading signals
            signals = insights['trading_signals']
            if not all(k in signals for k in ['technical', 'microstructure', 'flow']):
                logger.error("Invalid trading signals format")
                return False
                
            # Validate risk metrics
            risk = insights['risk_metrics']
            if not all(k in risk for k in ['market_quality', 'stability']):
                logger.error("Invalid risk metrics format")
                return False
                
            # Validate values are within expected ranges
            if not 0 <= market_state['volatility_level'] <= 1:
                logger.error("Invalid volatility level")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Insight validation error: {str(e)}")
            return False
    
    def analyze_microstructure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market microstructure
        
        Features:
        - Bid-ask spread analysis
        - Trade size distribution
        - Quote intensity
        - Trade clustering
        - Price discreteness
        """
        results = {}
        
        # Bid-ask spread analysis
        if all(col in df.columns for col in ['bid', 'ask']):
            spreads = df['ask'] - df['bid']
            results['spread_metrics'] = {
                'mean_spread': spreads.mean(),
                'spread_volatility': spreads.std(),
                'relative_spread': (spreads / df['price']).mean(),
                'spread_percentiles': spreads.quantile([0.25, 0.5, 0.75]).to_dict()
            }
        
        # Trade size analysis
        if 'volume' in df.columns:
            results['trade_size'] = {
                'mean_size': df['volume'].mean(),
                'size_volatility': df['volume'].std(),
                'size_skew': df['volume'].skew(),
                'size_distribution': self._fit_distribution(df['volume'])
            }
        
        # Quote intensity
        if 'timestamp' in df.columns:
            quote_times = pd.to_datetime(df['timestamp'])
            quote_intervals = quote_times.diff().dt.total_seconds()
            results['quote_intensity'] = {
                'mean_interval': quote_intervals.mean(),
                'interval_volatility': quote_intervals.std(),
                'quote_clustering': self._analyze_time_clustering(quote_intervals)
            }
        
        # Price discreteness
        if 'price' in df.columns:
            price_decimals = (df['price'] % 1).value_counts()
            results['price_discreteness'] = {
                'decimal_distribution': price_decimals.to_dict(),
                'price_clustering': self._analyze_price_clustering(df['price'])
            }
        
        return results
    
    def calculate_liquidity_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced liquidity metrics
        
        Metrics:
        - Amihud illiquidity
        - Kyle's lambda
        - Bid-ask spread components
        - Market depth
        - Resiliency
        """
        results = {}
        
        # Amihud illiquidity
        if all(col in df.columns for col in ['price', 'volume']):
            returns = df['price'].pct_change().abs()
            dollar_volume = df['price'] * df['volume']
            results['amihud'] = (returns / dollar_volume).mean()
        
        # Kyle's lambda
        if all(col in df.columns for col in ['price', 'volume']):
            price_changes = df['price'].diff()
            signed_volume = df['volume'] * np.sign(price_changes)
            model = stats.linregress(signed_volume, price_changes)
            results['kyle_lambda'] = {
                'lambda': model.slope,
                'r_squared': model.rvalue ** 2,
                'p_value': model.pvalue
            }
        
        # Bid-ask spread components
        if all(col in df.columns for col in ['bid', 'ask', 'price']):
            results['spread_components'] = self._decompose_spread(df)
        
        # Market depth
        if all(col in df.columns for col in ['bid_size', 'ask_size']):
            results['market_depth'] = {
                'total_depth': (df['bid_size'] + df['ask_size']).mean(),
                'depth_imbalance': ((df['ask_size'] - df['bid_size']) / 
                                  (df['ask_size'] + df['bid_size'])).mean()
            }
        
        # Resiliency
        if 'price' in df.columns:
            results['resiliency'] = self._calculate_resiliency(df)
        
        return results
    
    def analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Advanced volatility analysis
        
        Features:
        - Realized volatility
        - Implied volatility (if available)
        - Volatility components
        - Jump detection
        - Volatility forecasting
        """
        if 'price' not in df.columns:
            return {}
            
        returns = df['price'].pct_change().dropna()
        
        # Realized volatility
        realized_vol = returns.std() * np.sqrt(252)  # Annualized
        
        # GARCH modeling
        garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_result = garch_model.fit(disp='off')
        
        # Jump detection
        jumps = self._detect_jumps(returns)
        
        # Volatility components
        continuous_vol, jump_vol = self._decompose_volatility(returns)
        
        results = {
            'realized_volatility': realized_vol,
            'garch_params': {
                'omega': garch_result.params['omega'],
                'alpha': garch_result.params['alpha[1]'],
                'beta': garch_result.params['beta[1]']
            },
            'volatility_forecast': garch_result.forecast().variance.values[-1],
            'jumps': {
                'count': len(jumps),
                'mean_size': np.mean(jumps) if len(jumps) > 0 else 0,
                'times': jumps
            },
            'components': {
                'continuous': continuous_vol,
                'jump': jump_vol,
                'ratio': jump_vol / continuous_vol if continuous_vol != 0 else 0
            }
        }
        
        # Add implied volatility if available
        if 'implied_vol' in df.columns:
            results['implied_volatility'] = {
                'mean': df['implied_vol'].mean(),
                'term_structure': self._analyze_vol_term_structure(df)
            }
        
        return results
    
    def analyze_order_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze order flow toxicity and information content
        
        Metrics:
        - VPIN (Volume-synchronized Probability of Informed Trading)
        - Order flow imbalance
        - Trade initiation
        - Informed trading metrics
        """
        results = {}
        
        # Order flow imbalance
        if 'volume' in df.columns and 'price' in df.columns:
            price_changes = df['price'].diff()
            buy_volume = df['volume'][price_changes > 0].sum()
            sell_volume = df['volume'][price_changes < 0].sum()
            
            results['order_imbalance'] = {
                'buy_ratio': buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0,
                'net_flow': buy_volume - sell_volume,
                'flow_volatility': self._calculate_flow_volatility(df)
            }
        
        # VPIN calculation
        if 'volume' in df.columns:
            results['vpin'] = self._calculate_vpin(df)
        
        # Trade initiation analysis
        if all(col in df.columns for col in ['price', 'bid', 'ask']):
            results['trade_initiation'] = self._analyze_trade_initiation(df)
        
        # Information content
        results['information'] = {
            'price_impact': self._calculate_price_impact(df),
            'adverse_selection': self._estimate_adverse_selection(df)
        }
        
        return results
    
    def calculate_price_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate price impact using topological model
        
        Features:
        - Neural market impact prediction
        - Topological market structure analysis
        - Impact components and decay
        """
        if not all(col in df.columns for col in ['price', 'volume']):
            return {}
            
        # Get model predictions
        predictions = self.impact_model.predict(df)
        
        # Calculate volume-scaled impacts
        volume = df['volume']
        volume_scale = (volume / volume.mean()).values
        
        return {
            'permanent_impact': predictions['permanent_impact'] * volume_scale,
            'temporary_impact': predictions['temporary_impact'] * volume_scale,
            'decay_rate': predictions['decay_rate'],
            'volume_scaled': True
        }
    
    def analyze_market_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market efficiency and price discovery
        
        Metrics:
        - Random walk tests
        - Variance ratios
        - Serial correlation
        - Price discovery metrics
        """
        if 'price' not in df.columns:
            return {}
            
        returns = df['price'].pct_change().dropna()
        
        # Variance ratio tests
        vr_results = self._calculate_variance_ratios(returns)
        
        # Serial correlation analysis
        correlation_results = self._analyze_serial_correlation(returns)
        
        # Efficiency tests
        efficiency_results = {
            'runs_test': self._runs_test(returns),
            'adf_test': self._adf_test(df['price']),
            'hurst': self._calculate_hurst(returns)
        }
        
        # Price discovery (if multiple venues available)
        if 'venue' in df.columns:
            price_discovery = self._analyze_price_discovery(df)
        else:
            price_discovery = {}
        
        return {
            'variance_ratios': vr_results,
            'serial_correlation': correlation_results,
            'efficiency_tests': efficiency_results,
            'price_discovery': price_discovery
        }
    
    def _load_models(self) -> None:
        """Load required ML models"""
        from src.models.registry import ModelRegistry
        from src.models.market.regime.regime_detector import RegimeDetector
        
        # Initialize model registry
        self.model_registry = ModelRegistry(base_path="/models")
        
        # Load regime detection model
        try:
            model_info = self.model_registry.get_latest("regime_detector", min_accuracy=0.8)
            self.regime_model = self.model_registry.load_model(
                name="regime_detector",
                version=model_info["version"],
                model_class=RegimeDetector
            )
        except ValueError:
            # Initialize new model if none exists
            self.regime_model = RegimeDetector()
            self.regime_model.build({
                "input_dim": 32,
                "hidden_dim": 128,
                "n_regimes": 3
            })
            
    def analyze_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market regime using quantum attention model
        
        Features:
        - Quantum-based regime detection
        - Confidence scores
        - Attention patterns
        - Regime transitions
        """
        if 'price' not in df.columns:
            return {}
            
        # Get model predictions
        predictions = self.regime_model.predict(df)
        
        # Analyze transitions
        regime_sequence = predictions['regimes']
        transitions = self._calculate_regime_transitions(regime_sequence)
        
        # Calculate stability metrics
        stability = {
            'mean_confidence': float(np.mean(predictions['confidence'])),
            'regime_persistence': self._calculate_regime_persistence(regime_sequence),
            'attention_stability': float(np.mean(np.std(predictions['attention'], axis=0)))
        }
        
        return {
            'current_regime': regime_sequence[-1],
            'regime_probs': predictions['probabilities'][-1].tolist(),
            'confidence': float(predictions['confidence'][-1]),
            'attention_patterns': predictions['attention'][-1].tolist(),
            'transitions': transitions,
            'stability': stability
        }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced technical indicators
        
        Indicators:
        - Trend indicators
        - Momentum indicators
        - Volatility indicators
        - Volume indicators
        - Custom indicators
        """
        if not all(col in df.columns for col in ['price', 'volume', 'high', 'low']):
            return {}
            
        price = df['price'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        results = {
            'trend': {
                'adx': talib.ADX(high, low, price),
                'macd': talib.MACD(price)[0],
                'cci': talib.CCI(high, low, price),
                'aroon': talib.AROON(high, low)[0]
            },
            'momentum': {
                'rsi': talib.RSI(price),
                'stoch': talib.STOCH(high, low, price)[0],
                'mom': talib.MOM(price),
                'willr': talib.WILLR(high, low, price)
            },
            'volatility': {
                'atr': talib.ATR(high, low, price),
                'natr': talib.NATR(high, low, price),
                'trange': talib.TRANGE(high, low, price)
            },
            'volume': {
                'obv': talib.OBV(price, volume),
                'adosc': talib.ADOSC(high, low, price, volume),
                'mfi': talib.MFI(high, low, price, volume)
            }
        }
        
        # Add custom indicators
        results['custom'] = self._calculate_custom_indicators(df)
        
        return results
    
    def analyze_market_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze overall market quality
        
        Metrics:
        - Price efficiency
        - Liquidity quality
        - Trading costs
        - Market stability
        """
        results = {}
        
        # Price efficiency metrics
        if 'price' in df.columns:
            results['price_efficiency'] = {
                'variance_ratio': self._calculate_variance_ratio(df['price']),
                'autocorrelation': self._calculate_autocorrelation(df['price']),
                'price_impact': self._calculate_price_impact_efficiency(df)
            }
        
        # Liquidity quality
        if all(col in df.columns for col in ['bid', 'ask', 'volume']):
            results['liquidity_quality'] = {
                'spread_efficiency': self._analyze_spread_efficiency(df),
                'depth_stability': self._analyze_depth_stability(df),
                'resilience': self._calculate_market_resilience(df)
            }
        
        # Trading costs analysis
        if all(col in df.columns for col in ['price', 'volume']):
            results['trading_costs'] = {
                'implementation_shortfall': self._calculate_implementation_shortfall(df),
                'market_impact': self._calculate_market_impact_costs(df),
                'timing_cost': self._calculate_timing_costs(df)
            }
        
        # Market stability
        results['stability'] = {
            'price_stability': self._analyze_price_stability(df),
            'liquidity_stability': self._analyze_liquidity_stability(df),
            'structural_breaks': self.detect_structural_breaks(df['price'])
        }
        
        return results
    
    def _fit_distribution(self, series: pd.Series) -> Dict[str, float]:
        """Fit statistical distribution to data"""
        params = stats.norm.fit(series)
        return {
            'mean': params[0],
            'std': params[1],
            'skewness': stats.skew(series),
            'kurtosis': stats.kurtosis(series)
        }
    
    def _analyze_time_clustering(self, intervals: pd.Series) -> Dict[str, float]:
        """Analyze temporal clustering of events"""
        acf_values = acf(intervals, nlags=10)
        return {
            'clustering_coefficient': acf_values[1],
            'decay_rate': np.polyfit(range(len(acf_values)), np.log(np.abs(acf_values)), 1)[0]
        }
    
    def _analyze_price_clustering(self, prices: pd.Series) -> Dict[str, float]:
        """Analyze price clustering tendencies"""
        decimals = (prices % 1) * 100
        round_numbers = [0, 25, 50, 75, 100]
        clustering = {}
        for num in round_numbers:
            clustering[f'cluster_{num}'] = np.sum(np.abs(decimals - num) < 1) / len(decimals)
        return clustering
    
    def _decompose_spread(self, df: pd.DataFrame) -> Dict[str, float]:
        """Decompose bid-ask spread into components"""
        spreads = df['ask'] - df['bid']
        midpoint = (df['ask'] + df['bid']) / 2
        returns = midpoint.pct_change()
        
        # Estimate components using Roll's model
        roll_cov = np.cov(returns[1:], returns[:-1])[0,1]
        roll_spread = 2 * np.sqrt(-roll_cov) if roll_cov < 0 else np.nan
        
        return {
            'roll_implied': roll_spread,
            'effective_spread': 2 * np.abs(df['price'] - midpoint).mean(),
            'realized_spread': (df['price'].shift(-1) - df['price']).mean()
        }
    
    def _calculate_resiliency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market resiliency metrics"""
        if 'price' not in df.columns:
            return {}
            
        # Calculate price changes
        returns = df['price'].pct_change()
        
        # Fit mean reversion model
        model = stats.linregress(returns[:-1], returns[1:])
        
        return {
            'mean_reversion': -model.slope,
            'half_life': -np.log(2) / np.log(abs(model.slope)) if model.slope != 0 else np.inf,
            'r_squared': model.rvalue ** 2
        }
    
    def _detect_jumps(self, returns: pd.Series) -> List[datetime]:
        """Detect price jumps using threshold method"""
        # Calculate threshold
        std = returns.std()
        threshold = 4 * std  # 4 standard deviations
        
        # Find jumps
        jumps = returns[np.abs(returns) > threshold]
        return list(jumps.index)
    
    def _decompose_volatility(self, returns: pd.Series) -> Tuple[float, float]:
        """Decompose volatility into continuous and jump components"""
        # Threshold for jump detection
        threshold = 4 * returns.std()
        
        # Separate jumps
        jumps = returns[np.abs(returns) > threshold]
        continuous = returns[np.abs(returns) <= threshold]
        
        # Calculate components
        continuous_vol = continuous.var()
        jump_vol = jumps.var() if len(jumps) > 0 else 0
        
        return continuous_vol, jump_vol
    
    def _analyze_vol_term_structure(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility term structure"""
        if 'implied_vol' not in df.columns or 'maturity' not in df.columns:
            return {}
            
        # Group by maturity
        grouped = df.groupby('maturity')['implied_vol'].mean()
        
        # Fit term structure curve
        x = np.array([(date - df.index[0]).days for date in grouped.index])
        y = grouped.values
        slope, intercept = np.polyfit(x, y, 1)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'term_premium': y[-1] - y[0] if len(y) > 1 else 0
        }
    
    def _calculate_flow_volatility(self, df: pd.DataFrame) -> float:
        """Calculate order flow volatility"""
        if 'volume' not in df.columns:
            return 0.0
            
        # Calculate net order flow
        price_changes = df['price'].diff()
        signed_volume = df['volume'] * np.sign(price_changes)
        
        return signed_volume.std()
    
    def _calculate_vpin(self, df: pd.DataFrame, bucket_size: Optional[int] = None) -> float:
        """Calculate Volume-synchronized Probability of Informed Trading"""
        if 'volume' not in df.columns:
            return 0.0
            
        if bucket_size is None:
            bucket_size = len(df) // 50  # Default to 50 buckets
            
        # Calculate volume buckets
        df['cum_volume'] = df['volume'].cumsum()
        bucket_volume = df['volume'].sum() / bucket_size
        
        # Calculate buy/sell volume in each bucket
        price_changes = df['price'].diff()
        df['buy_volume'] = df['volume'] * (price_changes > 0)
        df['sell_volume'] = df['volume'] * (price_changes < 0)
        
        vpin_values = []
        for i in range(bucket_size):
            bucket = df[(df['cum_volume'] >= i * bucket_volume) & 
                       (df['cum_volume'] < (i + 1) * bucket_volume)]
            vpin = abs(bucket['buy_volume'].sum() - bucket['sell_volume'].sum()) / bucket['volume'].sum()
            vpin_values.append(vpin)
            
        return np.mean(vpin_values)
    
    def _analyze_trade_initiation(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze trade initiation patterns"""
        midpoint = (df['bid'] + df['ask']) / 2
        
        # Classify trades
        buy_initiated = df['price'] > midpoint
        sell_initiated = df['price'] < midpoint
        
        return {
            'buy_ratio': buy_initiated.mean(),
            'sell_ratio': sell_initiated.mean(),
            'midpoint_trades': 1 - (buy_initiated.mean() + sell_initiated.mean())
        }
    
    def _estimate_adverse_selection(self, df: pd.DataFrame) -> float:
        """Estimate adverse selection component of spread"""
        if not all(col in df.columns for col in ['bid', 'ask', 'price']):
            return 0.0
            
        spreads = df['ask'] - df['bid']
        midpoint = (df['ask'] + df['bid']) / 2
        midpoint_changes = midpoint.diff()
        
        # Estimate adverse selection using price impact
        trade_indicator = np.sign(df['price'] - midpoint)
        model = stats.linregress(trade_indicator[:-1], midpoint_changes[1:])
        
        return model.slope / spreads.mean() if spreads.mean() != 0 else 0
