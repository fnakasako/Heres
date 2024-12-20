from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
from enum import Enum

@dataclass
class LLMContext:
    """Context structure optimized for LLM consumption"""
    timestamp: datetime
    context_type: str
    summary: str  # Concise, token-optimized summary
    key_points: List[str]  # Bullet points for easy parsing
    confidence: float
    metadata: Dict[str, Any]  # RAG-friendly metadata

@dataclass
class LLMInsight:
    """Insight structure optimized for LLM strategy generation"""
    id: str
    timestamp: datetime
    insight_type: str
    summary: str
    confidence: float
    trading_relevance: float  # 0-1 score of relevance to trading
    context: List[LLMContext]  # Supporting contexts
    vectors: Dict[str, List[float]]  # Embedding vectors for RAG
    prompt_tags: List[str]  # Tags for prompt engineering

class InsightPriority(Enum):
    """Priority levels for insight processing"""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"         # Important but not urgent
    MEDIUM = "medium"     # Regular priority
    LOW = "low"          # Background information

class LLMBridge:
    """
    Bridge between esurio's mathematical insights and eualesco's LLM processing
    
    Key Features:
    1. Token Optimization
       - Concise, relevant summaries
       - Structured for minimal token usage
       - Priority-based content selection
    
    2. RAG Integration
       - Embedded vectors for similarity search
       - Metadata for efficient retrieval
       - Context hierarchies
    
    3. Prompt Engineering
       - Dynamic prompt templates
       - Context-aware formatting
       - Trading-specific structures
    
    4. Strategy Generation Focus
       - Trading-relevant filtering
       - Action-oriented formatting
       - Performance metric integration
    """
    
    def __init__(self, 
                 max_context_length: int = 2048,
                 confidence_threshold: float = 0.7,
                 relevance_threshold: float = 0.6):
        self.max_context_length = max_context_length
        self.confidence_threshold = confidence_threshold
        self.relevance_threshold = relevance_threshold
        
    def transform_insights(self, 
                         market_insights: List[Any],
                         priority: InsightPriority = InsightPriority.HIGH) -> List[LLMInsight]:
        """
        Transform esurio insights into LLM-optimized format
        
        Args:
            market_insights: Raw insights from esurio
            priority: Priority level for processing
            
        Returns:
            List of LLM-optimized insights
        """
        llm_insights = []
        
        for insight in market_insights:
            # Skip low confidence insights
            if insight.confidence < self.confidence_threshold:
                continue
                
            # Transform insight into LLM format
            llm_insight = self._transform_single_insight(insight, priority)
            
            # Skip if not trading relevant enough
            if llm_insight.trading_relevance < self.relevance_threshold:
                continue
                
            llm_insights.append(llm_insight)
            
        return llm_insights
    
    def _transform_single_insight(self, 
                                insight: Any,
                                priority: InsightPriority) -> LLMInsight:
        """Transform a single insight into LLM format"""
        # Generate concise summary
        summary = self._generate_summary(insight, priority)
        
        # Extract trading relevance
        trading_relevance = self._calculate_trading_relevance(insight)
        
        # Generate supporting contexts
        contexts = self._generate_contexts(insight, priority)
        
        # Generate embedding vectors for RAG
        vectors = self._generate_vectors(insight)
        
        # Generate prompt engineering tags
        prompt_tags = self._generate_prompt_tags(insight)
        
        return LLMInsight(
            id=f"insight_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            insight_type=insight.type,
            summary=summary,
            confidence=insight.confidence,
            trading_relevance=trading_relevance,
            context=contexts,
            vectors=vectors,
            prompt_tags=prompt_tags
        )
    
    def _generate_summary(self, insight: Any, priority: InsightPriority) -> str:
        """Generate token-optimized summary"""
        # Base summary components
        components = [
            f"Type: {insight.type}",
            f"Market Impact: {self._get_market_impact(insight)}",
            f"Key Finding: {self._get_key_finding(insight)}"
        ]
        
        # Add priority-specific details
        if priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]:
            components.extend([
                f"Action Required: {self._get_required_action(insight)}",
                f"Time Sensitivity: {self._get_time_sensitivity(insight)}"
            ])
            
        return " | ".join(components)
    
    def _calculate_trading_relevance(self, insight: Any) -> float:
        """Calculate trading relevance score"""
        factors = {
            'price_impact': self._estimate_price_impact(insight),
            'time_sensitivity': self._estimate_time_sensitivity(insight),
            'market_correlation': self._estimate_market_correlation(insight),
            'historical_significance': self._estimate_historical_significance(insight)
        }
        
        # Weighted average of factors
        weights = {
            'price_impact': 0.4,
            'time_sensitivity': 0.3,
            'market_correlation': 0.2,
            'historical_significance': 0.1
        }
        
        return sum(score * weights[factor] for factor, score in factors.items())
    
    def _generate_contexts(self, insight: Any, priority: InsightPriority) -> List[LLMContext]:
        """Generate supporting contexts"""
        contexts = []
        
        # Market context
        if 'market_data' in insight.supporting_evidence:
            contexts.append(self._create_market_context(insight))
            
        # Technical context
        if 'technical_analysis' in insight.supporting_evidence:
            contexts.append(self._create_technical_context(insight))
            
        # Theoretical context
        if priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]:
            contexts.append(self._create_theoretical_context(insight))
            
        # Risk context
        contexts.append(self._create_risk_context(insight))
        
        return contexts
    
    def _generate_vectors(self, insight: Any) -> Dict[str, List[float]]:
        """Generate embedding vectors for RAG"""
        vectors = {}
        
        # Market vector
        vectors['market'] = self._embed_market_data(insight)
        
        # Technical vector
        vectors['technical'] = self._embed_technical_data(insight)
        
        # Theoretical vector
        vectors['theoretical'] = self._embed_theoretical_data(insight)
        
        return vectors
    
    def _generate_prompt_tags(self, insight: Any) -> List[str]:
        """Generate tags for prompt engineering"""
        tags = []
        
        # Add type-based tags
        tags.extend(self._get_type_tags(insight))
        
        # Add market-based tags
        tags.extend(self._get_market_tags(insight))
        
        # Add strategy-based tags
        tags.extend(self._get_strategy_tags(insight))
        
        return tags
    
    def format_for_strategy_generation(self, 
                                     llm_insights: List[LLMInsight]) -> Dict[str, Any]:
        """
        Format insights specifically for strategy generation
        
        Returns a structured format optimized for LLM strategy generation:
        1. High-level market summary
        2. Key trading opportunities
        3. Risk factors
        4. Supporting evidence
        """
        return {
            'market_summary': self._generate_market_summary(llm_insights),
            'trading_opportunities': self._extract_trading_opportunities(llm_insights),
            'risk_factors': self._extract_risk_factors(llm_insights),
            'supporting_evidence': self._compile_supporting_evidence(llm_insights),
            'metadata': {
                'timestamp': datetime.now(),
                'confidence_range': self._get_confidence_range(llm_insights),
                'market_conditions': self._get_market_conditions(llm_insights)
            }
        }
    
    def _generate_market_summary(self, insights: List[LLMInsight]) -> Dict[str, Any]:
        """Generate concise market summary"""
        return {
            'overall_sentiment': self._calculate_overall_sentiment(insights),
            'key_drivers': self._identify_key_drivers(insights),
            'market_regime': self._identify_market_regime(insights),
            'anomalies': self._identify_anomalies(insights)
        }
    
    def _extract_trading_opportunities(self, insights: List[LLMInsight]) -> List[Dict[str, Any]]:
        """Extract and prioritize trading opportunities"""
        opportunities = []
        
        for insight in insights:
            if self._is_trading_opportunity(insight):
                opportunities.append({
                    'type': insight.insight_type,
                    'summary': insight.summary,
                    'confidence': insight.confidence,
                    'priority': self._calculate_opportunity_priority(insight),
                    'execution_parameters': self._extract_execution_parameters(insight)
                })
                
        return sorted(opportunities, 
                     key=lambda x: (x['priority'], x['confidence']), 
                     reverse=True)
    
    def _extract_risk_factors(self, insights: List[LLMInsight]) -> List[Dict[str, Any]]:
        """Extract and categorize risk factors"""
        return {
            'market_risks': self._extract_market_risks(insights),
            'execution_risks': self._extract_execution_risks(insights),
            'model_risks': self._extract_model_risks(insights),
            'systemic_risks': self._extract_systemic_risks(insights)
        }
    
    def _compile_supporting_evidence(self, insights: List[LLMInsight]) -> Dict[str, Any]:
        """Compile and structure supporting evidence"""
        return {
            'technical_evidence': self._compile_technical_evidence(insights),
            'fundamental_evidence': self._compile_fundamental_evidence(insights),
            'market_evidence': self._compile_market_evidence(insights),
            'theoretical_evidence': self._compile_theoretical_evidence(insights)
        }
