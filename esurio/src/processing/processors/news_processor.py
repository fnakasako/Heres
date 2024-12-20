from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from collections import defaultdict
import yfinance as yf
from textblob import TextBlob
import re
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

from .base_processor import BaseProcessor

class NewsProcessor(BaseProcessor):
    """Advanced news processor with NLP and sentiment analysis"""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Initialize NLP models
        self.nlp = spacy.load("en_core_web_lg")
        
        # Initialize sentiment analyzers
        self.finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1  # Use CPU
        )
        
        # Initialize topic modeling
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.lda = LatentDirichletAllocation(
            n_components=10,
            random_state=42
        )
        
        # Initialize entity graph
        self.entity_graph = nx.Graph()
        
        # Initialize Word2Vec for custom embeddings
        self.word2vec = None  # Will be trained on corpus
        
        # Cache for financial data
        self.price_cache = {}
        
    def _process_implementation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Implement news-specific processing logic.
        
        Args:
            df: Preprocessed DataFrame with news data
            
        Returns:
            Dictionary containing processed results
        """
        # Preprocess text
        df['processed_text'] = df['content'].apply(self._preprocess_text)
        
        # Build corpus for training
        corpus = self._build_corpus(df['processed_text'])
        
        # Train Word2Vec if not trained
        if self.word2vec is None:
            self._train_word2vec(corpus)
        
        # Process data
        results = {
            'sentiment_analysis': self._analyze_sentiment(df),
            'entity_analysis': self._analyze_entities(df),
            'topic_modeling': self._analyze_topics(df),
            'relationship_mapping': self._map_relationships(df),
            'impact_analysis': self._analyze_market_impact(df),
            'temporal_analysis': self._analyze_temporal_patterns(df),
            'source_analysis': self._analyze_sources(df),
            'narrative_analysis': self._analyze_narratives(df),
            'credibility_analysis': self._analyze_credibility(df),
            'complexity_analysis': self._analyze_complexity(df)
        }
        
        return results
        
    def generate_insights(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate news-specific insights from processed data.
        
        Args:
            processed_data: Dictionary containing processed news data
            
        Returns:
            Dictionary containing generated insights
        """
        insights = {
            'key_findings': {
                'sentiment': self._summarize_sentiment(processed_data['sentiment_analysis']),
                'topics': self._summarize_topics(processed_data['topic_modeling']),
                'entities': self._summarize_entities(processed_data['entity_analysis'])
            },
            'market_implications': {
                'impact': self._assess_market_impact(processed_data['impact_analysis']),
                'trends': self._identify_trends(processed_data),
                'risks': self._identify_risks(processed_data)
            },
            'source_assessment': {
                'credibility': processed_data['credibility_analysis'],
                'bias': self._assess_bias(processed_data['source_analysis']),
                'coverage': self._assess_coverage(processed_data['source_analysis'])
            },
            'narrative_insights': {
                'main_narratives': self._extract_main_narratives(processed_data['narrative_analysis']),
                'competing_views': self._identify_competing_views(processed_data),
                'evolution': self._track_narrative_evolution(processed_data)
            }
        }
        
        return insights
        
    def validate_insights(self, insights: Dict[str, Any]) -> bool:
        """
        Validate news-specific insights.
        
        Args:
            insights: Dictionary containing generated insights
            
        Returns:
            Boolean indicating if insights are valid
        """
        try:
            # Validate required sections
            required_sections = ['key_findings', 'market_implications', 'source_assessment', 'narrative_insights']
            if not all(section in insights for section in required_sections):
                logger.error("Missing required insight sections")
                return False
                
            # Validate key findings
            findings = insights['key_findings']
            if not all(k in findings for k in ['sentiment', 'topics', 'entities']):
                logger.error("Invalid key findings format")
                return False
                
            # Validate market implications
            implications = insights['market_implications']
            if not all(k in implications for k in ['impact', 'trends', 'risks']):
                logger.error("Invalid market implications format")
                return False
                
            # Validate source assessment
            assessment = insights['source_assessment']
            if not all(k in assessment for k in ['credibility', 'bias', 'coverage']):
                logger.error("Invalid source assessment format")
                return False
                
            # Validate narrative insights
            narratives = insights['narrative_insights']
            if not all(k in narratives for k in ['main_narratives', 'competing_views', 'evolution']):
                logger.error("Invalid narrative insights format")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Insight validation error: {str(e)}")
            return False
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenization and lemmatization
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)
    
    def _build_corpus(self, texts: pd.Series) -> List[List[str]]:
        """Build corpus for training"""
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Detect common phrases
        phrases = Phrases(tokenized_texts, min_count=3, threshold=10)
        bigram = Phraser(phrases)
        
        # Apply phrase detection
        corpus = [bigram[text] for text in tokenized_texts]
        
        return corpus
    
    def _train_word2vec(self, corpus: List[List[str]]) -> None:
        """Train Word2Vec model on corpus"""
        self.word2vec = Word2Vec(
            sentences=corpus,
            vector_size=300,
            window=10,
            min_count=2,
            workers=4,
            sg=1  # Skip-gram
        )
    
    def _analyze_sentiment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform advanced sentiment analysis
        
        Features:
        - FinBERT sentiment
        - Aspect-based sentiment
        - Temporal sentiment
        - Source-specific sentiment
        - Entity-level sentiment
        """
        results = {}
        
        # FinBERT sentiment
        finbert_sentiments = []
        for text in df['content']:
            sentiment = self.finbert(text[:512])[0]  # Truncate to max length
            finbert_sentiments.append({
                'label': sentiment['label'],
                'score': sentiment['score']
            })
        
        # Aggregate sentiments
        sentiment_df = pd.DataFrame(finbert_sentiments)
        results['overall_sentiment'] = {
            'positive_ratio': (sentiment_df['label'] == 'positive').mean(),
            'negative_ratio': (sentiment_df['label'] == 'negative').mean(),
            'neutral_ratio': (sentiment_df['label'] == 'neutral').mean(),
            'average_score': sentiment_df['score'].mean()
        }
        
        # Aspect-based sentiment
        results['aspect_sentiment'] = self._analyze_aspect_sentiment(df)
        
        # Temporal sentiment analysis
        results['temporal_sentiment'] = self._analyze_temporal_sentiment(df)
        
        # Source-specific sentiment
        if 'source' in df.columns:
            results['source_sentiment'] = self._analyze_source_sentiment(df)
        
        # Entity-level sentiment
        results['entity_sentiment'] = self._analyze_entity_sentiment(df)
        
        return results
    
    def _analyze_entities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze named entities and their relationships
        
        Features:
        - Entity extraction
        - Entity relationships
        - Entity importance
        - Entity clustering
        """
        entities = defaultdict(list)
        relationships = []
        
        for text in df['content']:
            doc = self.nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'GPE', 'MONEY', 'PERCENT']:
                    entities[ent.label_].append(ent.text)
                    
                    # Extract relationships between entities
                    for other_ent in doc.ents:
                        if other_ent != ent:
                            relationships.append((ent.text, other_ent.text))
        
        # Build entity graph
        G = nx.Graph()
        for e1, e2 in relationships:
            if G.has_edge(e1, e2):
                G[e1][e2]['weight'] += 1
            else:
                G.add_edge(e1, e2, weight=1)
        
        # Calculate entity metrics
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        communities = list(nx.community.greedy_modularity_communities(G))
        
        return {
            'entities': {k: list(set(v)) for k, v in entities.items()},
            'entity_metrics': {
                'centrality': centrality,
                'betweenness': betweenness,
                'communities': [list(c) for c in communities]
            },
            'relationship_graph': G
        }
    
    def _analyze_topics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform topic modeling and analysis
        
        Features:
        - LDA topics
        - Topic evolution
        - Topic relationships
        - Key phrases
        """
        # Transform texts to TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(df['processed_text'])
        
        # Fit LDA
        lda_output = self.lda.fit_transform(tfidf_matrix)
        
        # Get feature names
        feature_names = self.tfidf.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics.append({
                'id': topic_idx,
                'words': top_words,
                'weight': float(topic.sum())
            })
        
        # Analyze topic evolution if timestamp available
        if 'timestamp' in df.columns:
            topic_evolution = self._analyze_topic_evolution(df, lda_output)
        else:
            topic_evolution = {}
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(df['content'])
        
        return {
            'topics': topics,
            'topic_evolution': topic_evolution,
            'topic_coherence': self._calculate_topic_coherence(topics),
            'key_phrases': key_phrases
        }
    
    def _map_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Map relationships between entities and concepts
        
        Features:
        - Entity-entity relationships
        - Entity-topic relationships
        - Causal relationships
        - Temporal relationships
        """
        # Entity co-occurrence network
        entity_network = self._build_entity_network(df)
        
        # Topic-entity relationships
        topic_entity_relations = self._analyze_topic_entity_relations(df)
        
        # Causal relationship extraction
        causal_relations = self._extract_causal_relations(df)
        
        # Temporal relationship analysis
        temporal_relations = self._analyze_temporal_relations(df)
        
        return {
            'entity_network': {
                'nodes': list(entity_network.nodes()),
                'edges': list(entity_network.edges(data=True))
            },
            'topic_entity_relations': topic_entity_relations,
            'causal_relations': causal_relations,
            'temporal_relations': temporal_relations
        }
    
    def _analyze_market_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market impact of news
        
        Features:
        - Price impact
        - Volume impact
        - Sentiment correlation
        - Lead-lag relationships
        """
        results = {}
        
        # Extract mentioned tickers
        tickers = self._extract_tickers(df)
        
        # Analyze price impact
        if tickers:
            price_impact = self._analyze_price_impact(df, tickers)
            volume_impact = self._analyze_volume_impact(df, tickers)
            
            results.update({
                'price_impact': price_impact,
                'volume_impact': volume_impact,
                'sentiment_correlation': self._analyze_sentiment_correlation(df, tickers),
                'lead_lag': self._analyze_lead_lag_relationships(df, tickers)
            })
        
        return results
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal patterns in news
        
        Features:
        - Publication patterns
        - Topic evolution
        - Sentiment trends
        - Event clustering
        """
        if 'timestamp' not in df.columns:
            return {}
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return {
            'publication_patterns': self._analyze_publication_patterns(df),
            'topic_trends': self._analyze_topic_trends(df),
            'sentiment_trends': self._analyze_sentiment_trends(df),
            'event_clusters': self._cluster_events(df)
        }
    
    def _analyze_sources(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze news sources
        
        Features:
        - Source reliability
        - Bias detection
        - Coverage patterns
        - Source relationships
        """
        if 'source' not in df.columns:
            return {}
            
        return {
            'source_reliability': self._analyze_source_reliability(df),
            'source_bias': self._analyze_source_bias(df),
            'coverage_patterns': self._analyze_coverage_patterns(df),
            'source_relationships': self._analyze_source_relationships(df)
        }
    
    def _analyze_narratives(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze narrative patterns
        
        Features:
        - Narrative identification
        - Narrative evolution
        - Competing narratives
        - Narrative impact
        """
        return {
            'narratives': self._identify_narratives(df),
            'narrative_evolution': self._analyze_narrative_evolution(df),
            'competing_narratives': self._identify_competing_narratives(df),
            'narrative_impact': self._analyze_narrative_impact(df)
        }
    
    def _analyze_credibility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze content credibility
        
        Features:
        - Source credibility
        - Content verification
        - Fact checking
        - Consistency analysis
        """
        return {
            'source_credibility': self._assess_source_credibility(df),
            'content_verification': self._verify_content(df),
            'fact_checking': self._check_facts(df),
            'consistency': self._analyze_consistency(df)
        }
    
    def _analyze_complexity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze content complexity
        
        Features:
        - Readability metrics
        - Technical complexity
        - Semantic complexity
        - Structure analysis
        """
        return {
            'readability': self._analyze_readability(df),
            'technical_complexity': self._analyze_technical_complexity(df),
            'semantic_complexity': self._analyze_semantic_complexity(df),
            'structure': self._analyze_content_structure(df)
        }
