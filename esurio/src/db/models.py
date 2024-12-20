"""
Database models for Esurio Market Intelligence System.
"""

from datetime import datetime
from typing import Dict, List

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                       Integer, String, Table, Text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.schema import Index

Base = declarative_base()

class BaseModel(Base):
    """Base model with common fields."""
    
    __abstract__ = True
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    source = Column(String(50), nullable=False)
    metadata = Column(JSON, nullable=True)

# Market Data Models
class MarketData(BaseModel):
    """Market price and volume data."""
    
    __tablename__ = "market_data"
    
    symbol = Column(String(20), nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    exchange = Column(String(50))
    
    __table_args__ = (
        Index("ix_market_data_symbol_timestamp", "symbol", "timestamp"),
        Index("ix_market_data_timestamp", "timestamp"),
    )

class CryptoData(BaseModel):
    """Cryptocurrency market data."""
    
    __tablename__ = "crypto_data"
    
    pair = Column(String(20), nullable=False)
    price = Column(Float, nullable=False)
    volume_24h = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    exchange = Column(String(50))
    
    __table_args__ = (
        Index("ix_crypto_data_pair_timestamp", "pair", "timestamp"),
        Index("ix_crypto_data_timestamp", "timestamp"),
    )

class OrderBook(BaseModel):
    """Order book snapshots."""
    
    __tablename__ = "order_books"
    
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bids = Column(JSON, nullable=False)  # List of [price, size] pairs
    asks = Column(JSON, nullable=False)  # List of [price, size] pairs
    exchange = Column(String(50))
    depth = Column(Integer)
    
    __table_args__ = (
        Index("ix_order_books_symbol_timestamp", "symbol", "timestamp"),
    )

# News and Social Media Models
class NewsArticle(BaseModel):
    """News articles and press releases."""
    
    __tablename__ = "news_articles"
    
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    url = Column(String(1000), nullable=False, unique=True)
    published_at = Column(DateTime, nullable=False)
    author = Column(String(100))
    category = Column(String(50))
    sentiment_score = Column(Float)
    
    entities = relationship("NewsEntity", back_populates="article")
    
    __table_args__ = (
        Index("ix_news_articles_published_at", "published_at"),
    )

class NewsEntity(BaseModel):
    """Named entities extracted from news articles."""
    
    __tablename__ = "news_entities"
    
    article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=False)
    entity_type = Column(String(50), nullable=False)  # company, person, location, etc.
    entity_value = Column(String(200), nullable=False)
    confidence = Column(Float)
    
    article = relationship("NewsArticle", back_populates="entities")
    
    __table_args__ = (
        Index("ix_news_entities_entity_type_value", "entity_type", "entity_value"),
    )

class SocialPost(BaseModel):
    """Social media posts and sentiment."""
    
    __tablename__ = "social_posts"
    
    platform = Column(String(50), nullable=False)
    post_id = Column(String(100), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    author = Column(String(100))
    published_at = Column(DateTime, nullable=False)
    sentiment_score = Column(Float)
    engagement_metrics = Column(JSON)  # likes, shares, comments, etc.
    
    entities = relationship("SocialEntity", back_populates="post")
    
    __table_args__ = (
        Index("ix_social_posts_published_at", "published_at"),
        Index("ix_social_posts_platform_author", "platform", "author"),
    )

class SocialEntity(BaseModel):
    """Named entities extracted from social posts."""
    
    __tablename__ = "social_entities"
    
    post_id = Column(Integer, ForeignKey("social_posts.id"), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_value = Column(String(200), nullable=False)
    confidence = Column(Float)
    
    post = relationship("SocialPost", back_populates="entities")
    
    __table_args__ = (
        Index("ix_social_entities_entity_type_value", "entity_type", "entity_value"),
    )

# Economic Data Models
class EconomicIndicator(BaseModel):
    """Economic indicators and metrics."""
    
    __tablename__ = "economic_indicators"
    
    indicator = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    frequency = Column(String(20), nullable=False)  # daily, monthly, quarterly, etc.
    region = Column(String(50))
    unit = Column(String(20))
    adjustment = Column(String(50))  # seasonally adjusted, etc.
    
    __table_args__ = (
        Index("ix_economic_indicators_indicator_timestamp", "indicator", "timestamp"),
        Index("ix_economic_indicators_region_timestamp", "region", "timestamp"),
    )

# Supply Chain Models
class SupplyChainMetric(BaseModel):
    """Supply chain and logistics metrics."""
    
    __tablename__ = "supply_chain_metrics"
    
    metric_type = Column(String(50), nullable=False)  # container_rates, port_congestion, etc.
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    region = Column(String(50), nullable=False)
    sub_region = Column(String(50))
    route = Column(String(100))
    unit = Column(String(20))
    
    __table_args__ = (
        Index("ix_supply_chain_metrics_type_timestamp", "metric_type", "timestamp"),
        Index("ix_supply_chain_metrics_region_timestamp", "region", "timestamp"),
    )

# Earnings Models
class EarningsEvent(BaseModel):
    """Corporate earnings events and estimates."""
    
    __tablename__ = "earnings_events"
    
    symbol = Column(String(20), nullable=False)
    event_date = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    eps_estimate = Column(Float)
    eps_actual = Column(Float)
    revenue_estimate = Column(Float)
    revenue_actual = Column(Float)
    whisper_eps = Column(Float)
    pre_announcement = Column(Boolean, default=False)
    
    __table_args__ = (
        Index("ix_earnings_events_symbol_date", "symbol", "event_date"),
        Index("ix_earnings_events_date", "event_date"),
    )

class SECFiling(BaseModel):
    """SEC filings and documents."""
    
    __tablename__ = "sec_filings"
    
    company = Column(String(20), nullable=False)
    form_type = Column(String(20), nullable=False)
    filing_date = Column(DateTime, nullable=False)
    period_end = Column(DateTime)
    accession_number = Column(String(50), unique=True)
    document_url = Column(String(500))
    extracted_text = Column(Text)
    filing_sentiment = Column(Float)
    
    __table_args__ = (
        Index("ix_sec_filings_company_type_date", "company", "form_type", "filing_date"),
        Index("ix_sec_filings_date", "filing_date"),
    )

# Feature Store Models
class Feature(BaseModel):
    """Computed features for ML models."""
    
    __tablename__ = "features"
    
    feature_key = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    entity = Column(String(100), nullable=False)  # symbol, region, etc.
    feature_type = Column(String(50))  # technical, fundamental, sentiment, etc.
    window = Column(Integer)  # time window for feature computation
    
    __table_args__ = (
        Index("ix_features_key_entity_timestamp", "feature_key", "entity", "timestamp"),
        Index("ix_features_timestamp", "timestamp"),
    )

# Insight Models
class Insight(BaseModel):
    """Generated market insights."""
    
    __tablename__ = "insights"
    
    insight_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    entities = Column(JSON)  # List of related entities
    signals = Column(JSON)  # Contributing signals
    frameworks = Column(JSON)  # Theoretical frameworks used
    
    __table_args__ = (
        Index("ix_insights_type_timestamp", "insight_type", "timestamp"),
        Index("ix_insights_confidence", "confidence"),
    )

# Strategy Models
class Strategy(BaseModel):
    """Trading strategies."""
    
    __tablename__ = "strategies"
    
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    parameters = Column(JSON)
    performance_metrics = Column(JSON)
    risk_metrics = Column(JSON)
    status = Column(String(20))  # active, inactive, testing, etc.
    last_execution = Column(DateTime)
    
    signals = relationship("Signal", back_populates="strategy")

class Signal(BaseModel):
    """Trading signals generated by strategies."""
    
    __tablename__ = "signals"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    signal_type = Column(String(20), nullable=False)  # buy, sell, etc.
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    confidence = Column(Float, nullable=False)
    parameters = Column(JSON)
    execution_metrics = Column(JSON)
    
    strategy = relationship("Strategy", back_populates="signals")
    
    __table_args__ = (
        Index("ix_signals_strategy_timestamp", "strategy_id", "timestamp"),
        Index("ix_signals_symbol_timestamp", "symbol", "timestamp"),
    )
