# Esurio Database Layer

This directory contains the database components of the Esurio system, providing persistent storage and data processing capabilities for market intelligence data.

## Architecture

### Core Components

1. **Models** (`models.py`)
   - SQLAlchemy ORM models
   - Schema definitions
   - Relationship mappings
   - Data validation rules
   - Index optimizations

2. **Session Management** (`session.py`)
   - Connection pooling
   - Transaction handling
   - Context managers
   - Error handling
   - Performance optimization

### Data Models

1. **Market Data**
   - `MarketData`: Price and volume data
   - `CryptoData`: Cryptocurrency data
   - `OrderBook`: Order book snapshots

2. **News & Social**
   - `NewsArticle`: News articles and releases
   - `NewsEntity`: Named entities from news
   - `SocialPost`: Social media content
   - `SocialEntity`: Entities from social posts

3. **Economic Data**
   - `EconomicIndicator`: Economic metrics
   - `SupplyChainMetric`: Supply chain data

4. **Corporate Data**
   - `EarningsEvent`: Earnings reports
   - `SECFiling`: Regulatory filings

5. **Analysis Data**
   - `Feature`: Computed features
   - `Insight`: Generated insights
   - `Strategy`: Trading strategies
   - `Signal`: Trading signals

## Role in Data Pipeline

The database layer serves as a primary data processing component in the esurio system, providing an alternative to the models directory for data processing and transformation:

### Database vs Models Directory

While both the database layer and models directory handle data processing, they serve different but complementary purposes:

1. **Database Layer Processing**
   - Persistent storage and retrieval
   - Real-time data transformation
   - Relationship management
   - Data validation and normalization
   - Feature computation and storage
   - Historical data management
   - Transaction consistency
   - Cross-entity aggregations

2. **Models Directory Processing**
   - Machine learning model inference
   - Neural network operations
   - Batch predictions
   - Model-specific transformations
   - Feature engineering pipelines
   - Model artifacts storage
   - Inference caching
   - Model versioning

### Choosing Between Database and Models

1. **Use Database Layer When:**
   - Need persistent storage
   - Require ACID transactions
   - Working with relationships
   - Need real-time processing
   - Performing data aggregations
   - Managing historical data
   - Ensuring data consistency
   - Computing features

2. **Use Models Directory When:**
   - Running ML inference
   - Applying neural networks
   - Batch processing predictions
   - Working with model artifacts
   - Need model-specific transforms
   - Handling inference caching
   - Managing model versions
   - Computing ML features

### Interaction Between Components

The database layer and models directory work together in the data processing pipeline:

1. **Feature Engineering**
   - DB Layer: Computes and stores basic features
   - Models: Applies complex transformations
   - DB Layer: Stores transformed features
   - Models: Uses features for inference

2. **Data Flow**
   ```
   Raw Data → DB Layer → Basic Features → Models → Advanced Features → DB Layer
                      ↓                                                ↓
                   Storage                                         Storage
   ```

3. **Processing Pipeline**
   - DB Layer validates and normalizes data
   - Models apply ML transformations
   - DB Layer stores results
   - Models use stored features
   - DB Layer tracks lineage

4. **Caching Strategy**
   - DB Layer: Persistent feature store
   - Models: In-memory inference cache
   - DB Layer: Query result cache
   - Models: Model artifact cache

5. **Version Control**
   - DB Layer: Data schema versions
   - Models: Model versions
   - DB Layer: Feature versions
   - Models: Transform versions

### Integration Examples

Here are practical examples of how the database layer and models directory work together:

1. **Market Regime Detection**
```python
# DB Layer: Fetch and preprocess data
with session_scope() as session:
    # Get historical price data
    prices = session.query(MarketData)\
        .filter(MarketData.symbol == symbol)\
        .order_by(MarketData.timestamp)\
        .all()
    
    # Compute basic features
    features = compute_basic_features(prices)
    
    # Store basic features
    session.add_all([
        Feature(
            feature_key=f"basic_{k}",
            value=v,
            entity=symbol
        ) for k, v in features.items()
    ])

# Models: Apply ML transformation
regime_model = load_model('regime_detection')
regime_prediction = regime_model.predict(features)

# DB Layer: Store results
with session_scope() as session:
    session.add(
        Insight(
            insight_type="market_regime",
            content=regime_prediction,
            confidence=0.85
        )
    )
```

2. **Sentiment Analysis Pipeline**
```python
# DB Layer: Get news data
with session_scope() as session:
    articles = session.query(NewsArticle)\
        .filter(NewsArticle.published_at > cutoff_date)\
        .all()
    
    # Extract text features
    texts = [article.content for article in articles]

# Models: Apply NLP model
sentiment_model = load_model('sentiment_analysis')
sentiments = sentiment_model.predict(texts)

# DB Layer: Store enriched data
with session_scope() as session:
    for article, sentiment in zip(articles, sentiments):
        article.sentiment_score = sentiment
    session.commit()
```

3. **Feature Engineering Pipeline**
```python
# DB Layer: Get raw data
with session_scope() as session:
    raw_data = session.query(MarketData).all()
    
    # Compute time-series features
    basic_features = compute_ts_features(raw_data)
    
    # Store basic features
    store_features(session, basic_features)

# Models: Generate advanced features
ml_model = load_model('feature_engineering')
advanced_features = ml_model.transform(basic_features)

# DB Layer: Store advanced features
with session_scope() as session:
    store_features(session, advanced_features)
```

These examples demonstrate how:
- DB Layer handles data access and storage
- Models perform complex transformations
- Results flow back to DB Layer
- Features are incrementally built
- Both components maintain state

The database layer serves as a critical data processing component in the esurio system, handling both storage and transformation of market intelligence data:

### 1. Data Flow
```
Spiders → DB Layer → Processors → DB Layer → Quant Layer
         ↑_______________________________↓
```

The DB layer acts as both a sink and source, processing data at each stage:
- Validates and normalizes incoming spider data
- Provides processed data to processors
- Stores processed results and insights
- Serves as feature store for ML models
- Maintains data lineage and relationships

### 2. Processing Capabilities

1. **Data Normalization**
   - Type conversion and validation
   - Schema enforcement
   - Relationship mapping
   - Constraint checking
   - Format standardization

2. **Data Enrichment**
   - Automatic timestamp handling
   - Metadata attachment
   - Relationship inference
   - Index maintenance
   - Audit trail creation

3. **Data Aggregation**
   - Time-based grouping
   - Cross-entity aggregation
   - Statistical computations
   - Feature generation
   - Metric calculation

4. **Data Quality**
   - Constraint enforcement
   - Relationship validation
   - Duplicate detection
   - Anomaly identification
   - Consistency checks

### 3. Processing Workflow

1. **Ingestion Processing**
   - Schema validation
   - Type conversion
   - Relationship mapping
   - Constraint checking
   - Metadata enrichment

2. **Query Processing**
   - Query optimization
   - Result filtering
   - Join optimization
   - Aggregation computation
   - Cache management

3. **Update Processing**
   - Transaction management
   - Constraint validation
   - Relationship maintenance
   - Index updates
   - Audit logging

## Data Operations

### 1. Data Ingestion
```python
with session_scope() as session:
    # Create new market data entry
    market_data = MarketData(
        symbol="AAPL",
        price=150.0,
        volume=1000000,
        timestamp=datetime.utcnow()
    )
    session.add(market_data)
```

### 2. Data Validation
- Type checking
- Constraint validation
- Relationship integrity
- Index optimization
- Error handling

### 3. Data Transformation
- Automatic timestamps
- JSON field handling
- Relationship management
- Metadata tracking
- Audit logging

### 4. Data Access
```python
# Using context manager
with session_scope() as session:
    # Query with filtering
    results = session.query(MarketData)\
        .filter(MarketData.symbol == "AAPL")\
        .order_by(MarketData.timestamp.desc())\
        .limit(100)\
        .all()
```

## Integration

### With Spiders
- Receives scraped data
- Validates format
- Stores persistently
- Tracks metadata
- Manages relationships

### With Processors
- Provides data access
- Handles transactions
- Manages connections
- Ensures consistency
- Optimizes queries

### With Quant Layer
- Stores features
- Persists insights
- Manages strategies
- Tracks signals
- Records performance

## Performance Features

1. **Connection Pooling**
   - Configurable pool size
   - Connection recycling
   - Timeout handling
   - Overflow management

2. **Query Optimization**
   - Strategic indexing
   - Query planning
   - Result caching
   - Batch processing

3. **Transaction Management**
   - ACID compliance
   - Automatic rollback
   - Deadlock handling
   - Connection verification

## Configuration

Database settings in `config/base_config.yaml`:
```yaml
database:
  url: postgresql://localhost/esurio
  pool_size: 5
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 3600
  echo: false
```

## Usage Examples

### Basic Operations
```python
# Get database session
with session_scope() as session:
    # Create
    new_insight = Insight(
        insight_type="market_regime",
        content="Bullish trend detected",
        confidence=0.85
    )
    session.add(new_insight)
    
    # Read
    insights = session.query(Insight)\
        .filter(Insight.confidence > 0.8)\
        .all()
    
    # Update
    insight.confidence = 0.9
    
    # Delete
    session.delete(insight)
```

### Complex Queries
```python
# Join operations
results = session.query(
    MarketData, NewsArticle
).join(
    NewsArticle,
    MarketData.timestamp == NewsArticle.published_at
).filter(
    MarketData.symbol == "AAPL"
).all()

# Aggregations
metrics = session.query(
    MarketData.symbol,
    func.avg(MarketData.price).label('avg_price'),
    func.sum(MarketData.volume).label('total_volume')
).group_by(
    MarketData.symbol
).all()
```

## Error Handling

1. **Connection Errors**
   - Automatic retry
   - Connection validation
   - Pool management
   - Error logging

2. **Transaction Errors**
   - Automatic rollback
   - State recovery
   - Deadlock resolution
   - Constraint handling

3. **Query Errors**
   - Type validation
   - Constraint checking
   - Result verification
   - Error reporting

## Best Practices

1. **Session Management**
   - Always use context managers
   - Keep sessions short-lived
   - Handle errors properly
   - Commit explicitly

2. **Query Optimization**
   - Use appropriate indexes
   - Minimize joins
   - Batch operations
   - Profile queries

3. **Data Integrity**
   - Validate inputs
   - Use transactions
   - Handle constraints
   - Maintain relationships

## Troubleshooting and Common Patterns

### 1. Processing Pipeline Issues

1. **Data Validation Failures**
```python
# Problem: Invalid data from spiders
try:
    with session_scope() as session:
        # Validate before processing
        if not validate_spider_data(data):
            # Log and store invalid data for analysis
            store_invalid_data(session, data)
            raise ValueError("Invalid spider data")
            
        # Process valid data
        process_spider_data(session, data)
except ValueError as e:
    handle_validation_error(e)
```

2. **Processing Chain Breaks**
```python
# Problem: Incomplete processing chain
def process_with_recovery(data):
    try:
        with session_scope() as session:
            # Store processing state
            state = ProcessingState(
                data_id=data.id,
                stage="started"
            )
            session.add(state)
            
            # Process with checkpoints
            basic_features = process_basic(data)
            state.stage = "basic_complete"
            
            advanced_features = process_advanced(basic_features)
            state.stage = "advanced_complete"
            
            session.commit()
            
    except Exception as e:
        # Resume from last checkpoint
        resume_processing(data.id, state.stage)
```

### 2. Common Processing Patterns

1. **Batch Processing**
```python
def process_in_batches(data, batch_size=1000):
    with session_scope() as session:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Process batch
            processed = process_batch(batch)
            
            # Store results
            session.bulk_insert_mappings(
                ProcessedData,
                processed
            )
            
            # Commit each batch
            session.commit()
```

2. **Incremental Processing**
```python
def process_incremental(last_processed=None):
    with session_scope() as session:
        # Get new data since last processing
        query = session.query(RawData)
        if last_processed:
            query = query.filter(
                RawData.timestamp > last_processed
            )
            
        # Process in chunks
        for chunk in query.yield_per(1000):
            process_chunk(chunk)
            last_processed = chunk.timestamp
            
        return last_processed
```

### 3. Performance Patterns

1. **Efficient Joins**
```python
# Problem: Slow joins
# Solution: Use joins efficiently
def get_enriched_data():
    with session_scope() as session:
        return session.query(
            MarketData.symbol,
            MarketData.price,
            Insight.content
        ).outerjoin(
            Insight,
            and_(
                MarketData.symbol == Insight.symbol,
                MarketData.timestamp == Insight.timestamp
            )
        ).filter(
            MarketData.timestamp >= start_date
        ).options(
            joinedload(MarketData.insights)
        ).all()
```

2. **Caching Strategy**
```python
# Problem: Repeated computations
# Solution: Implement caching
def get_processed_data(key, compute_func):
    with session_scope() as session:
        # Check cache
        cached = session.query(CachedResult)\
            .filter(CachedResult.key == key)\
            .first()
            
        if cached and not is_stale(cached):
            return cached.result
            
        # Compute if not cached
        result = compute_func()
        
        # Update cache
        store_in_cache(session, key, result)
        return result
```

These patterns help handle common scenarios when using the database layer for data processing, including:
- Error recovery
- Batch processing
- Incremental updates
- Performance optimization
- State management
