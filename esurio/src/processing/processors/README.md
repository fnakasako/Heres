# Esurio Processors

This directory contains the data processing components of the Esurio system, implementing sophisticated analysis pipelines for different types of market intelligence data.

## Architecture

### Base Processor
The `BaseProcessor` (`base_processor.py`) provides the core functionality and interface that all specialized processors inherit from. It handles:
- Data validation and preprocessing
- Missing value handling with configurable strategies
- Metrics tracking and reporting
- Database schema verification
- Caching mechanism
- Error handling and logging

### Specialized Processors

1. **Market Processor** (`market_processor.py`)
   - Market microstructure analysis
   - Technical indicators
   - Order flow analysis
   - Price impact assessment
   - Market quality metrics
   - Regime analysis

2. **News Processor** (`news_processor.py`)
   - NLP and sentiment analysis
   - Entity extraction and relationship mapping
   - Topic modeling
   - Narrative analysis
   - Source credibility assessment
   - Market impact analysis

3. **Social Processor** (`social_processor.py`)
   - Network analysis
   - Sentiment analysis
   - Influence measurement
   - Trend detection
   - Behavioral analysis
   - Impact assessment

4. **Economic Processor** (`economic_processor.py`)
   - Macroeconomic trend analysis
   - Regime detection
   - Leading indicators
   - Structural change analysis
   - Risk assessment
   - Forecasting

5. **Supply Processor** (`supply_processor.py`)
   - Supply chain network analysis
   - Disruption detection
   - Efficiency metrics
   - Bottleneck analysis
   - Inventory optimization
   - Cost analysis

## Data Flow

1. Data is received from spiders in a standardized format
2. Base processor validates and preprocesses the data
3. Specialized processor implements domain-specific analysis
4. Results are transformed into actionable insights
5. Insights are validated and returned

## Integration

Processors integrate with:
- Spider output format for data ingestion
- Database for persistent storage
- Cache for performance optimization
- Metrics system for monitoring
- Logging system for debugging

## Usage

Each processor follows a consistent pattern:

```python
# Initialize processor
processor = SpecializedProcessor()

# Process data
results = processor.process(data)

# Access insights
insights = results["insights"]
metrics = results["_metrics"]
```

## Adding New Processors

1. Create new class inheriting from `BaseProcessor`
2. Implement required abstract methods:
   - `_process_implementation()`
   - `generate_insights()`
   - `validate_insights()`
3. Add domain-specific analysis methods
4. Update validation rules in config

## Configuration

Processors are configured through:
- Base configuration (`config/base_config.yaml`)
- Validation rules
- Missing value strategies
- Processing parameters

## Error Handling

Processors implement comprehensive error handling:
- Input validation
- Processing errors
- Database errors
- Cache errors
- Insight validation

## Metrics

Each processor tracks:
- Processing time
- Success/failure rates
- Data quality metrics
- Domain-specific metrics
