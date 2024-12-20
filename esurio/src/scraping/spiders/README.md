# Esurio Spiders

This directory contains the web scraping components of the Esurio system, implementing specialized spiders for collecting different types of market intelligence data.

## Architecture

### Base Spider
The `BaseSpider` (`base_spider.py`) provides the core functionality and interface that all specialized spiders inherit from. It handles:
- Rate limiting
- Error handling and retries
- Proxy management
- Data validation
- Metrics collection
- Logging

### Specialized Spiders

1. **Market Spider** (`market_spider.py`)
   - Price data
   - Order book data
   - Trading volume
   - Market depth
   - Technical indicators
   - Exchange-specific data

2. **News Spider** (`news_spider.py`)
   - Financial news articles
   - Press releases
   - Regulatory filings
   - Company announcements
   - Industry reports
   - Market commentary

3. **Social Spider** (`social_spider.py`)
   - Social media posts
   - Forum discussions
   - Expert opinions
   - Market sentiment
   - Trading ideas
   - Community signals

4. **Economic Spider** (`economic_spider.py`)
   - Economic indicators
   - Central bank data
   - Government statistics
   - Policy announcements
   - Research reports
   - Forecast data

5. **Supply Spider** (`supply_spider.py`)
   - Supply chain data
   - Logistics information
   - Inventory levels
   - Shipping data
   - Production metrics
   - Supplier information

6. **Crypto Spider** (`crypto_spider.py`)
   - Cryptocurrency prices
   - Exchange data
   - Blockchain metrics
   - Network statistics
   - DeFi protocols
   - On-chain data

7. **Earnings Spider** (`earnings_spider.py`)
   - Earnings reports
   - Financial statements
   - Company metrics
   - Analyst estimates
   - Guidance updates
   - Conference calls

## Data Format

All spiders output data in a standardized format:
```python
{
    # Main data
    "data_field1": value1,
    "data_field2": value2,
    
    # Metadata and metrics (prefixed with _)
    "_timestamp": "2023-01-01T00:00:00Z",
    "_source": "source_name",
    "_metrics": {
        "requests": 10,
        "success_rate": 0.95,
        "latency": 0.5
    }
}
```

## Integration

Spiders integrate with:
- Processors for data analysis
- Database for storage
- Cache for performance
- Rate limiters for compliance
- Proxy system for reliability

## Usage

```python
# Initialize spider
spider = SpecializedSpider()

# Configure spider
spider.configure(settings)

# Run spider
results = spider.run()

# Access data and metrics
data = results["data"]
metrics = results["_metrics"]
```

## Adding New Spiders

1. Create new class inheriting from `BaseSpider`
2. Implement required methods:
   - `parse()`
   - `start_requests()`
3. Add spider-specific parsing logic
4. Update validation rules
5. Add error handling

## Configuration

Spiders are configured through:
- Scraping targets (`config/scraping_targets.yaml`)
- Rate limits
- Proxy settings
- Validation rules
- Retry policies

## Error Handling

Spiders implement robust error handling:
- Network errors
- Rate limiting
- Parse errors
- Validation errors
- Proxy failures

## Metrics

Each spider tracks:
- Request counts
- Success rates
- Response times
- Error rates
- Data quality metrics
- Resource usage

## Rate Limiting

Spiders respect rate limits through:
- Per-domain limits
- Concurrent request limits
- Backoff strategies
- Retry policies
- Queue management

## Validation

Data validation includes:
- Schema validation
- Data type checking
- Required fields
- Value ranges
- Format checking
- Relationship validation
