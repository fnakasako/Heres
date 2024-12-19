"""
Configuration settings for the Financial Insights project.
"""

# API Settings
SEC_EDGAR_RATE_LIMIT = 0.1  # Maximum requests per second for SEC EDGAR
NEWS_API_RATE_LIMIT = 1.0   # Maximum requests per second for News API

# File Paths
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"

# Scraping Settings
DEFAULT_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Analysis Settings
SENTIMENT_ANALYSIS = {
    'min_document_length': 100,  # Minimum number of words for sentiment analysis
    'language': 'english',
    'additional_positive_terms': [
        'breakthrough', 'innovation', 'patent', 'partnership',
        'expansion', 'milestone', 'superior', 'exceptional'
    ],
    'additional_negative_terms': [
        'investigation', 'lawsuit', 'penalty', 'fine',
        'recall', 'deficit', 'downturn', 'restructuring'
    ]
}

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'financial_insights.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'scrapers': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'analysis': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Cache Settings
CACHE = {
    'enabled': True,
    'backend': 'redis',
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
    'ttl': 86400  # Cache TTL in seconds (24 hours)
}

# Database Settings
DATABASE = {
    'engine': 'sqlite',  # or 'postgresql'
    'name': 'financial_insights.db',
    'user': '',  # Required for PostgreSQL
    'password': '',  # Required for PostgreSQL
    'host': '',  # Required for PostgreSQL
    'port': '',  # Required for PostgreSQL
}

# Feature Flags
FEATURES = {
    'use_cache': True,
    'store_raw_data': True,
    'parallel_processing': True,
    'advanced_sentiment_analysis': True,
    'real_time_updates': False
}

# Error Handling
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
TIMEOUT = 30  # seconds

# Processing Settings
BATCH_SIZE = 100
MAX_WORKERS = 4  # Number of parallel workers for processing

# Export Settings
EXPORT_FORMATS = ['json', 'csv']
DEFAULT_EXPORT_FORMAT = 'json'
EXPORT_COMPRESSION = True
