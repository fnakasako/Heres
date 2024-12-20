"""
Base Processor for Esurio Market Intelligence System.
Provides common functionality for data processing and analysis.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.common.config import load_config
from src.common.logging_util import setup_logging
from src.db.session import session_scope

logger = setup_logging()

class BaseProcessor(ABC):
    """
    Abstract base class for all data processors in the Esurio system.
    Provides core functionality and interface requirements for specialized processors.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the base processor.

        Args:
            config: Optional configuration dictionary. If None, loads from default config.
        """
        self.config = config or load_config()
        self.batch_size = self.config.get('BATCH_SIZE', 1000)
        self.cache = {}
        
        # Spider-specific configurations
        self.scraping_config = self.config.get("scraping", {})
        self.validation_rules = self.config.get("validation", {})
        
        # Processing metrics
        self.processed_items = 0
        self.failed_items = 0
        self.processing_start_time = None
        
        self._initialize_processor()

    def _initialize_processor(self) -> None:
        """Initialize processor-specific resources and connections."""
        logger.info(f"Initializing {self.__class__.__name__}")
        self._setup_database()
        self._setup_cache()
        self._load_models()

    def _setup_database(self) -> None:
        """Set up database connections and verify schema."""
        with session_scope() as session:
            self._verify_schema(session)

    def _setup_cache(self) -> None:
        """Initialize caching mechanism."""
        self.cache_ttl = self.config.get('CACHE_TTL_SECONDS', 3600)
        self.cache = {}

    def _load_models(self) -> None:
        """Load any required ML models or preprocessing tools."""
        pass

    @abstractmethod
    def process(self, data: Union[pd.DataFrame, Dict, List]) -> Dict[str, Any]:
        """
        Process incoming data and generate insights.

        Args:
            data: Input data in supported format (DataFrame, Dict, or List)

        Returns:
            Dictionary containing processed results and insights
        """
        if self.processing_start_time is None:
            self.processing_start_time = datetime.now()
        
        try:
            # Validate input
            if not self.validate(data):
                self.failed_items += 1
                raise ValueError("Data validation failed")
            
            # Preprocess data
            df = self.preprocess(data)
            
            # Process data (to be implemented by subclasses)
            results = self._process_implementation(df)
            
            # Generate insights
            insights = self.generate_insights(results)
            
            # Validate insights
            if not self.validate_insights(insights):
                self.failed_items += 1
                raise ValueError("Insights validation failed")
            
            self.processed_items += 1
            
            # Add processing metrics
            processing_time = (datetime.now() - self.processing_start_time).total_seconds()
            metrics = {
                "processed_items": self.processed_items,
                "failed_items": self.failed_items,
                "processing_time": processing_time,
                "success_rate": self.processed_items / (self.processed_items + self.failed_items) if (self.processed_items + self.failed_items) > 0 else 0
            }
            
            return {
                "results": results,
                "insights": insights,
                "_metrics": metrics
            }
            
        except Exception as e:
            self.failed_items += 1
            logger.error(f"Processing error in {self.__class__.__name__}: {str(e)}")
            raise

    @abstractmethod
    def _process_implementation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Actual processing implementation to be provided by subclasses.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Dictionary containing processed results
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate input data format and content based on spider validation rules.

        Args:
            data: Input data to validate

        Returns:
            Boolean indicating if data is valid
        """
        if not isinstance(data, dict):
            logger.error(f"Invalid data type: expected dict, got {type(data)}")
            return False

        try:
            validation_rules = self.config.get("validation", {}).get(self.__class__.__name__, {})
            
            # Check required fields
            required_fields = validation_rules.get("required_fields", [])
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Check field types and constraints
            field_rules = validation_rules.get("fields", {})
            for field, value in data.items():
                if field in field_rules:
                    rule = field_rules[field]
                    
                    # Type validation
                    expected_type = rule.get("type")
                    if expected_type and not isinstance(value, expected_type):
                        logger.error(f"Invalid type for {field}: expected {expected_type}, got {type(value)}")
                        return False
                    
                    # Range validation
                    if "min" in rule and value < rule["min"]:
                        logger.error(f"Value for {field} below minimum: {value} < {rule['min']}")
                        return False
                    if "max" in rule and value > rule["max"]:
                        logger.error(f"Value for {field} above maximum: {value} > {rule['max']}")
                        return False
                    
                    # Pattern validation
                    if "pattern" in rule and not rule["pattern"].match(str(value)):
                        logger.error(f"Value for {field} does not match pattern: {value}")
                        return False

            # Validate metrics if present
            metrics = data.get("_metrics", {})
            if metrics:
                if not all(isinstance(v, (int, float)) for v in metrics.values()):
                    logger.error("Invalid metrics format")
                    return False

            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

    def preprocess(self, data: Any) -> pd.DataFrame:
        """
        Prepare scraped data for processing.

        Args:
            data: Raw input data from spider

        Returns:
            Preprocessed DataFrame
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict from spider, got {type(data)}")

        # Extract main data and metrics
        main_data = {k: v for k, v in data.items() if not k.startswith('_')}
        metrics = data.get('_metrics', {})
        
        # Convert to DataFrame
        if isinstance(main_data, dict):
            df = pd.DataFrame([main_data])
        elif isinstance(main_data, list):
            df = pd.DataFrame(main_data)
        else:
            raise ValueError(f"Unsupported data structure in spider output")

        # Add metrics as columns with '_metric_' prefix
        for metric_name, metric_value in metrics.items():
            df[f'_metric_{metric_name}'] = metric_value

        # Clean the DataFrame
        df = self._clean_dataframe(df)
        
        # Add processing metadata
        df['_processed_at'] = pd.Timestamp.now(tz='UTC')
        df['_processor'] = self.__class__.__name__
        
        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare DataFrame from spider data for processing.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates (excluding metric and metadata columns)
        data_cols = [col for col in df.columns if not col.startswith('_')]
        df = df.drop_duplicates(subset=data_cols)
        
        # Handle missing values (skip metric columns)
        df = self._handle_missing_values(df)
        
        # Convert timestamps
        df = self._standardize_timestamps(df)
        
        # Validate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not col.startswith('_'):  # Skip metric columns
                # Remove invalid numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check for anomalies
                mean, std = df[col].mean(), df[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in DataFrame based on spider data characteristics.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with handled missing values
        """
        # Skip metric and metadata columns
        data_cols = [col for col in df.columns if not col.startswith('_')]
        
        for col in data_cols:
            if col in df.columns:  # Check if column still exists
                # Get missing value strategy from config
                missing_strategy = self.config.get("missing_value_strategies", {}).get(col, "default")
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    if missing_strategy == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    elif missing_strategy == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif missing_strategy == "zero":
                        df[col] = df[col].fillna(0)
                    elif missing_strategy == "interpolate":
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    else:  # default for numeric
                        df[col] = df[col].fillna(df[col].mean())
                        
                elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    if missing_strategy == "mode":
                        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN")
                    elif missing_strategy == "constant":
                        fill_value = self.config.get("missing_value_constants", {}).get(col, "UNKNOWN")
                        df[col] = df[col].fillna(fill_value)
                    elif missing_strategy == "ffill":
                        df[col] = df[col].fillna(method='ffill')
                    elif missing_strategy == "bfill":
                        df[col] = df[col].fillna(method='bfill')
                    else:  # default for categorical/object
                        df[col] = df[col].fillna("UNKNOWN")
                        
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    if missing_strategy == "ffill":
                        df[col] = df[col].fillna(method='ffill')
                    elif missing_strategy == "bfill":
                        df[col] = df[col].fillna(method='bfill')
                    else:  # default for datetime
                        df[col] = df[col].fillna(pd.Timestamp.now(tz='UTC'))
        
        return df

    def _standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamp columns to standard format.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized timestamps
        """
        timestamp_cols = df.select_dtypes(include=['datetime64']).columns
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col], utc=True)
        return df

    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Retrieve cached result if available and not expired.

        Args:
            key: Cache key

        Returns:
            Cached value if available and valid, None otherwise
        """
        if key in self.cache:
            result, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return result
            else:
                del self.cache[key]
        return None

    def cache_result(self, key: str, value: Any) -> None:
        """
        Cache processing result with timestamp.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, datetime.now())

    def _verify_schema(self, session: Session) -> None:
        """
        Verify database schema matches expected structure.

        Args:
            session: SQLAlchemy session
        """
        try:
            # Get table names from database
            table_names = session.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()
            table_names = [t[0] for t in table_names]
            
            # Get expected tables from config
            expected_tables = self.config.get("database", {}).get("tables", {})
            
            # Verify all expected tables exist
            for table_name, schema in expected_tables.items():
                if table_name not in table_names:
                    logger.error(f"Missing required table: {table_name}")
                    raise ValueError(f"Missing required table: {table_name}")
                
                # Verify table schema
                columns = session.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'").fetchall()
                column_dict = {col[0]: col[1] for col in columns}
                
                for col_name, col_type in schema.items():
                    if col_name not in column_dict:
                        logger.error(f"Missing column {col_name} in table {table_name}")
                        raise ValueError(f"Missing column {col_name} in table {table_name}")
                    
                    if col_type.lower() != column_dict[col_name].lower():
                        logger.error(f"Invalid type for column {col_name} in table {table_name}: expected {col_type}, got {column_dict[col_name]}")
                        raise ValueError(f"Invalid type for column {col_name} in table {table_name}")
            
            logger.info("Database schema verification successful")
            
        except Exception as e:
            logger.error(f"Schema verification failed: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up resources and connections."""
        try:
            # Log final metrics
            if self.processing_start_time is not None:
                total_time = (datetime.now() - self.processing_start_time).total_seconds()
                logger.info(
                    "Processor cleanup",
                    extra={
                        "processor": self.__class__.__name__,
                        "total_time": total_time,
                        "processed_items": self.processed_items,
                        "failed_items": self.failed_items,
                        "success_rate": self.processed_items / (self.processed_items + self.failed_items) if (self.processed_items + self.failed_items) > 0 else 0
                    }
                )
            
            # Clear cache
            self.cache.clear()
            
            # Reset metrics
            self.processed_items = 0
            self.failed_items = 0
            self.processing_start_time = None
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self) -> 'BaseProcessor':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()

    @abstractmethod
    def generate_insights(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights from processed data.

        Args:
            processed_data: Dictionary containing processed data

        Returns:
            Dictionary containing generated insights
        """
        raise NotImplementedError

    @abstractmethod
    def validate_insights(self, insights: Dict[str, Any]) -> bool:
        """
        Validate generated insights.

        Args:
            insights: Dictionary containing generated insights

        Returns:
            Boolean indicating if insights are valid
        """
        raise NotImplementedError
