from abc import ABC, abstractmethod
from typing import Any, Dict, List
import time
from functools import wraps

class ScraperException(Exception):
    """Custom exception for scraping errors."""
    pass

def rate_limit(calls_per_second: float):
    """Decorator to implement rate limiting for API calls."""
    def decorator(func):
        last_call_time = 0
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_call_time
            current_time = time.time()
            time_diff = current_time - last_call_time
            if time_diff < 1/calls_per_second:
                time.sleep(1/calls_per_second - time_diff)
            result = func(*args, **kwargs)
            last_call_time = time.time()
            return result
        return wrapper
    return decorator

def robust_scrape(retries: int = 3):
    """Decorator to implement retry logic with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise ScraperException(f"Failed after {retries} attempts: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            return None
        return wrapper
    return decorator

class BaseScraper(ABC):
    """Abstract base class for all scrapers."""
    
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
    
    @abstractmethod
    def scrape(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Main scraping method to be implemented by concrete scrapers.
        
        Args:
            **kwargs: Additional arguments specific to the scraper implementation
            
        Returns:
            List[Dict[str, Any]]: List of scraped data items
        """
        pass
    
    @abstractmethod
    def clean(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean the scraped data.
        
        Args:
            data: Raw scraped data
            
        Returns:
            List[Dict[str, Any]]: Cleaned data
        """
        pass
    
    def run(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the complete scraping pipeline.
        
        Args:
            **kwargs: Additional arguments for the scrape method
            
        Returns:
            List[Dict[str, Any]]: Processed data
        """
        raw_data = self.scrape(**kwargs)
        cleaned_data = self.clean(raw_data)
        self.data = cleaned_data
        return cleaned_data
