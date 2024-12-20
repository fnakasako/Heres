"""
Base Spider for Esurio Market Intelligence System.
Provides common functionality for all specialized spiders.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import scrapy
from scrapy import Request, Spider
from scrapy.exceptions import NotConfigured
from scrapy.http import Response, TextResponse
from scrapy.utils.response import response_status_message

from src.common.config import load_config
from src.common.logging_util import setup_logging

logger = setup_logging()

class BaseSpider(Spider, ABC):
    """
    Base spider class implementing common functionality for all spiders.
    
    Features:
    - Configuration management
    - Rate limiting
    - Error handling and retries
    - Data validation
    - Proxy support
    - Metrics collection
    """

    name: str = "base_spider"
    
    def __init__(self, *args, **kwargs):
        """Initialize spider with configuration and common attributes."""
        super().__init__(*args, **kwargs)
        
        # Load configuration
        self.config = load_config()
        self.scraping_config = self.config.get("scraping", {})
        self.validation_rules = self.config.get("validation", {})
        
        # Set up rate limiting
        self.rate_limits = self.config.get("rate_limits", {})
        self.default_rate_limit = self.rate_limits.get("default", 2)
        
        # Error handling configuration
        self.retry_attempts = self.scraping_config.get("retry", {}).get("max_retries", 3)
        self.backoff_factor = self.scraping_config.get("retry", {}).get("backoff_factor", 1.5)
        self.retry_status_codes = self.scraping_config.get("retry", {}).get("status_forcelist", [500, 502, 503, 504])
        
        # Proxy configuration
        self.proxy_config = self.config.get("proxies", {})
        self.proxy_enabled = self.proxy_config.get("enabled", False)
        
        # Metrics
        self.requests_made = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = datetime.now()

    @abstractmethod
    def parse(self, response: Response) -> Any:
        """
        Abstract method to be implemented by specific spiders.
        
        Args:
            response: Scrapy Response object
            
        Returns:
            Parsed data in the format specific to the spider
        """
        pass

    def start_requests(self) -> List[Request]:
        """Generate initial requests for the spider."""
        raise NotImplementedError("Subclasses must implement start_requests()")

    def make_request(
        self, 
        url: str,
        callback: callable,
        method: str = "GET",
        headers: Optional[Dict] = None,
        body: Optional[Union[str, bytes]] = None,
        meta: Optional[Dict] = None,
        dont_filter: bool = False,
        errback: Optional[callable] = None,
        priority: int = 0,
    ) -> Request:
        """
        Create a new request with common configuration and middleware.
        
        Args:
            url: Target URL
            callback: Function to process the response
            method: HTTP method
            headers: Request headers
            body: Request body
            meta: Request metadata
            dont_filter: Whether to filter duplicate requests
            errback: Function to handle errors
            priority: Request priority
            
        Returns:
            Configured Request object
        """
        # Update headers with common configuration
        headers = headers or {}
        headers.update({
            "User-Agent": self.scraping_config.get("user_agent", "Esurio Bot 0.1.0"),
        })
        
        # Set up request metadata
        meta = meta or {}
        meta.update({
            "retry_times": 0,
            "max_retries": self.retry_attempts,
            "backoff_factor": self.backoff_factor,
            "dont_retry": False,
            "proxy_enabled": self.proxy_enabled,
            "download_timeout": self.scraping_config.get("timeout", {}).get("read", 30),
        })
        
        # Create request
        return Request(
            url=url,
            callback=callback,
            method=method,
            headers=headers,
            body=body,
            meta=meta,
            dont_filter=dont_filter,
            errback=errback or self.handle_error,
            priority=priority,
        )

    def validate_data(self, data: Dict, rules: Dict) -> bool:
        """
        Validate scraped data against defined rules.
        
        Args:
            data: Scraped data dictionary
            rules: Validation rules dictionary
            
        Returns:
            bool: Whether the data is valid
            
        Raises:
            ValueError: If validation fails
        """
        for field, value in data.items():
            if field in rules:
                rule = rules[field]
                
                # Check type
                if "type" in rule and not isinstance(value, rule["type"]):
                    raise ValueError(f"Invalid type for {field}: expected {rule['type']}, got {type(value)}")
                
                # Check range
                if "min" in rule and value < rule["min"]:
                    raise ValueError(f"Value for {field} below minimum: {value} < {rule['min']}")
                if "max" in rule and value > rule["max"]:
                    raise ValueError(f"Value for {field} above maximum: {value} > {rule['max']}")
                
                # Check pattern
                if "pattern" in rule and not rule["pattern"].match(str(value)):
                    raise ValueError(f"Value for {field} does not match pattern: {value}")
        
        return True

    def handle_error(self, failure):
        """
        Handle request failures and errors.
        
        Args:
            failure: Twisted Failure object
            
        Returns:
            None or new Request for retry
        """
        request = failure.request
        self.failed_requests += 1
        
        # Log error
        logger.error(
            f"Request failed: {failure.value}",
            extra={
                "spider": self.name,
                "url": request.url,
                "error": str(failure.value),
                "traceback": failure.getTraceback(),
            }
        )
        
        # Check if we should retry
        retry_times = request.meta.get("retry_times", 0)
        if retry_times < self.retry_attempts:
            # Calculate backoff delay
            delay = self.backoff_factor ** retry_times
            
            # Create new request with incremented retry count
            request.meta["retry_times"] = retry_times + 1
            request.meta["download_delay"] = delay
            
            logger.info(
                f"Retrying request (attempt {retry_times + 1}/{self.retry_attempts})",
                extra={
                    "spider": self.name,
                    "url": request.url,
                    "delay": delay,
                }
            )
            
            return request.copy()
        
        logger.error(
            f"Max retries reached for request",
            extra={
                "spider": self.name,
                "url": request.url,
                "max_retries": self.retry_attempts,
            }
        )
        return None

    def closed(self, reason):
        """
        Called when spider is closed.
        
        Args:
            reason: Why the spider was closed
        """
        duration = (datetime.now() - self.start_time).total_seconds()
        
        # Log final metrics
        logger.info(
            "Spider closed",
            extra={
                "spider": self.name,
                "reason": reason,
                "duration": duration,
                "requests_made": self.requests_made,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": self.successful_requests / max(self.requests_made, 1),
            }
        )

    def parse_json_response(self, response: TextResponse) -> Dict:
        """
        Parse JSON response with error handling.
        
        Args:
            response: Scrapy Response object
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValueError: If response is not valid JSON
        """
        try:
            return response.json()
        except ValueError as e:
            logger.error(
                "Failed to parse JSON response",
                extra={
                    "spider": self.name,
                    "url": response.url,
                    "error": str(e),
                    "response_text": response.text[:1000],  # First 1000 chars
                }
            )
            raise

    def update_metrics(self, success: bool = True):
        """
        Update spider metrics.
        
        Args:
            success: Whether the request was successful
        """
        self.requests_made += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
