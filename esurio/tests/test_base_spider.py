"""
Tests for the Base Spider of Esurio Market Intelligence System.
"""

import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from scrapy import Request
from scrapy.http import Response, TextResponse
from scrapy.utils.test import get_crawler

from src.scraping.spiders.base_spider import BaseSpider

class TestSpider(BaseSpider):
    """Test implementation of BaseSpider."""
    name = "test_spider"
    
    def parse(self, response):
        """Test parse method."""
        return {"test": "data"}

class TestBaseSpider(unittest.TestCase):
    """Test suite for BaseSpider."""
    
    def setUp(self):
        """Set up test environment."""
        self.crawler = get_crawler(TestSpider)
        self.spider = TestSpider.from_crawler(self.crawler)
        
        # Mock configuration
        self.mock_config = {
            "scraping": {
                "retry": {
                    "max_retries": 3,
                    "backoff_factor": 1.5,
                    "status_forcelist": [500, 502, 503, 504]
                },
                "timeout": {
                    "read": 30
                },
                "user_agent": "Test Bot 1.0"
            },
            "rate_limits": {
                "default": 2
            },
            "proxies": {
                "enabled": False
            },
            "validation": {
                "test_data": {
                    "field1": {
                        "type": int,
                        "min": 0,
                        "max": 100
                    },
                    "field2": {
                        "type": str,
                        "pattern": r"^test_.*$"
                    }
                }
            }
        }
        
        # Patch configuration loading
        self.config_patcher = patch(
            "src.scraping.spiders.base_spider.load_config",
            return_value=self.mock_config
        )
        self.mock_load_config = self.config_patcher.start()
        
        # Reset spider metrics
        self.spider.requests_made = 0
        self.spider.successful_requests = 0
        self.spider.failed_requests = 0
        self.spider.start_time = datetime.now()

    def tearDown(self):
        """Clean up after tests."""
        self.config_patcher.stop()

    def test_initialization(self):
        """Test spider initialization."""
        self.assertEqual(self.spider.name, "test_spider")
        self.assertEqual(self.spider.default_rate_limit, 2)
        self.assertEqual(self.spider.retry_attempts, 3)
        self.assertEqual(self.spider.backoff_factor, 1.5)
        self.assertFalse(self.spider.proxy_enabled)

    def test_make_request(self):
        """Test request creation."""
        url = "http://test.com"
        callback = lambda x: x
        
        request = self.spider.make_request(
            url=url,
            callback=callback,
            method="GET",
            headers={"Custom": "Header"},
            meta={"test": "meta"}
        )
        
        self.assertIsInstance(request, Request)
        self.assertEqual(request.url, url)
        self.assertEqual(request.callback, callback)
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.headers[b"Custom"], b"Header")
        self.assertEqual(request.meta["test"], "meta")
        self.assertEqual(request.meta["retry_times"], 0)
        self.assertEqual(request.meta["max_retries"], 3)
        self.assertEqual(request.meta["backoff_factor"], 1.5)
        self.assertFalse(request.meta["dont_retry"])
        self.assertFalse(request.meta["proxy_enabled"])

    def test_validate_data(self):
        """Test data validation."""
        # Valid data
        valid_data = {
            "field1": 50,
            "field2": "test_value"
        }
        self.assertTrue(
            self.spider.validate_data(valid_data, self.mock_config["validation"]["test_data"])
        )
        
        # Invalid type
        invalid_type = {
            "field1": "not_an_int",
            "field2": "test_value"
        }
        with self.assertRaises(ValueError):
            self.spider.validate_data(
                invalid_type,
                self.mock_config["validation"]["test_data"]
            )
        
        # Invalid range
        invalid_range = {
            "field1": 150,  # > max
            "field2": "test_value"
        }
        with self.assertRaises(ValueError):
            self.spider.validate_data(
                invalid_range,
                self.mock_config["validation"]["test_data"]
            )
        
        # Invalid pattern
        invalid_pattern = {
            "field1": 50,
            "field2": "invalid_pattern"
        }
        with self.assertRaises(ValueError):
            self.spider.validate_data(
                invalid_pattern,
                self.mock_config["validation"]["test_data"]
            )

    def test_handle_error(self):
        """Test error handling."""
        # Mock failure
        mock_failure = MagicMock()
        mock_failure.request = Request(url="http://test.com")
        mock_failure.value = Exception("Test error")
        
        # First retry
        result = self.spider.handle_error(mock_failure)
        self.assertIsInstance(result, Request)
        self.assertEqual(result.meta["retry_times"], 1)
        
        # Max retries exceeded
        mock_failure.request.meta["retry_times"] = 3
        result = self.spider.handle_error(mock_failure)
        self.assertIsNone(result)

    def test_parse_json_response(self):
        """Test JSON response parsing."""
        # Valid JSON
        valid_json = {"test": "data"}
        mock_response = TextResponse(
            url="http://test.com",
            body=json.dumps(valid_json).encode(),
            encoding="utf-8"
        )
        result = self.spider.parse_json_response(mock_response)
        self.assertEqual(result, valid_json)
        
        # Invalid JSON
        invalid_json = "not json"
        mock_response = TextResponse(
            url="http://test.com",
            body=invalid_json.encode(),
            encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            self.spider.parse_json_response(mock_response)

    def test_update_metrics(self):
        """Test metrics updating."""
        # Successful request
        self.spider.update_metrics(success=True)
        self.assertEqual(self.spider.requests_made, 1)
        self.assertEqual(self.spider.successful_requests, 1)
        self.assertEqual(self.spider.failed_requests, 0)
        
        # Failed request
        self.spider.update_metrics(success=False)
        self.assertEqual(self.spider.requests_made, 2)
        self.assertEqual(self.spider.successful_requests, 1)
        self.assertEqual(self.spider.failed_requests, 1)

    def test_closed(self):
        """Test spider closing."""
        # Mock some metrics
        self.spider.requests_made = 10
        self.spider.successful_requests = 8
        self.spider.failed_requests = 2
        
        # Close spider
        self.spider.closed(reason="finished")
        
        # Metrics should remain unchanged
        self.assertEqual(self.spider.requests_made, 10)
        self.assertEqual(self.spider.successful_requests, 8)
        self.assertEqual(self.spider.failed_requests, 2)

if __name__ == "__main__":
    unittest.main()
