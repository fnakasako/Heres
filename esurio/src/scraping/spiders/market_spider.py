"""
Market Spider for Esurio Market Intelligence System.
Handles market data collection from various financial data sources.
"""

import json
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from scrapy import Request
from scrapy.exceptions import NotConfigured
from scrapy.http import Response

from src.scraping.spiders.base_spider import BaseSpider

class MarketSpider(BaseSpider):
    """
    Spider for collecting market data from various sources.
    
    Features:
    - Real-time and historical price data
    - Multiple data sources (Alpha Vantage, Yahoo Finance)
    - Configurable symbols and update frequencies
    - Data validation and normalization
    """

    name = "market_spider"
    
    def __init__(self, *args, **kwargs):
        """Initialize market spider with source-specific configurations."""
        super().__init__(*args, **kwargs)
        
        # Load market data specific configuration
        market_config = next(
            (cfg for cfg in self.config.get("market_data", [])
             if cfg["name"] == "stock_prices"),
            {}
        )
        
        if not market_config:
            raise NotConfigured("Market data configuration not found")
        
        self.sources = market_config.get("sources", [])
        if not self.sources:
            raise NotConfigured("No market data sources configured")
        
        # Source-specific API configurations
        self.source_configs = {
            source["name"]: {
                "url": source["url"],
                "api_key": source.get("api_key"),
                "endpoints": source.get("endpoints", []),
                "symbols": source.get("symbols", []),
                "rate_limit": source.get("rate_limit", self.default_rate_limit)
            }
            for source in self.sources
        }
        
        # Validation rules specific to market data
        self.market_validation_rules = self.validation_rules.get("market_data", {})

    def start_requests(self) -> Generator[Request, None, None]:
        """Generate initial requests for each configured data source and symbol."""
        for source_name, config in self.source_configs.items():
            if source_name == "alpha_vantage":
                for symbol in config["symbols"]:
                    # Real-time quote
                    yield self.make_alpha_vantage_request(symbol, "GLOBAL_QUOTE")
                    # Intraday data
                    yield self.make_alpha_vantage_request(symbol, "TIME_SERIES_INTRADAY")
            
            elif source_name == "yahoo_finance":
                for symbol in config["symbols"]:
                    yield self.make_yahoo_finance_request(symbol)

    def make_alpha_vantage_request(self, symbol: str, function: str) -> Request:
        """
        Create Alpha Vantage API request.
        
        Args:
            symbol: Stock symbol
            function: API function to call
            
        Returns:
            Configured request object
        """
        config = self.source_configs["alpha_vantage"]
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": config["api_key"],
        }
        
        if function == "TIME_SERIES_INTRADAY":
            params.update({
                "interval": "1min",
                "outputsize": "compact"
            })
        
        url = f"{config['url']}?{self._build_query_string(params)}"
        
        return self.make_request(
            url=url,
            callback=self.parse_alpha_vantage,
            meta={
                "symbol": symbol,
                "function": function,
                "source": "alpha_vantage"
            },
            priority=1 if function == "GLOBAL_QUOTE" else 0
        )

    def make_yahoo_finance_request(self, symbol: str) -> Request:
        """
        Create Yahoo Finance API request.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Configured request object
        """
        config = self.source_configs["yahoo_finance"]
        params = {
            "symbol": symbol,
            "interval": "1m",
            "range": "1d"
        }
        
        url = f"{config['url']}/{symbol}?{self._build_query_string(params)}"
        
        return self.make_request(
            url=url,
            callback=self.parse_yahoo_finance,
            meta={
                "symbol": symbol,
                "source": "yahoo_finance"
            }
        )

    def parse_alpha_vantage(self, response: Response) -> Dict[str, Any]:
        """
        Parse Alpha Vantage API response.
        
        Args:
            response: API response
            
        Returns:
            Normalized market data
        """
        try:
            data = self.parse_json_response(response)
            symbol = response.meta["symbol"]
            function = response.meta["function"]
            
            if "Error Message" in data:
                self.logger.error(
                    f"Alpha Vantage API error for {symbol}",
                    extra={
                        "error": data["Error Message"],
                        "symbol": symbol,
                        "function": function
                    }
                )
                self.update_metrics(success=False)
                return None
            
            if function == "GLOBAL_QUOTE":
                quote_data = data.get("Global Quote", {})
                parsed_data = {
                    "symbol": symbol,
                    "price": float(quote_data.get("05. price", 0)),
                    "volume": int(quote_data.get("06. volume", 0)),
                    "timestamp": datetime.now().isoformat(),
                    "source": "alpha_vantage"
                }
            
            else:  # TIME_SERIES_INTRADAY
                time_series = data.get("Time Series (1min)", {})
                latest_time = max(time_series.keys()) if time_series else None
                if latest_time:
                    latest_data = time_series[latest_time]
                    parsed_data = {
                        "symbol": symbol,
                        "price": float(latest_data.get("4. close", 0)),
                        "volume": int(latest_data.get("5. volume", 0)),
                        "timestamp": latest_time,
                        "source": "alpha_vantage"
                    }
                else:
                    self.logger.warning(f"No time series data for {symbol}")
                    return None
            
            # Validate data
            if self.validate_data(parsed_data, self.market_validation_rules):
                self.update_metrics(success=True)
                return parsed_data
            
        except Exception as e:
            self.logger.error(
                f"Error parsing Alpha Vantage response for {symbol}",
                extra={
                    "error": str(e),
                    "symbol": symbol,
                    "function": function
                }
            )
            self.update_metrics(success=False)
            return None

    def parse_yahoo_finance(self, response: Response) -> Dict[str, Any]:
        """
        Parse Yahoo Finance API response.
        
        Args:
            response: API response
            
        Returns:
            Normalized market data
        """
        try:
            data = self.parse_json_response(response)
            symbol = response.meta["symbol"]
            
            quote = data.get("chart", {}).get("result", [{}])[0].get("quote", {})
            if not quote:
                self.logger.warning(f"No quote data for {symbol}")
                return None
            
            # Get the latest price
            timestamps = quote.get("timestamp", [])
            closes = quote.get("close", [])
            volumes = quote.get("volume", [])
            
            if not (timestamps and closes and volumes):
                self.logger.warning(f"Missing price data for {symbol}")
                return None
            
            latest_idx = -1
            parsed_data = {
                "symbol": symbol,
                "price": float(closes[latest_idx]),
                "volume": int(volumes[latest_idx]),
                "timestamp": datetime.fromtimestamp(timestamps[latest_idx]).isoformat(),
                "source": "yahoo_finance"
            }
            
            # Validate data
            if self.validate_data(parsed_data, self.market_validation_rules):
                self.update_metrics(success=True)
                return parsed_data
            
        except Exception as e:
            self.logger.error(
                f"Error parsing Yahoo Finance response for {symbol}",
                extra={
                    "error": str(e),
                    "symbol": symbol
                }
            )
            self.update_metrics(success=False)
            return None

    @staticmethod
    def _build_query_string(params: Dict[str, Any]) -> str:
        """
        Build URL query string from parameters.
        
        Args:
            params: Dictionary of query parameters
            
        Returns:
            Formatted query string
        """
        return "&".join(f"{k}={v}" for k, v in params.items())
