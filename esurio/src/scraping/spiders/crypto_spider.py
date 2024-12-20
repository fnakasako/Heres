"""
Crypto Spider for Esurio Market Intelligence System.
Handles cryptocurrency market data collection.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlencode

from scrapy import Request
from scrapy.exceptions import NotConfigured
from scrapy.http import Response

from src.scraping.spiders.base_spider import BaseSpider

class CryptoSpider(BaseSpider):
    """
    Spider for collecting cryptocurrency market data.
    
    Features:
    - Real-time price data
    - Order book depth
    - Trading volume
    - Multiple exchange support (Coinbase, Binance)
    """

    name = "crypto_spider"
    
    def __init__(self, *args, **kwargs):
        """Initialize crypto spider with source-specific configurations."""
        super().__init__(*args, **kwargs)
        
        # Load crypto specific configuration
        crypto_config = next(
            (cfg for cfg in self.config.get("market_data", [])
             if cfg["name"] == "crypto_prices"),
            {}
        )
        
        if not crypto_config:
            raise NotConfigured("Crypto configuration not found")
        
        self.sources = crypto_config.get("sources", [])
        if not self.sources:
            raise NotConfigured("No crypto sources configured")
        
        # Exchange-specific configurations
        self.coinbase_config = next(
            (source for source in self.sources if source["name"] == "coinbase"),
            {}
        )
        
        self.binance_config = next(
            (source for source in self.sources if source["name"] == "binance"),
            {}
        )
        
        if not (self.coinbase_config or self.binance_config):
            raise NotConfigured("No valid crypto exchanges configured")
        
        # Initialize trading pairs
        self.coinbase_pairs = self.coinbase_config.get("symbols", [])
        self.binance_pairs = self.binance_config.get("symbols", [])

    def start_requests(self) -> Generator[Request, None, None]:
        """Generate initial requests for each exchange and trading pair."""
        # Coinbase requests
        if self.coinbase_config:
            for pair in self.coinbase_pairs:
                yield self.make_coinbase_ticker_request(pair)
                yield self.make_coinbase_orderbook_request(pair)
        
        # Binance requests
        if self.binance_config:
            for pair in self.binance_pairs:
                yield self.make_binance_ticker_request(pair)
                yield self.make_binance_orderbook_request(pair)

    def make_coinbase_ticker_request(self, pair: str) -> Request:
        """
        Create Coinbase ticker request.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Configured request object
        """
        url = f"{self.coinbase_config['url']}/products/{pair}/ticker"
        
        headers = self._get_coinbase_headers()
        
        return self.make_request(
            url=url,
            headers=headers,
            callback=self.parse_coinbase_ticker,
            meta={
                "pair": pair,
                "exchange": "coinbase"
            },
            errback=self.handle_coinbase_error,
            priority=2  # High priority for price data
        )

    def make_coinbase_orderbook_request(self, pair: str) -> Request:
        """
        Create Coinbase order book request.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Configured request object
        """
        url = f"{self.coinbase_config['url']}/products/{pair}/book?level=2"
        
        headers = self._get_coinbase_headers()
        
        return self.make_request(
            url=url,
            headers=headers,
            callback=self.parse_coinbase_orderbook,
            meta={
                "pair": pair,
                "exchange": "coinbase"
            },
            errback=self.handle_coinbase_error,
            priority=1
        )

    def make_binance_ticker_request(self, pair: str) -> Request:
        """
        Create Binance ticker request.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Configured request object
        """
        params = {
            "symbol": pair
        }
        
        url = f"{self.binance_config['url']}/ticker/24hr?{urlencode(params)}"
        
        headers = self._get_binance_headers()
        
        return self.make_request(
            url=url,
            headers=headers,
            callback=self.parse_binance_ticker,
            meta={
                "pair": pair,
                "exchange": "binance"
            },
            errback=self.handle_binance_error,
            priority=2
        )

    def make_binance_orderbook_request(self, pair: str) -> Request:
        """
        Create Binance order book request.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Configured request object
        """
        params = {
            "symbol": pair,
            "limit": 100  # Depth of order book
        }
        
        url = f"{self.binance_config['url']}/depth?{urlencode(params)}"
        
        headers = self._get_binance_headers()
        
        return self.make_request(
            url=url,
            headers=headers,
            callback=self.parse_binance_orderbook,
            meta={
                "pair": pair,
                "exchange": "binance"
            },
            errback=self.handle_binance_error,
            priority=1
        )

    def parse_coinbase_ticker(self, response: Response) -> Dict[str, Any]:
        """
        Parse Coinbase ticker response.
        
        Args:
            response: API response
            
        Returns:
            Normalized ticker data
        """
        try:
            data = self.parse_json_response(response)
            pair = response.meta["pair"]
            
            processed_data = {
                "exchange": "coinbase",
                "pair": pair,
                "price": float(data["price"]),
                "volume_24h": float(data["volume"]),
                "timestamp": data["time"],
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
                "metadata": {
                    "trade_id": data.get("trade_id"),
                    "size": float(data.get("size", 0))
                }
            }
            
            if self._validate_ticker_data(processed_data):
                self.update_metrics(success=True)
                return processed_data
            
            self.update_metrics(success=False)
            return None
            
        except Exception as e:
            self.logger.error(
                f"Error parsing Coinbase ticker for {response.meta['pair']}",
                extra={"error": str(e)}
            )
            self.update_metrics(success=False)
            return None

    def parse_coinbase_orderbook(self, response: Response) -> Dict[str, Any]:
        """
        Parse Coinbase order book response.
        
        Args:
            response: API response
            
        Returns:
            Normalized order book data
        """
        try:
            data = self.parse_json_response(response)
            pair = response.meta["pair"]
            
            processed_data = {
                "exchange": "coinbase",
                "pair": pair,
                "timestamp": datetime.now().isoformat(),
                "bids": [
                    {"price": float(bid[0]), "size": float(bid[1])}
                    for bid in data.get("bids", [])
                ],
                "asks": [
                    {"price": float(ask[0]), "size": float(ask[1])}
                    for ask in data.get("asks", [])
                ],
                "metadata": {
                    "sequence": data.get("sequence"),
                    "num_bids": len(data.get("bids", [])),
                    "num_asks": len(data.get("asks", []))
                }
            }
            
            if self._validate_orderbook_data(processed_data):
                self.update_metrics(success=True)
                return processed_data
            
            self.update_metrics(success=False)
            return None
            
        except Exception as e:
            self.logger.error(
                f"Error parsing Coinbase orderbook for {response.meta['pair']}",
                extra={"error": str(e)}
            )
            self.update_metrics(success=False)
            return None

    def parse_binance_ticker(self, response: Response) -> Dict[str, Any]:
        """
        Parse Binance ticker response.
        
        Args:
            response: API response
            
        Returns:
            Normalized ticker data
        """
        try:
            data = self.parse_json_response(response)
            pair = response.meta["pair"]
            
            processed_data = {
                "exchange": "binance",
                "pair": pair,
                "price": float(data["lastPrice"]),
                "volume_24h": float(data["volume"]),
                "timestamp": datetime.fromtimestamp(int(data["closeTime"]) / 1000).isoformat(),
                "bid": float(data["bidPrice"]),
                "ask": float(data["askPrice"]),
                "metadata": {
                    "price_change": float(data["priceChange"]),
                    "price_change_percent": float(data["priceChangePercent"]),
                    "weighted_avg_price": float(data["weightedAvgPrice"]),
                    "trades": int(data["count"])
                }
            }
            
            if self._validate_ticker_data(processed_data):
                self.update_metrics(success=True)
                return processed_data
            
            self.update_metrics(success=False)
            return None
            
        except Exception as e:
            self.logger.error(
                f"Error parsing Binance ticker for {response.meta['pair']}",
                extra={"error": str(e)}
            )
            self.update_metrics(success=False)
            return None

    def parse_binance_orderbook(self, response: Response) -> Dict[str, Any]:
        """
        Parse Binance order book response.
        
        Args:
            response: API response
            
        Returns:
            Normalized order book data
        """
        try:
            data = self.parse_json_response(response)
            pair = response.meta["pair"]
            
            processed_data = {
                "exchange": "binance",
                "pair": pair,
                "timestamp": datetime.now().isoformat(),
                "bids": [
                    {"price": float(bid[0]), "size": float(bid[1])}
                    for bid in data.get("bids", [])
                ],
                "asks": [
                    {"price": float(ask[0]), "size": float(ask[1])}
                    for ask in data.get("asks", [])
                ],
                "metadata": {
                    "lastUpdateId": data.get("lastUpdateId"),
                    "num_bids": len(data.get("bids", [])),
                    "num_asks": len(data.get("asks", []))
                }
            }
            
            if self._validate_orderbook_data(processed_data):
                self.update_metrics(success=True)
                return processed_data
            
            self.update_metrics(success=False)
            return None
            
        except Exception as e:
            self.logger.error(
                f"Error parsing Binance orderbook for {response.meta['pair']}",
                extra={"error": str(e)}
            )
            self.update_metrics(success=False)
            return None

    def _get_coinbase_headers(self) -> Dict[str, str]:
        """
        Get Coinbase API headers.
        
        Returns:
            Headers dictionary
        """
        return {
            "Accept": "application/json",
            "CB-ACCESS-KEY": self.coinbase_config["api_key"],
            "CB-ACCESS-TIMESTAMP": str(int(datetime.now().timestamp())),
            "CB-ACCESS-PASSPHRASE": self.coinbase_config.get("passphrase", "")
        }

    def _get_binance_headers(self) -> Dict[str, str]:
        """
        Get Binance API headers.
        
        Returns:
            Headers dictionary
        """
        return {
            "X-MBX-APIKEY": self.binance_config["api_key"]
        }

    def handle_coinbase_error(self, failure):
        """Handle Coinbase API errors."""
        request = failure.request
        
        try:
            response = failure.value.response
            data = self.parse_json_response(response)
            error = data.get("message", "Unknown error")
        except:
            error = str(failure.value)
        
        self.logger.error(
            "Coinbase API error",
            extra={
                "url": request.url,
                "error": error,
                "pair": request.meta["pair"]
            }
        )
        
        return None

    def handle_binance_error(self, failure):
        """Handle Binance API errors."""
        request = failure.request
        
        try:
            response = failure.value.response
            data = self.parse_json_response(response)
            error = data.get("msg", "Unknown error")
        except:
            error = str(failure.value)
        
        self.logger.error(
            "Binance API error",
            extra={
                "url": request.url,
                "error": error,
                "pair": request.meta["pair"]
            }
        )
        
        return None

    def _validate_ticker_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate ticker data.
        
        Args:
            data: Ticker data dictionary
            
        Returns:
            Whether the data is valid
        """
        required_fields = ["exchange", "pair", "price", "volume_24h", "timestamp"]
        if not all(field in data for field in required_fields):
            return False
        
        try:
            # Price and volume should be positive
            if data["price"] <= 0 or data["volume_24h"] < 0:
                return False
            
            # Validate timestamp
            datetime.fromisoformat(data["timestamp"])
            
            # Bid/ask spread should make sense
            if data.get("bid") and data.get("ask"):
                if data["bid"] > data["ask"]:
                    return False
            
            return True
            
        except (ValueError, TypeError):
            return False

    def _validate_orderbook_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate order book data.
        
        Args:
            data: Order book data dictionary
            
        Returns:
            Whether the data is valid
        """
        required_fields = ["exchange", "pair", "timestamp", "bids", "asks"]
        if not all(field in data for field in required_fields):
            return False
        
        try:
            # Validate timestamp
            datetime.fromisoformat(data["timestamp"])
            
            # Validate bids and asks
            if not (isinstance(data["bids"], list) and isinstance(data["asks"], list)):
                return False
            
            # Validate order entries
            for bid in data["bids"]:
                if not all(key in bid for key in ["price", "size"]):
                    return False
                if bid["price"] <= 0 or bid["size"] <= 0:
                    return False
            
            for ask in data["asks"]:
                if not all(key in ask for key in ["price", "size"]):
                    return False
                if ask["price"] <= 0 or ask["size"] <= 0:
                    return False
            
            # Verify bid/ask spread
            if data["bids"] and data["asks"]:
                highest_bid = max(bid["price"] for bid in data["bids"])
                lowest_ask = min(ask["price"] for ask in data["asks"])
                if highest_bid >= lowest_ask:
                    return False
            
            return True
            
        except (ValueError, TypeError):
            return False
