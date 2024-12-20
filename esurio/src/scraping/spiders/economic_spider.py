"""
Economic Spider for Esurio Market Intelligence System.
Handles collection of macroeconomic indicators and data.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlencode

from scrapy import Request
from scrapy.exceptions import NotConfigured
from scrapy.http import Response

from src.scraping.spiders.base_spider import BaseSpider

class EconomicSpider(BaseSpider):
    """
    Spider for collecting macroeconomic data.
    
    Features:
    - FRED API integration
    - World Bank API integration
    - Economic indicator tracking
    - Time series handling
    """

    name = "economic_spider"
    
    def __init__(self, *args, **kwargs):
        """Initialize economic spider with source-specific configurations."""
        super().__init__(*args, **kwargs)
        
        # Load economic data specific configuration
        economic_config = next(
            (cfg for cfg in self.config.get("economic_data", [])
             if cfg["name"] == "macro_indicators"),
            {}
        )
        
        if not economic_config:
            raise NotConfigured("Economic data configuration not found")
        
        self.sources = economic_config.get("sources", [])
        if not self.sources:
            raise NotConfigured("No economic data sources configured")
        
        # Source-specific configurations
        self.fred_config = next(
            (source for source in self.sources if source["name"] == "fred"),
            {}
        )
        
        self.world_bank_config = next(
            (source for source in self.sources if source["name"] == "world_bank"),
            {}
        )
        
        if not self.fred_config and not self.world_bank_config:
            raise NotConfigured("No valid economic data sources configured")
        
        # Initialize series tracking
        self.fred_series = self.fred_config.get("series", [])
        self.world_bank_indicators = self.world_bank_config.get("indicators", [])

    def start_requests(self) -> Generator[Request, None, None]:
        """Generate initial requests for each configured data source."""
        # FRED requests
        if self.fred_config:
            for series_id in self.fred_series:
                yield self.make_fred_request(series_id)
        
        # World Bank requests
        if self.world_bank_config:
            for indicator in self.world_bank_indicators:
                yield self.make_world_bank_request(indicator)

    def make_fred_request(self, series_id: str) -> Request:
        """
        Create FRED API request.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Configured request object
        """
        params = {
            "series_id": series_id,
            "api_key": self.fred_config["api_key"],
            "file_type": "json",
            "observation_start": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "observation_end": datetime.now().strftime("%Y-%m-%d"),
            "frequency": "m",  # Monthly
            "units": "lin"  # Linear units
        }
        
        url = f"{self.fred_config['url']}/series/observations?{urlencode(params)}"
        
        return self.make_request(
            url=url,
            callback=self.parse_fred,
            meta={
                "series_id": series_id,
                "source": "fred"
            },
            errback=self.handle_fred_error
        )

    def make_world_bank_request(self, indicator: str) -> Request:
        """
        Create World Bank API request.
        
        Args:
            indicator: World Bank indicator code
            
        Returns:
            Configured request object
        """
        params = {
            "format": "json",
            "per_page": 1000,
            "date": f"{datetime.now().year-5}:{datetime.now().year}"
        }
        
        if self.world_bank_config.get("api_key"):
            params["api_key"] = self.world_bank_config["api_key"]
        
        url = f"{self.world_bank_config['url']}/countries/all/indicators/{indicator}?{urlencode(params)}"
        
        return self.make_request(
            url=url,
            callback=self.parse_world_bank,
            meta={
                "indicator": indicator,
                "source": "world_bank"
            },
            errback=self.handle_world_bank_error
        )

    def parse_fred(self, response: Response) -> Dict[str, Any]:
        """
        Parse FRED API response.
        
        Args:
            response: API response
            
        Returns:
            Normalized economic data
        """
        try:
            data = self.parse_json_response(response)
            series_id = response.meta["series_id"]
            
            if "observations" not in data:
                self.logger.error(
                    f"No data found for FRED series: {series_id}",
                    extra={"error": data.get("error_message")}
                )
                self.update_metrics(success=False)
                return None
            
            # Process observations
            observations = []
            for obs in data["observations"]:
                try:
                    value = float(obs["value"]) if obs["value"] != "." else None
                    if value is not None:
                        observations.append({
                            "date": obs["date"],
                            "value": value,
                            "status": obs.get("status", "final")
                        })
                except (ValueError, KeyError) as e:
                    self.logger.warning(
                        f"Error processing FRED observation",
                        extra={"error": str(e), "observation": obs}
                    )
                    continue
            
            processed_data = {
                "series_id": series_id,
                "source": "fred",
                "frequency": "monthly",
                "units": data.get("units", "lin"),
                "observations": observations,
                "metadata": {
                    "title": data.get("title"),
                    "notes": data.get("notes"),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            self.update_metrics(success=True)
            return processed_data
            
        except Exception as e:
            self.logger.error(
                f"Error parsing FRED response for {response.meta['series_id']}",
                extra={"error": str(e)}
            )
            self.update_metrics(success=False)
            return None

    def parse_world_bank(self, response: Response) -> Dict[str, Any]:
        """
        Parse World Bank API response.
        
        Args:
            response: API response
            
        Returns:
            Normalized economic data
        """
        try:
            data = self.parse_json_response(response)
            indicator = response.meta["indicator"]
            
            if not data or len(data) < 2:  # World Bank API returns metadata in [0]
                self.logger.error(
                    f"No data found for World Bank indicator: {indicator}"
                )
                self.update_metrics(success=False)
                return None
            
            # Extract metadata and data
            metadata = data[0]
            observations = []
            
            for entry in data[1]:
                try:
                    value = float(entry["value"]) if entry["value"] is not None else None
                    if value is not None:
                        observations.append({
                            "date": entry["date"],
                            "value": value,
                            "country": entry["country"]["value"]
                        })
                except (ValueError, KeyError) as e:
                    self.logger.warning(
                        f"Error processing World Bank observation",
                        extra={"error": str(e), "entry": entry}
                    )
                    continue
            
            processed_data = {
                "indicator": indicator,
                "source": "world_bank",
                "frequency": "yearly",
                "observations": observations,
                "metadata": {
                    "name": metadata.get("name"),
                    "description": metadata.get("description"),
                    "source_note": metadata.get("sourceNote"),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            self.update_metrics(success=True)
            return processed_data
            
        except Exception as e:
            self.logger.error(
                f"Error parsing World Bank response for {response.meta['indicator']}",
                extra={"error": str(e)}
            )
            self.update_metrics(success=False)
            return None

    def handle_fred_error(self, failure):
        """
        Handle FRED API errors.
        
        Args:
            failure: Request failure
        """
        request = failure.request
        
        try:
            response = failure.value.response
            data = self.parse_json_response(response)
            error = data.get("error_message", "Unknown error")
        except:
            error = str(failure.value)
        
        self.logger.error(
            "FRED API error",
            extra={
                "url": request.url,
                "error": error,
                "series_id": request.meta["series_id"]
            }
        )
        
        return None

    def handle_world_bank_error(self, failure):
        """
        Handle World Bank API errors.
        
        Args:
            failure: Request failure
        """
        request = failure.request
        
        try:
            response = failure.value.response
            data = self.parse_json_response(response)
            error = data.get("message", {}).get("error", "Unknown error")
        except:
            error = str(failure.value)
        
        self.logger.error(
            "World Bank API error",
            extra={
                "url": request.url,
                "error": error,
                "indicator": request.meta["indicator"]
            }
        )
        
        return None

    def _validate_economic_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate economic data.
        
        Args:
            data: Economic data dictionary
            
        Returns:
            Whether the data is valid
        """
        if not data.get("observations"):
            return False
        
        # Check for required metadata
        if not all(key in data for key in ["source", "frequency"]):
            return False
        
        # Validate observations
        for obs in data["observations"]:
            if not all(key in obs for key in ["date", "value"]):
                return False
            
            try:
                # Validate date format
                datetime.strptime(obs["date"], "%Y-%m-%d")
                
                # Validate value is numeric
                float(obs["value"])
            except (ValueError, TypeError):
                return False
        
        return True
