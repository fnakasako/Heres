"""
Supply Chain Spider for Esurio Market Intelligence System.
Handles collection of logistics and supply chain data.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlencode

from scrapy import Request
from scrapy.exceptions import NotConfigured
from scrapy.http import Response

from src.scraping.spiders.base_spider import BaseSpider

class SupplySpider(BaseSpider):
    """
    Spider for collecting supply chain and logistics data.
    
    Features:
    - Shipping rates tracking
    - Port congestion monitoring
    - Vessel tracking
    - Regional supply chain metrics
    """

    name = "supply_spider"
    
    def __init__(self, *args, **kwargs):
        """Initialize supply spider with source-specific configurations."""
        super().__init__(*args, **kwargs)
        
        # Load supply chain specific configuration
        supply_config = next(
            (cfg for cfg in self.config.get("supply_chain", [])
             if cfg["name"] == "logistics_data"),
            {}
        )
        
        if not supply_config:
            raise NotConfigured("Supply chain configuration not found")
        
        self.sources = supply_config.get("sources", [])
        if not self.sources:
            raise NotConfigured("No supply chain sources configured")
        
        # Source-specific configurations
        self.logistics_config = next(
            (source for source in self.sources if source["name"] == "shipping_rates"),
            {}
        )
        
        if not self.logistics_config:
            raise NotConfigured("Logistics configuration not found")
        
        # Initialize tracking parameters
        self.metrics = self.logistics_config.get("metrics", [])
        self.regions = self.logistics_config.get("regions", [])

    def start_requests(self) -> Generator[Request, None, None]:
        """Generate initial requests for each metric and region combination."""
        for metric in self.metrics:
            for region in self.regions:
                yield self.make_logistics_request(metric, region)

    def make_logistics_request(self, metric: str, region: str) -> Request:
        """
        Create logistics data request.
        
        Args:
            metric: Supply chain metric to collect
            region: Geographic region
            
        Returns:
            Configured request object
        """
        params = {
            "metric": metric,
            "region": region,
            "api_key": self.logistics_config["api_key"],
            "date_from": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "date_to": datetime.now().strftime("%Y-%m-%d")
        }
        
        url = f"{self.logistics_config['url']}/data?{urlencode(params)}"
        
        return self.make_request(
            url=url,
            callback=self.parse_logistics_data,
            meta={
                "metric": metric,
                "region": region
            },
            errback=self.handle_logistics_error,
            priority=self._get_metric_priority(metric)
        )

    def parse_logistics_data(self, response: Response) -> Dict[str, Any]:
        """
        Parse logistics API response.
        
        Args:
            response: API response
            
        Returns:
            Normalized supply chain data
        """
        try:
            data = self.parse_json_response(response)
            metric = response.meta["metric"]
            region = response.meta["region"]
            
            if "data" not in data:
                self.logger.error(
                    f"No data found for {metric} in {region}",
                    extra={"error": data.get("error")}
                )
                self.update_metrics(success=False)
                return None
            
            # Process time series data
            observations = []
            for entry in data["data"]:
                try:
                    processed_entry = self._process_metric_entry(entry, metric)
                    if processed_entry:
                        observations.append(processed_entry)
                except Exception as e:
                    self.logger.warning(
                        f"Error processing {metric} entry",
                        extra={"error": str(e), "entry": entry}
                    )
                    continue
            
            processed_data = {
                "metric": metric,
                "region": region,
                "source": "logistics_api",
                "frequency": "daily",
                "observations": observations,
                "metadata": {
                    "description": data.get("description"),
                    "unit": data.get("unit"),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            # Validate before returning
            if self._validate_supply_data(processed_data):
                self.update_metrics(success=True)
                return processed_data
            else:
                self.logger.error(
                    f"Invalid data format for {metric} in {region}",
                    extra={"data": processed_data}
                )
                self.update_metrics(success=False)
                return None
            
        except Exception as e:
            self.logger.error(
                f"Error parsing logistics data for {response.meta['metric']}",
                extra={"error": str(e)}
            )
            self.update_metrics(success=False)
            return None

    def _process_metric_entry(self, entry: Dict[str, Any], metric: str) -> Optional[Dict[str, Any]]:
        """
        Process individual metric entry based on type.
        
        Args:
            entry: Raw metric entry
            metric: Metric type
            
        Returns:
            Processed entry or None if invalid
        """
        try:
            if metric == "container_rates":
                return {
                    "date": entry["date"],
                    "value": float(entry["rate"]),
                    "currency": entry.get("currency", "USD"),
                    "container_type": entry.get("container_type", "40ft"),
                    "route": entry.get("route")
                }
            
            elif metric == "port_congestion":
                return {
                    "date": entry["date"],
                    "value": float(entry["congestion_index"]),
                    "port": entry["port"],
                    "vessel_count": int(entry.get("vessel_count", 0)),
                    "waiting_time": float(entry.get("waiting_time", 0))
                }
            
            elif metric == "vessel_tracking":
                return {
                    "date": entry["date"],
                    "value": float(entry["utilization_rate"]),
                    "vessel_type": entry.get("vessel_type"),
                    "capacity": float(entry.get("capacity", 0)),
                    "route": entry.get("route")
                }
            
            else:
                self.logger.warning(f"Unknown metric type: {metric}")
                return None
            
        except (KeyError, ValueError) as e:
            self.logger.warning(
                f"Error processing {metric} entry",
                extra={"error": str(e), "entry": entry}
            )
            return None

    def handle_logistics_error(self, failure):
        """
        Handle logistics API errors.
        
        Args:
            failure: Request failure
        """
        request = failure.request
        
        try:
            response = failure.value.response
            data = self.parse_json_response(response)
            error = data.get("error", {}).get("message", "Unknown error")
        except:
            error = str(failure.value)
        
        self.logger.error(
            "Logistics API error",
            extra={
                "url": request.url,
                "error": error,
                "metric": request.meta["metric"],
                "region": request.meta["region"]
            }
        )
        
        return None

    def _validate_supply_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate supply chain data.
        
        Args:
            data: Supply chain data dictionary
            
        Returns:
            Whether the data is valid
        """
        if not data.get("observations"):
            return False
        
        # Check required fields
        required_fields = ["metric", "region", "source", "frequency"]
        if not all(field in data for field in required_fields):
            return False
        
        # Validate observations based on metric type
        metric = data["metric"]
        for obs in data["observations"]:
            try:
                # Common validation
                if not all(key in obs for key in ["date", "value"]):
                    return False
                
                # Metric-specific validation
                if metric == "container_rates":
                    if not all(key in obs for key in ["currency", "container_type"]):
                        return False
                    if obs["value"] <= 0:  # Rates should be positive
                        return False
                
                elif metric == "port_congestion":
                    if not all(key in obs for key in ["port", "vessel_count"]):
                        return False
                    if not (0 <= obs["value"] <= 100):  # Congestion index range
                        return False
                
                elif metric == "vessel_tracking":
                    if not all(key in obs for key in ["vessel_type", "capacity"]):
                        return False
                    if not (0 <= obs["value"] <= 100):  # Utilization rate range
                        return False
                
            except (KeyError, TypeError):
                return False
        
        return True

    @staticmethod
    def _get_metric_priority(metric: str) -> int:
        """
        Get priority level for different metrics.
        
        Args:
            metric: Metric type
            
        Returns:
            Priority level (higher number = higher priority)
        """
        priorities = {
            "container_rates": 3,  # Highest priority - most volatile
            "port_congestion": 2,
            "vessel_tracking": 1
        }
        return priorities.get(metric, 0)
