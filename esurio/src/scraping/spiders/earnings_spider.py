"""
Earnings Spider for Esurio Market Intelligence System.
Handles collection of corporate earnings data.
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlencode

from scrapy import Request
from scrapy.exceptions import NotConfigured
from scrapy.http import Response, TextResponse

from src.scraping.spiders.base_spider import BaseSpider

class EarningsSpider(BaseSpider):
    """
    Spider for collecting corporate earnings data.
    
    Features:
    - Earnings calendar tracking
    - SEC filings monitoring
    - Earnings whispers analysis
    - Historical earnings data
    """

    name = "earnings_spider"
    
    def __init__(self, *args, **kwargs):
        """Initialize earnings spider with source-specific configurations."""
        super().__init__(*args, **kwargs)
        
        # Load earnings specific configuration
        earnings_config = next(
            (cfg for cfg in self.config.get("earnings_data", [])
             if cfg["name"] == "earnings_calendar"),
            {}
        )
        
        if not earnings_config:
            raise NotConfigured("Earnings configuration not found")
        
        self.sources = earnings_config.get("sources", [])
        if not self.sources:
            raise NotConfigured("No earnings sources configured")
        
        # Source-specific configurations
        self.whispers_config = next(
            (source for source in self.sources if source["name"] == "earnings_whispers"),
            {}
        )
        
        self.sec_config = next(
            (source for source in self.sources if source["name"] == "sec_filings"),
            {}
        )
        
        if not (self.whispers_config or self.sec_config):
            raise NotConfigured("No valid earnings sources configured")
        
        # Initialize tracking parameters
        self.companies = self.sec_config.get("companies", [])
        self.form_types = self.sec_config.get("form_types", [])
        self.selectors = self.whispers_config.get("selectors", {})

    def start_requests(self) -> Generator[Request, None, None]:
        """Generate initial requests for each data source."""
        # Earnings Whispers calendar
        yield self.make_earnings_calendar_request()
        
        # SEC filings for each company
        for company in self.companies:
            for form_type in self.form_types:
                yield self.make_sec_filing_request(company, form_type)

    def make_earnings_calendar_request(self) -> Request:
        """
        Create earnings calendar request.
        
        Returns:
            Configured request object
        """
        url = self.whispers_config["url"]
        
        return self.make_request(
            url=url,
            callback=self.parse_earnings_calendar,
            meta={
                "source": "earnings_whispers"
            },
            errback=self.handle_whispers_error,
            priority=2  # High priority for upcoming earnings
        )

    def make_sec_filing_request(self, company: str, form_type: str) -> Request:
        """
        Create SEC filing request.
        
        Args:
            company: Company ticker symbol
            form_type: SEC form type
            
        Returns:
            Configured request object
        """
        params = {
            "action": "getcompany",
            "CIK": company,
            "type": form_type,
            "dateb": "",
            "owner": "exclude",
            "count": 100
        }
        
        url = f"{self.sec_config['url']}?{urlencode(params)}"
        
        return self.make_request(
            url=url,
            callback=self.parse_sec_filings,
            meta={
                "company": company,
                "form_type": form_type,
                "source": "sec"
            },
            errback=self.handle_sec_error,
            priority=1
        )

    def parse_earnings_calendar(self, response: TextResponse) -> List[Dict[str, Any]]:
        """
        Parse earnings calendar HTML response.
        
        Args:
            response: HTML response
            
        Returns:
            List of normalized earnings events
        """
        try:
            events = []
            calendar_selector = self.selectors["calendar"]
            company_selector = self.selectors["company"]
            estimates_selector = self.selectors["estimates"]
            
            for event in response.css(calendar_selector):
                try:
                    company_info = event.css(company_selector)
                    estimates = event.css(estimates_selector)
                    
                    processed_event = {
                        "company": {
                            "ticker": self._extract_text(company_info.css(".ticker::text")),
                            "name": self._extract_text(company_info.css(".name::text")),
                            "market_cap": self._parse_market_cap(
                                company_info.css(".market-cap::text").get("")
                            )
                        },
                        "earnings": {
                            "date": self._parse_date(
                                event.css(".date::text").get("")
                            ),
                            "time": event.css(".time::text").get("").strip(),
                            "eps_estimate": self._parse_number(
                                estimates.css(".eps-estimate::text").get("")
                            ),
                            "revenue_estimate": self._parse_number(
                                estimates.css(".revenue-estimate::text").get("")
                            ),
                            "whisper_eps": self._parse_number(
                                estimates.css(".whisper-eps::text").get("")
                            )
                        },
                        "metadata": {
                            "source": "earnings_whispers",
                            "collected_at": datetime.now().isoformat()
                        }
                    }
                    
                    if self._validate_earnings_event(processed_event):
                        events.append(processed_event)
                    
                except Exception as e:
                    self.logger.warning(
                        "Error processing earnings event",
                        extra={"error": str(e), "event_html": event.get()}
                    )
                    continue
            
            self.update_metrics(success=True)
            return events
            
        except Exception as e:
            self.logger.error(
                "Error parsing earnings calendar",
                extra={"error": str(e)}
            )
            self.update_metrics(success=False)
            return []

    def parse_sec_filings(self, response: TextResponse) -> List[Dict[str, Any]]:
        """
        Parse SEC filings HTML response.
        
        Args:
            response: HTML response
            
        Returns:
            List of normalized SEC filings
        """
        try:
            filings = []
            company = response.meta["company"]
            form_type = response.meta["form_type"]
            
            # SEC EDGAR uses tables for filing listings
            for row in response.css("tr"):
                try:
                    cells = row.css("td")
                    if len(cells) < 4:  # Skip header rows
                        continue
                    
                    filing_type = cells[0].css("::text").get("").strip()
                    if filing_type != form_type:
                        continue
                    
                    filing = {
                        "company": company,
                        "form_type": form_type,
                        "filing_date": self._parse_date(
                            cells[3].css("::text").get("")
                        ),
                        "accession_number": self._extract_accession(
                            cells[4].css("a::attr(href)").get("")
                        ),
                        "document_url": self._build_sec_url(
                            cells[2].css("a::attr(href)").get("")
                        ),
                        "metadata": {
                            "source": "sec",
                            "collected_at": datetime.now().isoformat()
                        }
                    }
                    
                    if self._validate_sec_filing(filing):
                        filings.append(filing)
                    
                except Exception as e:
                    self.logger.warning(
                        f"Error processing SEC filing for {company}",
                        extra={"error": str(e), "row_html": row.get()}
                    )
                    continue
            
            self.update_metrics(success=True)
            return filings
            
        except Exception as e:
            self.logger.error(
                f"Error parsing SEC filings for {response.meta['company']}",
                extra={"error": str(e)}
            )
            self.update_metrics(success=False)
            return []

    def handle_whispers_error(self, failure):
        """Handle Earnings Whispers errors."""
        request = failure.request
        
        try:
            response = failure.value.response
            error = f"Status: {response.status}"
        except:
            error = str(failure.value)
        
        self.logger.error(
            "Earnings Whispers error",
            extra={
                "url": request.url,
                "error": error
            }
        )
        
        return None

    def handle_sec_error(self, failure):
        """Handle SEC EDGAR errors."""
        request = failure.request
        
        try:
            response = failure.value.response
            error = f"Status: {response.status}"
        except:
            error = str(failure.value)
        
        self.logger.error(
            "SEC EDGAR error",
            extra={
                "url": request.url,
                "error": error,
                "company": request.meta["company"],
                "form_type": request.meta["form_type"]
            }
        )
        
        return None

    def _validate_earnings_event(self, event: Dict[str, Any]) -> bool:
        """
        Validate earnings event data.
        
        Args:
            event: Earnings event dictionary
            
        Returns:
            Whether the event is valid
        """
        try:
            # Required fields
            if not all(key in event for key in ["company", "earnings", "metadata"]):
                return False
            
            company = event["company"]
            if not all(key in company for key in ["ticker", "name"]):
                return False
            
            earnings = event["earnings"]
            if not all(key in earnings for key in ["date", "time"]):
                return False
            
            # Validate date
            if not isinstance(earnings["date"], datetime):
                return False
            
            # Validate estimates
            for key in ["eps_estimate", "revenue_estimate", "whisper_eps"]:
                if key in earnings and earnings[key] is not None:
                    if not isinstance(earnings[key], (int, float)):
                        return False
            
            return True
            
        except (KeyError, TypeError):
            return False

    def _validate_sec_filing(self, filing: Dict[str, Any]) -> bool:
        """
        Validate SEC filing data.
        
        Args:
            filing: SEC filing dictionary
            
        Returns:
            Whether the filing is valid
        """
        try:
            # Required fields
            required_fields = [
                "company", "form_type", "filing_date",
                "accession_number", "document_url"
            ]
            if not all(key in filing for key in required_fields):
                return False
            
            # Validate date
            if not isinstance(filing["filing_date"], datetime):
                return False
            
            # Validate URLs
            if not all(
                url.startswith("http")
                for url in [filing["document_url"]]
            ):
                return False
            
            # Validate accession number format
            accession_pattern = re.compile(r'\d{10}-\d{2}-\d{6}')
            if not accession_pattern.match(filing["accession_number"]):
                return False
            
            return True
            
        except (KeyError, TypeError):
            return False

    @staticmethod
    def _extract_text(selector) -> str:
        """Extract and clean text from selector."""
        return selector.get("").strip()

    @staticmethod
    def _parse_market_cap(text: str) -> Optional[float]:
        """Parse market cap string to float value."""
        try:
            # Remove currency symbol and convert to number
            text = text.replace("$", "").strip().lower()
            
            # Handle suffixes
            multipliers = {
                "t": 1e12,  # trillion
                "b": 1e9,   # billion
                "m": 1e6    # million
            }
            
            for suffix, multiplier in multipliers.items():
                if text.endswith(suffix):
                    return float(text[:-1]) * multiplier
            
            return float(text)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        """Parse number string to float value."""
        try:
            return float(text.replace("$", "").replace(",", "").strip())
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_date(text: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        try:
            return datetime.strptime(text.strip(), "%Y-%m-%d")
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _extract_accession(url: str) -> str:
        """Extract accession number from SEC URL."""
        match = re.search(r'(\d{10}-\d{2}-\d{6})', url)
        return match.group(1) if match else ""

    @staticmethod
    def _build_sec_url(path: str) -> str:
        """Build complete SEC document URL."""
        return f"https://www.sec.gov{path}" if path else ""
