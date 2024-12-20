"""
Spider Runner for Esurio Market Intelligence System.
Manages and coordinates all data collection spiders.
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Type

from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
from scrapy.spiderloader import SpiderLoader
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor, task

from src.common.config import load_config
from src.common.logging_util import setup_logging
from src.scraping.spiders.base_spider import BaseSpider
from src.scraping.spiders.crypto_spider import CryptoSpider
from src.scraping.spiders.earnings_spider import EarningsSpider
from src.scraping.spiders.economic_spider import EconomicSpider
from src.scraping.spiders.market_spider import MarketSpider
from src.scraping.spiders.news_spider import NewsSpider
from src.scraping.spiders.social_spider import SocialSpider
from src.scraping.spiders.supply_spider import SupplySpider

logger = setup_logging()

class SpiderRunner:
    """
    Manages the execution of all data collection spiders.
    
    Features:
    - Concurrent spider execution
    - Scheduled runs
    - Resource management
    - Error handling and recovery
    """

    def __init__(self):
        """Initialize spider runner with configuration."""
        self.config = load_config()
        self.scraping_config = self.config.get("scraping", {})
        
        # Spider registry
        self.spiders: Dict[str, Type[BaseSpider]] = {
            "market": MarketSpider,
            "crypto": CryptoSpider,
            "news": NewsSpider,
            "social": SocialSpider,
            "economic": EconomicSpider,
            "supply": SupplySpider,
            "earnings": EarningsSpider
        }
        
        # Spider priorities (higher number = higher priority)
        self.priorities = {
            "market": 5,    # Highest - real-time market data
            "crypto": 5,    # Real-time crypto data
            "news": 4,      # Time-sensitive news
            "social": 3,    # Social media sentiment
            "earnings": 3,  # Earnings events
            "economic": 2,  # Economic indicators
            "supply": 1     # Supply chain data
        }
        
        # Initialize Scrapy settings
        self.settings = self._get_scrapy_settings()
        
        # Initialize crawler process
        self.process = CrawlerProcess(self.settings)
        
        # Track running spiders
        self.running_spiders: Dict[str, datetime] = {}
        
        # Setup signal handlers
        self._setup_signal_handlers()

    def run_spider(self, spider_name: str) -> None:
        """
        Run a specific spider.
        
        Args:
            spider_name: Name of spider to run
        """
        if spider_name not in self.spiders:
            logger.error(f"Unknown spider: {spider_name}")
            return
        
        try:
            spider_class = self.spiders[spider_name]
            self.process.crawl(
                spider_class,
                priority=self.priorities.get(spider_name, 0)
            )
            self.running_spiders[spider_name] = datetime.now()
            
            logger.info(
                f"Started spider: {spider_name}",
                extra={
                    "spider": spider_name,
                    "priority": self.priorities.get(spider_name, 0)
                }
            )
            
        except Exception as e:
            logger.error(
                f"Error running spider: {spider_name}",
                extra={
                    "spider": spider_name,
                    "error": str(e)
                }
            )

    def run_all_spiders(self) -> None:
        """Run all configured spiders concurrently."""
        try:
            # Sort spiders by priority
            sorted_spiders = sorted(
                self.spiders.keys(),
                key=lambda x: self.priorities.get(x, 0),
                reverse=True
            )
            
            for spider_name in sorted_spiders:
                self.run_spider(spider_name)
            
            logger.info(
                "Started all spiders",
                extra={
                    "spider_count": len(sorted_spiders),
                    "spiders": sorted_spiders
                }
            )
            
        except Exception as e:
            logger.error(
                "Error running spiders",
                extra={"error": str(e)}
            )

    def schedule_spider(
        self,
        spider_name: str,
        interval: int
    ) -> Optional[task.LoopingCall]:
        """
        Schedule a spider to run at regular intervals.
        
        Args:
            spider_name: Name of spider to schedule
            interval: Run interval in seconds
            
        Returns:
            Scheduled task or None if error
        """
        if spider_name not in self.spiders:
            logger.error(f"Unknown spider: {spider_name}")
            return None
        
        try:
            # Create looping task
            spider_task = task.LoopingCall(self.run_spider, spider_name)
            spider_task.start(interval)
            
            logger.info(
                f"Scheduled spider: {spider_name}",
                extra={
                    "spider": spider_name,
                    "interval": interval
                }
            )
            
            return spider_task
            
        except Exception as e:
            logger.error(
                f"Error scheduling spider: {spider_name}",
                extra={
                    "spider": spider_name,
                    "interval": interval,
                    "error": str(e)
                }
            )
            return None

    def schedule_all_spiders(self) -> List[task.LoopingCall]:
        """
        Schedule all spiders according to their configurations.
        
        Returns:
            List of scheduled tasks
        """
        tasks = []
        
        try:
            for spider_name in self.spiders:
                # Get spider-specific interval from config
                interval = self.scraping_config.get(
                    f"{spider_name}_interval",
                    self.scraping_config.get("default_interval", 300)  # 5 minutes default
                )
                
                task = self.schedule_spider(spider_name, interval)
                if task:
                    tasks.append(task)
            
            logger.info(
                "Scheduled all spiders",
                extra={
                    "task_count": len(tasks),
                    "spiders": list(self.spiders.keys())
                }
            )
            
            return tasks
            
        except Exception as e:
            logger.error(
                "Error scheduling spiders",
                extra={"error": str(e)}
            )
            return tasks

    def stop_spider(self, spider_name: str) -> None:
        """
        Stop a running spider.
        
        Args:
            spider_name: Name of spider to stop
        """
        if spider_name not in self.running_spiders:
            logger.warning(f"Spider not running: {spider_name}")
            return
        
        try:
            # Signal spider to stop
            self.process.stop()
            
            # Remove from running spiders
            del self.running_spiders[spider_name]
            
            logger.info(
                f"Stopped spider: {spider_name}",
                extra={"spider": spider_name}
            )
            
        except Exception as e:
            logger.error(
                f"Error stopping spider: {spider_name}",
                extra={
                    "spider": spider_name,
                    "error": str(e)
                }
            )

    def stop_all_spiders(self) -> None:
        """Stop all running spiders."""
        try:
            # Stop crawler process
            self.process.stop()
            
            # Clear running spiders
            self.running_spiders.clear()
            
            logger.info("Stopped all spiders")
            
        except Exception as e:
            logger.error(
                "Error stopping spiders",
                extra={"error": str(e)}
            )

    def _get_scrapy_settings(self) -> Settings:
        """
        Get Scrapy settings with custom configurations.
        
        Returns:
            Configured Scrapy settings
        """
        settings = get_project_settings()
        
        # Custom settings
        custom_settings = {
            "CONCURRENT_REQUESTS": self.scraping_config.get("max_concurrent_requests", 16),
            "CONCURRENT_REQUESTS_PER_DOMAIN": self.scraping_config.get("rate_limit_per_domain", 8),
            "DOWNLOAD_DELAY": self.scraping_config.get("download_delay", 1),
            "COOKIES_ENABLED": False,
            "TELNETCONSOLE_ENABLED": False,
            "LOG_LEVEL": "INFO",
            "USER_AGENT": self.scraping_config.get(
                "user_agent",
                "Esurio Market Intelligence Bot/0.1.0"
            ),
            # Retry settings
            "RETRY_ENABLED": True,
            "RETRY_TIMES": self.scraping_config.get("retry", {}).get("max_retries", 3),
            "RETRY_HTTP_CODES": self.scraping_config.get("retry", {}).get(
                "status_forcelist",
                [500, 502, 503, 504]
            ),
            # Timeout settings
            "DOWNLOAD_TIMEOUT": self.scraping_config.get("timeout", {}).get("read", 30),
            # Cache settings
            "HTTPCACHE_ENABLED": True,
            "HTTPCACHE_EXPIRATION_SECS": 60,
            "HTTPCACHE_DIR": "httpcache",
            "HTTPCACHE_IGNORE_HTTP_CODES": [401, 403, 404, 500, 502, 503, 504],
            # Extensions
            "EXTENSIONS": {
                "scrapy.extensions.telnet.TelnetConsole": None,
                "scrapy.extensions.corestats.CoreStats": 1,
                "scrapy.extensions.memusage.MemoryUsage": 1,
                "scrapy.extensions.logstats.LogStats": 1
            }
        }
        
        # Update settings
        settings.update(custom_settings)
        
        return settings

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self.stop_all_spiders()
            reactor.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point for spider runner."""
    parser = argparse.ArgumentParser(description="Esurio Spider Runner")
    parser.add_argument(
        "--spider",
        type=str,
        help="Name of spider to run (omit for all spiders)"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Schedule spiders to run periodically"
    )
    args = parser.parse_args()
    
    # Configure logging
    configure_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize runner
        runner = SpiderRunner()
        
        if args.schedule:
            if args.spider:
                # Schedule specific spider
                interval = runner.scraping_config.get(
                    f"{args.spider}_interval",
                    runner.scraping_config.get("default_interval", 300)
                )
                runner.schedule_spider(args.spider, interval)
            else:
                # Schedule all spiders
                runner.schedule_all_spiders()
        else:
            if args.spider:
                # Run specific spider
                runner.run_spider(args.spider)
            else:
                # Run all spiders
                runner.run_all_spiders()
        
        # Start reactor
        reactor.run()
        
    except Exception as e:
        logger.error(
            "Error in spider runner",
            extra={"error": str(e)}
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
