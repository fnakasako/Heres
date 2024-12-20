"""
News Spider for Esurio Market Intelligence System.
Handles financial news collection from various sources.
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlencode

from scrapy import Request
from scrapy.exceptions import NotConfigured
from scrapy.http import Response, TextResponse

from src.scraping.spiders.base_spider import BaseSpider

class NewsSpider(BaseSpider):
    """
    Spider for collecting financial news from various sources.
    
    Features:
    - Multiple news sources (NewsAPI, Reuters)
    - API and HTML scraping capabilities
    - Content extraction and cleaning
    - Metadata parsing
    """

    name = "news_spider"
    
    def __init__(self, *args, **kwargs):
        """Initialize news spider with source-specific configurations."""
        super().__init__(*args, **kwargs)
        
        # Load news specific configuration
        news_config = next(
            (cfg for cfg in self.config.get("news_data", [])
             if cfg["name"] == "financial_news"),
            {}
        )
        
        if not news_config:
            raise NotConfigured("News configuration not found")
        
        self.sources = news_config.get("sources", [])
        if not self.sources:
            raise NotConfigured("No news sources configured")
        
        # Source-specific configurations
        self.source_configs = {
            source["name"]: {
                "url": source["url"],
                "api_key": source.get("api_key"),
                "endpoints": source.get("endpoints", []),
                "categories": source.get("categories", []),
                "selectors": source.get("selectors", {}),
                "rate_limit": source.get("rate_limit", self.default_rate_limit)
            }
            for source in self.sources
        }

    def start_requests(self) -> Generator[Request, None, None]:
        """Generate initial requests for each configured news source."""
        for source_name, config in self.source_configs.items():
            if source_name == "newsapi":
                # Top headlines
                yield self.make_newsapi_request(
                    endpoint="top-headlines",
                    params={
                        "category": "business",
                        "language": "en"
                    }
                )
                
                # Everything endpoint with financial keywords
                yield self.make_newsapi_request(
                    endpoint="everything",
                    params={
                        "q": "finance OR stock market OR cryptocurrency",
                        "language": "en",
                        "sortBy": "publishedAt",
                        "from": (datetime.now() - timedelta(days=1)).isoformat()
                    }
                )
            
            elif source_name == "reuters":
                for category in config["categories"]:
                    yield self.make_reuters_request(category)

    def make_newsapi_request(self, endpoint: str, params: Dict[str, str]) -> Request:
        """
        Create NewsAPI request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Configured request object
        """
        config = self.source_configs["newsapi"]
        params["apiKey"] = config["api_key"]
        
        url = f"{config['url']}/{endpoint}?{urlencode(params)}"
        
        return self.make_request(
            url=url,
            callback=self.parse_newsapi,
            meta={
                "endpoint": endpoint,
                "source": "newsapi"
            },
            priority=1 if endpoint == "top-headlines" else 0
        )

    def make_reuters_request(self, category: str) -> Request:
        """
        Create Reuters scraping request.
        
        Args:
            category: News category
            
        Returns:
            Configured request object
        """
        config = self.source_configs["reuters"]
        url = f"{config['url']}/{category}"
        
        return self.make_request(
            url=url,
            callback=self.parse_reuters,
            meta={
                "category": category,
                "source": "reuters",
                "selectors": config["selectors"]
            }
        )

    def parse_newsapi(self, response: Response) -> List[Dict[str, Any]]:
        """
        Parse NewsAPI response.
        
        Args:
            response: API response
            
        Returns:
            List of normalized news articles
        """
        try:
            data = self.parse_json_response(response)
            
            if "status" not in data or data["status"] != "ok":
                self.logger.error(
                    "NewsAPI error",
                    extra={
                        "error": data.get("message", "Unknown error"),
                        "endpoint": response.meta["endpoint"]
                    }
                )
                self.update_metrics(success=False)
                return None
            
            articles = []
            for article in data.get("articles", []):
                try:
                    parsed_article = {
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "content": article.get("content"),
                        "url": article.get("url"),
                        "source": article.get("source", {}).get("name"),
                        "published_at": article.get("publishedAt"),
                        "author": article.get("author"),
                        "metadata": {
                            "source_id": article.get("source", {}).get("id"),
                            "url_to_image": article.get("urlToImage")
                        },
                        "collected_at": datetime.now().isoformat(),
                        "collector": "newsapi"
                    }
                    
                    # Clean and validate the article
                    if self._validate_article(parsed_article):
                        articles.append(parsed_article)
                
                except Exception as e:
                    self.logger.warning(
                        f"Error parsing individual article",
                        extra={
                            "error": str(e),
                            "article": article
                        }
                    )
                    continue
            
            self.update_metrics(success=True)
            return articles
            
        except Exception as e:
            self.logger.error(
                "Error parsing NewsAPI response",
                extra={
                    "error": str(e),
                    "endpoint": response.meta["endpoint"]
                }
            )
            self.update_metrics(success=False)
            return None

    def parse_reuters(self, response: TextResponse) -> List[Dict[str, Any]]:
        """
        Parse Reuters HTML response.
        
        Args:
            response: HTML response
            
        Returns:
            List of normalized news articles
        """
        try:
            selectors = response.meta["selectors"]
            category = response.meta["category"]
            
            articles = []
            for article in response.css(selectors["article"]):
                try:
                    # Extract article data
                    title = article.css(selectors["title"] + "::text").get()
                    content = " ".join(
                        p.strip() for p in
                        article.css(selectors["content"] + "::text").getall()
                    )
                    
                    # Extract timestamp
                    timestamp = article.css("time::attr(datetime)").get()
                    if not timestamp:
                        timestamp = datetime.now().isoformat()
                    
                    parsed_article = {
                        "title": title,
                        "content": content,
                        "url": response.urljoin(
                            article.css("a::attr(href)").get()
                        ),
                        "source": "Reuters",
                        "published_at": timestamp,
                        "category": category,
                        "metadata": {
                            "category": category,
                            "html_content": article.get()
                        },
                        "collected_at": datetime.now().isoformat(),
                        "collector": "reuters"
                    }
                    
                    # Clean and validate the article
                    if self._validate_article(parsed_article):
                        articles.append(parsed_article)
                
                except Exception as e:
                    self.logger.warning(
                        f"Error parsing individual Reuters article",
                        extra={
                            "error": str(e),
                            "article_html": article.get()
                        }
                    )
                    continue
            
            self.update_metrics(success=True)
            return articles
            
        except Exception as e:
            self.logger.error(
                "Error parsing Reuters page",
                extra={
                    "error": str(e),
                    "category": category,
                    "url": response.url
                }
            )
            self.update_metrics(success=False)
            return None

    def _validate_article(self, article: Dict[str, Any]) -> bool:
        """
        Validate article data.
        
        Args:
            article: Article data dictionary
            
        Returns:
            Whether the article is valid
        """
        # Required fields
        required_fields = ["title", "content", "url", "source"]
        if not all(article.get(field) for field in required_fields):
            return False
        
        # Title and content minimum length
        if len(article["title"]) < 10 or len(article["content"]) < 50:
            return False
        
        # URL format
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        if not url_pattern.match(article["url"]):
            return False
        
        return True

    def _clean_text(self, text: str) -> str:
        """
        Clean article text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        
        return text.strip()
