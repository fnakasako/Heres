"""
Social Spider for Esurio Market Intelligence System.
Handles social media data collection and sentiment analysis.
"""

import base64
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlencode

from scrapy import Request
from scrapy.exceptions import NotConfigured
from scrapy.http import Response, TextResponse

from src.scraping.spiders.base_spider import BaseSpider

class SocialSpider(BaseSpider):
    """
    Spider for collecting social media data.
    
    Features:
    - Twitter API v2 integration
    - Keyword and user-based tracking
    - Rate limit handling
    - Sentiment preprocessing
    """

    name = "social_spider"
    
    def __init__(self, *args, **kwargs):
        """Initialize social spider with source-specific configurations."""
        super().__init__(*args, **kwargs)
        
        # Load social media specific configuration
        social_config = next(
            (cfg for cfg in self.config.get("social_data", [])
             if cfg["name"] == "social_sentiment"),
            {}
        )
        
        if not social_config:
            raise NotConfigured("Social media configuration not found")
        
        self.sources = social_config.get("sources", [])
        if not self.sources:
            raise NotConfigured("No social media sources configured")
        
        # Twitter-specific configuration
        self.twitter_config = next(
            (source for source in self.sources if source["name"] == "twitter"),
            {}
        )
        
        if not self.twitter_config:
            raise NotConfigured("Twitter configuration not found")
        
        self.twitter_auth = {
            "api_key": self.twitter_config.get("api_key"),
            "api_secret": self.twitter_config.get("api_secret"),
            "access_token": None  # Will be set during authentication
        }
        
        self.keywords = self.twitter_config.get("keywords", [])
        self.influencers = self.twitter_config.get("influencers", [])
        
        # Cache for rate limiting
        self.rate_limit_reset = {}
        self.rate_limit_remaining = {}

    def start_requests(self) -> Generator[Request, None, None]:
        """Generate initial requests after authentication."""
        # First authenticate with Twitter
        yield self.make_twitter_auth_request()

    def make_twitter_auth_request(self) -> Request:
        """
        Create Twitter authentication request.
        
        Returns:
            Authentication request object
        """
        credentials = base64.b64encode(
            f"{self.twitter_auth['api_key']}:{self.twitter_auth['api_secret']}"
            .encode('ascii')
        ).decode('ascii')
        
        return self.make_request(
            url="https://api.twitter.com/oauth2/token",
            method="POST",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
            },
            body="grant_type=client_credentials",
            callback=self.handle_twitter_auth,
            priority=10  # High priority for authentication
        )

    def handle_twitter_auth(self, response: Response) -> Generator[Request, None, None]:
        """
        Handle Twitter authentication response.
        
        Args:
            response: Authentication response
            
        Yields:
            Search and user timeline requests
        """
        try:
            data = self.parse_json_response(response)
            
            if "access_token" not in data:
                self.logger.error(
                    "Twitter authentication failed",
                    extra={"error": data.get("error", "Unknown error")}
                )
                return
            
            self.twitter_auth["access_token"] = data["access_token"]
            
            # Start actual data collection
            # 1. Search tweets by keywords
            for keyword in self.keywords:
                yield self.make_twitter_search_request(keyword)
            
            # 2. Get tweets from influencers
            for influencer in self.influencers:
                yield self.make_twitter_user_request(influencer)
            
        except Exception as e:
            self.logger.error(
                "Error handling Twitter authentication",
                extra={"error": str(e)}
            )

    def make_twitter_search_request(self, query: str) -> Request:
        """
        Create Twitter search request.
        
        Args:
            query: Search query
            
        Returns:
            Search request object
        """
        params = {
            "query": query,
            "tweet.fields": "created_at,public_metrics,entities",
            "expansions": "author_id",
            "user.fields": "name,username,verified,public_metrics",
            "max_results": 100
        }
        
        url = f"{self.twitter_config['url']}/tweets/search/recent?{urlencode(params)}"
        
        return self.make_request(
            url=url,
            headers=self._get_twitter_headers(),
            callback=self.parse_twitter_search,
            meta={"query": query},
            errback=self.handle_twitter_error
        )

    def make_twitter_user_request(self, username: str) -> Request:
        """
        Create Twitter user timeline request.
        
        Args:
            username: Twitter username
            
        Returns:
            User timeline request object
        """
        # First get user ID
        params = {
            "usernames": username.lstrip("@"),
            "user.fields": "id,name,username,verified,public_metrics"
        }
        
        url = f"{self.twitter_config['url']}/users/by?{urlencode(params)}"
        
        return self.make_request(
            url=url,
            headers=self._get_twitter_headers(),
            callback=self.handle_user_lookup,
            meta={"username": username},
            errback=self.handle_twitter_error
        )

    def handle_user_lookup(self, response: Response) -> Optional[Request]:
        """
        Handle Twitter user lookup response.
        
        Args:
            response: User lookup response
            
        Returns:
            User timeline request
        """
        try:
            data = self.parse_json_response(response)
            
            if "data" not in data or not data["data"]:
                self.logger.warning(
                    f"User not found: {response.meta['username']}"
                )
                return None
            
            user_id = data["data"][0]["id"]
            
            # Now get user's tweets
            params = {
                "tweet.fields": "created_at,public_metrics,entities",
                "max_results": 100
            }
            
            url = f"{self.twitter_config['url']}/users/{user_id}/tweets?{urlencode(params)}"
            
            return self.make_request(
                url=url,
                headers=self._get_twitter_headers(),
                callback=self.parse_twitter_timeline,
                meta={"user_id": user_id, "username": response.meta["username"]},
                errback=self.handle_twitter_error
            )
            
        except Exception as e:
            self.logger.error(
                f"Error handling user lookup for {response.meta['username']}",
                extra={"error": str(e)}
            )
            return None

    def parse_twitter_search(self, response: Response) -> List[Dict[str, Any]]:
        """
        Parse Twitter search response.
        
        Args:
            response: Search response
            
        Returns:
            List of normalized tweets
        """
        try:
            data = self.parse_json_response(response)
            self._update_rate_limits(response)
            
            if "data" not in data:
                self.logger.warning(
                    f"No tweets found for query: {response.meta['query']}"
                )
                return []
            
            return self._process_tweets(
                data["data"],
                data.get("includes", {}).get("users", []),
                query=response.meta["query"]
            )
            
        except Exception as e:
            self.logger.error(
                f"Error parsing Twitter search response for {response.meta['query']}",
                extra={"error": str(e)}
            )
            return []

    def parse_twitter_timeline(self, response: Response) -> List[Dict[str, Any]]:
        """
        Parse Twitter user timeline response.
        
        Args:
            response: Timeline response
            
        Returns:
            List of normalized tweets
        """
        try:
            data = self.parse_json_response(response)
            self._update_rate_limits(response)
            
            if "data" not in data:
                self.logger.warning(
                    f"No tweets found for user: {response.meta['username']}"
                )
                return []
            
            return self._process_tweets(
                data["data"],
                data.get("includes", {}).get("users", []),
                username=response.meta["username"]
            )
            
        except Exception as e:
            self.logger.error(
                f"Error parsing Twitter timeline for {response.meta['username']}",
                extra={"error": str(e)}
            )
            return []

    def _process_tweets(
        self,
        tweets: List[Dict],
        users: List[Dict],
        query: Optional[str] = None,
        username: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process and normalize tweets.
        
        Args:
            tweets: Raw tweet data
            users: User data
            query: Search query if applicable
            username: Username if applicable
            
        Returns:
            List of normalized tweets
        """
        processed_tweets = []
        user_map = {user["id"]: user for user in users}
        
        for tweet in tweets:
            try:
                user = user_map.get(tweet.get("author_id"))
                if not user:
                    continue
                
                processed_tweet = {
                    "id": tweet["id"],
                    "text": self._clean_tweet_text(tweet["text"]),
                    "created_at": tweet["created_at"],
                    "metrics": tweet.get("public_metrics", {}),
                    "author": {
                        "id": user["id"],
                        "username": user["username"],
                        "name": user["name"],
                        "verified": user.get("verified", False),
                        "followers": user.get("public_metrics", {}).get("followers_count", 0)
                    },
                    "entities": tweet.get("entities", {}),
                    "metadata": {
                        "query": query,
                        "username": username,
                        "collected_at": datetime.now().isoformat()
                    }
                }
                
                # Add preliminary sentiment indicators
                processed_tweet.update(self._extract_sentiment_indicators(tweet["text"]))
                
                processed_tweets.append(processed_tweet)
                
            except Exception as e:
                self.logger.warning(
                    "Error processing individual tweet",
                    extra={"error": str(e), "tweet": tweet}
                )
                continue
        
        return processed_tweets

    def _get_twitter_headers(self) -> Dict[str, str]:
        """
        Get Twitter API headers.
        
        Returns:
            Headers dictionary
        """
        return {
            "Authorization": f"Bearer {self.twitter_auth['access_token']}",
            "Content-Type": "application/json"
        }

    def _update_rate_limits(self, response: Response):
        """
        Update rate limit tracking.
        
        Args:
            response: API response with rate limit headers
        """
        endpoint = response.url.split("/")[-1].split("?")[0]
        
        self.rate_limit_remaining[endpoint] = int(
            response.headers.get("x-rate-limit-remaining", 0)
        )
        self.rate_limit_reset[endpoint] = int(
            response.headers.get("x-rate-limit-reset", 0)
        )

    def handle_twitter_error(self, failure):
        """
        Handle Twitter API errors.
        
        Args:
            failure: Request failure
        """
        request = failure.request
        
        try:
            response = failure.value.response
            data = json.loads(response.body)
            error = data.get("errors", [{"message": "Unknown error"}])[0]["message"]
        except:
            error = str(failure.value)
        
        self.logger.error(
            "Twitter API error",
            extra={
                "url": request.url,
                "error": error,
                "meta": request.meta
            }
        )
        
        # Handle rate limiting
        if response.status == 429:
            endpoint = request.url.split("/")[-1].split("?")[0]
            reset_time = int(response.headers.get("x-rate-limit-reset", 0))
            
            if reset_time:
                delay = reset_time - datetime.now().timestamp()
                if delay > 0:
                    self.logger.info(
                        f"Rate limited for endpoint {endpoint}. "
                        f"Retrying in {delay} seconds"
                    )
                    request.meta["download_delay"] = delay
                    return request.copy()
        
        return None

    @staticmethod
    def _clean_tweet_text(text: str) -> str:
        """
        Clean tweet text.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text.strip()

    @staticmethod
    def _extract_sentiment_indicators(text: str) -> Dict[str, Any]:
        """
        Extract basic sentiment indicators from text.
        
        Args:
            text: Tweet text
            
        Returns:
            Dictionary of sentiment indicators
        """
        # This is a simple example - in practice, you'd use a proper NLP model
        positive_words = {"bullish", "up", "gain", "profit", "growth", "positive"}
        negative_words = {"bearish", "down", "loss", "crash", "negative", "risk"}
        
        words = set(text.lower().split())
        
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        return {
            "sentiment_indicators": {
                "positive_count": positive_count,
                "negative_count": negative_count,
                "net_sentiment": positive_count - negative_count
            }
        }
