import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class ConfigurationError(Exception):
    """Raised when there's an error in configuration"""
    pass

@lru_cache()
def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent

class Config:
    """Configuration manager that handles loading and accessing config values"""
    
    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML files and environment variables"""
        try:
            # Load base config
            base_config_path = get_project_root() / "config" / "base_config.yaml"
            with open(base_config_path, "r") as f:
                self._config = yaml.safe_load(f)

            # Load scraping targets
            targets_path = get_project_root() / "config" / "scraping_targets.yaml"
            with open(targets_path, "r") as f:
                targets_config = yaml.safe_load(f)
                self._config["scraping_targets"] = targets_config.get("targets", [])

            # Override with environment variables
            self._override_from_env()
            
            # Validate configuration
            self._validate_config()
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise ConfigurationError(f"Configuration loading failed: {str(e)}")

    def _override_from_env(self) -> None:
        """Override configuration values with environment variables"""
        env_mapping = {
            "POSTGRES_USER": ("database", "user"),
            "POSTGRES_PASSWORD": ("database", "password"),
            "POSTGRES_DB": ("database", "name"),
            "SCRAPING_CONCURRENCY": ("scraping", "concurrency"),
            "RATE_LIMIT_PER_DOMAIN": ("scraping", "rate_limits", "default_domain"),
            "LOG_LEVEL": ("logging", "level"),
        }

        for env_var, config_path in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if env_var in ["SCRAPING_CONCURRENCY", "RATE_LIMIT_PER_DOMAIN"]:
                    env_value = int(env_value)
                
                # Navigate to the correct config location and set the value
                config = self._config
                for key in config_path[:-1]:
                    config = config.setdefault(key, {})
                config[config_path[-1]] = env_value

    def _validate_config(self) -> None:
        """Validate the configuration"""
        required_keys = [
            ("database", "url"),
            ("scraping", "concurrency"),
            ("logging", "level"),
        ]

        for keys in required_keys:
            value = self.get(*keys)
            if value is None:
                raise ConfigurationError(f"Missing required configuration: {'.'.join(keys)}")

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        Example: config.get("database", "url")
        """
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_scraping_target(self, target_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific scraping target"""
        targets = self.get("scraping_targets", default=[])
        return next((t for t in targets if t["name"] == target_name), None)

    def get_all_scraping_targets(self) -> list:
        """Get all scraping targets"""
        return self.get("scraping_targets", default=[])

    def get_database_url(self) -> str:
        """Get the database URL with credentials"""
        return self.get("database", "url")

    def get_redis_url(self) -> str:
        """Get the Redis URL"""
        return os.getenv("REDIS_URL", "redis://localhost:6379")

    def get_feature_store_config(self) -> Dict[str, Any]:
        """Get feature store configuration"""
        return self.get("feature_store", default={})

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get("monitoring", default={})

    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get("logging", "level", default="INFO").upper() == "DEBUG"

# Global configuration instance
config = Config()

# Helper functions
def load_config() -> Dict[str, Any]:
    """Load and return the full configuration"""
    return config._config

def get_config() -> Config:
    """Get the global configuration instance"""
    return config
