import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger

from .config import get_config, get_project_root

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level name
        log_record['level'] = record.levelname
        
        # Add source location
        log_record['location'] = {
            'file': record.pathname,
            'line': record.lineno,
            'function': record.funcName
        }
        
        # Add process and thread info
        log_record['process'] = {
            'id': record.process,
            'name': record.processName
        }
        log_record['thread'] = {
            'id': record.thread,
            'name': record.threadName
        }

class LogManager:
    """Manages logging configuration and setup"""
    
    def __init__(self) -> None:
        self.config = get_config()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        # Get logging config
        log_config = self.config.get("logging", default={})
        log_level = log_config.get("level", "INFO").upper()
        
        # Create logs directory if it doesn't exist
        log_dir = get_project_root() / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Setup handlers
        handlers = []
        
        # Console handler
        if log_config.get("handlers", {}).get("console", {}).get("enabled", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            handlers.append(console_handler)
        
        # File handler
        file_config = log_config.get("handlers", {}).get("file", {})
        if file_config.get("enabled", True):
            log_file = log_dir / file_config.get("path", "esurio.log")
            max_bytes = file_config.get("max_size_mb", 100) * 1024 * 1024
            backup_count = file_config.get("backup_count", 5)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            handlers.append(file_handler)
        
        # Create formatter
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            json_ensure_ascii=False
        )
        
        # Add formatter to handlers and attach to root logger
        for handler in handlers:
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
        
        # Special handling for third-party loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('selenium').setLevel(logging.WARNING)
        logging.getLogger('scrapy').setLevel(logging.INFO)

class ContextLogger:
    """Logger that includes context information with each log message"""
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format message with context"""
        log_data = {
            'message': message,
            'context': self.context
        }
        if extra:
            log_data['extra'] = extra
        return log_data
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message with context"""
        self.logger.debug(json.dumps(self._format_message(message, extra)))
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message with context"""
        self.logger.info(json.dumps(self._format_message(message, extra)))
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message with context"""
        self.logger.warning(json.dumps(self._format_message(message, extra)))
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message with context"""
        self.logger.error(json.dumps(self._format_message(message, extra)))
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message with context"""
        self.logger.critical(json.dumps(self._format_message(message, extra)))

def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> ContextLogger:
    """Get a context logger instance"""
    return ContextLogger(name, context)

# Initialize logging on module import
log_manager = LogManager()

# Example usage:
# logger = get_logger(__name__, {'spider': 'market_prices', 'target': 'SPY'})
# logger.info('Starting market price scraping', {'url': 'https://example.com'})
