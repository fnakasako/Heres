"""
Database session management for Esurio Market Intelligence System.
"""

import contextlib
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.common.config import load_config
from src.common.logging_util import setup_logging
from src.db.models import Base

logger = setup_logging()

def get_database_url() -> str:
    """
    Get database URL from configuration.
    
    Returns:
        Database connection URL
    """
    config = load_config()
    db_config = config.get("database", {})
    
    return db_config.get("url", "postgresql://localhost/esurio")

def create_database_engine(url: str = None) -> Engine:
    """
    Create SQLAlchemy engine with proper configuration.
    
    Args:
        url: Optional database URL override
        
    Returns:
        Configured SQLAlchemy engine
    """
    if url is None:
        url = get_database_url()
    
    config = load_config()
    db_config = config.get("database", {})
    
    # Engine configuration
    engine_config = {
        # Connection pool settings
        "poolclass": QueuePool,
        "pool_size": db_config.get("pool_size", 5),
        "max_overflow": db_config.get("max_overflow", 10),
        "pool_timeout": db_config.get("pool_timeout", 30),
        "pool_recycle": db_config.get("pool_recycle", 3600),
        
        # Query execution settings
        "execution_options": {
            "isolation_level": "READ COMMITTED",
            "postgresql_readonly": False,
            "postgresql_deferrable": False
        },
        
        # Performance settings
        "echo": db_config.get("echo", False),
        "echo_pool": db_config.get("echo_pool", False),
        "future": True,
        
        # Connection settings
        "connect_args": {
            "connect_timeout": db_config.get("connect_timeout", 10),
            "application_name": "esurio"
        }
    }
    
    engine = create_engine(url, **engine_config)
    
    # Set up engine event listeners
    @event.listens_for(engine, "connect")
    def connect(dbapi_connection, connection_record):
        """Set session parameters on connection."""
        cursor = dbapi_connection.cursor()
        cursor.execute("SET timezone='UTC'")
        cursor.close()
    
    @event.listens_for(engine, "checkout")
    def checkout(dbapi_connection, connection_record, connection_proxy):
        """Verify connection is valid on checkout."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SELECT 1")
        except:
            # Replace invalid connection
            raise Exception(
                "Connection invalid, please ensure database is running"
            )
        finally:
            cursor.close()
    
    return engine

def get_session_factory(engine: Engine = None) -> sessionmaker:
    """
    Create session factory for database connections.
    
    Args:
        engine: Optional SQLAlchemy engine override
        
    Returns:
        Session factory
    """
    if engine is None:
        engine = create_database_engine()
    
    # Configure session factory
    factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )
    
    return factory

def get_session() -> Generator[Session, None, None]:
    """
    Get database session from factory.
    
    Yields:
        Active database session
    """
    factory = get_session_factory()
    session = factory()
    
    try:
        yield session
    finally:
        session.close()

@contextlib.contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Provides automatic commit/rollback and proper cleanup.
    
    Yields:
        Active database session
    """
    factory = get_session_factory()
    session = factory()
    
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(
            "Database error",
            extra={"error": str(e)}
        )
        raise
    finally:
        session.close()

def init_database(engine: Engine = None) -> None:
    """
    Initialize database schema.
    
    Args:
        engine: Optional SQLAlchemy engine override
    """
    if engine is None:
        engine = create_database_engine()
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema created successfully")
        
    except Exception as e:
        logger.error(
            "Error creating database schema",
            extra={"error": str(e)}
        )
        raise

def drop_database(engine: Engine = None) -> None:
    """
    Drop all database tables.
    
    Args:
        engine: Optional SQLAlchemy engine override
    """
    if engine is None:
        engine = create_database_engine()
    
    try:
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("Database schema dropped successfully")
        
    except Exception as e:
        logger.error(
            "Error dropping database schema",
            extra={"error": str(e)}
        )
        raise

class DatabaseManager:
    """
    Database connection and session manager.
    
    Provides centralized database management with connection pooling,
    session handling, and automatic cleanup.
    """
    
    def __init__(self, url: str = None):
        """
        Initialize database manager.
        
        Args:
            url: Optional database URL override
        """
        self.engine = create_database_engine(url)
        self.session_factory = get_session_factory(self.engine)
    
    def get_session(self) -> Session:
        """
        Get new database session.
        
        Returns:
            Active database session
        """
        return self.session_factory()
    
    @contextlib.contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        
        Yields:
            Active database session
        """
        session = self.get_session()
        
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(
                "Database error",
                extra={"error": str(e)}
            )
            raise
        finally:
            session.close()
    
    def init_database(self) -> None:
        """Initialize database schema."""
        init_database(self.engine)
    
    def drop_database(self) -> None:
        """Drop all database tables."""
        drop_database(self.engine)
    
    def dispose(self) -> None:
        """Dispose of database engine and connections."""
        self.engine.dispose()
        logger.info("Database connections disposed")

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions using global manager
def get_db() -> Generator[Session, None, None]:
    """Get database session from global manager."""
    return db_manager.session_scope()

def init_db() -> None:
    """Initialize database using global manager."""
    db_manager.init_database()

def drop_db() -> None:
    """Drop database using global manager."""
    db_manager.drop_database()

def dispose_db() -> None:
    """Dispose of global database manager."""
    db_manager.dispose()
