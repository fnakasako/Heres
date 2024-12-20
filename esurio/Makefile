# Esurio Market Intelligence System
# Advanced mathematical frameworks for market analysis

.PHONY: all install test lint format clean docs build deploy

# Environment variables
PYTHON := python3
POETRY := poetry
PYTEST := $(POETRY) run pytest
BLACK := $(POETRY) run black
ISORT := $(POETRY) run isort
MYPY := $(POETRY) run mypy
PYLINT := $(POETRY) run pylint

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
BUILD_DIR := build

# Default target
all: install test lint

# Installation
install:
	$(POETRY) install

install-dev:
	$(POETRY) install --with dev

# Testing
test:
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing

test-fast:
	$(PYTEST) $(TEST_DIR) -m "not slow"

test-quant:
	$(PYTEST) $(TEST_DIR)/quant -v

# Code quality
lint: format
	$(PYLINT) $(SRC_DIR)
	$(MYPY) $(SRC_DIR)

format:
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(ISORT) $(SRC_DIR) $(TEST_DIR)

# Mathematical framework validation
validate-math:
	$(PYTEST) $(TEST_DIR)/quant/test_mathematical_core.py -v
	$(PYTEST) $(TEST_DIR)/quant/test_neural_architectures.py -v

validate-strategies:
	$(PYTEST) $(TEST_DIR)/quant/test_strategy_generator.py -v
	$(PYTEST) $(TEST_DIR)/quant/test_backtester.py -v

validate-risk:
	$(PYTEST) $(TEST_DIR)/quant/test_risk_manager.py -v
	$(PYTEST) $(TEST_DIR)/quant/test_optimizer.py -v

# Data collection
scrape-market:
	$(PYTHON) -m src.scraping.runner --spider market

scrape-news:
	$(PYTHON) -m src.scraping.runner --spider news

scrape-social:
	$(PYTHON) -m src.scraping.runner --spider social

scrape-all:
	$(PYTHON) -m src.scraping.runner --all

# Analysis and insights
generate-insights:
	$(PYTHON) -m src.processing.quant.coordinator analyze

generate-strategies:
	$(PYTHON) -m src.processing.quant.coordinator strategies

backtest:
	$(PYTHON) -m src.processing.quant.backtester

optimize:
	$(PYTHON) -m src.processing.quant.optimizer

# Database
db-init:
	$(PYTHON) -m src.db.init

db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

# Development tools
dev-setup: install-dev
	pre-commit install
	$(PYTHON) -m spacy download en_core_web_lg

jupyter:
	$(POETRY) run jupyter notebook

# Documentation
docs:
	$(POETRY) run sphinx-build -b html $(DOCS_DIR)/source $(DOCS_DIR)/build

# Cleaning
clean:
	rm -rf $(BUILD_DIR)
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf .tox
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type f -name "*.bak" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +

# Building and deployment
build:
	$(POETRY) build

docker-build:
	docker build -t esurio .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Help
help:
	@echo "Esurio Market Intelligence System"
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make test          Run tests"
	@echo "  make test-quant    Run quantitative framework tests"
	@echo "  make lint          Run linters"
	@echo "  make format        Format code"
	@echo "  make validate-math Run mathematical validation tests"
	@echo "  make scrape-all    Run all scrapers"
	@echo "  make backtest      Run backtesting"
	@echo "  make optimize      Run strategy optimization"
	@echo "  make docs          Build documentation"
	@echo "  make clean         Clean build artifacts"
	@echo "  make build         Build package"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run with Docker Compose"

# Default target
.DEFAULT_GOAL := help
