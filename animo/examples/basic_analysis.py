"""
Example script demonstrating basic usage of the Financial Insights toolkit.
This script fetches SEC filings for a company and performs sentiment analysis.
"""

import sys
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scrapers.sec_edgar import SECEdgarScraper
from src.data_processing.pipeline import DataPipeline
from src.analysis.sentiment.analyzer import SentimentAnalyzer
from config.settings import LOGGING

# Configure logging
logging.basicConfig(**LOGGING['handlers']['default'])
logger = logging.getLogger(__name__)

def setup_pipeline() -> DataPipeline:
    """
    Set up the data processing pipeline with necessary components.
    
    Returns:
        DataPipeline: Configured pipeline instance
    """
    # Initialize components
    scraper = SECEdgarScraper()
    analyzer = SentimentAnalyzer()
    pipeline = DataPipeline()
    
    # Add components to pipeline
    pipeline.add_scraper(scraper)
    pipeline.add_analyzer(analyzer.analyze_batch)
    
    return pipeline

def analyze_company(
    company: str,
    filing_type: str = "10-K",
    lookback_days: int = 365
) -> Dict[str, Any]:
    """
    Analyze SEC filings for a specific company.
    
    Args:
        company: Company ticker symbol or CIK
        filing_type: Type of SEC filing to analyze
        lookback_days: Number of days to look back for filings
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Setup and run pipeline
    pipeline = setup_pipeline()
    
    logger.info(f"Starting analysis for {company}")
    logger.info(f"Fetching {filing_type} filings from {start_date.date()} to {end_date.date()}")
    
    try:
        results = pipeline.run(
            company=company,
            filing_type=filing_type,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Extract relevant information
        raw_data = results['raw_data']
        analysis_results = results['analysis_results'][0]  # First analyzer's results
        
        logger.info(f"Successfully analyzed {len(raw_data)} filings")
        
        return {
            'company': company,
            'filing_type': filing_type,
            'period': f"{start_date.date()} to {end_date.date()}",
            'num_filings': len(raw_data),
            'analysis': analysis_results
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {company}: {str(e)}")
        raise

def main():
    """Main execution function."""
    # Example usage
    companies = ["AAPL", "MSFT", "GOOGL"]
    
    for company in companies:
        try:
            results = analyze_company(company)
            
            # Print summary
            print(f"\nAnalysis Results for {company}")
            print("=" * 50)
            print(f"Period: {results['period']}")
            print(f"Number of filings analyzed: {results['num_filings']}")
            
            # Print sentiment analysis results
            if results['num_filings'] > 0:
                sentiment = results['analysis'].get(company, {})
                print("\nSentiment Analysis:")
                print(f"Average Polarity: {sentiment.get('average_polarity', 0):.3f}")
                print(f"Average Subjectivity: {sentiment.get('average_subjectivity', 0):.3f}")
                print(f"Positive Terms: {sentiment.get('total_positive_terms', 0)}")
                print(f"Negative Terms: {sentiment.get('total_negative_terms', 0)}")
                print(f"Overall Sentiment Ratio: {sentiment.get('overall_sentiment_ratio', 0):.3f}")
            
        except Exception as e:
            print(f"\nError analyzing {company}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
