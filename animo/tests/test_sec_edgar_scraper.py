"""
Tests for the SEC EDGAR scraper implementation.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from bs4 import BeautifulSoup

from src.scrapers.sec_edgar import SECEdgarScraper
from src.scrapers.base import ScraperException

# Sample test data
SAMPLE_FILING_HTML = """
<entry>
    <title>10-K filing for AAPL</title>
    <updated>2023-01-01T00:00:00Z</updated>
    <id>sec-filing/0000123456</id>
    <link href="https://www.sec.gov/filing/doc1" />
</entry>
"""

SAMPLE_FILING_CONTENT = """
<html>
    <body>
        <div class="filing-content">
            <p>This is a sample filing content with some positive growth 
            and profit indicators, but also mentions some risk factors.</p>
        </div>
    </body>
</html>
"""

@pytest.fixture
def scraper():
    """Create a scraper instance for testing."""
    return SECEdgarScraper()

@pytest.fixture
def mock_response():
    """Create a mock response object."""
    mock = Mock()
    mock.content = SAMPLE_FILING_HTML.encode()
    mock.text = SAMPLE_FILING_CONTENT
    mock.raise_for_status = Mock()
    return mock

def test_scraper_initialization(scraper):
    """Test scraper initialization."""
    assert isinstance(scraper, SECEdgarScraper)
    assert scraper.BASE_URL == "https://www.sec.gov/cgi-bin/browse-edgar"
    assert 'User-Agent' in scraper.session.headers

@patch('requests.Session.get')
def test_fetch_filing(mock_get, scraper, mock_response):
    """Test fetching a single filing."""
    mock_get.return_value = mock_response
    
    content = scraper._fetch_filing("https://www.sec.gov/filing/doc1")
    
    assert content == SAMPLE_FILING_CONTENT
    mock_get.assert_called_once_with("https://www.sec.gov/filing/doc1")
    mock_response.raise_for_status.assert_called_once()

@patch('requests.Session.get')
def test_scrape_filings(mock_get, scraper, mock_response):
    """Test scraping multiple filings."""
    # Mock both the initial request and the filing content request
    mock_get.return_value = mock_response
    
    results = scraper.scrape(
        company="AAPL",
        filing_type="10-K",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    assert len(results) == 1
    assert results[0]['company'] == "AAPL"
    assert results[0]['filing_type'] == "10-K"
    assert results[0]['accession_number'] == "0000123456"
    assert 'content' in results[0]

@patch('requests.Session.get')
def test_scrape_error_handling(mock_get, scraper):
    """Test error handling during scraping."""
    mock_get.side_effect = Exception("API Error")
    
    with pytest.raises(ScraperException):
        scraper.scrape(company="AAPL")

def test_clean_data(scraper):
    """Test cleaning of scraped data."""
    test_data = [{
        'content': SAMPLE_FILING_CONTENT,
        'accession_number': '0000123456',
        'filing_type': '10-K',
        'filing_date': '2023-01-01',
        'company': 'AAPL'
    }]
    
    cleaned_data = scraper.clean(test_data)
    
    assert len(cleaned_data) == 1
    assert 'text' in cleaned_data[0]
    assert 'filing_content' not in cleaned_data[0]
    assert 'growth' in cleaned_data[0]['text'].lower()
    assert 'risk' in cleaned_data[0]['text'].lower()

def test_rate_limiting(scraper):
    """Test rate limiting functionality."""
    with patch('time.sleep') as mock_sleep:
        for _ in range(3):
            scraper._fetch_filing("https://test.url")
        
        # Should have called sleep at least once due to rate limiting
        assert mock_sleep.called

def test_invalid_date_format(scraper):
    """Test handling of invalid date formats."""
    with pytest.raises(ScraperException):
        scraper.scrape(
            company="AAPL",
            start_date="invalid-date"
        )

def test_empty_response_handling(scraper, mock_response):
    """Test handling of empty responses."""
    mock_response.content = "<feed></feed>".encode()
    
    with patch('requests.Session.get', return_value=mock_response):
        results = scraper.scrape(company="AAPL")
        
        assert isinstance(results, list)
        assert len(results) == 0

if __name__ == '__main__':
    pytest.main([__file__])
