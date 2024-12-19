from typing import Any, Dict, List
import requests
from bs4 import BeautifulSoup
from .base import BaseScraper, rate_limit, robust_scrape, ScraperException

class SECEdgarScraper(BaseScraper):
    """Scraper for SEC EDGAR financial filings."""
    
    BASE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    
    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        # SEC EDGAR requires a user agent
        self.session.headers.update({
            'User-Agent': 'FinancialInsights 1.0',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
    
    @rate_limit(0.1)  # Maximum 10 requests per second
    @robust_scrape(retries=3)
    def _fetch_filing(self, url: str) -> str:
        """
        Fetch a single filing document.
        
        Args:
            url: URL of the filing
            
        Returns:
            str: Raw HTML content of the filing
        """
        response = self.session.get(url)
        response.raise_for_status()
        return response.text
    
    def scrape(self, 
               company: str,
               filing_type: str = "10-K",
               start_date: str = None,
               end_date: str = None) -> List[Dict[str, Any]]:
        """
        Scrape SEC filings for a given company.
        
        Args:
            company: Company name or CIK
            filing_type: Type of filing to fetch (e.g., 10-K, 10-Q)
            start_date: Start date for filing search (YYYY-MM-DD)
            end_date: End date for filing search (YYYY-MM-DD)
            
        Returns:
            List[Dict[str, Any]]: List of filing data
        """
        params = {
            'company': company,
            'type': filing_type,
            'owner': 'exclude',
            'start': start_date,
            'end': end_date,
            'output': 'atom'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            entries = soup.find_all('entry')
            
            filings = []
            for entry in entries:
                filing_data = {
                    'title': entry.title.text,
                    'filing_date': entry.updated.text,
                    'filing_type': filing_type,
                    'company': company,
                    'link': entry.link['href'],
                    'accession_number': entry.id.text.split('/')[-1]
                }
                
                # Fetch the actual filing document
                filing_content = self._fetch_filing(filing_data['link'])
                filing_data['content'] = filing_content
                filings.append(filing_data)
            
            return filings
            
        except requests.exceptions.RequestException as e:
            raise ScraperException(f"Failed to fetch SEC filings: {str(e)}")
    
    def clean(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean the scraped SEC filing data.
        
        Args:
            data: Raw filing data
            
        Returns:
            List[Dict[str, Any]]: Cleaned filing data
        """
        cleaned_data = []
        for filing in data:
            # Parse the HTML content
            soup = BeautifulSoup(filing['content'], 'html.parser')
            
            # Remove unnecessary tags
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            # Extract text and clean it
            text = soup.get_text(separator=' ')
            text = ' '.join(text.split())  # Remove extra whitespace
            
            cleaned_filing = {
                'title': filing['title'],
                'filing_date': filing['filing_date'],
                'filing_type': filing['filing_type'],
                'company': filing['company'],
                'accession_number': filing['accession_number'],
                'text': text
            }
            cleaned_data.append(cleaned_filing)
        
        return cleaned_data
