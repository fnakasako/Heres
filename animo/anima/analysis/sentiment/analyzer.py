from typing import List, Dict, Any
import re
from textblob import TextBlob
from collections import defaultdict

class SentimentAnalyzer:
    """
    Analyzer for performing sentiment analysis on financial texts.
    """
    
    def __init__(self):
        # Common financial terms that might indicate positive/negative sentiment
        self.positive_terms = {
            'growth', 'profit', 'increase', 'improved', 'success', 'positive',
            'strong', 'gain', 'opportunity', 'innovative', 'leading', 'achieved',
            'exceeded', 'record', 'robust', 'favorable'
        }
        
        self.negative_terms = {
            'loss', 'decline', 'decrease', 'risk', 'negative', 'weak',
            'challenging', 'difficult', 'uncertain', 'volatility', 'adverse',
            'failed', 'litigation', 'below', 'unfavorable', 'discontinued'
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and extra whitespace.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def _calculate_term_frequency(self, text: str) -> Dict[str, int]:
        """
        Calculate frequency of positive and negative terms.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict[str, int]: Dictionary with term frequencies
        """
        words = text.split()
        frequencies = {
            'positive': sum(1 for word in words if word in self.positive_terms),
            'negative': sum(1 for word in words if word in self.negative_terms)
        }
        return frequencies
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis on a single document.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        cleaned_text = self._clean_text(text)
        
        # Use TextBlob for general sentiment analysis
        blob = TextBlob(cleaned_text)
        
        # Calculate term frequencies
        term_freq = self._calculate_term_frequency(cleaned_text)
        
        # Calculate sentiment metrics
        return {
            'polarity': blob.sentiment.polarity,  # Range: -1 (negative) to 1 (positive)
            'subjectivity': blob.sentiment.subjectivity,  # Range: 0 (objective) to 1 (subjective)
            'positive_terms': term_freq['positive'],
            'negative_terms': term_freq['negative'],
            'sentiment_ratio': term_freq['positive'] / (term_freq['negative'] + 1),  # Avoid division by zero
            'word_count': len(cleaned_text.split())
        }
    
    def analyze_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform sentiment analysis on a batch of documents.
        
        Args:
            documents: List of documents with text content
            
        Returns:
            List[Dict[str, Any]]: Sentiment analysis results for each document
        """
        results = []
        for doc in documents:
            # Analyze the document text
            sentiment_results = self.analyze_document(doc['text'])
            
            # Combine original metadata with sentiment analysis results
            analysis_result = {
                'document_id': doc.get('accession_number', ''),
                'filing_type': doc.get('filing_type', ''),
                'filing_date': doc.get('filing_date', ''),
                'company': doc.get('company', ''),
                'sentiment_analysis': sentiment_results
            }
            results.append(analysis_result)
        
        return results
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate sentiment analysis results across multiple documents.
        
        Args:
            results: List of sentiment analysis results
            
        Returns:
            Dict[str, Any]: Aggregated sentiment metrics
        """
        if not results:
            return {}
        
        # Group results by company
        company_results = defaultdict(list)
        for result in results:
            company_results[result['company']].append(result['sentiment_analysis'])
        
        # Calculate aggregated metrics for each company
        aggregated = {}
        for company, analyses in company_results.items():
            avg_polarity = sum(a['polarity'] for a in analyses) / len(analyses)
            avg_subjectivity = sum(a['subjectivity'] for a in analyses) / len(analyses)
            total_positive = sum(a['positive_terms'] for a in analyses)
            total_negative = sum(a['negative_terms'] for a in analyses)
            
            aggregated[company] = {
                'average_polarity': avg_polarity,
                'average_subjectivity': avg_subjectivity,
                'total_positive_terms': total_positive,
                'total_negative_terms': total_negative,
                'overall_sentiment_ratio': total_positive / (total_negative + 1),
                'document_count': len(analyses)
            }
        
        return aggregated
