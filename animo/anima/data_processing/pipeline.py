from typing import List, Any, Dict, Callable
from ..scrapers.base import BaseScraper

class DataPipeline:
    """
    A pipeline for orchestrating data collection, processing, and analysis.
    """
    
    def __init__(self):
        self.scrapers: List[BaseScraper] = []
        self.processors: List[Callable] = []
        self.analyzers: List[Callable] = []
        self.data: List[Dict[str, Any]] = []
    
    def add_scraper(self, scraper: BaseScraper) -> None:
        """
        Add a scraper to the pipeline.
        
        Args:
            scraper: An instance of BaseScraper
        """
        if not isinstance(scraper, BaseScraper):
            raise TypeError("Scraper must be an instance of BaseScraper")
        self.scrapers.append(scraper)
    
    def add_processor(self, processor: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]) -> None:
        """
        Add a data processor to the pipeline.
        
        Args:
            processor: A function that takes and returns a list of dictionaries
        """
        self.processors.append(processor)
    
    def add_analyzer(self, analyzer: Callable[[List[Dict[str, Any]]], Any]) -> None:
        """
        Add an analyzer to the pipeline.
        
        Args:
            analyzer: A function that processes the data and returns analysis results
        """
        self.analyzers.append(analyzer)
    
    def run_scrapers(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Run all registered scrapers.
        
        Args:
            **kwargs: Arguments to pass to each scraper
            
        Returns:
            List[Dict[str, Any]]: Combined results from all scrapers
        """
        all_data = []
        for scraper in self.scrapers:
            try:
                scraper_data = scraper.run(**kwargs)
                all_data.extend(scraper_data)
            except Exception as e:
                print(f"Error running scraper {scraper.__class__.__name__}: {str(e)}")
        return all_data
    
    def run_processors(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run all registered processors on the data.
        
        Args:
            data: Data to process
            
        Returns:
            List[Dict[str, Any]]: Processed data
        """
        processed_data = data
        for processor in self.processors:
            try:
                processed_data = processor(processed_data)
            except Exception as e:
                print(f"Error running processor {processor.__name__}: {str(e)}")
        return processed_data
    
    def run_analyzers(self, data: List[Dict[str, Any]]) -> List[Any]:
        """
        Run all registered analyzers on the data.
        
        Args:
            data: Data to analyze
            
        Returns:
            List[Any]: Results from all analyzers
        """
        results = []
        for analyzer in self.analyzers:
            try:
                result = analyzer(data)
                results.append(result)
            except Exception as e:
                print(f"Error running analyzer {analyzer.__name__}: {str(e)}")
        return results
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Args:
            **kwargs: Arguments to pass to scrapers
            
        Returns:
            Dict[str, Any]: Dictionary containing raw data, processed data, and analysis results
        """
        # Run scrapers
        raw_data = self.run_scrapers(**kwargs)
        self.data = raw_data
        
        # Run processors
        processed_data = self.run_processors(raw_data)
        
        # Run analyzers
        analysis_results = self.run_analyzers(processed_data)
        
        return {
            'raw_data': raw_data,
            'processed_data': processed_data,
            'analysis_results': analysis_results
        }
