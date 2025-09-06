"""
Benchmark Scraper for Laptop Recommender System
Fetches CPU and GPU benchmark data from PassMark websites
Uses Knuth-Morris-Pratt algorithm for efficient string searching
"""

import requests
import pandas as pd
import numpy as np
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import json
import os
from urllib.parse import quote
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkScraper:
    """
    Scraper for CPU and GPU benchmark data from PassMark websites.
    """
    
    def __init__(self, preprocessor=None):
        """Initialize the benchmark scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize preprocessor for extraction methods
        if preprocessor is None:
            try:
                from data_preprocessing import LaptopDataPreprocessor
                self.preprocessor = LaptopDataPreprocessor()
            except ImportError:
                self.preprocessor = None
                logger.warning("Could not import LaptopDataPreprocessor, using built-in extraction methods")
        else:
            self.preprocessor = preprocessor
        
        # Cache for benchmark data
        self.cpu_benchmarks = {}
        self.gpu_benchmarks = {}
        
        # Passmark URLs
        self.cpu_url = "https://www.cpubenchmark.net/cpu_list.php"
        self.gpu_url = "https://www.videocardbenchmark.net/gpu_list.php"
        
        # Load cached data if available
        self.load_cached_benchmarks()
        
        # If no cached data, fetch benchmarks immediately
        if not self.cpu_benchmarks:
            self.fetch_cpu_benchmarks()
        if not self.gpu_benchmarks:
            self.fetch_gpu_benchmarks()
    
    def load_cached_benchmarks(self):
        """Load cached benchmark data from files."""
        try:
            if os.path.exists('data/cpu_benchmarks.json'):
                with open('data/cpu_benchmarks.json', 'r') as f:
                    self.cpu_benchmarks = json.load(f)
                logger.info(f"Loaded {len(self.cpu_benchmarks)} cached CPU benchmarks")
            
            if os.path.exists('data/gpu_benchmarks.json'):
                with open('data/gpu_benchmarks.json', 'r') as f:
                    self.gpu_benchmarks = json.load(f)
                logger.info(f"Loaded {len(self.gpu_benchmarks)} cached GPU benchmarks")
                
        except Exception as e:
            logger.warning(f"Could not load cached benchmarks: {e}")
    
    def save_cached_benchmarks(self):
        """Save benchmark data to cache files."""
        try:
            os.makedirs('data', exist_ok=True)
            
            with open('data/cpu_benchmarks.json', 'w') as f:
                json.dump(self.cpu_benchmarks, f, indent=2)
            
            with open('data/gpu_benchmarks.json', 'w') as f:
                json.dump(self.gpu_benchmarks, f, indent=2)
                
            logger.info("Benchmark data cached successfully")
            
        except Exception as e:
            logger.error(f"Could not save cached benchmarks: {e}")
    
    def normalize_processor_name(self, processor_name: str) -> str:
        """
        Normalize processor name for better matching.
        
        Args:
            processor_name (str): Raw processor name
            
        Returns:
            str: Normalized processor name
        """
        if pd.isna(processor_name) or processor_name == 'Unknown':
            return 'Unknown'
        
        # Convert to lowercase and remove extra spaces
        normalized = str(processor_name).lower().strip()
        
        # Remove common suffixes and prefixes
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        normalized = re.sub(r'processor', '', normalized)
        normalized = re.sub(r'cpu', '', normalized)
        normalized = re.sub(r'apu', '', normalized)
        normalized = re.sub(r'dual', '', normalized)
        normalized = re.sub(r'quad', '', normalized)
        # Don't remove 'core' as it's part of the benchmark dictionary keys
        normalized = re.sub(r'@\s*\d+\.?\d*ghz', '', normalized)  # Remove clock speeds
        normalized = re.sub(r'\s+', ' ', normalized).strip()  # Clean up spaces
        
        # Handle manufacturer prefix consistency
        if 'core i' in normalized and not normalized.startswith('intel'):
            normalized = 'intel ' + normalized
        elif 'celeron' in normalized and not normalized.startswith('intel'):
            normalized = 'intel ' + normalized
        elif 'pentium' in normalized and not normalized.startswith('intel'):
            normalized = 'intel ' + normalized
        elif 'ryzen' in normalized and not normalized.startswith('amd'):
            normalized = 'amd ' + normalized
        elif 'athlon' in normalized and not normalized.startswith('amd'):
            normalized = 'amd ' + normalized
        
        return normalized
    
    def normalize_gpu_name(self, gpu_name: str) -> str:
        """
        Normalize GPU name for better matching.
        
        Args:
            gpu_name (str): Raw GPU name
            
        Returns:
            str: Normalized GPU name
        """
        if pd.isna(gpu_name) or gpu_name == 'Unknown' or gpu_name == 0:
            return 'Unknown'
        
        # Convert to lowercase and remove extra spaces
        normalized = str(gpu_name).lower().strip()
        
        # Remove common suffixes and prefixes
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        normalized = re.sub(r'graphics', '', normalized)
        normalized = re.sub(r'card', '', normalized)
        normalized = re.sub(r'gpu', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()  # Clean up spaces
        
        return normalized
    
    def fetch_cpu_benchmarks(self) -> Dict[str, int]:
        """
        Fetch CPU benchmark data from PassMark.
        
        Returns:
            Dict[str, int]: Dictionary mapping processor names to benchmark scores
        """
        logger.info("Fetching CPU benchmarks from PassMark...")
        
        try:
            # Add delay to be respectful to the server
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(self.cpu_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the CPU table
            cpu_table = soup.find('table', {'id': 'cputable'})
            if not cpu_table:
                logger.warning("CPU table not found, trying alternative selectors...")
                cpu_table = soup.find('table', {'class': 'chart'})
            
            if not cpu_table:
                logger.error("Could not find CPU benchmark table")
                return {}
            
            cpu_benchmarks = {}
            
            # Parse table rows
            rows = cpu_table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 3:
                    try:
                        # Extract CPU name and benchmark score
                        cpu_name = cells[0].get_text(strip=True)
                        benchmark_score = cells[1].get_text(strip=True)
                        
                        # Clean up the benchmark score
                        benchmark_score = re.sub(r'[^\d]', '', benchmark_score)
                        
                        if cpu_name and benchmark_score:
                            score = int(benchmark_score)
                            normalized_name = self.normalize_processor_name(cpu_name)
                            cpu_benchmarks[normalized_name] = score
                            
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing CPU row: {e}")
                        continue
            
            self.cpu_benchmarks = cpu_benchmarks
            logger.info(f"Successfully fetched {len(cpu_benchmarks)} CPU benchmarks from PassMark")
            
            # Save to cache
            self.save_cached_benchmarks()
            
            return cpu_benchmarks
            
        except requests.RequestException as e:
            logger.error(f"Error fetching CPU benchmarks from PassMark: {e}")
            logger.info("Using cached data if available...")
            return self.cpu_benchmarks
        except Exception as e:
            logger.error(f"Unexpected error fetching CPU benchmarks: {e}")
            return {}
    
    def fetch_gpu_benchmarks(self) -> Dict[str, int]:
        """
        Fetch GPU benchmark data from PassMark.
        
        Returns:
            Dict[str, int]: Dictionary mapping GPU names to benchmark scores
        """
        logger.info("Fetching GPU benchmarks from PassMark...")
        
        try:
            # Add delay to be respectful to the server
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(self.gpu_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the GPU table
            gpu_table = soup.find('table', {'id': 'cputable'})
            if not gpu_table:
                logger.warning("GPU table not found, trying alternative selectors...")
                gpu_table = soup.find('table', {'class': 'chart'})
            
            if not gpu_table:
                logger.error("Could not find GPU benchmark table")
                return {}
            
            gpu_benchmarks = {}
            
            # Parse table rows
            rows = gpu_table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 3:
                    try:
                        # Extract GPU name and benchmark score
                        gpu_name = cells[0].get_text(strip=True)
                        benchmark_score = cells[1].get_text(strip=True)
                        
                        # Clean up the benchmark score
                        benchmark_score = re.sub(r'[^\d]', '', benchmark_score)
                        
                        if gpu_name and benchmark_score:
                            score = int(benchmark_score)
                            normalized_name = self.normalize_gpu_name(gpu_name)
                            gpu_benchmarks[normalized_name] = score
                            
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing GPU row: {e}")
                        continue
            
            self.gpu_benchmarks = gpu_benchmarks
            logger.info(f"Successfully fetched {len(gpu_benchmarks)} GPU benchmarks from PassMark")
            
            # Save to cache
            self.save_cached_benchmarks()
            
            return gpu_benchmarks
            
        except requests.RequestException as e:
            logger.error(f"Error fetching GPU benchmarks from PassMark: {e}")
            logger.info("Using cached data if available...")
            return self.gpu_benchmarks
        except Exception as e:
            logger.error(f"Unexpected error fetching GPU benchmarks: {e}")
            return {}
    
    def search_cpu_benchmark(self, processor_name: str) -> Optional[int]:
        """
        Search for CPU benchmark score on PassMark website.
        
        Args:
            processor_name (str): Processor name to search for
            
        Returns:
            Optional[int]: Benchmark score if found, None otherwise
        """
        try:
            # Normalize processor name for search
            search_term = self.normalize_processor_name(processor_name)
            
            # Create search URL
            search_url = f"https://www.cpubenchmark.net/cpu.php?cpu={quote(search_term)}"
            
            # Add delay to be respectful
            time.sleep(random.uniform(2, 4))
            
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for benchmark score in the page
            # Common patterns for PassMark CPU pages
            score_patterns = [
                r'PassMark CPU Mark: (\d+)',
                r'CPU Mark: (\d+)',
                r'Benchmark: (\d+)',
                r'Score: (\d+)'
            ]
            
            page_text = soup.get_text()
            for pattern in score_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
            
            # If no score found, try to find it in the cached data
            if search_term in self.cpu_benchmarks:
                return self.cpu_benchmarks[search_term]
            
            return None
            
        except Exception as e:
            logger.debug(f"Error searching CPU benchmark for {processor_name}: {e}")
            return None
    
    def search_gpu_benchmark(self, gpu_name: str) -> Optional[int]:
        """
        Search for GPU benchmark score on PassMark website.
        
        Args:
            gpu_name (str): GPU name to search for
            
        Returns:
            Optional[int]: Benchmark score if found, None otherwise
        """
        try:
            # Normalize GPU name for search
            search_term = self.normalize_gpu_name(gpu_name)
            
            # Create search URL
            search_url = f"https://www.videocardbenchmark.net/gpu.php?gpu={quote(search_term)}"
            
            # Add delay to be respectful
            time.sleep(random.uniform(2, 4))
            
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for benchmark score in the page
            # Common patterns for PassMark GPU pages
            score_patterns = [
                r'PassMark G3D Mark: (\d+)',
                r'G3D Mark: (\d+)',
                r'Benchmark: (\d+)',
                r'Score: (\d+)'
            ]
            
            page_text = soup.get_text()
            for pattern in score_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
            
            # If no score found, try to find it in the cached data
            if search_term in self.gpu_benchmarks:
                return self.gpu_benchmarks[search_term]
            
            return None
            
        except Exception as e:
            logger.debug(f"Error searching GPU benchmark for {gpu_name}: {e}")
            return None
    
    def get_cpu_benchmark_score(self, processor_name: str) -> int:
        """
        Get benchmark score for a processor using regex pattern matching.
        
        Args:
            processor_name (str): Processor name
            
        Returns:
            int: Benchmark score (0 if not found)
        """
        if not self.cpu_benchmarks:
            self.fetch_cpu_benchmarks()
        
        normalized_name = self.normalize_processor_name(processor_name)
        
        # Try exact match first
        if normalized_name in self.cpu_benchmarks:
            return self.cpu_benchmarks[normalized_name]
        
        # Use regex pattern matching for fuzzy matching
        best_pattern = self._find_best_pattern_match(normalized_name, list(self.cpu_benchmarks.keys()))
        
        if best_pattern:
            return self.cpu_benchmarks[best_pattern]
        
        # Return 0 if no match found (instead of None)
        return 0
    
    def get_gpu_benchmark_score(self, gpu_name: str) -> int:
        """
        Get benchmark score for a GPU using regex pattern matching.
        
        Args:
            gpu_name (str): GPU name
            
        Returns:
            int: Benchmark score (0 if not found)
        """
        if not self.gpu_benchmarks:
            self.fetch_gpu_benchmarks()
        
        normalized_name = self.normalize_gpu_name(gpu_name)
        
        # Try exact match first
        if normalized_name in self.gpu_benchmarks:
            return self.gpu_benchmarks[normalized_name]
        
        # Use regex pattern matching for fuzzy matching
        best_pattern = self._find_best_pattern_match(normalized_name, list(self.gpu_benchmarks.keys()))
        
        if best_pattern:
            return self.gpu_benchmarks[best_pattern]
        
        # If no match found in cache, try to search online
        logger.info(f"GPU benchmark not found in cache for {gpu_name}, searching online...")
        online_score = self.search_gpu_benchmark(gpu_name)
        
        if online_score:
            # Cache the result
            self.gpu_benchmarks[normalized_name] = online_score
            self.save_cached_benchmarks()
            return online_score
        
        # Return 0 if no match found (instead of None)
        return 0
    
    def _find_best_pattern_match(self, search_text: str, patterns: List[str]) -> Optional[str]:
        """
        Find the best matching pattern using regex and similarity scoring.
        
        Args:
            search_text (str): Text to search for
            patterns (List[str]): List of patterns to match against
            
        Returns:
            Optional[str]: Best matching pattern or None
        """
        if not patterns or not search_text:
            return None
        
        best_pattern = None
        best_score = 0.0
        
        # Split search text into words for better matching
        search_words = set(search_text.lower().split())
        
        for pattern in patterns:
            pattern_words = set(pattern.lower().split())
            
            # Calculate word overlap score
            common_words = search_words.intersection(pattern_words)
            if common_words:
                # Score based on word overlap ratio and pattern length
                overlap_ratio = len(common_words) / max(len(search_words), len(pattern_words))
                length_score = min(len(pattern), len(search_text)) / max(len(pattern), len(search_text))
                score = overlap_ratio * 0.7 + length_score * 0.3
                
                if score > best_score and score > 0.3:  # Minimum threshold
                    best_score = score
                    best_pattern = pattern
        
        return best_pattern
    
    
    
    
    def _extract_processor_name_from_text(self, text: str) -> Optional[str]:
        """
        Extract processor model from text using regex patterns.
        
        Args:
            text (str): Text containing processor information
            
        Returns:
            Optional[str]: Processor model, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # Processor patterns - comprehensive patterns to capture complete processor names
        processor_patterns = [
            # Intel Core i series - various formats
            r'(intel\s+core\s+i[3579]-\d+[a-z]*\d*)',  # Intel Core i7-5950HQ, i5-1135G7
            r'(intel\s+core\s+i[3579]\s+\d+[a-z]*\d*)',  # Intel Core i7 5950HQ (space instead of dash)
            r'(core\s+i[3579]-\d+[a-z]*\d*)',  # Core i7-5950HQ (without Intel)
            r'(core\s+i[3579]\s+\d+[a-z]*\d*)',  # Core i7 5950HQ (without Intel, space)
            r'(i[3579]-\d+[a-z]*\d*)',  # i7-5950HQ (minimal format)
            
            # AMD Ryzen series - various formats
            r'(amd\s+ryzen\s+[3579]\s+\d+[a-z]*\d*)',  # AMD Ryzen 5 5500U
            r'(amd\s+ryzen\s+[3579]-\d+[a-z]*\d*)',  # AMD Ryzen 5-5500U (with dash)
            r'(ryzen\s+[3579]\s+\d+[a-z]*\d*)',  # Ryzen 5 5500U (without AMD)
            r'(ryzen\s+[3579]-\d+[a-z]*\d*)',  # Ryzen 5-5500U (without AMD, with dash)
            
            # Intel Celeron series
            r'(intel\s+celeron\s+[a-z]*\d*)',  # Intel Celeron N4020
            r'(celeron\s+[a-z]*\d*)',  # Celeron N4020 (without Intel)
            
            # Intel Pentium series
            r'(intel\s+pentium\s+[a-z]*\d*)',  # Intel Pentium Gold 7505
            r'(pentium\s+[a-z]*\d*)',  # Pentium Gold 7505 (without Intel)
            
            # AMD Athlon series
            r'(amd\s+athlon\s+\d+[a-z]*)',  # AMD Athlon 300U
            r'(athlon\s+\d+[a-z]*)',  # Athlon 300U (without AMD)
            
            # Apple M series
            r'(apple\s+m\d+\s*[a-z]*)',  # Apple M1 Pro
            r'(m\d+\s*[a-z]*)',  # M1 Pro (without Apple)
            
            # AMD A series
            r'(amd\s+a\d+\s+[a-z]*\d*)',  # AMD A10-7850K
            r'(a\d+\s+[a-z]*\d*)',  # A10-7850K (without AMD)
        ]
        
        for pattern in processor_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                return matches[0].title()
        
        return None
    
    def _extract_gpu_name_from_text(self, text: str) -> Optional[str]:
        """
        Extract GPU model from text using regex patterns.
        
        Args:
            text (str): Text containing GPU information
            
        Returns:
            Optional[str]: GPU model, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # GPU patterns - comprehensive patterns to capture complete GPU names
        gpu_patterns = [
            # Intel Graphics - various formats
            r'(intel\s+uhd\s+graphics\s*\d*)',  # Intel UHD Graphics 600
            r'(intel\s+uhd\s+graphics)',  # Intel UHD Graphics (without number)
            r'(uhd\s+graphics\s*\d*)',  # UHD Graphics 600 (without Intel)
            r'(uhd\s+graphics)',  # UHD Graphics (without Intel, without number)
            
            r'(intel\s+iris\s+xe\s+graphics\s*[a-z0-9]*)',  # Intel Iris Xe Graphics G7
            r'(intel\s+iris\s+xe\s+graphics)',  # Intel Iris Xe Graphics (without suffix)
            r'(iris\s+xe\s+graphics\s*[a-z0-9]*)',  # Iris Xe Graphics G7 (without Intel)
            r'(iris\s+xe\s+graphics)',  # Iris Xe Graphics (without Intel, without suffix)
            
            r'(intel\s+iris\s+pro\s+graphics\s*[a-z0-9]*)',  # Intel Iris Pro Graphics 5200
            r'(intel\s+iris\s+graphics\s*[a-z0-9]*)',  # Intel Iris Graphics 6100
            r'(intel\s+hd\s+graphics\s*\d*)',  # Intel HD Graphics 4000
            
            # AMD Radeon Graphics - various formats
            r'(amd\s+radeon\s+vega\s*\d+)',  # AMD Radeon Vega 7
            r'(radeon\s+vega\s*\d+)',  # Radeon Vega 7 (without AMD)
            
            r'(amd\s+radeon\s+rx\s*\d{4}[a-z]*)',  # AMD Radeon RX 5500M
            r'(radeon\s+rx\s*\d{4}[a-z]*)',  # Radeon RX 5500M (without AMD)
            
            r'(amd\s+radeon\s+pro\s*\d{4}[a-z]*)',  # AMD Radeon Pro 5500M
            r'(radeon\s+pro\s*\d{4}[a-z]*)',  # Radeon Pro 5500M (without AMD)
            
            r'(amd\s+radeon\s+graphics)',  # AMD Radeon Graphics
            r'(radeon\s+graphics)',  # Radeon Graphics (without AMD)
            
            # NVIDIA Graphics - various formats
            r'(nvidia\s+geforce\s+gtx\s*\d{4}[a-z]*)',  # NVIDIA GeForce GTX 1650
            r'(geforce\s+gtx\s*\d{4}[a-z]*)',  # GeForce GTX 1650 (without NVIDIA)
            r'(gtx\s*\d{4}[a-z]*)',  # GTX 1650 (minimal format)
            
            r'(nvidia\s+geforce\s+rtx\s*\d{4}[a-z]*)',  # NVIDIA GeForce RTX 3060
            r'(geforce\s+rtx\s*\d{4}[a-z]*)',  # GeForce RTX 3060 (without NVIDIA)
            r'(rtx\s*\d{4}[a-z]*)',  # RTX 3060 (minimal format)
            
            r'(nvidia\s+quadro\s*[a-z0-9]+)',  # NVIDIA Quadro T1000
            r'(quadro\s*[a-z0-9]+)',  # Quadro T1000 (without NVIDIA)
            
            r'(nvidia\s+geforce\s+mx\s*\d{3}[a-z]*)',  # NVIDIA GeForce MX350
            r'(geforce\s+mx\s*\d{3}[a-z]*)',  # GeForce MX350 (without NVIDIA)
            r'(mx\s*\d{3}[a-z]*)',  # MX350 (minimal format)
        ]
        
        for pattern in gpu_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                return matches[0].title()
        
        return None
    
    def _get_cpu_benchmark_from_columns(self, row: pd.Series) -> int:
        """
        Extract CPU information from title_y, features, and details columns and get benchmark score.
        
        Args:
            row (pd.Series): DataFrame row containing the columns
            
        Returns:
            int: CPU benchmark score
        """
        # Combine text from all relevant columns
        text_parts = []
        
        # Add title_y if it exists
        if 'title_y' in row.index and pd.notna(row['title_y']):
            text_parts.append(str(row['title_y']))
        
        # Add features if it exists
        if 'features' in row.index and pd.notna(row['features']):
            text_parts.append(str(row['features']))
        
        # Add details if it exists
        if 'details' in row.index and pd.notna(row['details']):
            text_parts.append(str(row['details']))
        
        # Add details_parsed if it exists (processed details)
        if 'details_parsed' in row.index and pd.notna(row['details_parsed']):
            details_text = str(row['details_parsed'])
            text_parts.append(details_text)
        
        # Combine all text
        combined_text = ' '.join(text_parts)
        
        # First extract the processor model name from the combined text
        if self.preprocessor:
            processor_model = self.preprocessor._extract_processor_name_from_text(combined_text)
        else:
            processor_model = self._extract_processor_name_from_text(combined_text)
        
        if processor_model:
            # Use the extracted processor name with regex pattern matching
            return self.get_cpu_benchmark_score(processor_model)
        else:
            # Fallback: use the combined text if no specific processor model found
            return self.get_cpu_benchmark_score(combined_text)
    
    def _get_gpu_benchmark_from_columns(self, row: pd.Series) -> int:
        """
        Extract GPU information from title_y, features, and details columns and get benchmark score.
        
        Args:
            row (pd.Series): DataFrame row containing the columns
            
        Returns:
            int: GPU benchmark score
        """
        # Combine text from all relevant columns
        text_parts = []
        
        # Add title_y if it exists
        if 'title_y' in row.index and pd.notna(row['title_y']):
            text_parts.append(str(row['title_y']))
        
        # Add features if it exists
        if 'features' in row.index and pd.notna(row['features']):
            text_parts.append(str(row['features']))
        
        # Add details if it exists
        if 'details' in row.index and pd.notna(row['details']):
            text_parts.append(str(row['details']))
        
        # Add details_parsed if it exists (processed details)
        if 'details_parsed' in row.index and pd.notna(row['details_parsed']):
            details_text = str(row['details_parsed'])
            text_parts.append(details_text)
        
        # Combine all text
        combined_text = ' '.join(text_parts)
        
        # First extract the GPU model name from the combined text
        if self.preprocessor:
            gpu_model = self.preprocessor._extract_gpu_name_from_text(combined_text)
        else:
            gpu_model = self._extract_gpu_name_from_text(combined_text)
        
        if gpu_model:
            # Use the extracted GPU name with regex pattern matching
            return self.get_gpu_benchmark_score(gpu_model)
        else:
            # Fallback: use the combined text if no specific GPU model found
            return self.get_gpu_benchmark_score(combined_text)
    
    def debug_cpu_matching(self, text: str) -> Dict:
        """
        Debug function to test CPU matching and see what's happening.
        
        Args:
            text (str): Text containing CPU information
            
        Returns:
            Dict: Debug information about the matching process
        """
        # Simple debug function that just uses the actual function
        score = self.get_cpu_benchmark_score(text)
        normalized = self.normalize_processor_name(text)
        
        return {
            'input_text': text,
            'normalized_name': normalized,
            'final_score': score,
            'match_type': 'actual_function'
        }
    

    def add_benchmark_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add CPU and GPU benchmark scores and specifications to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe with processor and GPU information
            
        Returns:
            pd.DataFrame: Dataframe with added benchmark scores and specifications
        """
        logger.info("Adding benchmark scores and specifications to dataset...")
        
        # Ensure we have benchmark data
        if not self.cpu_benchmarks:
            self.fetch_cpu_benchmarks()
        if not self.gpu_benchmarks:
            self.fetch_gpu_benchmarks()
        
        # First, extract specifications from text columns
        df_with_specs = self.add_specifications_from_columns(df)
        
        # Add CPU benchmark scores by searching in title_y, features, and details columns
        logger.info(f"Adding CPU benchmark scores for {len(df_with_specs)} laptops...")
        cpu_scores = []
        for i, (_, row) in enumerate(df_with_specs.iterrows()):
            if i % 100 == 0:  # Progress indicator every 100 rows
                logger.info(f"Processing CPU benchmarks: {i}/{len(df_with_specs)} ({i/len(df_with_specs)*100:.1f}%)")
            cpu_scores.append(self._get_cpu_benchmark_from_columns(row))
        df_with_specs['cpu_benchmark_score'] = cpu_scores
        
        # Add GPU benchmark scores by searching in title_y, features, and details columns
        logger.info(f"Adding GPU benchmark scores for {len(df_with_specs)} laptops...")
        gpu_scores = []
        for i, (_, row) in enumerate(df_with_specs.iterrows()):
            if i % 100 == 0:  # Progress indicator every 100 rows
                logger.info(f"Processing GPU benchmarks: {i}/{len(df_with_specs)} ({i/len(df_with_specs)*100:.1f}%)")
            gpu_scores.append(self._get_gpu_benchmark_from_columns(row))
        df_with_specs['gpu_benchmark_score'] = gpu_scores
        logger.info("CPU and GPU benchmark scores processing completed!")
        
        # Calculate total benchmark score (weighted combination)
        df_with_specs['total_benchmark_score'] = (
            df_with_specs['cpu_benchmark_score'] * 0.7 +  # CPU has higher weight
            df_with_specs['gpu_benchmark_score'] * 0.3
        ).round(0)
        
        # Add performance tier based on total benchmark score
        def get_performance_tier(score):
            if score >= 20000:
                return 'Ultra High'
            elif score >= 15000:
                return 'High'
            elif score >= 10000:
                return 'Medium-High'
            elif score >= 7000:
                return 'Medium'
            elif score >= 4000:
                return 'Low-Medium'
            else:
                return 'Low'
        
        df_with_specs['performance_tier'] = df_with_specs['total_benchmark_score'].apply(get_performance_tier)
        
        # Add gaming capability score
        def get_gaming_capability(row):
            cpu_score = row['cpu_benchmark_score']
            gpu_score = row['gpu_benchmark_score']
            
            if gpu_score >= 8000 and cpu_score >= 12000:
                return 'High-End Gaming'
            elif gpu_score >= 5000 and cpu_score >= 8000:
                return 'Mid-Range Gaming'
            elif gpu_score >= 3000 and cpu_score >= 6000:
                return 'Casual Gaming'
            elif gpu_score >= 1500:
                return 'Light Gaming'
            else:
                return 'No Gaming'
        
        df_with_specs['gaming_capability'] = df_with_specs.apply(get_gaming_capability, axis=1)
        
        logger.info("Benchmark scores and specifications added successfully")
        return df_with_specs
    
    def add_specifications_from_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and add CPU/GPU specifications from title_y, features, and details columns.
        
        Args:
            df (pd.DataFrame): Input dataframe with title_y, features, and details columns
            
        Returns:
            pd.DataFrame: Dataframe with added CPU/GPU specification columns
        """
        logger.info("Extracting CPU/GPU specifications from columns...")
        
        df_specs = df.copy()
        
        # Combine text from all relevant columns for each row
        def combine_text_columns(row):
            text_parts = []
            
            # Add title_y if it exists
            if 'title_y' in row.index and pd.notna(row['title_y']):
                text_parts.append(str(row['title_y']))
            
            # Add features if it exists
            if 'features' in row.index and pd.notna(row['features']):
                text_parts.append(str(row['features']))
            
            # Add details if it exists
            if 'details' in row.index and pd.notna(row['details']):
                text_parts.append(str(row['details']))
            
            # Add details_parsed if it exists (processed details)
            if 'details_parsed' in row.index and pd.notna(row['details_parsed']):
                details_text = str(row['details_parsed'])
                text_parts.append(details_text)
            
            return ' '.join(text_parts)
        
        # Extract CPU and GPU specifications for each row using preprocessor methods if available
        if self.preprocessor:
            df_specs['processor_model'] = df_specs.apply(
                lambda row: self.preprocessor._extract_processor_name_from_text(combine_text_columns(row)), axis=1
            )
            
            df_specs['gpu_model'] = df_specs.apply(
                lambda row: self.preprocessor._extract_gpu_name_from_text(combine_text_columns(row)), axis=1
            )
        else:
            # Fallback to built-in methods
            df_specs['processor_model'] = df_specs.apply(
                lambda row: self._extract_processor_name_from_text(combine_text_columns(row)), axis=1
            )
            
            df_specs['gpu_model'] = df_specs.apply(
                lambda row: self._extract_gpu_name_from_text(combine_text_columns(row)), axis=1
            )
        
        logger.info("CPU/GPU specifications extracted successfully")
        logger.info(f"Processor models found: {df_specs['processor_model'].notna().sum()}/{len(df_specs)} rows")
        logger.info(f"GPU models found: {df_specs['gpu_model'].notna().sum()}/{len(df_specs)} rows")
        
        return df_specs
    
    def get_benchmark_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about benchmark scores in the dataset.
        
        Args:
            df (pd.DataFrame): Dataframe with benchmark scores
            
        Returns:
            Dict: Statistics about benchmark scores
        """
        stats = {
            'cpu_benchmark_stats': {
                'mean': float(df['cpu_benchmark_score'].mean()),
                'median': float(df['cpu_benchmark_score'].median()),
                'min': int(df['cpu_benchmark_score'].min()),
                'max': int(df['cpu_benchmark_score'].max()),
                'std': float(df['cpu_benchmark_score'].std())
            },
            'gpu_benchmark_stats': {
                'mean': float(df['gpu_benchmark_score'].mean()),
                'median': float(df['gpu_benchmark_score'].median()),
                'min': int(df['gpu_benchmark_score'].min()),
                'max': int(df['gpu_benchmark_score'].max()),
                'std': float(df['gpu_benchmark_score'].std())
            },
            'total_benchmark_stats': {
                'mean': float(df['total_benchmark_score'].mean()),
                'median': float(df['total_benchmark_score'].median()),
                'min': int(df['total_benchmark_score'].min()),
                'max': int(df['total_benchmark_score'].max()),
                'std': float(df['total_benchmark_score'].std())
            },
            'performance_tier_distribution': df['performance_tier'].value_counts().to_dict(),
            'gaming_capability_distribution': df['gaming_capability'].value_counts().to_dict()
        }
        
        return stats
    


def main():
    """
    Main function to test the benchmark scraper with regex pattern matching.
    """
    scraper = BenchmarkScraper()
    
    # Test with sample data using the correct column structure
    test_data = pd.DataFrame({
        'title_y': [
            'HP Pavilion Laptop with Intel Core i5-1135G7 and Intel Iris Xe Graphics',
            'Lenovo ThinkPad with AMD Ryzen 5 5500U and AMD Radeon Vega 7',
            'Dell XPS with Intel Core i7-1165G7 and Intel Iris Xe Graphics G7',
            'Acer Aspire with Intel Celeron N4020 and Intel UHD Graphics 600',
            'ASUS VivoBook with AMD Ryzen 7 5700U and AMD Radeon Vega 8'
        ],
        'features': [
            'Intel Core i5-1135G7 processor, Intel Iris Xe Graphics',
            'AMD Ryzen 5 5500U processor, AMD Radeon Vega 7 graphics',
            'Intel Core i7-1165G7 processor, Intel Iris Xe Graphics G7',
            'Intel Celeron N4020 processor, Intel UHD Graphics 600',
            'AMD Ryzen 7 5700U processor, AMD Radeon Vega 8 graphics'
        ],
        'details': [
            '{"processor": "Intel Core i5-1135G7", "graphics": "Intel Iris Xe Graphics"}',
            '{"processor": "AMD Ryzen 5 5500U", "graphics": "AMD Radeon Vega 7"}',
            '{"processor": "Intel Core i7-1165G7", "graphics": "Intel Iris Xe Graphics G7"}',
            '{"processor": "Intel Celeron N4020", "graphics": "Intel UHD Graphics 600"}',
            '{"processor": "AMD Ryzen 7 5700U", "graphics": "AMD Radeon Vega 8"}'
        ]
    })
    
    # Test processor matching functionality
    print("Testing processor matching functionality...")
    test_processors = [
        'Intel Core i5-1135G7',
        'AMD Ryzen 5 5500U',
        'Intel Core i7-1165G7',
        'Intel Celeron N4020',
        'AMD Ryzen 7 5700U',
        'Intel Core i9-11900H',
        'AMD Ryzen 9 5900H',
        'Intel Pentium Gold 7505',
        'AMD Athlon 300U',
        'Apple M1 Pro'
    ]
    
    print("\n=== Testing Processor Matching ===")
    for processor in test_processors:
        score = scraper.get_cpu_benchmark_score(processor)
        print(f"{processor}: {score}")
    
    # Test debug functionality for specific processors
    print("\n=== Testing Debug Functionality ===")
    test_processors = [
        'AMD Ryzen 5 5500U',
        'Intel Core i7-1165G7', 
        'Intel Celeron N4020',
        'AMD Ryzen 7 5700U'
    ]
    
    for processor in test_processors:
        print(f"\n--- Testing: {processor} ---")
        debug_result = scraper.debug_cpu_matching(processor)
        print(f"Input: {debug_result['input_text']}")
        print(f"Normalized: {debug_result['normalized_name']}")
        print(f"Final Score: {debug_result['final_score']}")
        print(f"Match Type: {debug_result['match_type']}")
    
    # Add benchmark scores
    result = scraper.add_benchmark_scores(test_data)
    
    print("\nSample benchmark results:")
    print(result[['title_y', 'cpu_benchmark_score', 'gpu_benchmark_score', 
                 'total_benchmark_score', 'performance_tier', 'gaming_capability']])
    
    print("\nSample CPU/GPU specification results:")
    spec_columns = ['processor_model', 'gpu_model']
    available_spec_cols = [col for col in spec_columns if col in result.columns]
    if available_spec_cols:
        print(result[['title_y'] + available_spec_cols])
    
    # Get statistics
    stats = scraper.get_benchmark_statistics(result)
    print("\nBenchmark statistics:")
    print(json.dumps(stats, indent=2))
    
    # Show all columns in the result
    print(f"\nTotal columns in result: {len(result.columns)}")
    print("All columns:")
    for i, col in enumerate(result.columns):
        print(f"  {i+1:2d}. {col}")


if __name__ == "__main__":
    main()
