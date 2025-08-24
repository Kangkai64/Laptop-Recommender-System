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
from urllib.parse import urljoin, quote
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnuthMorrisPratt:
    """
    Implementation of the Knuth-Morris-Pratt string search algorithm
    for efficient pattern matching in CPU and GPU benchmark searches.
    """
    
    def __init__(self):
        """Initialize the KMP algorithm."""
        pass
    
    def compute_lps(self, pattern: str) -> List[int]:
        """
        Compute the Longest Proper Prefix which is also Suffix (LPS) array.
        
        Args:
            pattern (str): The pattern to search for
            
        Returns:
            List[int]: LPS array for the pattern
        """
        lps = [0] * len(pattern)
        length = 0  # Length of the previous longest prefix suffix
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    def search(self, text: str, pattern: str) -> List[int]:
        """
        Search for pattern in text using KMP algorithm.
        
        Args:
            text (str): The text to search in
            pattern (str): The pattern to search for
            
        Returns:
            List[int]: List of starting indices where pattern is found
        """
        if not pattern or not text:
            return []
        
        n, m = len(text), len(pattern)
        lps = self.compute_lps(pattern)
        i = j = 0  # i for text, j for pattern
        matches = []
        
        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return matches
    
    def search_case_insensitive(self, text: str, pattern: str) -> List[int]:
        """
        Case-insensitive search using KMP algorithm.
        
        Args:
            text (str): The text to search in
            pattern (str): The pattern to search for
            
        Returns:
            List[int]: List of starting indices where pattern is found
        """
        return self.search(text.lower(), pattern.lower())
    
    def find_best_match(self, text: str, patterns: List[str]) -> Tuple[str, float]:
        """
        Find the best matching pattern in text using KMP algorithm.
        
        Args:
            text (str): The text to search in
            patterns (List[str]): List of patterns to search for
            
        Returns:
            Tuple[str, float]: (best_pattern, match_score)
        """
        if not patterns:
            return "", 0.0
        
        best_pattern = ""
        best_score = 0.0
        
        for pattern in patterns:
            matches = self.search_case_insensitive(text, pattern)
            if matches:
                # Calculate match score based on pattern length and number of matches
                score = len(pattern) * len(matches) / len(text)
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
        
        return best_pattern, best_score

class BenchmarkScraper:
    """
    Scraper for CPU and GPU benchmark data from PassMark websites.
    """
    
    def __init__(self):
        """Initialize the benchmark scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize KMP algorithm for efficient string searching
        self.kmp = KnuthMorrisPratt()
        
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
        Get benchmark score for a processor using KMP algorithm for efficient matching.
        
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
        
        # Use KMP algorithm for efficient pattern matching
        cpu_patterns = list(self.cpu_benchmarks.keys())
        best_pattern, match_score = self.kmp.find_best_match(normalized_name, cpu_patterns)
        
        if best_pattern and match_score > 0.1:  # Threshold for acceptable match
            return self.cpu_benchmarks[best_pattern]
        
        # If no match found in cache, try to search online
        logger.info(f"CPU benchmark not found in cache for {processor_name}, searching online...")
        online_score = self.search_cpu_benchmark(processor_name)
        
        if online_score:
            # Cache the result
            self.cpu_benchmarks[normalized_name] = online_score
            self.save_cached_benchmarks()
            return online_score
        
        # Default score for unknown processors
        return 3000
    
    def get_gpu_benchmark_score(self, gpu_name: str) -> int:
        """
        Get benchmark score for a GPU using KMP algorithm for efficient matching.
        
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
        
        # Use KMP algorithm for efficient pattern matching
        gpu_patterns = list(self.gpu_benchmarks.keys())
        best_pattern, match_score = self.kmp.find_best_match(normalized_name, gpu_patterns)
        
        if best_pattern and match_score > 0.1:  # Threshold for acceptable match
            return self.gpu_benchmarks[best_pattern]
        
        # If no match found in cache, try to search online
        logger.info(f"GPU benchmark not found in cache for {gpu_name}, searching online...")
        online_score = self.search_gpu_benchmark(gpu_name)
        
        if online_score:
            # Cache the result
            self.gpu_benchmarks[normalized_name] = online_score
            self.save_cached_benchmarks()
            return online_score
        
        # Default score for unknown GPUs (integrated graphics)
        return 500
    
    def _extract_cpu_benchmark_with_regex(self, text: str) -> int:
        """
        Extract CPU benchmark score using regex patterns for precise matching.
        
        Args:
            text (str): Text containing CPU information
            
        Returns:
            int: CPU benchmark score
        """
        if not self.cpu_benchmarks:
            self.fetch_cpu_benchmarks()
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Define regex patterns for different CPU families with exact model matching
        cpu_patterns = {
            # Intel Core i3 patterns - exact model matching
            r'intel\s+core\s+i3[-\s](\d{4}[a-z0-9]+)': ('core i3', 'core i3-{}'),
            r'i3[-\s](\d{4}[a-z0-9]+)': ('core i3', 'core i3-{}'),
            
            # Intel Core i5 patterns - exact model matching
            r'intel\s+core\s+i5[-\s](\d{4}[a-z0-9]+)': ('core i5', 'core i5-{}'),
            r'i5[-\s](\d{4}[a-z0-9]+)': ('core i5', 'core i5-{}'),
            
            # Intel Core i7 patterns - exact model matching
            r'intel\s+core\s+i7[-\s](\d{4}[a-z0-9]+)': ('core i7', 'core i7-{}'),
            r'i7[-\s](\d{4}[a-z0-9]+)': ('core i7', 'core i7-{}'),
            
            # Intel Core i9 patterns - exact model matching
            r'intel\s+core\s+i9[-\s](\d{4}[a-z0-9]+)': ('core i9', 'core i9-{}'),
            r'i9[-\s](\d{4}[a-z0-9]+)': ('core i9', 'core i9-{}'),
            
            # Intel Celeron patterns - exact model matching
            r'intel\s+celeron[-\s]([a-z0-9]+)': ('celeron', 'celeron {}'),
            r'celeron[-\s]([a-z0-9]+)': ('celeron', 'celeron {}'),
            
            # Intel Pentium patterns - exact model matching
            r'intel\s+pentium[-\s]([a-z0-9\s]+)': ('pentium', 'pentium {}'),
            r'pentium[-\s]([a-z0-9\s]+)': ('pentium', 'pentium {}'),
            
            # AMD Ryzen patterns - exact model matching
            r'amd\s+ryzen\s+(\d+)[-\s](\d{4}[a-z]?)': ('ryzen', 'ryzen {} {}'),
            r'ryzen\s+(\d+)[-\s](\d{4}[a-z]?)': ('ryzen', 'ryzen {} {}'),
            
            # AMD Athlon patterns - exact model matching
            r'amd\s+athlon[-\s]([a-z0-9]+)': ('athlon', 'athlon {}'),
            r'athlon[-\s]([a-z0-9]+)': ('athlon', 'athlon {}'),
            
            # AMD A series patterns - exact model matching
            r'amd\s+a(\d+)[-\s]([a-z0-9]+)': ('a6', 'a{}-{}'),
            r'a(\d+)[-\s]([a-z0-9]+)': ('a6', 'a{}-{}'),
            
            # Apple M series patterns - exact model matching
            r'apple\s+m(\d+)': ('m1', 'm{}'),
            r'm(\d+)': ('m1', 'm{}'),
            
            # Additional patterns for better matching
            # Intel Core patterns with different spacing
            r'intel\s+core\s+i(\d+)[-\s](\d{4}[a-z0-9]+)': ('core i{}', 'core i{}-{}'),
            r'core\s+i(\d+)[-\s](\d{4}[a-z0-9]+)': ('core i{}', 'core i{}-{}'),
            
            # AMD Ryzen patterns with different spacing
            r'amd\s+ryzen\s+(\d+)\s+(\d{4}[a-z]?)': ('ryzen', 'ryzen {} {}'),
            r'ryzen\s+(\d+)\s+(\d{4}[a-z]?)': ('ryzen', 'ryzen {} {}'),
        }
        
        # Try to match specific models first
        for pattern, (family, model_format) in cpu_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Try to find exact model match
                for match in matches:
                    if isinstance(match, tuple):
                        # Handle patterns with multiple groups (like Ryzen)
                        try:
                            model_name = model_format.format(*match).lower()
                        except (IndexError, TypeError):
                            # Fallback for tuple formatting issues
                            model_name = f"{family}-{'-'.join(match)}".lower()
                    else:
                        # Handle single group matches
                        try:
                            model_name = model_format.format(match).lower()
                        except (IndexError, TypeError):
                            # Fallback for single group formatting issues
                            model_name = f"{family}-{match}".lower()
                    
                    # Try exact match first
                    if model_name in self.cpu_benchmarks:
                        return self.cpu_benchmarks[model_name]
                    
                    # Try family match
                    if family in self.cpu_benchmarks:
                        return self.cpu_benchmarks[family]
        
        # Use KMP algorithm for efficient pattern matching with cached data
        cpu_patterns_list = list(self.cpu_benchmarks.keys())
        best_pattern, match_score = self.kmp.find_best_match(text_lower, cpu_patterns_list)
        
        if best_pattern and match_score > 0.1:  # Threshold for acceptable match
            return self.cpu_benchmarks[best_pattern]
        
        # Fallback to family-based matching
        family_keywords = {
            'core i3': ['i3', 'core i3'],
            'core i5': ['i5', 'core i5'],
            'core i7': ['i7', 'core i7'],
            'core i9': ['i9', 'core i9'],
            'ryzen 3': ['ryzen 3'],
            'ryzen 5': ['ryzen 5'],
            'ryzen 7': ['ryzen 7'],
            'ryzen 9': ['ryzen 9'],
            'celeron': ['celeron'],
            'athlon': ['athlon'],
            'pentium': ['pentium'],
            'ryzen': ['ryzen']
        }
        
        for family, keywords in family_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return self.cpu_benchmarks.get(family, 3000)
        
        # Default score for unknown processors
        return 3000
    
    def _extract_gpu_benchmark_with_regex(self, text: str) -> int:
        """
        Extract GPU benchmark score using regex patterns for precise matching.
        
        Args:
            text (str): Text containing GPU information
            
        Returns:
            int: GPU benchmark score
        """
        if not self.gpu_benchmarks:
            self.fetch_gpu_benchmarks()
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Define regex patterns for different GPU families
        gpu_patterns = {
            # Intel UHD Graphics patterns
            r'intel\s+uhd\s+graphics\s*(\d*)': 'intel uhd graphics',
            r'uhd\s+graphics\s*(\d*)': 'intel uhd graphics',
            
            # Intel Iris Xe Graphics patterns
            r'intel\s+iris\s+xe\s+graphics\s*([a-z0-9]*)': 'intel iris xe graphics',
            r'iris\s+xe\s+graphics\s*([a-z0-9]*)': 'intel iris xe graphics',
            
            # AMD Radeon Vega patterns
            r'amd\s+radeon\s+vega\s*(\d+)': 'amd radeon vega',
            r'radeon\s+vega\s*(\d+)': 'amd radeon vega',
            
            # AMD Radeon Graphics patterns
            r'amd\s+radeon\s+graphics': 'amd radeon graphics',
            r'radeon\s+graphics': 'amd radeon graphics',
            
            # NVIDIA GeForce GTX patterns
            r'nvidia\s+geforce\s+gtx\s*(\d{4}[a-z]*)': 'geforce gtx',
            r'geforce\s+gtx\s*(\d{4}[a-z]*)': 'geforce gtx',
            r'gtx\s*(\d{4}[a-z]*)': 'geforce gtx',
            
            # NVIDIA GeForce RTX patterns
            r'nvidia\s+geforce\s+rtx\s*(\d{4}[a-z]*)': 'geforce rtx',
            r'geforce\s+rtx\s*(\d{4}[a-z]*)': 'geforce rtx',
            r'rtx\s*(\d{4}[a-z]*)': 'geforce rtx',
            
            # NVIDIA Quadro patterns
            r'nvidia\s+quadro\s*([a-z0-9]+)': 'quadro',
            r'quadro\s*([a-z0-9]+)': 'quadro',
            
            # AMD Radeon RX patterns
            r'amd\s+radeon\s+rx\s*(\d{4}[a-z]*)': 'radeon rx',
            r'radeon\s+rx\s*(\d{4}[a-z]*)': 'radeon rx',
            r'rx\s*(\d{4}[a-z]*)': 'radeon rx',
            
            # AMD Radeon Pro patterns
            r'amd\s+radeon\s+pro\s*(\d{4}[a-z]*)': 'radeon pro',
            r'radeon\s+pro\s*(\d{4}[a-z]*)': 'radeon pro',
        }
        
        # Try to match specific models first
        for pattern, family in gpu_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Try to find exact model match
                for match in matches:
                    if match:  # If we have a specific model number
                        # Clean up the match and create model name
                        clean_match = match.strip()
                        if family == 'intel uhd graphics':
                            model_name = f"{family} {clean_match}"
                        elif family == 'intel iris xe graphics':
                            model_name = f"{family} {clean_match}"
                        elif family == 'amd radeon vega':
                            model_name = f"amd radeon vega {clean_match}"
                        elif family == 'geforce gtx':
                            model_name = f"geforce gtx {clean_match}"
                        elif family == 'geforce rtx':
                            model_name = f"geforce rtx {clean_match}"
                        elif family == 'quadro':
                            model_name = f"quadro {clean_match}"
                        elif family == 'radeon rx':
                            model_name = f"radeon rx {clean_match}"
                        elif family == 'radeon pro':
                            model_name = f"radeon pro {clean_match}"
                        else:
                            model_name = f"{family} {clean_match}"
                        
                        if model_name in self.gpu_benchmarks:
                            return self.gpu_benchmarks[model_name]
                    
                    # Try family match
                    if family in self.gpu_benchmarks:
                        return self.gpu_benchmarks[family]
        
        # Use KMP algorithm for efficient pattern matching with cached data
        gpu_patterns_list = list(self.gpu_benchmarks.keys())
        best_pattern, match_score = self.kmp.find_best_match(text_lower, gpu_patterns_list)
        
        if best_pattern and match_score > 0.1:  # Threshold for acceptable match
            return self.gpu_benchmarks[best_pattern]
        
        # Fallback to family-based matching
        family_keywords = {
            'intel uhd graphics': ['uhd', 'intel uhd'],
            'intel iris xe graphics': ['iris xe', 'intel iris'],
            'amd radeon vega': ['radeon vega', 'vega'],
            'amd radeon graphics': ['radeon', 'amd radeon'],
            'geforce gtx': ['gtx', 'geforce gtx'],
            'geforce rtx': ['rtx', 'geforce rtx'],
            'quadro': ['quadro'],
            'radeon rx': ['rx', 'radeon rx'],
            'radeon pro': ['radeon pro']
        }
        
        for family, keywords in family_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return self.gpu_benchmarks.get(family, 500)
        
        # Default score for unknown GPUs (integrated graphics)
        return 500
    
    def _extract_ram_from_text(self, text: str) -> Optional[float]:
        """
        Extract RAM capacity from text using regex patterns.
        
        Args:
            text (str): Text containing RAM information
            
        Returns:
            Optional[float]: RAM capacity in GB, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # RAM patterns with various formats
        ram_patterns = [
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ddr\d*|ram|memory)',  # 8GB DDR4, 16GB RAM
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ddr\d*)',  # 8GB DDR4
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ram)',  # 8GB RAM
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:memory)',  # 8GB Memory
            r'(\d+(?:\.\d+)?)\s*gb',  # 8GB (fallback)
            r'(\d+(?:\.\d+)?)\s*tb\s*(?:ddr\d*|ram|memory)',  # 1TB DDR4
            r'(\d+(?:\.\d+)?)\s*tb',  # 1TB (fallback)
        ]
        
        for pattern in ram_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                value = float(matches[0])
                # Convert TB to GB
                if 'tb' in text_str and 'gb' not in text_str:
                    value *= 1024
                return value
        
        return None
    
    def _extract_storage_from_text(self, text: str) -> Optional[float]:
        """
        Extract storage capacity from text using regex patterns.
        
        Args:
            text (str): Text containing storage information
            
        Returns:
            Optional[float]: Storage capacity in GB, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # Storage patterns with various formats - prioritize storage-specific terms
        storage_patterns = [
            # High priority: explicit storage terms with capacity
            r'(\d+(?:\.\d+)?)\s*tb\s*(?:ssd|hdd|hard\s*drive|storage|flash\s*storage|nvme|pcie)',  # 1TB SSD, 2TB NVMe
            r'(\d+(?:\.\d+)?)\s*tb\s*(?:ssd|hdd|nvme|pcie)',  # 1TB SSD, 2TB NVMe
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ssd|hdd|hard\s*drive|storage|flash\s*storage|nvme|pcie)',  # 512GB SSD, 1TB HDD
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ssd|hdd|nvme|pcie)',  # 512GB SSD
            
            # Medium priority: storage with less specific terms
            r'(\d+(?:\.\d+)?)\s*tb\s*(?:hard\s*drive|storage)',  # 1TB hard drive
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:hard\s*drive|storage)',  # 512GB hard drive
            
            # Lower priority: just numbers with storage context
            r'(\d+(?:\.\d+)?)\s*tb',  # 1TB (fallback, but check context)
            r'(\d+(?:\.\d+)?)\s*gb',  # 512GB (fallback, but check context)
        ]
        
        # First try high-priority patterns
        for i, pattern in enumerate(storage_patterns):
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                value = float(matches[0])
                
                # Check if this pattern matched TB or GB
                # Look for the pattern in the original text to see the unit
                pattern_match = re.search(pattern, text_str, re.IGNORECASE)
                if pattern_match:
                    matched_text = pattern_match.group(0).lower()
                    # If the pattern contains TB, convert to GB
                    if 'tb' in matched_text and 'gb' not in matched_text[:matched_text.find('tb')]:
                        value *= 1024
                
                # For lower priority patterns, verify it's actually storage
                if i >= 6:  # Lower priority patterns
                    # Check if this might be RAM instead of storage
                    if any(term in text_str for term in ['ram', 'memory', 'ddr', 'lpddr']):
                        # Skip if this looks like RAM
                        continue
                
                return value
        
        return None
    
    def _extract_screen_size_from_text(self, text: str) -> Optional[float]:
        """
        Extract screen size from text using regex patterns.
        
        Args:
            text (str): Text containing screen size information
            
        Returns:
            Optional[float]: Screen size in inches, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # Screen size patterns
        screen_patterns = [
            r'(\d+(?:\.\d+)?)\s*inch',  # 15.6 inch
            r'(\d+(?:\.\d+)?)\s*"',  # 15.6"
            r'(\d+(?:\.\d+)?)\s*in',  # 15.6 in
            r'(\d+(?:\.\d+)?)\s*inches',  # 15.6 inches
        ]
        
        for pattern in screen_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                return float(matches[0])
        
        return None
    
    def _extract_storage_type_from_text(self, text: str) -> Optional[str]:
        """
        Extract storage type from text using regex patterns.
        
        Args:
            text (str): Text containing storage type information
            
        Returns:
            Optional[str]: Storage type (SSD, HDD, Hybrid, etc.), None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # Storage type patterns
        if 'ssd' in text_str:
            if 'nvme' in text_str:
                return 'NVMe SSD'
            elif 'pcie' in text_str:
                return 'PCIe SSD'
            else:
                return 'SSD'
        elif 'hdd' in text_str or 'hard drive' in text_str or 'hard disk' in text_str:
            return 'HDD'
        elif 'hybrid' in text_str or 'sshdd' in text_str:
            return 'Hybrid'
        elif 'emmc' in text_str:
            return 'eMMC'
        elif 'flash' in text_str:
            return 'Flash Storage'
        
        return None
    
    def _extract_ram_type_from_text(self, text: str) -> Optional[str]:
        """
        Extract RAM type from text using regex patterns.
        
        Args:
            text (str): Text containing RAM type information
            
        Returns:
            Optional[str]: RAM type (DDR4, DDR5, LPDDR4, etc.), None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # RAM type patterns
        ram_types = ['ddr5', 'ddr4', 'ddr3', 'lpddr5', 'lpddr4', 'lpddr3']
        
        for ram_type in ram_types:
            if ram_type in text_str:
                return ram_type.upper()
        
        return None
    
    def _extract_processor_model_from_text(self, text: str) -> Optional[str]:
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
        
        # Processor patterns
        processor_patterns = [
            r'(intel\s+core\s+i[3579]-\d+[a-z]*\d*)',  # Intel Core i5-1135G7
            r'(amd\s+ryzen\s+[3579]\s+\d+[a-z]*\d*)',  # AMD Ryzen 5 5500U
            r'(intel\s+celeron\s+[a-z]*\d*)',  # Intel Celeron N4020
            r'(intel\s+pentium\s+[a-z]*\d*)',  # Intel Pentium Gold 7505
            r'(amd\s+athlon\s+\d+[a-z]*)',  # AMD Athlon 300U
            r'(apple\s+m\d+\s*[a-z]*)',  # Apple M1 Pro
        ]
        
        for pattern in processor_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                return matches[0].title()
        
        return None
    
    def _extract_gpu_model_from_text(self, text: str) -> Optional[str]:
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
        
        # GPU patterns
        gpu_patterns = [
            r'(intel\s+uhd\s+graphics\s*\d*)',  # Intel UHD Graphics 600
            r'(intel\s+iris\s+xe\s+graphics\s*[a-z0-9]*)',  # Intel Iris Xe Graphics G7
            r'(amd\s+radeon\s+vega\s*\d+)',  # AMD Radeon Vega 7
            r'(amd\s+radeon\s+graphics)',  # AMD Radeon Graphics
            r'(nvidia\s+geforce\s+gtx\s*\d{4}[a-z]*)',  # NVIDIA GeForce GTX 1650
            r'(nvidia\s+geforce\s+rtx\s*\d{4}[a-z]*)',  # NVIDIA GeForce RTX 3060
            r'(nvidia\s+quadro\s*[a-z0-9]+)',  # NVIDIA Quadro T1000
            r'(amd\s+radeon\s+rx\s*\d{4}[a-z]*)',  # AMD Radeon RX 5500M
            r'(amd\s+radeon\s+pro\s*\d{4}[a-z]*)',  # AMD Radeon Pro 5500M
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
        
        # Extract CPU information using regex patterns
        return self._extract_cpu_benchmark_with_regex(combined_text)
    
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
        
        # Extract GPU information using regex patterns
        return self._extract_gpu_benchmark_with_regex(combined_text)
    
    def debug_cpu_matching(self, text: str) -> Dict:
        """
        Debug function to test CPU regex matching and see what's happening.
        
        Args:
            text (str): Text containing CPU information
            
        Returns:
            Dict: Debug information about the matching process
        """
        if not self.cpu_benchmarks:
            self.fetch_cpu_benchmarks()
        
        debug_info = {
            'input_text': text,
            'text_lower': text.lower(),
            'matches_found': [],
            'model_names_tried': [],
            'benchmark_scores_found': [],
            'final_score': None
        }
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Define regex patterns for different CPU families with exact model matching
        cpu_patterns = {
            # Intel Core i3 patterns - exact model matching
            r'intel\s+core\s+i3[-\s](\d{4}[a-z0-9]+)': ('core i3', 'core i3-{}'),
            r'i3[-\s](\d{4}[a-z0-9]+)': ('core i3', 'core i3-{}'),
            
            # Intel Core i5 patterns - exact model matching
            r'intel\s+core\s+i5[-\s](\d{4}[a-z0-9]+)': ('core i5', 'core i5-{}'),
            r'i5[-\s](\d{4}[a-z0-9]+)': ('core i5', 'core i5-{}'),
            
            # Intel Core i7 patterns - exact model matching
            r'intel\s+core\s+i7[-\s](\d{4}[a-z0-9]+)': ('core i7', 'core i7-{}'),
            r'i7[-\s](\d{4}[a-z0-9]+)': ('core i7', 'core i7-{}'),
            
            # Intel Core i9 patterns - exact model matching
            r'intel\s+core\s+i9[-\s](\d{4}[a-z0-9]+)': ('core i9', 'core i9-{}'),
            r'i9[-\s](\d{4}[a-z0-9]+)': ('core i9', 'core i9-{}'),
            
            # Intel Celeron patterns - exact model matching
            r'intel\s+celeron[-\s]([a-z0-9]+)': ('celeron', 'celeron {}'),
            r'celeron[-\s]([a-z0-9]+)': ('celeron', 'celeron {}'),
            
            # Intel Pentium patterns - exact model matching
            r'intel\s+pentium[-\s]([a-z0-9\s]+)': ('pentium', 'pentium {}'),
            r'pentium[-\s]([a-z0-9\s]+)': ('pentium', 'pentium {}'),
            
            # AMD Ryzen patterns - exact model matching
            r'amd\s+ryzen\s+(\d+)[-\s](\d{4}[a-z]?)': ('ryzen', 'ryzen {} {}'),
            r'ryzen\s+(\d+)[-\s](\d{4}[a-z]?)': ('ryzen', 'ryzen {} {}'),
            
            # AMD Athlon patterns - exact model matching
            r'amd\s+athlon[-\s]([a-z0-9]+)': ('athlon', 'athlon {}'),
            r'athlon[-\s]([a-z0-9]+)': ('athlon', 'athlon {}'),
            
            # AMD A series patterns - exact model matching
            r'amd\s+a(\d+)[-\s]([a-z0-9]+)': ('a6', 'a{}-{}'),
            r'a(\d+)[-\s]([a-z0-9]+)': ('a6', 'a{}-{}'),
            
            # Apple M series patterns - exact model matching
            r'apple\s+m(\d+)': ('m1', 'm{}'),
            r'm(\d+)': ('m1', 'm{}'),
            
            # Additional patterns for better matching
            # Intel Core patterns with different spacing
            r'intel\s+core\s+i(\d+)[-\s](\d{4}[a-z0-9]+)': ('core i{}', 'core i{}-{}'),
            r'core\s+i(\d+)[-\s](\d{4}[a-z0-9]+)': ('core i{}', 'core i{}-{}'),
            
            # AMD Ryzen patterns with different spacing
            r'amd\s+ryzen\s+(\d+)\s+(\d{4}[a-z]?)': ('ryzen', 'ryzen {} {}'),
            r'ryzen\s+(\d+)\s+(\d{4}[a-z]?)': ('ryzen', 'ryzen {} {}'),
        }
        
        # Try to match specific models first
        for pattern, (family, model_format) in cpu_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                debug_info['matches_found'].append({
                    'pattern': pattern,
                    'family': family,
                    'model_format': model_format,
                    'matches': matches
                })
                
                # Try to find exact model match
                for match in matches:
                    if isinstance(match, tuple):
                        # Handle patterns with multiple groups (like Ryzen)
                        try:
                            model_name = model_format.format(*match).lower()
                        except (IndexError, TypeError):
                            # Fallback for tuple formatting issues
                            model_name = f"{family}-{'-'.join(match)}".lower()
                    else:
                        # Handle single group matches
                        try:
                            model_name = model_format.format(match).lower()
                        except (IndexError, TypeError):
                            # Fallback for single group formatting issues
                            model_name = f"{family}-{match}".lower()
                    
                    debug_info['model_names_tried'].append(model_name)
                    
                    # Try exact match first
                    if model_name in self.cpu_benchmarks:
                        score = self.cpu_benchmarks[model_name]
                        debug_info['benchmark_scores_found'].append({
                            'model_name': model_name,
                            'score': score,
                            'match_type': 'exact'
                        })
                        debug_info['final_score'] = score
                        return debug_info
                    
                    # Try family match
                    if family in self.cpu_benchmarks:
                        score = self.cpu_benchmarks[family]
                        debug_info['benchmark_scores_found'].append({
                            'model_name': family,
                            'score': score,
                            'match_type': 'family'
                        })
                        debug_info['final_score'] = score
                        return debug_info
        
        # Fallback to family-based matching
        family_keywords = {
            'core i3': ['i3', 'core i3'],
            'core i5': ['i5', 'core i5'],
            'core i7': ['i7', 'core i7'],
            'core i9': ['i9', 'core i9'],
            'ryzen 3': ['ryzen 3'],
            'ryzen 5': ['ryzen 5'],
            'ryzen 7': ['ryzen 7'],
            'ryzen 9': ['ryzen 9'],
            'celeron': ['celeron'],
            'athlon': ['athlon'],
            'pentium': ['pentium'],
            'ryzen': ['ryzen']
        }
        
        for family, keywords in family_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    score = self.cpu_benchmarks.get(family, 3000)
                    debug_info['benchmark_scores_found'].append({
                        'model_name': family,
                        'score': score,
                        'match_type': 'keyword_fallback'
                    })
                    debug_info['final_score'] = score
                    return debug_info
        
        # Default score for unknown processors
        default_score = self.cpu_benchmarks.get('unknown', 3000)
        debug_info['benchmark_scores_found'].append({
            'model_name': 'unknown',
            'score': default_score,
            'match_type': 'default'
        })
        debug_info['final_score'] = default_score
        return debug_info
    
    def get_specification_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about extracted specifications in the dataset.
        
        Args:
            df (pd.DataFrame): Dataframe with specification columns
            
        Returns:
            Dict: Statistics about specifications
        """
        stats = {}
        
        # RAM statistics
        if 'ram_gb' in df.columns:
            ram_stats = df['ram_gb'].describe()
            stats['ram_stats'] = {
                'mean': float(ram_stats['mean']) if not pd.isna(ram_stats['mean']) else None,
                'median': float(df['ram_gb'].median()) if not df['ram_gb'].isna().all() else None,
                'min': int(ram_stats['min']) if not pd.isna(ram_stats['min']) else None,
                'max': int(ram_stats['max']) if not pd.isna(ram_stats['max']) else None,
                'std': float(ram_stats['std']) if not pd.isna(ram_stats['std']) else None,
                'total_found': int(df['ram_gb'].notna().sum()),
                'total_rows': len(df)
            }
            
            if 'ram_category' in df.columns:
                stats['ram_category_distribution'] = df['ram_category'].value_counts().to_dict()
        
        # Storage statistics
        if 'storage_gb' in df.columns:
            storage_stats = df['storage_gb'].describe()
            stats['storage_stats'] = {
                'mean': float(storage_stats['mean']) if not pd.isna(storage_stats['mean']) else None,
                'median': float(df['storage_gb'].median()) if not df['storage_gb'].isna().all() else None,
                'min': int(storage_stats['min']) if not pd.isna(storage_stats['min']) else None,
                'max': int(storage_stats['max']) if not pd.isna(storage_stats['max']) else None,
                'std': float(storage_stats['std']) if not pd.isna(storage_stats['std']) else None,
                'total_found': int(df['storage_gb'].notna().sum()),
                'total_rows': len(df)
            }
            
            if 'storage_category' in df.columns:
                stats['storage_category_distribution'] = df['storage_category'].value_counts().to_dict()
            
            if 'storage_type' in df.columns:
                stats['storage_type_distribution'] = df['storage_type'].value_counts().to_dict()
        
        # Screen size statistics
        if 'screen_size_inches' in df.columns:
            screen_stats = df['screen_size_inches'].describe()
            stats['screen_stats'] = {
                'mean': float(screen_stats['mean']) if not pd.isna(screen_stats['mean']) else None,
                'median': float(df['screen_size_inches'].median()) if not df['screen_size_inches'].isna().all() else None,
                'min': float(screen_stats['min']) if not pd.isna(screen_stats['min']) else None,
                'max': float(screen_stats['max']) if not pd.isna(screen_stats['max']) else None,
                'std': float(screen_stats['std']) if not pd.isna(screen_stats['std']) else None,
                'total_found': int(df['screen_size_inches'].notna().sum()),
                'total_rows': len(df)
            }
            
            if 'screen_category' in df.columns:
                stats['screen_category_distribution'] = df['screen_category'].value_counts().to_dict()
        
        # Processor and GPU model distributions
        if 'processor_model' in df.columns:
            stats['processor_model_distribution'] = df['processor_model'].value_counts().head(10).to_dict()
        
        if 'gpu_model' in df.columns:
            stats['gpu_model_distribution'] = df['gpu_model'].value_counts().head(10).to_dict()
        
        if 'ram_type' in df.columns:
            stats['ram_type_distribution'] = df['ram_type'].value_counts().to_dict()
        
        return stats

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
        df_with_specs['cpu_benchmark_score'] = df_with_specs.apply(self._get_cpu_benchmark_from_columns, axis=1)
        
        # Add GPU benchmark scores by searching in title_y, features, and details columns
        df_with_specs['gpu_benchmark_score'] = df_with_specs.apply(self._get_gpu_benchmark_from_columns, axis=1)
        
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
        Extract and add laptop specifications from title_y, features, and details columns.
        
        Args:
            df (pd.DataFrame): Input dataframe with title_y, features, and details columns
            
        Returns:
            pd.DataFrame: Dataframe with added specification columns
        """
        logger.info("Extracting laptop specifications from columns...")
        
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
        
        # Extract specifications for each row
        df_specs['ram_gb'] = df_specs.apply(
            lambda row: self._extract_ram_from_text(combine_text_columns(row)), axis=1
        )
        
        df_specs['storage_gb'] = df_specs.apply(
            lambda row: self._extract_storage_from_text(combine_text_columns(row)), axis=1
        )
        
        df_specs['screen_size_inches'] = df_specs.apply(
            lambda row: self._extract_screen_size_from_text(combine_text_columns(row)), axis=1
        )
        
        df_specs['storage_type'] = df_specs.apply(
            lambda row: self._extract_storage_type_from_text(combine_text_columns(row)), axis=1
        )
        
        df_specs['ram_type'] = df_specs.apply(
            lambda row: self._extract_ram_type_from_text(combine_text_columns(row)), axis=1
        )
        
        df_specs['processor_model'] = df_specs.apply(
            lambda row: self._extract_processor_model_from_text(combine_text_columns(row)), axis=1
        )
        
        df_specs['gpu_model'] = df_specs.apply(
            lambda row: self._extract_gpu_model_from_text(combine_text_columns(row)), axis=1
        )
        
        # Add storage categories
        def get_storage_category(storage_gb):
            if pd.isna(storage_gb):
                return 'Unknown'
            elif storage_gb < 256:
                return 'Small (<256GB)'
            elif storage_gb < 512:
                return 'Medium (256-512GB)'
            elif storage_gb < 1000:
                return 'Large (512GB-1TB)'
            else:
                return 'Very Large (>1TB)'
        
        df_specs['storage_category'] = df_specs['storage_gb'].apply(get_storage_category)
        
        # Add storage display values (in appropriate units)
        def get_storage_display(storage_gb):
            if pd.isna(storage_gb):
                return None
            elif storage_gb >= 1024:
                return f"{storage_gb/1024:.1f}TB"
            else:
                return f"{storage_gb:.0f}GB"
        
        df_specs['storage_display'] = df_specs['storage_gb'].apply(get_storage_display)
        
        # Add RAM categories
        def get_ram_category(ram_gb):
            if pd.isna(ram_gb):
                return 'Unknown'
            elif ram_gb < 8:
                return 'Low (<8GB)'
            elif ram_gb < 16:
                return 'Medium (8-16GB)'
            elif ram_gb < 32:
                return 'High (16-32GB)'
            else:
                return 'Very High (>32GB)'
        
        df_specs['ram_category'] = df_specs['ram_gb'].apply(get_ram_category)
        
        # Add screen size categories
        def get_screen_category(screen_inches):
            if pd.isna(screen_inches):
                return 'Unknown'
            elif screen_inches < 13:
                return 'Small (<13")'
            elif screen_inches < 15:
                return 'Medium (13-15")'
            elif screen_inches < 17:
                return 'Large (15-17")'
            else:
                return 'Very Large (>17")'
        
        df_specs['screen_category'] = df_specs['screen_size_inches'].apply(get_screen_category)
        
        logger.info("Specifications extracted successfully")
        logger.info(f"RAM found: {df_specs['ram_gb'].notna().sum()}/{len(df_specs)} rows")
        logger.info(f"Storage found: {df_specs['storage_gb'].notna().sum()}/{len(df_specs)} rows")
        logger.info(f"Screen size found: {df_specs['screen_size_inches'].notna().sum()}/{len(df_specs)} rows")
        
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
    
    def test_kmp_performance(self, test_data: List[str]) -> Dict:
        """
        Test the performance of KMP algorithm vs traditional string matching.
        
        Args:
            test_data (List[str]): List of processor/GPU names to test
            
        Returns:
            Dict: Performance comparison results
        """
        import time
        
        logger.info("Testing KMP algorithm performance...")
        
        # Test traditional string matching
        start_time = time.time()
        traditional_matches = 0
        for item in test_data:
            for pattern in list(self.cpu_benchmarks.keys())[:50]:  # Test with first 50 patterns
                if pattern in item.lower() or item.lower() in pattern:
                    traditional_matches += 1
                    break
        traditional_time = time.time() - start_time
        
        # Test KMP algorithm
        start_time = time.time()
        kmp_matches = 0
        for item in test_data:
            patterns = list(self.cpu_benchmarks.keys())[:50]  # Test with first 50 patterns
            best_pattern, score = self.kmp.find_best_match(item, patterns)
            if best_pattern:
                kmp_matches += 1
        kmp_time = time.time() - start_time
        
        performance_results = {
            'traditional_matching': {
                'time': traditional_time,
                'matches': traditional_matches,
                'avg_time_per_item': traditional_time / len(test_data) if test_data else 0
            },
            'kmp_matching': {
                'time': kmp_time,
                'matches': kmp_matches,
                'avg_time_per_item': kmp_time / len(test_data) if test_data else 0
            },
            'speedup': traditional_time / kmp_time if kmp_time > 0 else float('inf'),
            'efficiency_gain': ((traditional_time - kmp_time) / traditional_time * 100) if traditional_time > 0 else 0
        }
        
        logger.info(f"KMP algorithm is {performance_results['speedup']:.2f}x faster than traditional matching")
        logger.info(f"Efficiency gain: {performance_results['efficiency_gain']:.2f}%")
        
        return performance_results


def main():
    """
    Main function to test the benchmark scraper with KMP algorithm.
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
    
    # Test KMP algorithm performance
    print("Testing KMP algorithm performance...")
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
    
    performance_results = scraper.test_kmp_performance(test_processors)
    print(f"\nKMP Performance Results:")
    print(f"Traditional matching time: {performance_results['traditional_matching']['time']:.4f}s")
    print(f"KMP matching time: {performance_results['kmp_matching']['time']:.4f}s")
    print(f"Speedup: {performance_results['speedup']:.2f}x")
    print(f"Efficiency gain: {performance_results['efficiency_gain']:.2f}%")
    
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
        print(f"Final Score: {debug_result['final_score']}")
        print(f"Match Type: {debug_result['benchmark_scores_found'][-1]['match_type'] if debug_result['benchmark_scores_found'] else 'No match'}")
        print(f"Model Names Tried: {debug_result['model_names_tried']}")
        if debug_result['matches_found']:
            print(f"Regex Matches: {debug_result['matches_found']}")
    
    # Add benchmark scores
    result = scraper.add_benchmark_scores(test_data)
    
    print("\nSample benchmark results:")
    print(result[['title_y', 'cpu_benchmark_score', 'gpu_benchmark_score', 
                 'total_benchmark_score', 'performance_tier', 'gaming_capability']])
    
    print("\nSample specification results:")
    spec_columns = ['ram_gb', 'storage_gb', 'screen_size_inches', 'storage_type', 
                   'ram_type', 'processor_model', 'gpu_model', 'storage_category', 
                   'ram_category', 'screen_category']
    available_spec_cols = [col for col in spec_columns if col in result.columns]
    if available_spec_cols:
        print(result[['title_y'] + available_spec_cols])
    
    # Get statistics
    stats = scraper.get_benchmark_statistics(result)
    print("\nBenchmark statistics:")
    print(json.dumps(stats, indent=2))
    
    # Get specification statistics
    spec_stats = scraper.get_specification_statistics(result)
    print("\nSpecification statistics:")
    print(json.dumps(spec_stats, indent=2))
    
    # Show all columns in the result
    print(f"\nTotal columns in result: {len(result.columns)}")
    print("All columns:")
    for i, col in enumerate(result.columns):
        print(f"  {i+1:2d}. {col}")


if __name__ == "__main__":
    main()
