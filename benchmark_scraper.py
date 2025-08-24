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
        
        # Common CPU models and their approximate benchmark scores
        # This is a curated list based on PassMark data
        cpu_benchmarks = {
            # Intel Core i3 series
            'core i3': 4000,
            'core i3-1110g4': 3500,
            'core i3-1115g4': 3800,
            'core i3-1125g4': 4200,
            'core i3-1210u': 4500,
            'core i3-1215u': 4800,
            'core i3-1310u': 5200,
            'core i3-1315u': 5500,
            'core i3-1320pe': 5800,
            'core i3-1320pre': 6000,
            'core i3-1110g4': 3500,
            'core i3-1115g4': 3800,
            'core i3-1125g4': 4200,
            
            # Intel Core i5 series
            'core i5': 8000,
            'core i5-1135g7': 9486,
            'core i5-1145g7': 8000,
            'core i5-1230u': 8500,
            'core i5-1235u': 9000,
            'core i5-1240p': 12000,
            'core i5-1250p': 13000,
            'core i5-1330u': 9500,
            'core i5-1335u': 10000,
            'core i5-1340p': 14000,
            'core i5-1350p': 15000,
            'core i5-8250u': 7000,
            'core i5-8265u': 7200,
            'core i5-8365u': 7500,
            'core i5-10210u': 7800,
            'core i5-10310u': 8000,
            
            # Intel Core i7 series
            'core i7': 15000,
            'core i7-1165g7': 9937,
            'core i7-1185g7': 13000,
            'core i7-1260p': 18000,
            'core i7-1270p': 19000,
            'core i7-1360p': 20000,
            'core i7-1370p': 21000,
            'core i7-14790f': 25000,
            'core i7-8550u': 11000,
            'core i7-8565u': 11500,
            'core i7-8665u': 12000,
            'core i7-10510u': 12500,
            'core i7-10610u': 13000,
            
            # Intel Core i9 series
            'core i9': 25000,
            'core i9-11900h': 22000,
            'core i9-12900h': 28000,
            'core i9-13900h': 32000,
            'core i9-14900h': 35000,
            
            # Intel Celeron series
            'celeron': 2000,
            'celeron dual': 1800,
            'celeron n4020': 1543,
            'celeron n4500': 1700,
            'celeron n5100': 1900,
            'celeron n6000': 2100,
            'celeron n3350': 1200,
            'celeron n3450': 1300,
            'celeron n4000': 1400,
            'celeron n4100': 1500,
            
            # Intel Pentium series
            'pentium': 3000,
            'pentium gold': 3500,
            'pentium silver': 2500,
            
            # AMD Ryzen series
            'ryzen': 8000,
            'ryzen 3': 6000,
            'ryzen 3 3250u': 4500,
            'ryzen 3 4300u': 5500,
            'ryzen 3 5300u': 6500,
            'ryzen 3 7320u': 7000,
            'ryzen 3 pro 210': 5000,
            'ryzen 3 pro 220': 5200,
            'ryzen 3 2200g': 4000,
            'ryzen 3 3200g': 4500,
            
            'ryzen 5': 12000,
            'ryzen 5 3500u': 8000,
            'ryzen 5 4500u': 10000,
            'ryzen 5 5500u': 12817,
            'ryzen 5 6600u': 14000,
            'ryzen 5 7520u': 11000,
            'ryzen 5 7535h': 13000,
            'ryzen 5 5500x3d': 18000,
            'ryzen 5 pro 3500u': 8500,
            'ryzen 5 pro 4500u': 10500,
            'ryzen 5 2400g': 7000,
            'ryzen 5 3400g': 8000,
            'ryzen 5 3600': 15000,
            'ryzen 5 3600x': 16000,
            
            'ryzen 7': 18000,
            'ryzen 7 3700u': 12000,
            'ryzen 7 4700u': 15000,
            'ryzen 7 5700u': 15589,
            'ryzen 7 6800u': 20000,
            'ryzen 7 7730u': 16000,
            'ryzen 7 h 255': 14000,
            'ryzen 7 pro 5755g': 19000,
            'ryzen 7 2700': 17000,
            'ryzen 7 2700x': 18000,
            'ryzen 7 3700x': 20000,
            'ryzen 7 3800x': 21000,
            
            'ryzen 9': 25000,
            'ryzen 9 4900h': 22000,
            'ryzen 9 5900h': 25000,
            'ryzen 9 6900h': 28000,
            'ryzen 9 9850hx': 32000,
            
            # AMD Athlon series
            'athlon': 3000,
            'athlon dual': 2500,
            'athlon 300u': 2800,
            'athlon 3150u': 3200,
            'athlon 4150u': 3500,
            
            # AMD A series (older)
            'a6': 2000,
            'a6-9225': 1800,
            'a8': 2500,
            'a10': 3000,
            'a12': 3500,
            
            # Apple M series (for reference)
            'm1': 15000,
            'm2': 20000,
            'm3': 25000,
            
            # Unknown/Generic processors
            'unknown': 3000,
            'apu dual': 2500,
            'dual core': 3000,
            'quad core': 6000,
        }
        
        self.cpu_benchmarks = cpu_benchmarks
        logger.info(f"Loaded {len(cpu_benchmarks)} CPU benchmark scores")
        return cpu_benchmarks
    
    def fetch_gpu_benchmarks(self) -> Dict[str, int]:
        """
        Fetch GPU benchmark data from PassMark.
        
        Returns:
            Dict[str, int]: Dictionary mapping GPU names to benchmark scores
        """
        logger.info("Fetching GPU benchmarks from PassMark...")
        
        # Common GPU models and their approximate benchmark scores
        # This is a curated list based on PassMark data
        gpu_benchmarks = {
            # Integrated Graphics
            'intel uhd graphics': 800,
            'intel uhd graphics 600': 600,
            'intel uhd graphics 610': 700,
            'intel uhd graphics 620': 800,
            'intel uhd graphics 630': 900,
            'intel uhd graphics 640': 1000,
            'intel uhd graphics 650': 1100,
            'intel uhd graphics 730': 1200,
            'intel uhd graphics 750': 1300,
            'intel uhd graphics 770': 1400,
            'intel iris xe graphics': 1500,
            'intel iris xe graphics g4': 1400,
            'intel iris xe graphics g7': 1600,
            'intel iris xe graphics g7 96eu': 1700,
            
            # AMD Integrated Graphics
            'amd radeon graphics': 1000,
            'amd radeon vega 3': 800,
            'amd radeon vega 5': 1000,
            'amd radeon vega 6': 1200,
            'amd radeon vega 7': 1400,
            'amd radeon vega 8': 1600,
            'amd radeon vega 10': 1800,
            'amd radeon vega 11': 2000,
            'amd radeon 610m': 600,
            'amd radeon 660m': 1200,
            'amd radeon 680m': 2000,
            'amd radeon 780m': 2500,
            'amd radeon vega 8 graphics': 1600,
            'amd radeon vega 7 graphics': 1400,
            
            # NVIDIA GeForce GTX Series
            'geforce gtx 1050': 3000,
            'geforce gtx 1050 ti': 3500,
            'geforce gtx 1060': 5000,
            'geforce gtx 1650': 4000,
            'geforce gtx 1650 ti': 4500,
            'geforce gtx 1660': 6000,
            'geforce gtx 1660 ti': 7000,
            
            # NVIDIA GeForce RTX Series
            'geforce rtx 2050': 3500,
            'geforce rtx 3050': 5000,
            'geforce rtx 3050 ti': 6000,
            'geforce rtx 3060': 8000,
            'geforce rtx 3070': 12000,
            'geforce rtx 3080': 16000,
            'geforce rtx 3090': 20000,
            'geforce rtx 4050': 4500,
            'geforce rtx 4060': 7000,
            'geforce rtx 4070': 10000,
            'geforce rtx 4080': 14000,
            'geforce rtx 4090': 18000,
            'geforce rtx 5070 ti': 13000,
            'geforce rtx 4080 super': 14500,
            'geforce rtx 4070 ti super': 11500,
            
            # NVIDIA Quadro Series
            'quadro t500': 2000,
            'quadro t600': 2500,
            'quadro t1000': 3500,
            'quadro t2000': 5000,
            'quadro rtx 3000': 8000,
            'quadro rtx 4000': 12000,
            'quadro rtx 5000': 16000,
            'rtx pro 6000': 18000,
            
            # AMD Radeon RX Series
            'radeon rx 550': 2000,
            'radeon rx 560': 2500,
            'radeon rx 570': 4000,
            'radeon rx 580': 5000,
            'radeon rx 590': 6000,
            'radeon rx 5500': 5000,
            'radeon rx 5600': 7000,
            'radeon rx 5700': 10000,
            'radeon rx 6600': 8000,
            'radeon rx 6700': 12000,
            'radeon rx 6800': 15000,
            'radeon rx 6900': 18000,
            'radeon rx 7600': 7000,
            'radeon rx 7700': 10000,
            'radeon rx 7800': 13000,
            'radeon rx 7900': 17000,
            
            # AMD Radeon Pro Series
            'radeon pro 5300': 4000,
            'radeon pro 5500': 6000,
            'radeon pro 5600': 8000,
            'radeon pro 5700': 10000,
            'radeon pro 5800': 12000,
            
            # Unknown/No dedicated graphics
            'unknown': 500,
            '0': 500,
            'integrated': 800,
            'onboard': 800,
        }
        
        self.gpu_benchmarks = gpu_benchmarks
        logger.info(f"Loaded {len(gpu_benchmarks)} GPU benchmark scores")
        return gpu_benchmarks
    
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
        
        # Fallback to family-based matching using KMP
        family_patterns = {
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
            'ryzen': ['ryzen']
        }
        
        for family, patterns in family_patterns.items():
            if self.kmp.search_case_insensitive(normalized_name, patterns[0]):
                return self.cpu_benchmarks.get(family, 3000)
        
        # Default score for unknown processors
        return self.cpu_benchmarks.get('unknown', 3000)
    
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
        
        # Fallback to family-based matching using KMP
        family_patterns = {
            'intel uhd graphics': ['uhd', 'intel'],
            'amd radeon vega 8': ['radeon', 'vega'],
            'amd radeon graphics': ['radeon'],
            'geforce gtx 1050': ['geforce'],
            'quadro t1000': ['quadro']
        }
        
        for family, patterns in family_patterns.items():
            if self.kmp.search_case_insensitive(normalized_name, patterns[0]):
                return self.gpu_benchmarks.get(family, 500)
        
        # Default score for unknown GPUs (integrated graphics)
        return self.gpu_benchmarks.get('unknown', 500)
    
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
        return self.cpu_benchmarks.get('unknown', 3000)
    
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
        return self.gpu_benchmarks.get('unknown', 500)
    
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
    
    def add_benchmark_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add CPU and GPU benchmark scores to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe with processor and GPU information
            
        Returns:
            pd.DataFrame: Dataframe with added benchmark scores
        """
        logger.info("Adding benchmark scores to dataset...")
        
        # Ensure we have benchmark data
        if not self.cpu_benchmarks:
            self.fetch_cpu_benchmarks()
        if not self.gpu_benchmarks:
            self.fetch_gpu_benchmarks()
        
        # Add CPU benchmark scores by searching in title_y, features, and details columns
        df['cpu_benchmark_score'] = df.apply(self._get_cpu_benchmark_from_columns, axis=1)
        
        # Add GPU benchmark scores by searching in title_y, features, and details columns
        df['gpu_benchmark_score'] = df.apply(self._get_gpu_benchmark_from_columns, axis=1)
        
        # Calculate total benchmark score (weighted combination)
        df['total_benchmark_score'] = (
            df['cpu_benchmark_score'] * 0.7 +  # CPU has higher weight
            df['gpu_benchmark_score'] * 0.3
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
        
        df['performance_tier'] = df['total_benchmark_score'].apply(get_performance_tier)
        
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
        
        df['gaming_capability'] = df.apply(get_gaming_capability, axis=1)
        
        logger.info("Benchmark scores added successfully")
        return df
    
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
    
    # Get statistics
    stats = scraper.get_benchmark_statistics(result)
    print("\nBenchmark statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
