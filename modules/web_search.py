#!/usr/bin/env python3
"""
Web Search Module - Provides real web search capability to PHOENIX.
Uses DuckDuckGo for privacy-focused searches.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

class WebSearchModule:
    """
    Provides web search and information retrieval capabilities.
    """

    def __init__(self, memory_manager=None):
        """
        Initialize the Web Search Module.

        Args:
            memory_manager: Memory system for caching results
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.WebSearch")

        # DuckDuckGo search URL
        self.search_url = "https://html.duckduckgo.com/html/"

        # Headers to appear as a regular browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        self.logger.info("Web Search Module initialized")

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a web search and return results.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results with title, url, and snippet
        """
        try:
            self.logger.info(f"Searching for: {query}")

            # Search using DuckDuckGo (requires POST)
            data = {'q': query}
            response = requests.post(self.search_url, data=data, headers=self.headers)

            if response.status_code != 200:
                self.logger.error(f"Search failed with status {response.status_code}")
                # Fallback: try a simple Google search URL
                return self._fallback_search(query)

            # Parse results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Find result links - DuckDuckGo structure
            for result in soup.find_all('div', class_='result')[:num_results]:
                # Get title and URL
                title_elem = result.find('a', class_='result__a')
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                url = title_elem.get('href', '')

                # Get snippet
                snippet_elem = result.find('a', class_='result__snippet')
                if not snippet_elem:
                    snippet_elem = result.find('span', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''

                if title and url:
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'timestamp': datetime.now().isoformat()
                    })

            # Cache results in memory
            if self.memory_manager and results:
                self.memory_manager.learn_fact(
                    f"Web search for '{query}' returned {len(results)} results",
                    category='web_searches'
                )

            self.logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []

    def _fallback_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Fallback search method when primary search fails.

        Args:
            query: Search query

        Returns:
            List of manually constructed search results
        """
        self.logger.info("Using fallback search method")

        # For now, return informative message about search
        # In a real implementation, this could use an alternative search API
        results = [
            {
                'title': f'Search for: {query}',
                'url': f'https://www.google.com/search?q={query.replace(" ", "+")}',
                'snippet': f'Direct Google search link for "{query}". Click to search in browser.',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': f'DuckDuckGo Search: {query}',
                'url': f'https://duckduckgo.com/?q={query.replace(" ", "+")}',
                'snippet': f'Direct DuckDuckGo search link for "{query}".',
                'timestamp': datetime.now().isoformat()
            }
        ]

        return results

    def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract text content from a webpage.

        Args:
            url: URL to fetch

        Returns:
            Extracted text content or None if failed
        """
        try:
            self.logger.info(f"Fetching content from: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return None

            # Parse and extract text
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get text
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text[:5000]  # Limit to first 5000 chars

        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None

    def search_and_extract(self, query: str, extract_content: bool = False) -> Dict[str, Any]:
        """
        Search and optionally extract content from top result.

        Args:
            query: Search query
            extract_content: Whether to fetch content from top result

        Returns:
            Search results and extracted content
        """
        results = self.search(query)

        response = {
            'query': query,
            'results': results,
            'extracted_content': None,
            'success': len(results) > 0
        }

        if extract_content and results:
            # Get content from top result
            top_url = results[0]['url']
            content = self.fetch_page_content(top_url)
            if content:
                response['extracted_content'] = {
                    'url': top_url,
                    'title': results[0]['title'],
                    'content': content
                }

        return response

    def verify_information(self, claim: str) -> Dict[str, Any]:
        """
        Verify a claim by searching for information.

        Args:
            claim: Information to verify

        Returns:
            Verification results
        """
        # Search for the claim
        results = self.search(claim, num_results=3)

        verification = {
            'claim': claim,
            'sources_found': len(results),
            'sources': results,
            'confidence': 0.0
        }

        if results:
            # Simple confidence based on number of sources
            verification['confidence'] = min(len(results) / 3.0, 1.0)

        return verification

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Return actual capabilities of this module.

        Returns:
            Dictionary of capabilities
        """
        return {
            'web_search': True,
            'fetch_content': True,
            'verify_claims': True,
            'real_time_data': True,
            'api_access': False,  # No direct API access yet
            'cached_results': self.memory_manager is not None
        }