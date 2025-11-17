#!/usr/bin/env python3
"""
Web Browser Module - Full autonomous web browsing capabilities.
This module allows PHOENIX to navigate the web, read content, and extract information.
"""

import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
import json
import re
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import time

class WebBrowser:
    """
    Full web browsing capabilities for PHOENIX.
    Can navigate, search, extract content, and browse autonomously.
    """

    def __init__(self, memory_manager=None):
        """
        Initialize the Web Browser.

        Args:
            memory_manager: Memory manager for storing browsing history
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.WebBrowser")

        # Current browsing session
        self.current_url = None
        self.current_content = None
        self.current_links = []
        self.browsing_history = []

        # Headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Selenium driver (lazy loaded)
        self.driver = None

        self.logger.info("Web Browser initialized")

    def navigate(self, url: str, use_selenium: bool = False) -> Dict[str, Any]:
        """
        Navigate to a URL and extract content.

        Args:
            url: URL to navigate to
            use_selenium: Use Selenium for JavaScript-heavy sites

        Returns:
            Page content and metadata
        """
        try:
            self.current_url = url
            self.browsing_history.append(url)

            if use_selenium:
                content = self._fetch_with_selenium(url)
            else:
                content = self._fetch_with_requests(url)

            if content['success']:
                self.current_content = content['text']
                self.current_links = content['links']

                # Store in memory if available
                if self.memory_manager:
                    self.memory_manager.learn_fact(
                        f"Visited: {content['title']}",
                        category='web_browsing'
                    )

                self.logger.info(f"Successfully navigated to {url}")
            else:
                self.logger.error(f"Failed to navigate to {url}: {content.get('error')}")

            return content

        except Exception as e:
            self.logger.error(f"Navigation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _fetch_with_requests(self, url: str) -> Dict[str, Any]:
        """Fetch page using requests library."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                links.append({
                    'text': link.get_text(strip=True),
                    'url': absolute_url
                })

            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else urlparse(url).netloc

            # Extract main content if possible
            main_content = None
            for selector in ['main', 'article', '[role="main"]', '#content', '.content']:
                main = soup.select_one(selector)
                if main:
                    main_content = main.get_text(separator=' ', strip=True)
                    break

            return {
                'success': True,
                'url': url,
                'title': title_text,
                'text': main_content or text[:5000],  # Limit text length
                'full_text': text,
                'links': links[:50],  # Limit number of links
                'html': response.text
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    def _fetch_with_selenium(self, url: str) -> Dict[str, Any]:
        """Fetch page using Selenium for JavaScript rendering."""
        try:
            if not self.driver:
                self._init_selenium()

            self.driver.get(url)

            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Give JavaScript time to execute
            time.sleep(2)

            # Get page source
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # Extract content similar to requests method
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Extract links
            links = []
            link_elements = self.driver.find_elements(By.TAG_NAME, "a")
            for elem in link_elements[:50]:
                try:
                    href = elem.get_attribute('href')
                    text_content = elem.text
                    if href:
                        links.append({
                            'text': text_content,
                            'url': href
                        })
                except:
                    continue

            title = self.driver.title

            return {
                'success': True,
                'url': url,
                'title': title,
                'text': text[:5000],
                'full_text': text,
                'links': links,
                'html': html
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    def _init_selenium(self):
        """Initialize Selenium WebDriver."""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = webdriver.Firefox(options=options)
            self.logger.info("Selenium WebDriver initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Selenium: {e}")
            self.logger.info("Install Firefox and geckodriver for full JS support")

    def search(self, query: str, engine: str = "duckduckgo") -> Dict[str, Any]:
        """
        Search the web and return actual results with content.

        Args:
            query: Search query
            engine: Search engine to use (google, duckduckgo, bing)

        Returns:
            Search results with content
        """
        search_urls = {
            'google': f'https://www.google.com/search?q={query.replace(" ", "+")}',
            'duckduckgo': f'https://html.duckduckgo.com/html/?q={query.replace(" ", "+")}',
            'bing': f'https://www.bing.com/search?q={query.replace(" ", "+")}'
        }

        url = search_urls.get(engine, search_urls['duckduckgo'])

        self.logger.info(f"Searching for: {query} using {engine}")

        # Get search results page
        search_page = self._fetch_with_requests(url)

        if not search_page['success']:
            return search_page

        # Parse search results
        soup = BeautifulSoup(search_page['html'], 'html.parser')
        results = []

        if engine == 'duckduckgo':
            # DuckDuckGo result parsing
            for result in soup.select('.result'):
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                url_elem = result.select_one('.result__url')

                if title_elem and url_elem:
                    result_data = {
                        'title': title_elem.get_text(strip=True),
                        'url': url_elem.get('href', ''),
                        'snippet': snippet_elem.get_text(strip=True) if snippet_elem else '',
                        'source': 'duckduckgo'
                    }
                    results.append(result_data)

        elif engine == 'google':
            # Google result parsing
            for g in soup.select('.g'):
                title_elem = g.select_one('h3')
                snippet_elem = g.select_one('.VwiC3b')
                link_elem = g.select_one('a')

                if title_elem and link_elem:
                    result_data = {
                        'title': title_elem.get_text(strip=True),
                        'url': link_elem.get('href', ''),
                        'snippet': snippet_elem.get_text(strip=True) if snippet_elem else '',
                        'source': 'google'
                    }
                    results.append(result_data)

        # Fetch content from top results
        for i, result in enumerate(results[:3]):  # Get content from top 3 results
            try:
                content = self._fetch_with_requests(result['url'])
                if content['success']:
                    result['content'] = content['text'][:1000]  # First 1000 chars
                    result['full_content_available'] = True
            except:
                result['content'] = result['snippet']
                result['full_content_available'] = False

        return {
            'success': True,
            'query': query,
            'results': results,
            'total_results': len(results)
        }

    def extract_data(self, url: str, selectors: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract specific data from a webpage using CSS selectors.

        Args:
            url: URL to extract from
            selectors: Dictionary of field_name: css_selector

        Returns:
            Extracted data
        """
        content = self.navigate(url)

        if not content['success']:
            return content

        soup = BeautifulSoup(content['html'], 'html.parser')
        extracted = {}

        for field, selector in selectors.items():
            try:
                element = soup.select_one(selector)
                if element:
                    extracted[field] = element.get_text(strip=True)
                else:
                    extracted[field] = None
            except Exception as e:
                extracted[field] = f"Error: {e}"

        return {
            'success': True,
            'url': url,
            'data': extracted
        }

    def follow_link(self, link_text: str = None, link_index: int = None) -> Dict[str, Any]:
        """
        Follow a link from the current page.

        Args:
            link_text: Text of the link to follow
            link_index: Index of the link in current_links

        Returns:
            New page content
        """
        if not self.current_links:
            return {
                'success': False,
                'error': 'No links available on current page'
            }

        target_url = None

        if link_text:
            # Find link by text
            for link in self.current_links:
                if link_text.lower() in link['text'].lower():
                    target_url = link['url']
                    break

        elif link_index is not None:
            # Get link by index
            if 0 <= link_index < len(self.current_links):
                target_url = self.current_links[link_index]['url']

        if target_url:
            return self.navigate(target_url)
        else:
            return {
                'success': False,
                'error': f'Link not found: {link_text or link_index}'
            }

    def summarize_page(self) -> str:
        """
        Summarize the current page content.

        Returns:
            Summary of the page
        """
        if not self.current_content:
            return "No page loaded"

        # Basic summarization
        lines = self.current_content.split('.')[:5]  # First 5 sentences
        summary = '. '.join(lines) + '.'

        return f"Page Summary: {summary[:500]}"

    def get_page_info(self) -> Dict[str, Any]:
        """
        Get information about the current page.

        Returns:
            Current page info
        """
        return {
            'current_url': self.current_url,
            'has_content': bool(self.current_content),
            'num_links': len(self.current_links),
            'content_length': len(self.current_content) if self.current_content else 0,
            'history_length': len(self.browsing_history)
        }

    def close(self):
        """Close the browser and clean up resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.logger.info("Selenium WebDriver closed")

    def get_capabilities(self) -> Dict[str, bool]:
        """Return browser capabilities."""
        return {
            'navigate_urls': True,
            'search_web': True,
            'extract_content': True,
            'follow_links': True,
            'javascript_rendering': True,
            'data_extraction': True,
            'form_submission': False,  # TODO: Add form handling
            'download_files': False,   # TODO: Add download capability
            'handle_cookies': False    # TODO: Add cookie management
        }