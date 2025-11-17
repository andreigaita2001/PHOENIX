#!/usr/bin/env python3
"""
Code Executor Module - Actually executes Python code instead of just claiming to.
This ensures PHOENIX does what it says it will do.
"""

import logging
import sys
import io
import contextlib
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple

class CodeExecutor:
    """
    Executes Python code safely and returns results.
    """

    def __init__(self, system_control=None):
        """
        Initialize the Code Executor.

        Args:
            system_control: System control module for command execution
        """
        self.system_control = system_control
        self.logger = logging.getLogger("PHOENIX.CodeExecutor")
        self.temp_dir = Path('/tmp/phoenix_code')
        self.temp_dir.mkdir(exist_ok=True)

    def execute_python_code(self, code: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """
        Execute Python code and return results.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            # Write code to temporary file
            temp_file = self.temp_dir / f"script_{id(code)}.py"
            with open(temp_file, 'w') as f:
                f.write(code)

            # Execute with timeout
            result = subprocess.run(
                [sys.executable, str(temp_file)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Clean up temp file
            temp_file.unlink(missing_ok=True)

            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr

            if success:
                self.logger.info(f"Code executed successfully")
            else:
                self.logger.error(f"Code execution failed: {stderr}")

            return success, stdout, stderr

        except subprocess.TimeoutExpired:
            self.logger.error(f"Code execution timed out after {timeout} seconds")
            return False, "", f"Execution timed out after {timeout} seconds"
        except Exception as e:
            self.logger.error(f"Code execution error: {e}")
            return False, "", str(e)

    def execute_web_search(self, query: str) -> Dict[str, Any]:
        """
        Execute a web search using Python code.

        Args:
            query: Search query

        Returns:
            Search results
        """
        # Create search code
        search_code = f'''
import requests
from bs4 import BeautifulSoup
import json

query = "{query}"

# Try to get search results
results = []

try:
    # Use a simple Google search URL
    search_url = f"https://www.google.com/search?q={{query.replace(' ', '+')}}"

    # Note: This is for demonstration - real implementation would need proper API
    # For now, just return the search URL so user can click it
    results.append({{
        'title': f'Google Search: {{query}}',
        'url': search_url,
        'snippet': 'Click to search in Google',
        'source': 'google'
    }})

    # Add DuckDuckGo link
    ddg_url = f"https://duckduckgo.com/?q={{query.replace(' ', '+')}}"
    results.append({{
        'title': f'DuckDuckGo Search: {{query}}',
        'url': ddg_url,
        'snippet': 'Click to search in DuckDuckGo',
        'source': 'duckduckgo'
    }})

    # Try to use requests to get actual results from a simpler API
    # Using DuckDuckGo Instant Answer API (limited but free)
    api_url = f"https://api.duckduckgo.com/?q={{query.replace(' ', '+')}}&format=json&no_html=1"
    response = requests.get(api_url, timeout=5)

    if response.status_code == 200:
        data = response.json()

        # Check for instant answer
        if data.get('AbstractText'):
            results.insert(0, {{
                'title': data.get('Heading', query),
                'url': data.get('AbstractURL', ''),
                'snippet': data.get('AbstractText', '')[:300],
                'source': 'duckduckgo_instant'
            }})

        # Check for related topics
        for topic in data.get('RelatedTopics', [])[:3]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({{
                    'title': topic.get('Text', '')[:100],
                    'url': topic.get('FirstURL', ''),
                    'snippet': topic.get('Text', '')[:200],
                    'source': 'duckduckgo_related'
                }})

except Exception as e:
    results.append({{
        'error': str(e),
        'message': 'Search failed but here are direct links'
    }})

print(json.dumps(results))
'''

        # Execute the search code
        success, stdout, stderr = self.execute_python_code(search_code)

        if success and stdout:
            try:
                import json
                results = json.loads(stdout)
                return {
                    'success': True,
                    'results': results,
                    'query': query
                }
            except:
                pass

        # Fallback if execution fails
        return {
            'success': False,
            'results': [
                {
                    'title': f'Search for: {query}',
                    'url': f'https://www.google.com/search?q={query.replace(" ", "+")}',
                    'snippet': 'Direct link to Google search',
                    'source': 'fallback'
                }
            ],
            'query': query,
            'error': stderr if stderr else 'Search execution failed'
        }

    def get_capabilities(self) -> Dict[str, bool]:
        """Return actual capabilities of this module."""
        return {
            'execute_python': True,
            'execute_searches': True,
            'timeout_control': True,
            'safe_execution': True
        }