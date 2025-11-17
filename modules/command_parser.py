#!/usr/bin/env python3
"""
Command Parser Module - Interprets user commands and executes system actions.
This bridges the gap between what the user says and what PHOENIX actually does.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple


class CommandParser:
    """
    Parses natural language commands and determines what actions to take.
    """

    def __init__(self):
        self.logger = logging.getLogger("PHOENIX.CommandParser")

        # Define command patterns
        self.patterns = {
            'list_files': [
                r'list files?\s*(?:in\s+)?(.+)?',
                r'show files?\s*(?:in\s+)?(.+)?',
                r'ls\s*(.+)?'
            ],
            'gpu_info': [
                r'check (?:my\s+)?(?:gpu|GPU)',
                r'show (?:my\s+)?(?:gpu|GPU)(?:\s+info)?',
                r'(?:what is|what\'s|whats)\s+my\s+gpu',
                r'nvidia-smi',
                r'gpu status',
                r'graphics card'
            ],
            'cpu_info': [
                r'check (?:my\s+)?(?:cpu|CPU)',
                r'show (?:my\s+)?(?:cpu|CPU)(?:\s+info)?',
                r'(?:what is|what\'s|whats)\s+my\s+cpu'
            ],
            'system_info': [
                r'system (?:info|information|status)',
                r'show system (?:info|information|status)',
                r'check (?:my\s+)?system'
            ],
            'run_command': [
                r'run (?:the\s+)?command[:\s]+(.+)',
                r'execute (?:the\s+)?command[:\s]+(.+)',
                r'exec[:\s]+(.+)',
                r'^\$\s*(.+)'  # Direct command with $
            ],
            'process_list': [
                r'(?:show|list)\s+processes?',
                r'what(?:\'s| is) running',
                r'ps aux',
                r'top'
            ],
            'memory_info': [
                r'check (?:my\s+)?(?:memory|ram)',
                r'show (?:my\s+)?(?:memory|ram)(?:\s+usage)?',
                r'how much (?:memory|ram)'
            ],
            'disk_info': [
                r'check (?:my\s+)?(?:disk|storage)(?:\s+space)?',
                r'show (?:my\s+)?(?:disk|storage)(?:\s+usage)?',
                r'(?:my\s+)?storage',
                r'df',
                r'disk space'
            ]
        }

    def parse(self, command: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Parse a user command and determine the action to take.

        Args:
            command: User's natural language command

        Returns:
            Tuple of (action_type, parameters)
        """
        command_lower = command.lower().strip()

        # Check each pattern type
        for action_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    self.logger.info(f"Matched action: {action_type}")

                    # Extract parameters based on action type
                    params = {}

                    if action_type == 'list_files':
                        params['directory'] = match.group(1) if match.group(1) else '.'

                    elif action_type == 'run_command':
                        params['command'] = match.group(1)

                    return action_type, params

        # No pattern matched
        return None, {}

    def extract_system_command(self, text: str) -> Optional[str]:
        """
        Extract a system command from natural language.

        Args:
            text: Text that might contain a command

        Returns:
            Extracted command or None
        """
        # Look for commands in backticks or after keywords
        patterns = [
            r'`([^`]+)`',  # Commands in backticks
            r'command:\s*(.+)',  # After "command:"
            r'run:\s*(.+)',  # After "run:"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None