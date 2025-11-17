#!/usr/bin/env python3
"""
Self-Awareness Module - Enables PHOENIX to understand itself and its environment.
Handles complex, multi-part requests and provides introspection capabilities.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import importlib
import sys


class SelfAwareness:
    """
    Gives Phoenix the ability to understand itself, its architecture, and handle
    complex multi-part requests intelligently.
    """

    def __init__(self, system_control=None, memory=None):
        """
        Initialize Self-Awareness module.

        Args:
            system_control: System control module reference
            memory: Memory module reference
        """
        self.system_control = system_control
        self.memory = memory
        self.logger = logging.getLogger("PHOENIX.SelfAwareness")

        # Phoenix's self-knowledge
        self.phoenix_home = Path(__file__).parent.parent  # PHOENIX directory
        self.architecture = self._discover_architecture()
        self.capabilities = self._discover_capabilities()

        self.logger.info("Self-Awareness module initialized")

    def _discover_architecture(self) -> Dict[str, Any]:
        """
        Discover PHOENIX's architecture.

        Returns:
            Architecture information
        """
        architecture = {
            'location': str(self.phoenix_home),
            'modules': {},
            'config': {},
            'model': {}
        }

        # Find all modules
        modules_dir = self.phoenix_home / 'modules'
        if modules_dir.exists():
            architecture['modules'] = {
                f.stem: str(f) for f in modules_dir.glob('*.py')
                if not f.name.startswith('__')
            }

        # Find configuration
        config_file = self.phoenix_home / 'config' / 'phoenix_config.yaml'
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    architecture['config'] = {
                        'model': config.get('llm', {}).get('model', 'unknown'),
                        'provider': config.get('llm', {}).get('provider', 'unknown'),
                        'version': config.get('system', {}).get('version', 'unknown')
                    }
                    architecture['model'] = {
                        'name': config.get('llm', {}).get('model', 'unknown'),
                        'temperature': config.get('llm', {}).get('temperature', 0.7),
                        'max_tokens': config.get('llm', {}).get('max_tokens', 4096)
                    }
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")

        # Get running model info
        if self.system_control:
            success, stdout, _ = self.system_control.run_command("ollama list | grep qwen")
            if success and stdout:
                architecture['model']['active'] = stdout.strip().split()[0]

        return architecture

    def _discover_capabilities(self) -> List[str]:
        """
        Discover what PHOENIX can do.

        Returns:
            List of capabilities
        """
        capabilities = []

        # Check which modules are loaded
        for module_name in ['system_control', 'memory_manager', 'advanced_learning',
                           'automation', 'intelligent_executor', 'system_discovery']:
            module_path = self.phoenix_home / 'modules' / f'{module_name}.py'
            if module_path.exists():
                capabilities.append(module_name.replace('_', ' ').title())

        return capabilities

    def understand_self_request(self, request: str) -> Dict[str, Any]:
        """
        Understand a request about PHOENIX itself.

        Args:
            request: User's request

        Returns:
            Understanding of what's being asked
        """
        request_lower = request.lower()

        understanding = {
            'wants_location': any(word in request_lower for word in ['where', 'location', 'folder', 'directory', 'path']),
            'wants_architecture': any(word in request_lower for word in ['architecture', 'structure', 'modules', 'components']),
            'wants_model': any(word in request_lower for word in ['model', 'llm', 'ai', 'brain']),
            'wants_capabilities': any(word in request_lower for word in ['can you', 'capabilities', 'able to', 'features']),
            'wants_exploration': any(word in request_lower for word in ['check', 'explore', 'find', 'discover', 'scan']),
            'wants_understanding': any(word in request_lower for word in ['understand', 'know', 'tell me', 'explain'])
        }

        # Determine primary intent
        understanding['primary_intent'] = 'general'
        for intent, wanted in understanding.items():
            if wanted and intent.startswith('wants_'):
                understanding['primary_intent'] = intent.replace('wants_', '')
                break

        return understanding

    def execute_self_exploration(self) -> str:
        """
        Execute a full self-exploration and return a comprehensive report.

        Returns:
            Self-exploration report
        """
        report = "🔍 **PHOENIX Self-Exploration Report**\n\n"

        # 1. Location
        report += "**📍 My Location:**\n"
        report += f"I'm installed at: `{self.phoenix_home}`\n\n"

        # Check directory structure
        if self.system_control:
            success, stdout, _ = self.system_control.run_command(f"tree {self.phoenix_home} -L 2 -d")
            if success and stdout:
                report += "**📁 My Structure:**\n```\n" + stdout[:500] + "\n```\n\n"

        # 2. Architecture
        report += "**🏗️ My Architecture:**\n"
        report += f"- **Modules:** {len(self.architecture['modules'])} modules found\n"
        for module_name in sorted(self.architecture['modules'].keys())[:10]:
            report += f"  • {module_name}\n"
        report += "\n"

        # 3. Model Information
        report += "**🧠 My AI Model:**\n"
        report += f"- **Configured Model:** {self.architecture['model']['name']}\n"
        report += f"- **Provider:** {self.architecture['config']['provider']}\n"
        report += f"- **Temperature:** {self.architecture['model']['temperature']}\n"
        report += f"- **Max Tokens:** {self.architecture['model']['max_tokens']}\n"

        # Check actual running model
        if self.system_control:
            success, stdout, _ = self.system_control.run_command("ollama list")
            if success and stdout:
                report += f"\n**Available Models:**\n```\n{stdout[:500]}\n```\n"

        # 4. Capabilities
        report += "**💪 My Capabilities:**\n"
        for capability in self.capabilities:
            report += f"- {capability}\n"

        # 5. Current Process
        if self.system_control:
            success, stdout, _ = self.system_control.run_command("ps aux | grep phoenix | grep -v grep | head -3")
            if success and stdout:
                report += f"\n**⚙️ My Process:**\n```\n{stdout}\n```\n"

        # 6. Memory Status
        if self.memory:
            stats = self.memory.get_stats()
            report += f"\n**🧠 My Memory:**\n"
            report += f"- Conversations: {stats['total_conversations']}\n"
            report += f"- Facts Learned: {stats['total_facts']}\n"
            report += f"- Knowledge Categories: {stats['knowledge_categories']}\n"

        # Store this self-knowledge
        if self.memory:
            self.memory.learn_fact(
                f"PHOENIX location: {self.phoenix_home}",
                'self_knowledge'
            )
            self.memory.learn_fact(
                f"PHOENIX model: {self.architecture['model']['name']}",
                'self_knowledge'
            )

        return report

    def handle_complex_request(self, request: str) -> List[Dict[str, Any]]:
        """
        Break down a complex, multi-part request into actionable steps.

        Args:
            request: Complex user request

        Returns:
            List of actions to take
        """
        actions = []

        # Parse the request into parts
        parts = []

        # Common conjunctions and separators
        separators = [' and ', ' then ', ' also ', '. ', ', ']

        current_part = request
        for separator in separators:
            if separator in current_part:
                splits = current_part.split(separator)
                parts.extend(splits[:-1])
                current_part = splits[-1]

        parts.append(current_part)

        # Analyze each part
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Determine action type for this part
            action = {
                'original': part,
                'type': 'unknown',
                'params': {}
            }

            part_lower = part.lower()

            if any(word in part_lower for word in ['check', 'find', 'locate', 'where']):
                if 'file' in part_lower or 'folder' in part_lower:
                    action['type'] = 'explore_files'
                elif 'location' in part_lower:
                    action['type'] = 'show_location'

            elif any(word in part_lower for word in ['understand', 'architecture', 'structure']):
                action['type'] = 'explain_architecture'

            elif any(word in part_lower for word in ['model', 'llm', 'running on']):
                action['type'] = 'show_model_info'

            elif any(word in part_lower for word in ['tell me', 'explain', 'what']):
                action['type'] = 'explain'

            actions.append(action)

        return actions

    def explain_error(self, command: str, error: str) -> str:
        """
        Analyze and explain why a command failed.

        Args:
            command: The command that failed
            error: The error message

        Returns:
            Explanation and possible solutions
        """
        explanation = "**❌ Command Failed - Analysis:**\n\n"

        # Analyze the error
        if 'not found' in error.lower():
            explanation += "**Problem:** The command or program doesn't exist.\n"

            # Extract what wasn't found
            if ':' in error:
                missing = error.split(':')[-1].strip()
                explanation += f"**Missing:** `{missing}`\n"

                # Suggest alternatives
                if missing == 'on.':
                    explanation += "\n**What happened:** I incorrectly parsed your request and tried to run 'on.' as a command.\n"
                    explanation += "**Real issue:** I need better understanding of complex requests.\n"

        elif 'permission denied' in error.lower():
            explanation += "**Problem:** Insufficient permissions.\n"
            explanation += "**Solution:** May need to use `sudo` or change file permissions.\n"

        elif 'no such file' in error.lower():
            explanation += "**Problem:** The file or directory doesn't exist.\n"
            explanation += "**Solution:** Check the path and ensure it exists.\n"

        else:
            explanation += f"**Raw error:** {error}\n"

        # Suggest what the user might have wanted
        explanation += "\n**What you probably wanted:**\n"

        if 'model' in command.lower() or 'architecture' in command.lower():
            explanation += "- Information about my AI model and architecture\n"
            explanation += "- My installation location and structure\n"
            explanation += "\n**Try instead:** Just ask me directly without 'run command'\n"

        return explanation

    def get_self_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about PHOENIX.

        Returns:
            Self information dictionary
        """
        return {
            'name': 'PHOENIX',
            'full_name': 'Personal Hybrid Operating Environment Network Intelligence eXtension',
            'version': self.architecture['config'].get('version', 'unknown'),
            'location': str(self.phoenix_home),
            'model': self.architecture['model'],
            'capabilities': self.capabilities,
            'modules_count': len(self.architecture['modules']),
            'status': 'operational'
        }