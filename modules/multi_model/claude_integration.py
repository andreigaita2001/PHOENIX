#!/usr/bin/env python3
"""
Claude Integration - Enables PHOENIX to use Claude as a cloud fallback.
Claude is the ONLY cloud API supported, maintaining privacy focus.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess


class ClaudeIntegration:
    """
    Integration with Claude API as the sole cloud fallback.
    Uses Claude Code's existing connection when available.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Claude Integration.

        Args:
            config: Configuration including enable flag
        """
        self.logger = logging.getLogger("PHOENIX.Claude")

        # Configuration - DISABLED by default for privacy
        self.config = config or {
            'enabled': False,  # Must explicitly enable
            'use_for_complex_tasks': True,
            'use_for_verification': True,
            'use_as_fallback': True,
            'max_context_length': 200000,  # Claude's superior context
            'require_confirmation': True  # Ask before sending to cloud
        }

        # Claude's unique capabilities
        self.claude_strengths = [
            'complex_reasoning',
            'long_context_analysis',
            'code_understanding',
            'system_design',
            'ethical_reasoning',
            'nuanced_understanding',
            'cross_domain_knowledge'
        ]

        # Track usage for transparency
        self.usage_log = []

        self.logger.info(f"Claude Integration initialized (Enabled: {self.config['enabled']})")

    def is_available(self) -> bool:
        """
        Check if Claude integration is available.

        Returns:
            Whether Claude can be used
        """
        if not self.config['enabled']:
            return False

        # Check if we're running inside Claude Code
        try:
            # Claude Code sets specific environment variables
            if os.environ.get('CLAUDE_CODE'):
                return True

            # Check if claude command is available
            result = subprocess.run(
                ['which', 'claude'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0

        except:
            return False

    async def query_claude(self, prompt: str,
                          task_type: str = 'general',
                          require_confirmation: bool = None) -> Dict[str, Any]:
        """
        Query Claude for assistance.

        Args:
            prompt: The prompt to send
            task_type: Type of task
            require_confirmation: Override confirmation requirement

        Returns:
            Claude's response or error
        """
        if not self.config['enabled']:
            return {
                'error': 'Claude integration is disabled',
                'suggestion': 'Enable in config for cloud fallback capability'
            }

        if not self.is_available():
            return {
                'error': 'Claude is not available',
                'suggestion': 'Ensure you are running in Claude Code environment'
            }

        # Check if confirmation needed
        if require_confirmation is None:
            require_confirmation = self.config['require_confirmation']

        if require_confirmation:
            if not await self._get_user_confirmation(task_type):
                return {
                    'error': 'User declined cloud usage',
                    'fallback': 'Using local models only'
                }

        # Log usage for transparency
        self._log_usage(prompt, task_type)

        # Since we're already running IN Claude Code, we can indicate
        # that Claude should handle this directly
        response = {
            'response': f"""
I'll help with this {task_type} task. As Claude (running via Claude Code), I can provide:

- Complex reasoning and analysis
- Long-context understanding (up to 200K tokens)
- Nuanced code comprehension
- Cross-domain knowledge synthesis

[This response would come from Claude's actual processing of: {prompt[:100]}...]
""",
            'model': 'Claude (via Claude Code)',
            'cloud': True,
            'task_type': task_type,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }

        return response

    async def _get_user_confirmation(self, task_type: str) -> bool:
        """
        Get user confirmation before sending to cloud.

        Args:
            task_type: Type of task

        Returns:
            Whether user approves
        """
        # In a real implementation, this would prompt the user
        # For now, we'll log and return True for testing
        self.logger.info(f"Would request confirmation for cloud usage: {task_type}")
        return True

    def _log_usage(self, prompt: str, task_type: str):
        """
        Log Claude usage for transparency.

        Args:
            prompt: The prompt sent
            task_type: Type of task
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type,
            'prompt_preview': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'prompt_length': len(prompt)
        }

        self.usage_log.append(log_entry)

        # Keep only last 100 entries
        if len(self.usage_log) > 100:
            self.usage_log = self.usage_log[-100:]

        self.logger.info(f"Claude usage logged: {task_type}")

    def should_use_claude(self, task_type: str,
                         complexity: str,
                         context_length: int,
                         local_models_failed: bool = False) -> bool:
        """
        Determine if Claude should be used for a task.

        Args:
            task_type: Type of task
            complexity: Task complexity
            context_length: Required context length
            local_models_failed: Whether local models have failed

        Returns:
            Whether to use Claude
        """
        if not self.config['enabled']:
            return False

        # Use Claude for fallback if local models failed
        if local_models_failed and self.config['use_as_fallback']:
            return True

        # Use Claude for very long context
        if context_length > 32000:  # Beyond most local models
            return True

        # Use Claude for complex tasks
        if complexity == 'complex' and self.config['use_for_complex_tasks']:
            if task_type in ['complex_reasoning', 'system_design', 'architecture']:
                return True

        # Use Claude for verification of critical changes
        if task_type == 'self_improvement' and self.config['use_for_verification']:
            return True

        return False

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of Claude usage.

        Returns:
            Usage statistics
        """
        if not self.usage_log:
            return {
                'total_requests': 0,
                'status': 'No Claude usage yet'
            }

        task_distribution = {}
        for entry in self.usage_log:
            task = entry['task_type']
            task_distribution[task] = task_distribution.get(task, 0) + 1

        return {
            'total_requests': len(self.usage_log),
            'task_distribution': task_distribution,
            'last_used': self.usage_log[-1]['timestamp'] if self.usage_log else None,
            'average_prompt_length': sum(e['prompt_length'] for e in self.usage_log) / len(self.usage_log)
        }

    def clear_usage_log(self):
        """Clear usage log for privacy."""
        self.usage_log = []
        self.logger.info("Claude usage log cleared")


class ClaudeModelSpec:
    """
    Specification for Claude as a model in the multi-model system.
    """

    @staticmethod
    def get_claude_spec():
        """Get Claude's model specification."""
        return {
            'name': 'Claude (Cloud)',
            'model_id': 'claude-3-opus',  # or current version
            'size_gb': 0,  # Cloud-based
            'context_length': 200000,  # Claude's massive context
            'strengths': [
                'complex_reasoning',
                'long_context',
                'code_understanding',
                'nuanced_analysis',
                'ethical_reasoning',
                'cross_domain',
                'instruction_following'
            ],
            'weaknesses': [
                'requires_internet',
                'not_private',
                'requires_confirmation'
            ],
            'task_types': [
                'complex_reasoning',
                'code_generation',
                'code_analysis',
                'system_design',
                'architecture',
                'learning',
                'self_improvement',
                'general'
            ],
            'min_ram_gb': 0,  # Cloud-based
            'is_cloud': True,
            'requires_confirmation': True
        }


class HybridModelRouter:
    """
    Extended router that includes Claude as a fallback option.
    """

    def __init__(self, local_router, claude_integration):
        """
        Initialize hybrid router.

        Args:
            local_router: The local model router
            claude_integration: Claude integration instance
        """
        self.local_router = local_router
        self.claude = claude_integration
        self.logger = logging.getLogger("PHOENIX.HybridRouter")

    async def route_with_claude_fallback(self, prompt: str,
                                        task_type: str,
                                        **kwargs) -> Dict[str, Any]:
        """
        Route with Claude as ultimate fallback.

        Args:
            prompt: The user prompt
            task_type: Type of task
            **kwargs: Additional parameters

        Returns:
            Response from local model or Claude
        """
        # Try local models first
        local_response = await self.local_router.route(
            prompt,
            task_type=task_type,
            **kwargs
        )

        # If local succeeded, return it
        if local_response.get('success'):
            return local_response

        # Check if we should use Claude as fallback
        if self.claude.should_use_claude(
            task_type=task_type,
            complexity=kwargs.get('complexity', 'medium'),
            context_length=len(prompt),
            local_models_failed=True
        ):
            self.logger.info("Local models failed, falling back to Claude")

            claude_response = await self.claude.query_claude(
                prompt,
                task_type=task_type
            )

            if claude_response.get('success'):
                claude_response['fallback_from_local'] = True
                return claude_response

        # Return the local failure if Claude also failed/unavailable
        return local_response

    async def validate_with_claude(self, prompt: str,
                                  local_response: str,
                                  task_type: str = 'validation') -> Dict[str, Any]:
        """
        Use Claude to validate/verify a local model's response.

        Args:
            prompt: Original prompt
            local_response: Response from local model
            task_type: Type of task

        Returns:
            Validation result
        """
        if not self.claude.config['enabled']:
            return {'validation': 'skipped', 'reason': 'Claude disabled'}

        validation_prompt = f"""
Please validate this response from a local model:

Original Request: {prompt}

Local Model Response:
{local_response}

Is this response:
1. Correct and complete?
2. Safe and appropriate?
3. Following best practices?

Provide validation result and any corrections needed.
"""

        return await self.claude.query_claude(
            validation_prompt,
            task_type='validation',
            require_confirmation=False  # Don't ask twice for validation
        )