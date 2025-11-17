#!/usr/bin/env python3
"""
Model Router - Intelligently routes tasks to appropriate models.
Implements task classification, model selection, and fallback strategies.
"""

import re
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class TaskType(Enum):
    """Types of tasks PHOENIX can handle."""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_DEBUG = "code_debug"
    SYSTEM_COMMAND = "system_command"
    GENERAL_QUERY = "general_query"
    COMPLEX_REASONING = "complex_reasoning"
    CONVERSATION = "conversation"
    SUMMARIZATION = "summarization"
    LEARNING = "learning"
    SELF_IMPROVEMENT = "self_improvement"
    DOCUMENTATION = "documentation"
    DATA_ANALYSIS = "data_analysis"


class ModelRouter:
    """
    Routes tasks to appropriate models based on task classification.
    """

    def __init__(self, model_manager, memory_manager=None):
        """
        Initialize the Model Router.

        Args:
            model_manager: The model manager instance
            memory_manager: Memory manager for learning routing patterns
        """
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.ModelRouter")

        # Task routing configuration
        self.routing_rules = self._initialize_routing_rules()

        # Performance tracking for adaptive routing
        self.routing_history = []
        self.adaptive_weights = {}

        # Fallback chain for each task type
        self.fallback_chains = self._initialize_fallback_chains()

    def _initialize_routing_rules(self) -> Dict[TaskType, Dict]:
        """Initialize routing rules for different task types."""
        return {
            TaskType.CODE_GENERATION: {
                'preferred_models': ['codellama-13b', 'deepseek-coder-6.7b', 'codellama-7b'],
                'keywords': ['write', 'create', 'implement', 'code', 'function', 'class'],
                'patterns': [r'write.*function', r'create.*script', r'implement.*algorithm'],
                'min_capability': 'code_generation'
            },
            TaskType.CODE_ANALYSIS: {
                'preferred_models': ['codellama-13b', 'qwen2.5-14b', 'deepseek-coder-6.7b'],
                'keywords': ['analyze', 'review', 'explain', 'understand', 'bug', 'error'],
                'patterns': [r'analyze.*code', r'find.*bug', r'explain.*function'],
                'min_capability': 'code_analysis'
            },
            TaskType.SYSTEM_COMMAND: {
                'preferred_models': ['qwen2.5-14b', 'qwen2.5-7b', 'mistral-7b'],
                'keywords': ['system', 'command', 'terminal', 'bash', 'linux', 'process'],
                'patterns': [r'run.*command', r'execute.*script', r'check.*system'],
                'min_capability': 'system_commands'
            },
            TaskType.COMPLEX_REASONING: {
                'preferred_models': ['qwen2.5-32b', 'qwen2.5-14b', 'llama3-8b'],
                'keywords': ['complex', 'analyze', 'reason', 'think', 'strategy', 'plan'],
                'patterns': [r'complex.*problem', r'strategic.*plan', r'deep.*analysis'],
                'min_capability': 'complex_reasoning'
            },
            TaskType.CONVERSATION: {
                'preferred_models': ['mistral-7b', 'llama3-8b', 'phi3-mini'],
                'keywords': ['chat', 'talk', 'discuss', 'hello', 'hi', 'thanks'],
                'patterns': [r'^hello', r'^hi', r'how are you'],
                'min_capability': 'general'
            },
            TaskType.SUMMARIZATION: {
                'preferred_models': ['llama3.2-3b', 'mistral-7b', 'qwen2.5-7b'],
                'keywords': ['summarize', 'summary', 'brief', 'tldr', 'overview'],
                'patterns': [r'summarize.*text', r'give.*summary', r'brief.*overview'],
                'min_capability': 'summarization'
            },
            TaskType.SELF_IMPROVEMENT: {
                'preferred_models': ['qwen2.5-14b', 'codellama-13b'],
                'keywords': ['improve', 'optimize', 'enhance', 'refactor', 'self'],
                'patterns': [r'improve.*code', r'optimize.*performance', r'self.*improvement'],
                'min_capability': 'code_generation',
                'ensemble': True  # Use multiple models for validation
            },
            TaskType.LEARNING: {
                'preferred_models': ['qwen2.5-14b', 'llama3-8b'],
                'keywords': ['learn', 'understand', 'pattern', 'knowledge', 'remember'],
                'patterns': [r'learn.*pattern', r'understand.*behavior'],
                'min_capability': 'analysis'
            },
            TaskType.GENERAL_QUERY: {
                'preferred_models': ['qwen2.5-7b', 'mistral-7b', 'llama3-8b'],
                'keywords': [],  # Default fallback
                'patterns': [],
                'min_capability': 'general'
            }
        }

    def _initialize_fallback_chains(self) -> Dict[TaskType, List[str]]:
        """Initialize fallback chains for each task type."""
        return {
            TaskType.CODE_GENERATION: [
                'codellama-13b', 'codellama-7b', 'deepseek-coder-6.7b',
                'qwen2.5-14b', 'qwen2.5-7b'
            ],
            TaskType.CODE_ANALYSIS: [
                'codellama-13b', 'qwen2.5-14b', 'deepseek-coder-6.7b',
                'qwen2.5-7b', 'mistral-7b'
            ],
            TaskType.SYSTEM_COMMAND: [
                'qwen2.5-14b', 'qwen2.5-7b', 'mistral-7b', 'llama3-8b'
            ],
            TaskType.COMPLEX_REASONING: [
                'qwen2.5-32b', 'qwen2.5-14b', 'llama3-8b', 'mistral-7b'
            ],
            TaskType.CONVERSATION: [
                'mistral-7b', 'llama3-8b', 'qwen2.5-7b', 'phi3-mini'
            ],
            TaskType.SUMMARIZATION: [
                'llama3.2-3b', 'mistral-7b', 'qwen2.5-7b', 'llama3-8b'
            ],
            TaskType.SELF_IMPROVEMENT: [
                'qwen2.5-14b', 'codellama-13b', 'qwen2.5-32b'
            ],
            TaskType.LEARNING: [
                'qwen2.5-14b', 'llama3-8b', 'qwen2.5-32b'
            ],
            TaskType.GENERAL_QUERY: [
                'qwen2.5-7b', 'mistral-7b', 'llama3-8b', 'phi3-mini'
            ]
        }

    def classify_task(self, prompt: str, context: Dict = None) -> TaskType:
        """
        Classify the task type based on prompt and context.

        Args:
            prompt: The user prompt
            context: Additional context (previous messages, etc.)

        Returns:
            Classified task type
        """
        prompt_lower = prompt.lower()

        # Check each task type's patterns and keywords
        scores = {}

        for task_type, rules in self.routing_rules.items():
            score = 0

            # Check keywords
            for keyword in rules['keywords']:
                if keyword in prompt_lower:
                    score += 2

            # Check patterns
            for pattern in rules['patterns']:
                if re.search(pattern, prompt_lower):
                    score += 3

            scores[task_type] = score

        # Context-based adjustments
        if context:
            if context.get('previous_task_type') == TaskType.CODE_GENERATION:
                scores[TaskType.CODE_ANALYSIS] += 2  # Likely follow-up

            if context.get('in_conversation'):
                scores[TaskType.CONVERSATION] += 2

        # Get task type with highest score
        if max(scores.values()) > 0:
            classified = max(scores, key=scores.get)
        else:
            classified = TaskType.GENERAL_QUERY

        self.logger.info(f"Classified task as: {classified.value}")
        return classified

    async def route(self, prompt: str,
                   context: Dict = None,
                   override_model: str = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Route a task to the appropriate model(s).

        Args:
            prompt: The user prompt
            context: Additional context
            override_model: Manual model selection
            **kwargs: Additional generation parameters

        Returns:
            Response from the selected model(s)
        """
        # Manual override
        if override_model:
            if override_model in self.model_manager.models:
                self.logger.info(f"Using override model: {override_model}")
                return await self.model_manager.generate(
                    prompt,
                    model_key=override_model,
                    **kwargs
                )
            else:
                self.logger.warning(f"Override model not found: {override_model}")

        # Classify the task
        task_type = self.classify_task(prompt, context)
        routing_rule = self.routing_rules[task_type]

        # Check if ensemble is needed
        if routing_rule.get('ensemble'):
            return await self._ensemble_route(prompt, task_type, **kwargs)

        # Try preferred models in order
        for model_key in routing_rule['preferred_models']:
            if self.model_manager.models[model_key].is_available:
                response = await self._try_model(
                    prompt,
                    model_key,
                    task_type,
                    **kwargs
                )

                if response and response.get('success'):
                    self._record_routing(task_type, model_key, True)
                    return response

        # Fallback chain
        return await self._fallback_route(prompt, task_type, **kwargs)

    async def _try_model(self, prompt: str,
                        model_key: str,
                        task_type: TaskType,
                        **kwargs) -> Optional[Dict]:
        """
        Try to generate with a specific model.

        Args:
            prompt: The prompt
            model_key: Model to try
            task_type: Type of task
            **kwargs: Generation parameters

        Returns:
            Response or None if failed
        """
        try:
            self.logger.info(f"Trying model {model_key} for {task_type.value}")

            response = await self.model_manager.generate(
                prompt,
                model_key=model_key,
                task_type=task_type.value,
                **kwargs
            )

            if response.get('success'):
                # Validate response quality
                if self._validate_response(response, task_type):
                    return response
                else:
                    self.logger.warning(f"Response from {model_key} failed validation")

        except Exception as e:
            self.logger.error(f"Model {model_key} failed: {e}")

        return None

    async def _ensemble_route(self, prompt: str,
                             task_type: TaskType,
                             **kwargs) -> Dict[str, Any]:
        """
        Route to multiple models and aggregate results.

        Args:
            prompt: The prompt
            task_type: Type of task
            **kwargs: Generation parameters

        Returns:
            Aggregated response
        """
        self.logger.info(f"Ensemble routing for {task_type.value}")

        routing_rule = self.routing_rules[task_type]
        model_keys = [
            m for m in routing_rule['preferred_models']
            if self.model_manager.models[m].is_available
        ][:3]  # Use top 3 available models

        if len(model_keys) < 2:
            # Not enough models for ensemble, fall back to single
            return await self._fallback_route(prompt, task_type, **kwargs)

        responses = []
        tasks = []

        # Create tasks for parallel execution
        for model_key in model_keys:
            task = self._try_model(prompt, model_key, task_type, **kwargs)
            tasks.append(task)

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get('success'):
                responses.append(result)

        if not responses:
            return {'error': 'Ensemble routing failed - no successful responses'}

        # Aggregate responses
        return self._aggregate_responses(responses, task_type)

    def _aggregate_responses(self, responses: List[Dict],
                            task_type: TaskType) -> Dict[str, Any]:
        """
        Aggregate multiple model responses.

        Args:
            responses: List of responses
            task_type: Type of task

        Returns:
            Aggregated response
        """
        if task_type in [TaskType.CODE_GENERATION, TaskType.SELF_IMPROVEMENT]:
            # For code, pick the most complete/correct looking one
            best = max(responses, key=lambda r: len(r['response']))

            # Add validation from other models
            validations = []
            for r in responses:
                if r != best:
                    validations.append({
                        'model': r['model'],
                        'agrees': self._responses_similar(best['response'], r['response'])
                    })

            best['ensemble_validation'] = validations
            return best

        elif task_type == TaskType.COMPLEX_REASONING:
            # Merge insights from multiple models
            merged = {
                'response': '\n\n=== Insights from Multiple Models ===\n\n',
                'models_used': [],
                'success': True,
                'ensemble': True
            }

            for i, r in enumerate(responses, 1):
                merged['response'] += f"Model {i} ({r['model']}):\n{r['response']}\n\n"
                merged['models_used'].append(r['model'])

            return merged

        else:
            # Default: return best performing model's response
            return responses[0]

    async def _fallback_route(self, prompt: str,
                             task_type: TaskType,
                             **kwargs) -> Dict[str, Any]:
        """
        Try fallback chain when preferred models fail.

        Args:
            prompt: The prompt
            task_type: Type of task
            **kwargs: Generation parameters

        Returns:
            Response or error
        """
        self.logger.info(f"Using fallback chain for {task_type.value}")

        fallback_chain = self.fallback_chains.get(task_type, [])

        for model_key in fallback_chain:
            if self.model_manager.models[model_key].is_available:
                response = await self._try_model(
                    prompt,
                    model_key,
                    task_type,
                    **kwargs
                )

                if response and response.get('success'):
                    response['fallback'] = True
                    self._record_routing(task_type, model_key, True, fallback=True)
                    return response

        # All models failed
        return {
            'error': f'All models failed for {task_type.value} task',
            'task_type': task_type.value,
            'attempted_models': fallback_chain
        }

    def _validate_response(self, response: Dict, task_type: TaskType) -> bool:
        """
        Validate that a response is appropriate for the task type.

        Args:
            response: The model response
            task_type: Expected task type

        Returns:
            Whether response is valid
        """
        if not response.get('success'):
            return False

        content = response.get('response', '')

        # Basic validation rules
        if task_type == TaskType.CODE_GENERATION:
            # Should contain code-like structures
            return any(keyword in content for keyword in ['def ', 'class ', 'import ', 'function'])

        elif task_type == TaskType.SYSTEM_COMMAND:
            # Should contain commands or system-related content
            return len(content) > 10 and not content.startswith("I can't")

        elif task_type == TaskType.SUMMARIZATION:
            # Should be reasonably concise
            return 50 < len(content) < 1000

        # Default: accept if response is not empty
        return len(content) > 10

    def _responses_similar(self, response1: str, response2: str) -> bool:
        """
        Check if two responses are similar (for ensemble validation).

        Args:
            response1: First response
            response2: Second response

        Returns:
            Whether responses are similar
        """
        # Simple similarity check - could be made more sophisticated
        if len(response1) == 0 or len(response2) == 0:
            return False

        # Check length similarity
        len_ratio = len(response1) / len(response2)
        if len_ratio < 0.5 or len_ratio > 2.0:
            return False

        # Check for common keywords (simplified)
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        overlap = len(words1.intersection(words2))
        similarity = overlap / min(len(words1), len(words2))

        return similarity > 0.3

    def _record_routing(self, task_type: TaskType,
                       model_key: str,
                       success: bool,
                       fallback: bool = False):
        """
        Record routing decision for learning.

        Args:
            task_type: Type of task
            model_key: Model that was used
            success: Whether it succeeded
            fallback: Whether this was a fallback
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type.value,
            'model': model_key,
            'success': success,
            'fallback': fallback
        }

        self.routing_history.append(record)

        # Update adaptive weights
        key = f"{task_type.value}:{model_key}"
        if key not in self.adaptive_weights:
            self.adaptive_weights[key] = {'successes': 0, 'attempts': 0}

        self.adaptive_weights[key]['attempts'] += 1
        if success:
            self.adaptive_weights[key]['successes'] += 1

        # Learn from routing patterns if memory manager available
        if self.memory_manager and success:
            self.memory_manager.learn_fact(
                f"Model {model_key} successful for {task_type.value}",
                'model_routing'
            )

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = {
            'total_routes': len(self.routing_history),
            'task_distribution': {},
            'model_performance': {},
            'fallback_rate': 0
        }

        if self.routing_history:
            # Task distribution
            for record in self.routing_history:
                task = record['task_type']
                stats['task_distribution'][task] = stats['task_distribution'].get(task, 0) + 1

            # Model performance
            for key, weights in self.adaptive_weights.items():
                task_type, model = key.split(':')
                if model not in stats['model_performance']:
                    stats['model_performance'][model] = {}

                success_rate = weights['successes'] / weights['attempts'] if weights['attempts'] > 0 else 0
                stats['model_performance'][model][task_type] = {
                    'success_rate': success_rate,
                    'attempts': weights['attempts']
                }

            # Fallback rate
            fallbacks = sum(1 for r in self.routing_history if r.get('fallback'))
            stats['fallback_rate'] = fallbacks / len(self.routing_history)

        return stats

    def optimize_routing(self):
        """
        Optimize routing rules based on historical performance.
        Updates preferred model orders based on success rates.
        """
        self.logger.info("Optimizing routing rules based on performance")

        for task_type in TaskType:
            # Get performance for each model on this task type
            model_scores = {}

            for model_key in self.model_manager.models:
                key = f"{task_type.value}:{model_key}"
                if key in self.adaptive_weights:
                    weights = self.adaptive_weights[key]
                    if weights['attempts'] >= 5:  # Minimum attempts for consideration
                        success_rate = weights['successes'] / weights['attempts']
                        model_scores[model_key] = success_rate

            # Reorder preferred models based on performance
            if model_scores:
                sorted_models = sorted(model_scores.keys(),
                                     key=lambda k: model_scores[k],
                                     reverse=True)

                # Update routing rules
                if task_type in self.routing_rules:
                    current_preferred = self.routing_rules[task_type]['preferred_models']
                    # Keep models not yet tested at the end
                    untested = [m for m in current_preferred if m not in model_scores]
                    self.routing_rules[task_type]['preferred_models'] = sorted_models + untested

                    self.logger.info(f"Optimized routing for {task_type.value}: {sorted_models[:3]}")