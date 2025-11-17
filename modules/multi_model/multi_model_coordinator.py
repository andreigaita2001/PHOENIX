#!/usr/bin/env python3
"""
Multi-Model Coordinator - High-level interface for multi-model intelligence.
Coordinates model management, routing, and provides unified interface.
"""

import os
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class MultiModelCoordinator:
    """
    High-level coordinator for PHOENIX's multi-model intelligence system.
    """

    def __init__(self, config: Dict = None, memory_manager=None):
        """
        Initialize the Multi-Model Coordinator.

        Args:
            config: Configuration dictionary
            memory_manager: Memory manager for learning
        """
        self.logger = logging.getLogger("PHOENIX.MultiModel")
        self.memory_manager = memory_manager

        # Default configuration
        self.config = config or {
            'auto_download': True,
            'auto_benchmark': True,
            'enable_ensemble': True,
            'enable_fallback': True,
            'max_parallel_models': 3,
            'default_timeout': 60,
            'optimize_routing_interval': 100  # Optimize after every 100 requests
        }

        # Initialize components
        from .model_manager import ModelManager
        from .model_router import ModelRouter

        self.model_manager = ModelManager(memory_manager=memory_manager)
        self.model_router = ModelRouter(self.model_manager, memory_manager)

        # Request counter for optimization
        self.request_count = 0

        # Context management for conversations
        self.conversation_context = {}

        self.logger.info("Multi-Model Coordinator initialized")

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the multi-model system.

        Returns:
            Initialization status
        """
        self.logger.info("Initializing multi-model system...")

        status = {
            'initialized': False,
            'available_models': 0,
            'errors': []
        }

        try:
            # Check available models
            available = sum(1 for m in self.model_manager.models.values() if m.is_available)
            status['available_models'] = available

            if available == 0:
                # Try to pull at least one model
                if self.config['auto_download']:
                    self.logger.info("No models available, downloading starter models...")
                    await self._download_starter_models()
                    available = sum(1 for m in self.model_manager.models.values() if m.is_available)
                    status['available_models'] = available

            # Run initial benchmark if configured
            if self.config['auto_benchmark'] and available > 0:
                self.logger.info("Running initial model benchmarks...")
                benchmark_results = self.model_manager.benchmark_models()
                status['benchmark_results'] = benchmark_results

            status['initialized'] = available > 0

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            status['errors'].append(str(e))

        return status

    async def _download_starter_models(self):
        """Download essential starter models."""
        starter_models = [
            'qwen2.5-7b',  # Good general model
            'mistral-7b',  # Fast and efficient
            'codellama-7b'  # For code tasks
        ]

        for model_key in starter_models:
            if model_key in self.model_manager.models:
                self.logger.info(f"Downloading {model_key}...")
                success = await self.model_manager.pull_model(model_key)
                if success:
                    self.logger.info(f"Successfully downloaded {model_key}")
                    break  # At least one model is enough to start

    async def query(self, prompt: str,
                   model: str = None,
                   task_type: str = None,
                   context: Dict = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Main query interface for multi-model system.

        Args:
            prompt: The user prompt
            model: Specific model to use (optional)
            task_type: Hint about task type (optional)
            context: Conversation context (optional)
            **kwargs: Additional generation parameters

        Returns:
            Model response with metadata
        """
        self.request_count += 1

        # Add conversation context if available
        if not context:
            context = self.conversation_context

        # Route the query
        response = await self.model_router.route(
            prompt,
            context=context,
            override_model=model,
            **kwargs
        )

        # Optimize routing periodically
        if self.request_count % self.config['optimize_routing_interval'] == 0:
            self.model_router.optimize_routing()

        # Update conversation context
        if response.get('success'):
            self._update_context(prompt, response)

        return response

    async def query_ensemble(self, prompt: str,
                           models: List[str] = None,
                           aggregation: str = 'best',
                           **kwargs) -> Dict[str, Any]:
        """
        Query multiple models and aggregate results.

        Args:
            prompt: The user prompt
            models: List of models to use (optional)
            aggregation: Aggregation method (best, vote, merge)
            **kwargs: Generation parameters

        Returns:
            Aggregated response
        """
        if not models:
            # Auto-select models based on task classification
            task_type = self.model_router.classify_task(prompt)
            routing_rule = self.model_router.routing_rules[task_type]
            models = [
                m for m in routing_rule['preferred_models']
                if self.model_manager.models[m].is_available
            ][:self.config['max_parallel_models']]

        return await self.model_manager.ensemble_generate(
            prompt,
            models,
            aggregation,
            **kwargs
        )

    async def specialized_query(self, prompt: str,
                               specialization: str,
                               **kwargs) -> Dict[str, Any]:
        """
        Query with specific specialization requirement.

        Args:
            prompt: The user prompt
            specialization: Required specialization (code, system, reasoning, etc.)
            **kwargs: Generation parameters

        Returns:
            Model response
        """
        # Map specialization to task type
        specialization_map = {
            'code': 'code_generation',
            'debug': 'code_debug',
            'system': 'system_command',
            'reasoning': 'complex_reasoning',
            'chat': 'conversation',
            'summary': 'summarization',
            'learn': 'learning',
            'improve': 'self_improvement'
        }

        task_type = specialization_map.get(specialization, 'general')

        # Select best model for specialization
        model_key = self.model_manager.select_model_for_task(
            task_type,
            len(prompt),
            kwargs.get('complexity', 'medium')
        )

        if model_key:
            return await self.model_manager.generate(
                prompt,
                model_key=model_key,
                task_type=task_type,
                **kwargs
            )

        return {'error': f'No suitable model for {specialization}'}

    def _update_context(self, prompt: str, response: Dict):
        """Update conversation context."""
        self.conversation_context = {
            'last_prompt': prompt,
            'last_response': response.get('response', ''),
            'last_model': response.get('model', ''),
            'last_task_type': response.get('task_type', ''),
            'timestamp': datetime.now().isoformat(),
            'in_conversation': True
        }

    def switch_model(self, model_key: str) -> bool:
        """
        Explicitly switch to a specific model.

        Args:
            model_key: Model to switch to

        Returns:
            Success status
        """
        if model_key not in self.model_manager.models:
            self.logger.error(f"Unknown model: {model_key}")
            return False

        if not self.model_manager.models[model_key].is_available:
            self.logger.error(f"Model not available: {model_key}")
            return False

        self.model_manager.current_model = model_key
        self.logger.info(f"Switched to model: {model_key}")
        return True

    async def add_model(self, model_id: str,
                       name: str = None,
                       task_types: List[str] = None) -> bool:
        """
        Add a new model to the system.

        Args:
            model_id: Ollama model ID
            name: Display name (optional)
            task_types: Task types this model handles

        Returns:
            Success status
        """
        from .model_manager import ModelSpec

        # Create model spec
        model_key = model_id.replace(':', '-')
        model_spec = ModelSpec(
            name=name or model_id,
            model_id=model_id,
            size_gb=5.0,  # Estimate
            context_length=8192,  # Default
            strengths=['custom'],
            weaknesses=['unknown'],
            task_types=task_types or ['general'],
            min_ram_gb=8
        )

        # Add to manager
        self.model_manager.models[model_key] = model_spec

        # Try to pull
        success = await self.model_manager.pull_model(model_key)

        if success:
            self.logger.info(f"Successfully added model: {model_id}")

            # Run benchmark
            if self.config['auto_benchmark']:
                self.model_manager.benchmark_models()

        return success

    def get_model_status(self, model_key: str = None) -> Dict[str, Any]:
        """
        Get status of a specific model or all models.

        Args:
            model_key: Specific model (optional)

        Returns:
            Model status information
        """
        if model_key:
            if model_key not in self.model_manager.models:
                return {'error': f'Unknown model: {model_key}'}

            model_spec = self.model_manager.models[model_key]
            return {
                'name': model_spec.name,
                'available': model_spec.is_available,
                'loaded': model_key in self.model_manager.active_models,
                'size_gb': model_spec.size_gb,
                'context_length': model_spec.context_length,
                'strengths': model_spec.strengths,
                'task_types': model_spec.task_types,
                'performance_score': model_spec.performance_score,
                'success_rate': model_spec.success_rate,
                'average_response_time': model_spec.average_response_time,
                'last_used': model_spec.last_used
            }
        else:
            return self.model_manager.get_status()

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return self.model_router.get_routing_stats()

    def get_recommendations(self, task_description: str) -> Dict[str, Any]:
        """
        Get model recommendations for a task.

        Args:
            task_description: Description of the task

        Returns:
            Recommendations
        """
        # Classify task
        task_type = self.model_router.classify_task(task_description)

        # Get routing rule
        routing_rule = self.model_router.routing_rules[task_type]

        # Find available models
        available_models = []
        for model_key in routing_rule['preferred_models']:
            if self.model_manager.models[model_key].is_available:
                available_models.append({
                    'model': model_key,
                    'name': self.model_manager.models[model_key].name,
                    'performance_score': self.model_manager.models[model_key].performance_score
                })

        return {
            'task_type': task_type.value,
            'recommended_models': available_models[:3],
            'ensemble_recommended': routing_rule.get('ensemble', False),
            'fallback_available': len(available_models) > 1
        }

    async def benchmark_all(self) -> Dict[str, Any]:
        """Run comprehensive benchmark on all available models."""
        self.logger.info("Running comprehensive benchmark...")
        return self.model_manager.benchmark_models()

    def save_state(self):
        """Save current state to disk."""
        state = {
            'request_count': self.request_count,
            'routing_history': self.model_router.routing_history[-1000:],  # Keep last 1000
            'adaptive_weights': self.model_router.adaptive_weights,
            'performance_data': self.model_manager.performance_data
        }

        state_file = self.model_manager.config_path / 'coordinator_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        self.logger.info("Saved coordinator state")

    def load_state(self):
        """Load state from disk."""
        state_file = self.model_manager.config_path / 'coordinator_state.json'

        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                self.request_count = state.get('request_count', 0)
                self.model_router.routing_history = state.get('routing_history', [])
                self.model_router.adaptive_weights = state.get('adaptive_weights', {})
                self.model_manager.performance_data = state.get('performance_data', {})

                self.logger.info("Loaded coordinator state")
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")

    def get_summary(self) -> str:
        """Get a human-readable summary of the multi-model system."""
        status = self.model_manager.get_status()
        routing_stats = self.model_router.get_routing_stats()

        summary = f"""
╔══════════════════════════════════════════════════════╗
║          PHOENIX Multi-Model Intelligence             ║
╚══════════════════════════════════════════════════════╝

📊 Model Status:
  Available Models: {status['available_models']}/{status['total_models']}
  Active Models: {status['active_models']}
  Current Model: {status['current_model'] or 'None'}

🎯 Routing Statistics:
  Total Requests: {routing_stats['total_routes']}
  Fallback Rate: {routing_stats['fallback_rate']:.1%}

💾 System Resources:
  Available RAM: {status['system_resources']['available_ram_gb']:.1f} GB
  CPU Usage: {status['system_resources']['cpu_percent']:.1f}%
  GPU Available: {'✓' if status['system_resources']['gpu_available'] else '✗'}

🏆 Top Performing Models:
"""
        # Add top models by performance
        models_by_perf = sorted(
            [(k, v) for k, v in status['models'].items()],
            key=lambda x: x[1]['performance_score'],
            reverse=True
        )[:3]

        for model_key, model_info in models_by_perf:
            if model_info['available']:
                summary += f"  • {model_info['name']}: {model_info['success_rate']:.1%} success rate\n"

        return summary