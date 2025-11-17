#!/usr/bin/env python3
"""
Model Manager - Manages multiple AI models for PHOENIX.
Handles model loading, switching, and lifecycle management.
"""

import os
import json
import time
import psutil
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess


@dataclass
class ModelSpec:
    """Specification for an AI model."""
    name: str
    model_id: str  # Ollama model name
    size_gb: float
    context_length: int
    strengths: List[str]
    weaknesses: List[str]
    task_types: List[str]
    min_ram_gb: float
    gpu_layers: int = -1  # -1 for auto
    temperature: float = 0.7
    is_available: bool = False
    last_used: Optional[str] = None
    performance_score: float = 0.0
    average_response_time: float = 0.0
    success_rate: float = 0.0


class ModelManager:
    """
    Manages multiple AI models for PHOENIX's multi-model intelligence.
    """

    def __init__(self, config_path: Path = None, memory_manager=None):
        """
        Initialize the Model Manager.

        Args:
            config_path: Path to model configuration
            memory_manager: Memory manager for model learning
        """
        self.logger = logging.getLogger("PHOENIX.ModelManager")
        self.memory_manager = memory_manager

        # Paths
        self.config_path = config_path or Path.home() / '.phoenix' / 'models'
        self.config_path.mkdir(parents=True, exist_ok=True)

        # Model registry
        self.models = self._initialize_model_registry()

        # Active models (loaded in memory)
        self.active_models: Dict[str, Any] = {}
        self.current_model = None

        # Performance tracking
        self.performance_data = {}
        self.load_performance_data()

        # Resource monitoring
        self.system_resources = self._get_system_resources()

        # Model loading thread pool
        self.loading_lock = threading.Lock()

        # Check available models
        self._check_available_models()

    def _initialize_model_registry(self) -> Dict[str, ModelSpec]:
        """Initialize the registry of available models."""
        models = {
            # Qwen family - Great for general reasoning and system tasks
            'qwen2.5-14b': ModelSpec(
                name='Qwen 2.5 14B',
                model_id='qwen2.5:14b-instruct',
                size_gb=10.0,
                context_length=32768,
                strengths=['reasoning', 'instruction_following', 'system_commands'],
                weaknesses=['resource_intensive'],
                task_types=['general', 'system', 'analysis', 'complex_reasoning'],
                min_ram_gb=16
            ),
            'qwen2.5-32b': ModelSpec(
                name='Qwen 2.5 32B',
                model_id='qwen2.5:32b-instruct',
                size_gb=20.0,
                context_length=32768,
                strengths=['complex_reasoning', 'long_context', 'analysis'],
                weaknesses=['very_resource_intensive', 'slower'],
                task_types=['complex_reasoning', 'deep_analysis', 'planning'],
                min_ram_gb=40
            ),
            'qwen2.5-7b': ModelSpec(
                name='Qwen 2.5 7B',
                model_id='qwen2.5:7b-instruct',
                size_gb=5.0,
                context_length=32768,
                strengths=['balanced', 'efficient'],
                weaknesses=['less_capable'],
                task_types=['general', 'quick_tasks'],
                min_ram_gb=8
            ),

            # CodeLlama family - Specialized for code
            'codellama-13b': ModelSpec(
                name='CodeLlama 13B',
                model_id='codellama:13b-instruct',
                size_gb=10.0,
                context_length=16384,
                strengths=['code_generation', 'code_analysis', 'debugging'],
                weaknesses=['not_general_purpose'],
                task_types=['code', 'refactoring', 'optimization', 'debugging'],
                min_ram_gb=16
            ),
            'codellama-7b': ModelSpec(
                name='CodeLlama 7B',
                model_id='codellama:7b-instruct',
                size_gb=5.0,
                context_length=16384,
                strengths=['code_generation', 'fast'],
                weaknesses=['limited_context'],
                task_types=['code', 'quick_fixes'],
                min_ram_gb=8
            ),

            # Mistral - Fast and efficient
            'mistral-7b': ModelSpec(
                name='Mistral 7B',
                model_id='mistral:7b-instruct',
                size_gb=5.0,
                context_length=8192,
                strengths=['fast', 'efficient', 'good_general'],
                weaknesses=['smaller_context'],
                task_types=['general', 'quick_response', 'chat'],
                min_ram_gb=8
            ),

            # Llama 3 family - Balanced and capable
            'llama3-8b': ModelSpec(
                name='Llama 3 8B',
                model_id='llama3:8b-instruct',
                size_gb=6.0,
                context_length=8192,
                strengths=['balanced', 'general_purpose'],
                weaknesses=['none_specific'],
                task_types=['general', 'conversation', 'analysis'],
                min_ram_gb=10
            ),
            'llama3.2-3b': ModelSpec(
                name='Llama 3.2 3B',
                model_id='llama3.2:3b-instruct',
                size_gb=2.5,
                context_length=128000,  # Huge context!
                strengths=['massive_context', 'efficient', 'fast'],
                weaknesses=['less_capable'],
                task_types=['long_document', 'quick_tasks', 'summarization'],
                min_ram_gb=4
            ),

            # Phi-3 - Lightweight
            'phi3-mini': ModelSpec(
                name='Phi-3 Mini',
                model_id='phi3:mini',
                size_gb=2.0,
                context_length=4096,
                strengths=['very_fast', 'lightweight'],
                weaknesses=['limited_capability'],
                task_types=['simple_tasks', 'quick_queries'],
                min_ram_gb=4,
                temperature=0.5
            ),

            # DeepSeek Coder - Specialized for code
            'deepseek-coder-6.7b': ModelSpec(
                name='DeepSeek Coder 6.7B',
                model_id='deepseek-coder:6.7b-instruct',
                size_gb=5.0,
                context_length=16384,
                strengths=['code_generation', 'multiple_languages'],
                weaknesses=['code_only'],
                task_types=['code', 'algorithms', 'data_structures'],
                min_ram_gb=8
            )
        }

        return models

    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resources."""
        return {
            'total_ram_gb': psutil.virtual_memory().total / (1024**3),
            'available_ram_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'gpu_available': self._check_gpu()
        }

    def _check_gpu(self) -> bool:
        """Check if GPU is available for inference."""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _check_available_models(self):
        """Check which models are available in Ollama."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                installed_models = result.stdout.lower()

                for model_key, model_spec in self.models.items():
                    # Check if model ID appears in installed list
                    # Need exact match, not just substring
                    model_id_parts = model_spec.model_id.replace(':', ' ')

                    # Check for exact model match
                    is_installed = False
                    for line in result.stdout.split('\n'):
                        if line and not line.startswith('NAME'):
                            # Parse ollama list output
                            parts = line.split()
                            if parts:
                                installed_name = parts[0]
                                if installed_name == model_spec.model_id:
                                    is_installed = True
                                    break

                    model_spec.is_available = is_installed
                    if is_installed:
                        self.logger.info(f"Model available: {model_spec.name}")
                    else:
                        self.logger.debug(f"Model not installed: {model_spec.name}")

        except Exception as e:
            self.logger.error(f"Failed to check available models: {e}")

    async def pull_model(self, model_key: str) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model_key: Key of the model to pull

        Returns:
            Success status
        """
        if model_key not in self.models:
            self.logger.error(f"Unknown model: {model_key}")
            return False

        model_spec = self.models[model_key]

        # Check if we have enough resources
        if model_spec.min_ram_gb > self.system_resources['available_ram_gb']:
            self.logger.warning(f"Insufficient RAM for {model_spec.name}")
            return False

        self.logger.info(f"Pulling model: {model_spec.name}")

        try:
            process = subprocess.Popen(
                ['ollama', 'pull', model_spec.model_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Stream output
            for line in process.stdout:
                if line.strip():
                    self.logger.info(f"Pull progress: {line.strip()}")

            process.wait()

            if process.returncode == 0:
                model_spec.is_available = True
                self.logger.info(f"Successfully pulled: {model_spec.name}")
                return True
            else:
                self.logger.error(f"Failed to pull model: {process.stderr.read()}")
                return False

        except Exception as e:
            self.logger.error(f"Error pulling model: {e}")
            return False

    def load_model(self, model_key: str) -> bool:
        """
        Load a model into memory.

        Args:
            model_key: Key of the model to load

        Returns:
            Success status
        """
        if model_key not in self.models:
            self.logger.error(f"Unknown model: {model_key}")
            return False

        model_spec = self.models[model_key]

        if not model_spec.is_available:
            self.logger.warning(f"Model not available: {model_spec.name}")
            return False

        with self.loading_lock:
            if model_key not in self.active_models:
                self.logger.info(f"Loading model: {model_spec.name}")

                # Here we would actually load the model
                # For Ollama, models are loaded on-demand
                self.active_models[model_key] = {
                    'spec': model_spec,
                    'loaded_at': datetime.now().isoformat(),
                    'request_count': 0
                }

                model_spec.last_used = datetime.now().isoformat()
                self.current_model = model_key

                self.logger.info(f"Model loaded: {model_spec.name}")
                return True
            else:
                self.logger.info(f"Model already loaded: {model_spec.name}")
                self.current_model = model_key
                return True

    def unload_model(self, model_key: str):
        """
        Unload a model from memory.

        Args:
            model_key: Key of the model to unload
        """
        if model_key in self.active_models:
            self.logger.info(f"Unloading model: {self.models[model_key].name}")
            del self.active_models[model_key]

            if self.current_model == model_key:
                self.current_model = None

    def select_model_for_task(self, task_type: str,
                            context_length: int = 0,
                            complexity: str = 'medium') -> str:
        """
        Select the best model for a given task.

        Args:
            task_type: Type of task (code, general, system, etc.)
            context_length: Required context length
            complexity: Task complexity (simple, medium, complex)

        Returns:
            Model key for best model
        """
        candidates = []

        for model_key, model_spec in self.models.items():
            if not model_spec.is_available:
                continue

            # Check if model handles this task type
            if task_type in model_spec.task_types or 'general' in model_spec.task_types:
                score = 0

                # Task type match
                if task_type in model_spec.task_types:
                    score += 10

                # Context length
                if model_spec.context_length >= context_length:
                    score += 5

                # Complexity match
                if complexity == 'complex' and 'complex_reasoning' in model_spec.strengths:
                    score += 8
                elif complexity == 'simple' and 'fast' in model_spec.strengths:
                    score += 8
                else:
                    score += 4

                # Performance history
                score += model_spec.performance_score

                # Resource efficiency
                if model_spec.min_ram_gb < self.system_resources['available_ram_gb'] * 0.5:
                    score += 3

                candidates.append((model_key, score))

        if not candidates:
            # Fallback to default
            return 'qwen2.5-14b' if self.models['qwen2.5-14b'].is_available else None

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[0][0]

        self.logger.info(f"Selected {self.models[selected].name} for {task_type} task")
        return selected

    async def generate(self, prompt: str,
                      model_key: str = None,
                      task_type: str = 'general',
                      **kwargs) -> Dict[str, Any]:
        """
        Generate a response using appropriate model.

        Args:
            prompt: The prompt to send
            model_key: Specific model to use (optional)
            task_type: Type of task for auto-selection
            **kwargs: Additional generation parameters

        Returns:
            Generation response with metadata
        """
        # Select model if not specified
        if not model_key:
            model_key = self.select_model_for_task(
                task_type=task_type,
                context_length=len(prompt),
                complexity=kwargs.get('complexity', 'medium')
            )

        if not model_key:
            return {'error': 'No suitable model available'}

        # Load model if needed
        if model_key not in self.active_models:
            if not self.load_model(model_key):
                return {'error': f'Failed to load model: {model_key}'}

        model_spec = self.models[model_key]
        start_time = time.time()

        try:
            # Use Ollama to generate
            result = subprocess.run(
                ['ollama', 'run', model_spec.model_id, prompt],
                capture_output=True,
                text=True,
                timeout=kwargs.get('timeout', 60)
            )

            response_time = time.time() - start_time

            if result.returncode == 0:
                response = {
                    'response': result.stdout,
                    'model': model_spec.name,
                    'model_key': model_key,
                    'response_time': response_time,
                    'task_type': task_type,
                    'success': True
                }

                # Update performance metrics
                self._update_performance_metrics(model_key, response_time, True)

                return response
            else:
                self.logger.error(f"Generation failed: {result.stderr}")
                self._update_performance_metrics(model_key, response_time, False)
                return {'error': result.stderr, 'model': model_spec.name}

        except subprocess.TimeoutExpired:
            self.logger.error(f"Model {model_spec.name} timed out")
            self._update_performance_metrics(model_key, kwargs.get('timeout', 60), False)
            return {'error': 'Generation timed out', 'model': model_spec.name}
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return {'error': str(e), 'model': model_spec.name}

    async def ensemble_generate(self, prompt: str,
                               model_keys: List[str],
                               aggregation: str = 'best') -> Dict[str, Any]:
        """
        Generate using multiple models and aggregate results.

        Args:
            prompt: The prompt to send
            model_keys: List of models to use
            aggregation: How to aggregate (best, vote, merge)

        Returns:
            Aggregated response
        """
        responses = []

        for model_key in model_keys:
            if self.models[model_key].is_available:
                response = await self.generate(prompt, model_key=model_key)
                if response.get('success'):
                    responses.append(response)

        if not responses:
            return {'error': 'No successful responses from ensemble'}

        if aggregation == 'best':
            # Return response with best performance score
            best = max(responses,
                      key=lambda r: self.models[r['model_key']].performance_score)
            return best

        elif aggregation == 'vote':
            # Find most common response pattern
            # This is simplified - real implementation would be more sophisticated
            return responses[0]

        elif aggregation == 'merge':
            # Merge responses intelligently
            merged = {
                'response': '\n\n'.join([r['response'] for r in responses]),
                'models': [r['model'] for r in responses],
                'aggregation': 'merged',
                'success': True
            }
            return merged

        return responses[0]

    def benchmark_models(self, test_suite: List[Dict] = None) -> Dict[str, Any]:
        """
        Benchmark all available models.

        Args:
            test_suite: List of test prompts and expected capabilities

        Returns:
            Benchmark results
        """
        if not test_suite:
            test_suite = [
                {
                    'prompt': 'Write a Python function to calculate factorial',
                    'task_type': 'code',
                    'check': lambda r: 'def' in r and 'factorial' in r
                },
                {
                    'prompt': 'Explain quantum computing in simple terms',
                    'task_type': 'general',
                    'check': lambda r: len(r) > 100
                },
                {
                    'prompt': 'How do I check system memory usage in Linux?',
                    'task_type': 'system',
                    'check': lambda r: any(cmd in r.lower() for cmd in ['free', 'top', 'htop'])
                }
            ]

        results = {}

        for model_key, model_spec in self.models.items():
            if not model_spec.is_available:
                continue

            self.logger.info(f"Benchmarking {model_spec.name}")
            model_results = {
                'scores': [],
                'response_times': [],
                'success_rate': 0
            }

            for test in test_suite:
                try:
                    response = asyncio.run(self.generate(
                        test['prompt'],
                        model_key=model_key,
                        task_type=test['task_type']
                    ))

                    if response.get('success'):
                        # Check response quality
                        passed = test['check'](response['response'])
                        model_results['scores'].append(1 if passed else 0)
                        model_results['response_times'].append(response['response_time'])
                except:
                    model_results['scores'].append(0)

            if model_results['scores']:
                model_results['success_rate'] = sum(model_results['scores']) / len(model_results['scores'])
                if model_results['response_times']:
                    model_results['avg_response_time'] = sum(model_results['response_times']) / len(model_results['response_times'])
                else:
                    model_results['avg_response_time'] = 0

            results[model_key] = model_results

            # Update model performance score
            model_spec.performance_score = model_results['success_rate'] * 10
            model_spec.average_response_time = model_results.get('avg_response_time', 0)

        self.save_performance_data()
        return results

    def _update_performance_metrics(self, model_key: str, response_time: float, success: bool):
        """Update performance metrics for a model."""
        if model_key not in self.performance_data:
            self.performance_data[model_key] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_response_time': 0
            }

        self.performance_data[model_key]['total_requests'] += 1
        if success:
            self.performance_data[model_key]['successful_requests'] += 1
        self.performance_data[model_key]['total_response_time'] += response_time

        # Update model spec
        model_spec = self.models[model_key]
        model_spec.success_rate = (
            self.performance_data[model_key]['successful_requests'] /
            self.performance_data[model_key]['total_requests']
        )
        model_spec.average_response_time = (
            self.performance_data[model_key]['total_response_time'] /
            self.performance_data[model_key]['total_requests']
        )

    def save_performance_data(self):
        """Save performance data to disk."""
        data_file = self.config_path / 'performance_data.json'
        with open(data_file, 'w') as f:
            json.dump(self.performance_data, f, indent=2)

    def load_performance_data(self):
        """Load performance data from disk."""
        data_file = self.config_path / 'performance_data.json'
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    self.performance_data = json.load(f)
            except:
                self.performance_data = {}

    def get_status(self) -> Dict[str, Any]:
        """Get current status of model manager."""
        return {
            'available_models': sum(1 for m in self.models.values() if m.is_available),
            'total_models': len(self.models),
            'active_models': len(self.active_models),
            'current_model': self.current_model,
            'system_resources': self.system_resources,
            'models': {
                k: {
                    'name': v.name,
                    'available': v.is_available,
                    'loaded': k in self.active_models,
                    'performance_score': v.performance_score,
                    'success_rate': v.success_rate
                }
                for k, v in self.models.items()
            }
        }