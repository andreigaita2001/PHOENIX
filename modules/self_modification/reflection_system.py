#!/usr/bin/env python3
"""
Reflection System Module - Learns from improvement attempts.
Tracks patterns of success and failure to improve future decisions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from collections import defaultdict


class ReflectionSystem:
    """
    Reflects on improvement attempts to learn and adapt strategies.
    """

    def __init__(self, memory_path: Path, memory_manager=None, llm_client=None):
        """
        Initialize the Reflection System.

        Args:
            memory_path: Path to store reflection data
            memory_manager: Memory manager for long-term learning
            llm_client: LLM client for meta-analysis
        """
        self.memory_path = memory_path
        self.memory_manager = memory_manager
        self.llm_client = llm_client
        self.logger = logging.getLogger("PHOENIX.ReflectionSystem")

        # Ensure memory directory exists
        self.memory_path.mkdir(parents=True, exist_ok=True)

        # Learning data structures
        self.improvement_history = []
        self.pattern_database = defaultdict(lambda: {
            'successes': 0,
            'failures': 0,
            'avg_confidence': 0,
            'avg_impact': 0,
            'contexts': []
        })

        # Meta-learning parameters
        self.learning_rate = 0.1
        self.strategy_weights = {
            'optimization': 1.0,
            'refactor': 1.0,
            'bugfix': 1.0,
            'feature': 1.0,
            'documentation': 1.0,
            'reliability': 1.0
        }

        # Performance metrics
        self.metrics = {
            'total_attempts': 0,
            'successful_improvements': 0,
            'failed_improvements': 0,
            'rolled_back': 0,
            'average_test_pass_rate': 0,
            'patterns_learned': 0,
            'adaptation_rate': 0
        }

        # Load historical data
        self._load_reflection_data()

    async def reflect_on_attempt(self, improvement: Any, test_results: Dict, was_applied: bool):
        """
        Reflect on an improvement attempt and learn from it.

        Args:
            improvement: The improvement that was attempted
            test_results: Results from testing
            was_applied: Whether the improvement was applied to production
        """
        self.logger.info(f"Reflecting on improvement: {improvement.description}")

        # Record the attempt
        attempt_record = {
            'timestamp': datetime.now().isoformat(),
            'improvement_id': improvement.id,
            'type': improvement.type,
            'module': improvement.module,
            'description': improvement.description,
            'confidence': improvement.confidence,
            'impact': improvement.impact,
            'risk': improvement.risk,
            'test_results': test_results,
            'was_applied': was_applied,
            'success': test_results.get('passed', False) and was_applied
        }

        self.improvement_history.append(attempt_record)
        self.metrics['total_attempts'] += 1

        if attempt_record['success']:
            self.metrics['successful_improvements'] += 1
            await self._learn_from_success(improvement, test_results)
        else:
            self.metrics['failed_improvements'] += 1
            if was_applied:
                self.metrics['rolled_back'] += 1
            await self._learn_from_failure(improvement, test_results)

        # Update pattern database
        self._update_patterns(improvement, attempt_record['success'])

        # Adjust strategy weights based on outcomes
        self._adjust_strategy_weights(improvement.type, attempt_record['success'])

        # Meta-analysis every 10 attempts
        if self.metrics['total_attempts'] % 10 == 0:
            await self._perform_meta_analysis()

        # Persist reflection data
        self._save_reflection_data()

    async def _learn_from_success(self, improvement: Any, test_results: Dict):
        """
        Learn from a successful improvement.

        Args:
            improvement: The successful improvement
            test_results: Test results showing success
        """
        self.logger.info(f"Learning from success: {improvement.type} in {improvement.module}")

        # Extract success patterns
        success_pattern = {
            'type': improvement.type,
            'module_characteristics': self._extract_module_characteristics(improvement.module),
            'code_patterns': self._extract_code_patterns(improvement),
            'test_coverage': test_results.get('tests_passed', 0) / max(test_results.get('tests_run', 1), 1),
            'performance_gain': test_results.get('performance_impact', {})
        }

        # Store in pattern database
        pattern_key = f"{improvement.type}:{improvement.module}"
        self.pattern_database[pattern_key]['successes'] += 1
        self.pattern_database[pattern_key]['contexts'].append(success_pattern)

        # If using memory manager, store the learning
        if self.memory_manager:
            self.memory_manager.learn_fact(
                f"Successfully applied {improvement.type} to {improvement.module}",
                'improvement_successes'
            )

        # Increase confidence for similar future improvements
        self.pattern_database[pattern_key]['avg_confidence'] = \
            (self.pattern_database[pattern_key]['avg_confidence'] + improvement.confidence) / 2

    async def _learn_from_failure(self, improvement: Any, test_results: Dict):
        """
        Learn from a failed improvement.

        Args:
            improvement: The failed improvement
            test_results: Test results showing failure
        """
        self.logger.info(f"Learning from failure: {improvement.type} in {improvement.module}")

        # Extract failure patterns
        failure_pattern = {
            'type': improvement.type,
            'module_characteristics': self._extract_module_characteristics(improvement.module),
            'failure_reasons': test_results.get('failures', []),
            'test_failures': test_results.get('tests_failed', 0),
            'error_type': self._categorize_error(test_results)
        }

        # Store in pattern database
        pattern_key = f"{improvement.type}:{improvement.module}"
        self.pattern_database[pattern_key]['failures'] += 1
        self.pattern_database[pattern_key]['contexts'].append(failure_pattern)

        # Analyze why it failed
        if self.llm_client:
            failure_analysis = await self._ai_failure_analysis(improvement, test_results)
            if failure_analysis:
                failure_pattern['ai_analysis'] = failure_analysis

        # Decrease confidence for similar future improvements
        self.pattern_database[pattern_key]['avg_confidence'] *= 0.9

    def _extract_module_characteristics(self, module_path: str) -> Dict:
        """Extract characteristics of a module."""
        return {
            'is_core': 'core' in module_path,
            'is_module': 'modules' in module_path,
            'depth': module_path.count('/'),
            'component': module_path.split('/')[-1].replace('.py', '') if '/' in module_path else module_path
        }

    def _extract_code_patterns(self, improvement: Any) -> Dict:
        """Extract patterns from the code changes."""
        patterns = {
            'lines_changed': len(improvement.proposed_code.split('\n')),
            'adds_error_handling': 'try' in improvement.proposed_code and 'try' not in improvement.current_code,
            'adds_async': 'async' in improvement.proposed_code and 'async' not in improvement.current_code,
            'adds_type_hints': ':' in improvement.proposed_code and '->' in improvement.proposed_code,
            'refactors_complexity': len(improvement.proposed_code) < len(improvement.current_code) * 0.8
        }
        return patterns

    def _categorize_error(self, test_results: Dict) -> str:
        """Categorize the type of error that occurred."""
        failures = test_results.get('failures', [])
        if not failures:
            return 'unknown'

        failure_str = str(failures[0]).lower()

        if 'syntax' in failure_str:
            return 'syntax_error'
        elif 'import' in failure_str:
            return 'import_error'
        elif 'attribute' in failure_str:
            return 'attribute_error'
        elif 'type' in failure_str:
            return 'type_error'
        elif 'value' in failure_str:
            return 'value_error'
        elif 'timeout' in failure_str:
            return 'performance_regression'
        elif 'assert' in failure_str:
            return 'test_failure'
        else:
            return 'runtime_error'

    def _update_patterns(self, improvement: Any, success: bool):
        """Update pattern recognition database."""
        pattern_key = f"{improvement.type}:{improvement.module}"

        # Calculate pattern strength
        pattern = self.pattern_database[pattern_key]
        total_attempts = pattern['successes'] + pattern['failures']

        if total_attempts > 0:
            pattern['success_rate'] = pattern['successes'] / total_attempts
            pattern['avg_impact'] = (pattern['avg_impact'] * (total_attempts - 1) +
                                    (1.0 if improvement.impact == 'high' else 0.5 if improvement.impact == 'medium' else 0.2)) / total_attempts

        self.metrics['patterns_learned'] = len([p for p in self.pattern_database.values()
                                               if p['successes'] + p['failures'] > 2])

    def _adjust_strategy_weights(self, strategy_type: str, success: bool):
        """Adjust weights for different improvement strategies."""
        if strategy_type in self.strategy_weights:
            if success:
                # Increase weight for successful strategies
                self.strategy_weights[strategy_type] = min(2.0,
                    self.strategy_weights[strategy_type] * (1 + self.learning_rate))
            else:
                # Decrease weight for failed strategies
                self.strategy_weights[strategy_type] = max(0.1,
                    self.strategy_weights[strategy_type] * (1 - self.learning_rate))

            # Normalize weights
            total_weight = sum(self.strategy_weights.values())
            for key in self.strategy_weights:
                self.strategy_weights[key] /= total_weight

    async def _ai_failure_analysis(self, improvement: Any, test_results: Dict) -> Optional[str]:
        """Use AI to analyze why an improvement failed."""
        if not self.llm_client:
            return None

        prompt = f"""Analyze why this code improvement failed.

Improvement Type: {improvement.type}
Module: {improvement.module}
Description: {improvement.description}

Original Code:
{improvement.current_code[:500]}

Proposed Code:
{improvement.proposed_code[:500]}

Test Results:
- Tests Run: {test_results.get('tests_run', 0)}
- Tests Failed: {test_results.get('tests_failed', 0)}
- Failures: {test_results.get('failures', [])}

Provide a brief analysis of:
1. Root cause of failure
2. What could be done differently
3. Lessons learned

Keep response under 200 words."""

        try:
            response = self.llm_client.generate(
                model='qwen2.5:14b-instruct',
                prompt=prompt,
                options={'temperature': 0.3}
            )
            return response['response']
        except Exception as e:
            self.logger.error(f"AI failure analysis failed: {e}")
            return None

    async def _perform_meta_analysis(self):
        """Perform meta-analysis on accumulated learning."""
        self.logger.info("Performing meta-analysis on improvement patterns")

        if self.metrics['total_attempts'] == 0:
            return

        # Calculate overall metrics
        self.metrics['success_rate'] = self.metrics['successful_improvements'] / self.metrics['total_attempts']

        # Find most successful patterns
        successful_patterns = []
        failed_patterns = []

        for pattern_key, pattern_data in self.pattern_database.items():
            total = pattern_data['successes'] + pattern_data['failures']
            if total > 2:  # Minimum attempts for significance
                success_rate = pattern_data['successes'] / total
                if success_rate > 0.7:
                    successful_patterns.append({
                        'pattern': pattern_key,
                        'success_rate': success_rate,
                        'attempts': total
                    })
                elif success_rate < 0.3:
                    failed_patterns.append({
                        'pattern': pattern_key,
                        'success_rate': success_rate,
                        'attempts': total
                    })

        # Store insights
        insights = {
            'timestamp': datetime.now().isoformat(),
            'total_attempts': self.metrics['total_attempts'],
            'overall_success_rate': self.metrics['success_rate'],
            'successful_patterns': successful_patterns,
            'failed_patterns': failed_patterns,
            'strategy_weights': self.strategy_weights.copy(),
            'adaptation_rate': self._calculate_adaptation_rate()
        }

        # Save insights
        insights_file = self.memory_path / f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2)

        self.logger.info(f"Meta-analysis complete. Success rate: {self.metrics['success_rate']:.2%}")

    def _calculate_adaptation_rate(self) -> float:
        """Calculate how well the system is adapting over time."""
        if len(self.improvement_history) < 10:
            return 0.0

        # Compare recent success rate to overall
        recent_attempts = self.improvement_history[-10:]
        recent_successes = sum(1 for a in recent_attempts if a['success'])
        recent_rate = recent_successes / len(recent_attempts)

        overall_rate = self.metrics['success_rate']

        # Positive value means improving
        adaptation_rate = recent_rate - overall_rate
        self.metrics['adaptation_rate'] = adaptation_rate

        return adaptation_rate

    def get_recommendation(self, improvement_type: str, module: str) -> Dict[str, Any]:
        """
        Get recommendation based on past learning.

        Args:
            improvement_type: Type of improvement being considered
            module: Module to improve

        Returns:
            Recommendation with confidence and reasoning
        """
        pattern_key = f"{improvement_type}:{module}"
        pattern = self.pattern_database.get(pattern_key)

        if not pattern or pattern['successes'] + pattern['failures'] == 0:
            # No history, use strategy weights
            return {
                'recommended': True,
                'confidence': self.strategy_weights.get(improvement_type, 0.5),
                'reasoning': "No prior experience with this pattern",
                'risk_level': 'unknown'
            }

        success_rate = pattern['successes'] / (pattern['successes'] + pattern['failures'])

        recommendation = {
            'recommended': success_rate > 0.5,
            'confidence': pattern['avg_confidence'],
            'success_rate': success_rate,
            'attempts': pattern['successes'] + pattern['failures'],
            'reasoning': f"Based on {pattern['successes'] + pattern['failures']} previous attempts",
            'risk_level': 'low' if success_rate > 0.7 else 'medium' if success_rate > 0.3 else 'high'
        }

        # Add specific warnings for high-risk patterns
        if 'core' in module and success_rate < 0.5:
            recommendation['warning'] = "Core module modifications have low success rate"
            recommendation['risk_level'] = 'high'

        return recommendation

    def _save_reflection_data(self):
        """Save reflection data to disk."""
        data = {
            'improvement_history': self.improvement_history[-100:],  # Keep last 100
            'pattern_database': dict(self.pattern_database),
            'strategy_weights': self.strategy_weights,
            'metrics': self.metrics
        }

        reflection_file = self.memory_path / 'reflection_data.json'
        with open(reflection_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_reflection_data(self):
        """Load reflection data from disk."""
        reflection_file = self.memory_path / 'reflection_data.json'

        if reflection_file.exists():
            try:
                with open(reflection_file, 'r') as f:
                    data = json.load(f)

                self.improvement_history = data.get('improvement_history', [])

                # Reconstruct pattern database
                for key, value in data.get('pattern_database', {}).items():
                    self.pattern_database[key] = value

                self.strategy_weights = data.get('strategy_weights', self.strategy_weights)
                self.metrics = data.get('metrics', self.metrics)

                self.logger.info(f"Loaded reflection data: {self.metrics['total_attempts']} attempts")

            except Exception as e:
                self.logger.error(f"Failed to load reflection data: {e}")

    def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of what the system has learned.

        Returns:
            Dictionary containing learning insights
        """
        if self.metrics['total_attempts'] == 0:
            return {
                'status': 'No learning data available',
                'attempts': 0
            }

        # Find best and worst performing strategies
        strategy_performance = {}
        for pattern_key, pattern_data in self.pattern_database.items():
            strategy = pattern_key.split(':')[0]
            total = pattern_data['successes'] + pattern_data['failures']
            if total > 0:
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'successes': 0, 'attempts': 0}
                strategy_performance[strategy]['successes'] += pattern_data['successes']
                strategy_performance[strategy]['attempts'] += total

        best_strategies = []
        worst_strategies = []
        for strategy, perf in strategy_performance.items():
            rate = perf['successes'] / perf['attempts'] if perf['attempts'] > 0 else 0
            if rate > 0.7:
                best_strategies.append({'strategy': strategy, 'success_rate': rate})
            elif rate < 0.3:
                worst_strategies.append({'strategy': strategy, 'success_rate': rate})

        return {
            'total_attempts': self.metrics['total_attempts'],
            'success_rate': self.metrics.get('success_rate', 0),
            'patterns_learned': self.metrics['patterns_learned'],
            'adaptation_rate': self.metrics.get('adaptation_rate', 0),
            'best_strategies': best_strategies,
            'worst_strategies': worst_strategies,
            'current_weights': self.strategy_weights,
            'recent_trend': 'improving' if self.metrics.get('adaptation_rate', 0) > 0 else 'declining'
        }