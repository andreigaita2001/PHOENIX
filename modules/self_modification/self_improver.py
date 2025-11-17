#!/usr/bin/env python3
"""
Self-Improver Module - The core orchestrator for PHOENIX self-modification.
This is the cutting-edge, research-grade implementation for autonomous AI improvement.
"""

import os
import ast
import json
import logging
import asyncio
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import git
from dataclasses import dataclass, asdict


@dataclass
class Improvement:
    """Represents a potential improvement to the codebase."""
    id: str
    type: str  # 'optimization', 'feature', 'bugfix', 'refactor'
    module: str
    description: str
    current_code: str
    proposed_code: str
    confidence: float
    impact: str  # 'low', 'medium', 'high'
    risk: str  # 'low', 'medium', 'high'
    rationale: str
    timestamp: str = None
    status: str = 'proposed'  # 'proposed', 'testing', 'applied', 'rejected'
    test_results: Dict = None
    metrics: Dict = None

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(f"{self.module}{self.description}{datetime.now()}".encode()).hexdigest()[:12]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class SelfImprover:
    """
    The main self-modification orchestrator for PHOENIX.
    Coordinates all aspects of autonomous self-improvement.
    """

    def __init__(self, config: Dict[str, Any], llm_client=None, system_control=None, memory=None):
        """
        Initialize the Self-Improver with future compatibility in mind.

        Args:
            config: Configuration dictionary
            llm_client: LLM client for reasoning
            system_control: System control module
            memory: Memory module for learning
        """
        self.config = config
        self.llm_client = llm_client
        self.system_control = system_control
        self.memory = memory
        self.logger = logging.getLogger("PHOENIX.SelfImprover")

        # Paths
        self.phoenix_root = Path(__file__).parent.parent.parent
        self.sandbox_dir = self.phoenix_root / 'sandbox'
        self.improvements_dir = self.phoenix_root / 'improvements'
        self.backup_dir = self.phoenix_root / 'backups'

        # Initialize directories
        self.sandbox_dir.mkdir(exist_ok=True)
        self.improvements_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)

        # Initialize components (lazy loading for modularity)
        self._code_analyzer = None
        self._modification_engine = None
        self._test_framework = None
        self._reflection_system = None
        self._safety_monitor = None

        # Improvement tracking
        self.improvements_history = self._load_history()
        self.active_improvements = []
        self.metrics = {
            'total_attempts': 0,
            'successful_improvements': 0,
            'failed_attempts': 0,
            'rollbacks': 0,
            'performance_gains': []
        }

        # Plugin compatibility
        self.plugin_interfaces = {
            'analyzers': [],  # Additional code analyzers
            'modifiers': [],  # Additional modification strategies
            'validators': [],  # Additional validation methods
            'metrics': []     # Additional success metrics
        }

        # Multi-model support preparation
        self.model_specializations = {
            'code_generation': 'qwen2.5:14b-instruct',  # Best for code
            'reasoning': 'qwen2.5:14b-instruct',  # Best for logic
            'documentation': 'qwen2.5:14b-instruct',  # Best for docs
            'safety': 'qwen2.5:14b-instruct'  # Best for safety checks
        }

        # Research mode settings
        self.research_mode = config.get('research_mode', True)
        self.experimental_features = {
            'neural_architecture_search': False,  # Future: evolve network structure
            'meta_learning': False,  # Future: learn to learn
            'curriculum_learning': False,  # Future: progressive skill development
            'adversarial_improvement': False  # Future: compete with self
        }

        self.logger.info("Self-Improver initialized with future compatibility")

    @property
    def code_analyzer(self):
        """Lazy load code analyzer."""
        if not self._code_analyzer:
            from .code_analyzer import CodeAnalyzer
            self._code_analyzer = CodeAnalyzer(self.phoenix_root, self.llm_client)
        return self._code_analyzer

    @property
    def modification_engine(self):
        """Lazy load modification engine."""
        if not self._modification_engine:
            from .modification_engine import ModificationEngine
            self._modification_engine = ModificationEngine(self.llm_client, self.memory)
        return self._modification_engine

    @property
    def test_framework(self):
        """Lazy load test framework."""
        if not self._test_framework:
            from .test_framework import TestFramework
            self._test_framework = TestFramework(self.sandbox_dir, self.system_control)
        return self._test_framework

    @property
    def reflection_system(self):
        """Lazy load reflection system."""
        if not self._reflection_system:
            from .reflection_system import ReflectionSystem
            self._reflection_system = ReflectionSystem(self.memory, self.improvements_history)
        return self._reflection_system

    @property
    def safety_monitor(self):
        """Lazy load safety monitor."""
        if not self._safety_monitor:
            from .safety_monitor import SafetyMonitor
            self._safety_monitor = SafetyMonitor(self.config.get('safety', {}))
        return self._safety_monitor

    # ============= CORE SELF-MODIFICATION PIPELINE =============

    async def autonomous_improvement_cycle(self) -> str:
        """
        Execute a full autonomous improvement cycle.
        This is the main entry point for self-modification.

        Returns:
            Report of improvements made
        """
        self.logger.info("Starting autonomous improvement cycle")
        report = "🧬 **PHOENIX Self-Modification Cycle**\n\n"

        try:
            # Phase 1: Analysis
            report += "**Phase 1: Code Analysis**\n"
            analysis = await self._analyze_codebase()
            report += f"  ✓ Analyzed {analysis['modules_analyzed']} modules\n"
            report += f"  ✓ Found {len(analysis['improvement_opportunities'])} opportunities\n\n"

            # Phase 2: Improvement Generation
            report += "**Phase 2: Generating Improvements**\n"
            improvements = await self._generate_improvements(analysis['improvement_opportunities'])
            report += f"  ✓ Generated {len(improvements)} improvement proposals\n\n"

            # Phase 3: Safety Validation
            report += "**Phase 3: Safety Validation**\n"
            safe_improvements = await self._validate_safety(improvements)
            report += f"  ✓ {len(safe_improvements)} passed safety checks\n\n"

            # Phase 4: Testing
            report += "**Phase 4: Testing in Sandbox**\n"
            tested_improvements = await self._test_improvements(safe_improvements)
            report += f"  ✓ {len(tested_improvements)} passed tests\n\n"

            # Phase 5: Application
            report += "**Phase 5: Applying Improvements**\n"
            applied = await self._apply_improvements(tested_improvements)
            report += f"  ✓ Successfully applied {len(applied)} improvements\n\n"

            # Phase 6: Reflection
            report += "**Phase 6: Learning from Results**\n"
            learnings = await self._reflect_and_learn(applied)
            report += f"  ✓ Extracted {len(learnings)} learnings\n\n"

            # Update metrics
            self.metrics['total_attempts'] += len(improvements)
            self.metrics['successful_improvements'] += len(applied)
            self.metrics['failed_attempts'] += len(improvements) - len(applied)

            report += self._generate_summary(applied)

            # Store in memory
            if self.memory:
                self.memory.learn_fact(
                    f"Self-modification cycle: {len(applied)} improvements applied",
                    'self_modification'
                )

        except Exception as e:
            self.logger.error(f"Improvement cycle failed: {e}")
            report += f"\n❌ Cycle interrupted: {e}"
            await self._emergency_rollback()

        return report

    async def _analyze_codebase(self) -> Dict[str, Any]:
        """
        Analyze the entire PHOENIX codebase for improvement opportunities.

        Returns:
            Analysis results
        """
        analysis = {
            'modules_analyzed': 0,
            'improvement_opportunities': [],
            'code_metrics': {},
            'dependency_graph': {}
        }

        # Analyze each module
        modules_path = self.phoenix_root / 'modules'
        for module_file in modules_path.glob('**/*.py'):
            if '__pycache__' in str(module_file):
                continue

            module_analysis = await self.code_analyzer.analyze_module(module_file)
            analysis['modules_analyzed'] += 1

            # Extract improvement opportunities
            for opportunity in module_analysis.get('opportunities', []):
                analysis['improvement_opportunities'].append({
                    'module': str(module_file.relative_to(self.phoenix_root)),
                    'type': opportunity['type'],
                    'description': opportunity['description'],
                    'location': opportunity.get('location'),
                    'priority': opportunity.get('priority', 'medium')
                })

            # Collect metrics
            analysis['code_metrics'][str(module_file.stem)] = module_analysis.get('metrics', {})

        # Build dependency graph
        analysis['dependency_graph'] = await self.code_analyzer.build_dependency_graph()

        return analysis

    async def _generate_improvements(self, opportunities: List[Dict]) -> List[Improvement]:
        """
        Generate concrete improvements from opportunities.

        Args:
            opportunities: List of improvement opportunities

        Returns:
            List of Improvement objects
        """
        improvements = []

        for opportunity in opportunities[:10]:  # Limit to 10 per cycle for safety
            improvement = await self.modification_engine.generate_improvement(opportunity)
            if improvement:
                improvements.append(improvement)

        return improvements

    async def _validate_safety(self, improvements: List[Improvement]) -> List[Improvement]:
        """
        Validate that improvements are safe to apply.

        Args:
            improvements: List of proposed improvements

        Returns:
            List of safe improvements
        """
        safe_improvements = []

        for improvement in improvements:
            safety_check = await self.safety_monitor.validate_improvement(improvement)

            if safety_check['is_safe']:
                safe_improvements.append(improvement)
            else:
                self.logger.warning(f"Rejected unsafe improvement: {improvement.description}")
                self.logger.warning(f"Reason: {safety_check['reason']}")

        return safe_improvements

    async def _test_improvements(self, improvements: List[Improvement]) -> List[Improvement]:
        """
        Test improvements in sandbox environment.

        Args:
            improvements: List of improvements to test

        Returns:
            List of improvements that passed testing
        """
        tested_improvements = []

        for improvement in improvements:
            test_result = await self.test_framework.test_improvement(improvement)

            if test_result['passed']:
                improvement.test_results = test_result
                improvement.status = 'tested'
                tested_improvements.append(improvement)
            else:
                self.logger.warning(f"Improvement failed testing: {improvement.description}")
                self.logger.warning(f"Failures: {test_result.get('failures')}")

        return tested_improvements

    async def _apply_improvements(self, improvements: List[Improvement]) -> List[Improvement]:
        """
        Apply tested improvements to the actual codebase.

        Args:
            improvements: List of tested improvements

        Returns:
            List of successfully applied improvements
        """
        applied = []

        # Create backup before applying
        backup_id = await self._create_backup()

        for improvement in improvements:
            try:
                # Apply the improvement
                success = await self._apply_single_improvement(improvement)

                if success:
                    improvement.status = 'applied'
                    applied.append(improvement)
                    self.logger.info(f"Applied improvement: {improvement.description}")

                    # Commit to git
                    await self._commit_improvement(improvement)

            except Exception as e:
                self.logger.error(f"Failed to apply improvement: {e}")
                # Rollback this specific change
                await self._rollback_improvement(improvement, backup_id)

        return applied

    async def _apply_single_improvement(self, improvement: Improvement) -> bool:
        """
        Apply a single improvement to the codebase.

        Args:
            improvement: The improvement to apply

        Returns:
            Success status
        """
        module_path = self.phoenix_root / improvement.module

        # Read current file
        with open(module_path, 'r') as f:
            current_content = f.read()

        # Apply the modification
        modified_content = current_content.replace(
            improvement.current_code,
            improvement.proposed_code
        )

        # Write back
        with open(module_path, 'w') as f:
            f.write(modified_content)

        # Verify the change
        with open(module_path, 'r') as f:
            new_content = f.read()

        return improvement.proposed_code in new_content

    async def _reflect_and_learn(self, improvements: List[Improvement]) -> List[Dict]:
        """
        Reflect on improvements and learn from the experience.

        Args:
            improvements: List of applied improvements

        Returns:
            List of learnings
        """
        learnings = await self.reflection_system.reflect_on_improvements(improvements)

        # Store learnings in memory
        if self.memory:
            for learning in learnings:
                self.memory.learn_fact(
                    f"Self-modification learning: {learning['insight']}",
                    'self_modification_learning'
                )

        # Update improvement history
        for improvement in improvements:
            self.improvements_history.append(asdict(improvement))

        self._save_history()

        return learnings

    # ============= PLUGIN COMPATIBILITY =============

    def register_plugin(self, plugin_type: str, plugin_instance: Any):
        """
        Register a plugin for extended functionality.

        Args:
            plugin_type: Type of plugin ('analyzer', 'modifier', 'validator', 'metric')
            plugin_instance: The plugin instance
        """
        if plugin_type in ['analyzer', 'analyzers']:
            self.plugin_interfaces['analyzers'].append(plugin_instance)
        elif plugin_type in ['modifier', 'modifiers']:
            self.plugin_interfaces['modifiers'].append(plugin_instance)
        elif plugin_type in ['validator', 'validators']:
            self.plugin_interfaces['validators'].append(plugin_instance)
        elif plugin_type in ['metric', 'metrics']:
            self.plugin_interfaces['metrics'].append(plugin_instance)

        self.logger.info(f"Registered {plugin_type} plugin: {plugin_instance.__class__.__name__}")

    # ============= MULTI-MODEL SUPPORT =============

    async def select_model_for_task(self, task_type: str) -> str:
        """
        Select the best model for a specific task.

        Args:
            task_type: Type of task

        Returns:
            Model name
        """
        return self.model_specializations.get(task_type, 'qwen2.5:14b-instruct')

    # ============= SAFETY & ROLLBACK =============

    async def _create_backup(self) -> str:
        """
        Create a backup of the current codebase.

        Returns:
            Backup ID
        """
        backup_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / backup_id

        # Use git to create backup
        if self.system_control:
            self.system_control.run_command(
                f"cp -r {self.phoenix_root}/modules {backup_path}"
            )

        self.logger.info(f"Created backup: {backup_id}")
        return backup_id

    async def _rollback_improvement(self, improvement: Improvement, backup_id: str):
        """
        Rollback a specific improvement.

        Args:
            improvement: The improvement to rollback
            backup_id: Backup to restore from
        """
        backup_path = self.backup_dir / backup_id / improvement.module
        target_path = self.phoenix_root / improvement.module

        if backup_path.exists():
            with open(backup_path, 'r') as f:
                backup_content = f.read()
            with open(target_path, 'w') as f:
                f.write(backup_content)

            self.logger.info(f"Rolled back improvement: {improvement.description}")
            self.metrics['rollbacks'] += 1

    async def _emergency_rollback(self):
        """Emergency rollback of all recent changes."""
        self.logger.warning("Performing emergency rollback!")

        # Find most recent backup
        backups = sorted(self.backup_dir.glob('*'), reverse=True)
        if backups:
            latest_backup = backups[0]
            if self.system_control:
                self.system_control.run_command(
                    f"cp -r {latest_backup}/* {self.phoenix_root}/modules/"
                )
            self.logger.info("Emergency rollback completed")

    async def _commit_improvement(self, improvement: Improvement):
        """
        Commit an improvement to git.

        Args:
            improvement: The improvement to commit
        """
        try:
            repo = git.Repo(self.phoenix_root)
            repo.index.add([improvement.module])
            commit_message = f"[SelfMod] {improvement.type}: {improvement.description}\n\n{improvement.rationale}"
            repo.index.commit(commit_message)
            self.logger.info(f"Committed improvement to git: {improvement.id}")
        except Exception as e:
            self.logger.error(f"Failed to commit to git: {e}")

    # ============= UTILITIES =============

    def _generate_summary(self, improvements: List[Improvement]) -> str:
        """Generate a summary of improvements."""
        if not improvements:
            return "No improvements applied in this cycle."

        summary = "**📈 Improvements Summary:**\n\n"
        for imp in improvements:
            summary += f"• **{imp.type.title()}**: {imp.description}\n"
            summary += f"  - Module: {imp.module}\n"
            summary += f"  - Impact: {imp.impact}, Risk: {imp.risk}\n"
            summary += f"  - Confidence: {imp.confidence:.0%}\n\n"

        return summary

    def _load_history(self) -> List[Dict]:
        """Load improvement history."""
        history_file = self.improvements_dir / 'history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return []

    def _save_history(self):
        """Save improvement history."""
        history_file = self.improvements_dir / 'history.json'
        with open(history_file, 'w') as f:
            json.dump(self.improvements_history, f, indent=2)

    def get_status(self) -> Dict[str, Any]:
        """Get self-improvement status."""
        return {
            'metrics': self.metrics,
            'active_improvements': len(self.active_improvements),
            'total_history': len(self.improvements_history),
            'plugin_count': sum(len(plugins) for plugins in self.plugin_interfaces.values()),
            'research_mode': self.research_mode,
            'experimental_features': self.experimental_features
        }