#!/usr/bin/env python3
"""
Test Framework Module - Tests improvements in a sandboxed environment.
Ensures modifications don't break existing functionality.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import unittest
import asyncio
import importlib.util


class TestFramework:
    """
    Tests improvements in isolation before applying to production code.
    """

    def __init__(self, sandbox_dir: Path, system_control=None):
        """
        Initialize the Test Framework.

        Args:
            sandbox_dir: Directory for sandboxed testing
            system_control: System control module for running commands
        """
        self.sandbox_dir = sandbox_dir
        self.system_control = system_control
        self.logger = logging.getLogger("PHOENIX.TestFramework")

        # Ensure sandbox exists
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Test types
        self.test_types = {
            'unit': self._run_unit_tests,
            'integration': self._run_integration_tests,
            'performance': self._run_performance_tests,
            'security': self._run_security_tests
        }

    async def test_improvement(self, improvement: Any) -> Dict[str, Any]:
        """
        Test an improvement in the sandbox.

        Args:
            improvement: The improvement to test

        Returns:
            Test results dictionary
        """
        self.logger.info(f"Testing improvement: {improvement.description}")

        # Create sandbox environment
        sandbox_path = await self._create_sandbox(improvement)

        try:
            # Apply improvement to sandbox
            await self._apply_to_sandbox(improvement, sandbox_path)

            # Run various tests
            results = {
                'passed': True,
                'tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'failures': [],
                'performance_impact': None,
                'sandbox_path': str(sandbox_path)
            }

            # Run unit tests
            unit_results = await self._run_unit_tests(sandbox_path, improvement)
            results['tests_run'] += unit_results['tests_run']
            results['tests_passed'] += unit_results['tests_passed']
            results['tests_failed'] += unit_results['tests_failed']
            if unit_results['failures']:
                results['failures'].extend(unit_results['failures'])
                results['passed'] = False

            # Run integration tests if unit tests pass
            if results['passed']:
                integration_results = await self._run_integration_tests(sandbox_path, improvement)
                results['tests_run'] += integration_results.get('tests_run', 0)
                results['tests_passed'] += integration_results.get('tests_passed', 0)
                results['tests_failed'] += integration_results.get('tests_failed', 0)

            # Check performance impact
            if results['passed']:
                performance_impact = await self._run_performance_tests(sandbox_path, improvement)
                results['performance_impact'] = performance_impact

            # Security scan
            if results['passed']:
                security_results = await self._run_security_tests(sandbox_path, improvement)
                if not security_results['passed']:
                    results['passed'] = False
                    results['failures'].append(f"Security: {security_results['reason']}")

        except Exception as e:
            self.logger.error(f"Testing failed: {e}")
            results = {
                'passed': False,
                'tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 1,
                'failures': [str(e)],
                'performance_impact': None,
                'error': str(e)
            }

        finally:
            # Clean up sandbox (unless debugging)
            if not os.environ.get('PHOENIX_DEBUG'):
                await self._cleanup_sandbox(sandbox_path)

        return results

    async def _create_sandbox(self, improvement: Any) -> Path:
        """
        Create a sandboxed copy of the module.

        Args:
            improvement: The improvement being tested

        Returns:
            Path to sandbox directory
        """
        # Create unique sandbox directory
        sandbox_name = f"sandbox_{improvement.id}"
        sandbox_path = self.sandbox_dir / sandbox_name

        # Copy the module to sandbox
        module_path = Path(improvement.module)
        if module_path.is_absolute():
            source_path = module_path
        else:
            source_path = Path.cwd() / module_path

        if source_path.exists():
            sandbox_module = sandbox_path / module_path.name
            sandbox_module.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, sandbox_module)

            # Copy dependencies if needed
            # TODO: Copy module dependencies

        return sandbox_path

    async def _apply_to_sandbox(self, improvement: Any, sandbox_path: Path):
        """
        Apply the improvement to the sandboxed code.

        Args:
            improvement: The improvement to apply
            sandbox_path: Path to sandbox
        """
        module_path = Path(improvement.module)
        sandbox_module = sandbox_path / module_path.name

        if sandbox_module.exists():
            with open(sandbox_module, 'r') as f:
                content = f.read()

            # Apply the modification
            modified_content = content.replace(
                improvement.current_code,
                improvement.proposed_code
            )

            with open(sandbox_module, 'w') as f:
                f.write(modified_content)

            self.logger.info(f"Applied improvement to sandbox: {sandbox_module}")

    async def _run_unit_tests(self, sandbox_path: Path, improvement: Any) -> Dict[str, Any]:
        """
        Run unit tests on the modified code.

        Args:
            sandbox_path: Path to sandbox
            improvement: The improvement being tested

        Returns:
            Test results
        """
        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'failures': []
        }

        # Try to import and test the modified module
        module_path = sandbox_path / Path(improvement.module).name

        if module_path.exists():
            # Basic syntax check
            try:
                with open(module_path, 'r') as f:
                    compile(f.read(), str(module_path), 'exec')
                results['tests_run'] += 1
                results['tests_passed'] += 1
                self.logger.info("Syntax check passed")
            except SyntaxError as e:
                results['tests_run'] += 1
                results['tests_failed'] += 1
                results['failures'].append(f"Syntax error: {e}")
                self.logger.error(f"Syntax error: {e}")
                return results

            # Try to import the module
            try:
                spec = importlib.util.spec_from_file_location("test_module", module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                results['tests_run'] += 1
                results['tests_passed'] += 1
                self.logger.info("Module import successful")
            except Exception as e:
                results['tests_run'] += 1
                results['tests_failed'] += 1
                results['failures'].append(f"Import failed: {e}")
                self.logger.error(f"Import failed: {e}")
                return results

            # Run existing unit tests if available
            test_file = sandbox_path.parent.parent / 'tests' / f"test_{Path(improvement.module).stem}.py"
            if test_file.exists():
                result = await self._run_pytest(test_file, sandbox_path)
                results['tests_run'] += result['tests_run']
                results['tests_passed'] += result['tests_passed']
                results['tests_failed'] += result['tests_failed']
                results['failures'].extend(result['failures'])

        return results

    async def _run_integration_tests(self, sandbox_path: Path, improvement: Any) -> Dict[str, Any]:
        """Run integration tests."""
        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'failures': []
        }

        # Check if modified module works with rest of system
        # TODO: Implement integration testing

        return results

    async def _run_performance_tests(self, sandbox_path: Path, improvement: Any) -> Dict[str, Any]:
        """
        Measure performance impact of the improvement.

        Args:
            sandbox_path: Path to sandbox
            improvement: The improvement being tested

        Returns:
            Performance metrics
        """
        metrics = {
            'execution_time_change': 0,
            'memory_usage_change': 0,
            'complexity_change': 0
        }

        # TODO: Implement performance testing
        # - Measure execution time before/after
        # - Measure memory usage
        # - Calculate complexity metrics

        return metrics

    async def _run_security_tests(self, sandbox_path: Path, improvement: Any) -> Dict[str, Any]:
        """
        Run security tests on the modified code.

        Args:
            sandbox_path: Path to sandbox
            improvement: The improvement being tested

        Returns:
            Security test results
        """
        results = {'passed': True, 'reason': None}

        # Check for dangerous patterns
        dangerous_patterns = [
            'eval(',
            'exec(',
            '__import__',
            'os.system',
            'subprocess.call',
            'pickle.loads'
        ]

        module_path = sandbox_path / Path(improvement.module).name
        if module_path.exists():
            with open(module_path, 'r') as f:
                content = f.read()

            for pattern in dangerous_patterns:
                if pattern in improvement.proposed_code and pattern not in improvement.current_code:
                    results['passed'] = False
                    results['reason'] = f"Introduces potentially dangerous pattern: {pattern}"
                    break

        return results

    async def _run_pytest(self, test_file: Path, sandbox_path: Path) -> Dict[str, Any]:
        """Run pytest on a test file."""
        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'failures': []
        }

        if self.system_control:
            # Run pytest
            cmd = f"cd {sandbox_path} && python -m pytest {test_file} -v --tb=short"
            success, stdout, stderr = self.system_control.run_command(cmd)

            # Parse results
            if 'passed' in stdout:
                # Extract test counts from pytest output
                import re
                match = re.search(r'(\d+) passed', stdout)
                if match:
                    results['tests_passed'] = int(match.group(1))
                    results['tests_run'] = results['tests_passed']

            if 'failed' in stdout:
                match = re.search(r'(\d+) failed', stdout)
                if match:
                    results['tests_failed'] = int(match.group(1))
                    results['tests_run'] += results['tests_failed']
                results['failures'].append(stderr or "Test failures")

        return results

    async def _cleanup_sandbox(self, sandbox_path: Path):
        """Clean up sandbox directory."""
        try:
            shutil.rmtree(sandbox_path)
            self.logger.info(f"Cleaned up sandbox: {sandbox_path}")
        except Exception as e:
            self.logger.error(f"Failed to clean up sandbox: {e}")

    def create_test_suite(self, improvement: Any) -> unittest.TestSuite:
        """
        Create a test suite for an improvement.

        Args:
            improvement: The improvement to test

        Returns:
            Test suite
        """
        suite = unittest.TestSuite()

        # Add basic tests
        suite.addTest(SyntaxTest(improvement))
        suite.addTest(ImportTest(improvement))

        # Add specific tests based on improvement type
        if improvement.type == 'optimization':
            suite.addTest(PerformanceTest(improvement))
        elif improvement.type == 'reliability':
            suite.addTest(ErrorHandlingTest(improvement))

        return suite


class SyntaxTest(unittest.TestCase):
    """Test for syntax validity."""

    def __init__(self, improvement):
        super().__init__()
        self.improvement = improvement

    def test_syntax(self):
        """Test that the modified code has valid syntax."""
        try:
            compile(self.improvement.proposed_code, '<string>', 'exec')
        except SyntaxError:
            self.fail("Proposed code has syntax errors")


class ImportTest(unittest.TestCase):
    """Test for importability."""

    def __init__(self, improvement):
        super().__init__()
        self.improvement = improvement

    def test_import(self):
        """Test that the module can be imported."""
        # This would be run in the sandbox
        pass


class PerformanceTest(unittest.TestCase):
    """Test for performance improvements."""

    def __init__(self, improvement):
        super().__init__()
        self.improvement = improvement

    def test_performance(self):
        """Test that performance is improved."""
        # This would measure execution time
        pass


class ErrorHandlingTest(unittest.TestCase):
    """Test for error handling."""

    def __init__(self, improvement):
        super().__init__()
        self.improvement = improvement

    def test_error_handling(self):
        """Test that errors are properly handled."""
        # This would test error scenarios
        pass