#!/usr/bin/env python3
"""
Safety Monitor Module - Validates improvements are safe before applying.
Ensures system stability and prevents harmful modifications.
"""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from datetime import datetime
import json


class SafetyMonitor:
    """
    Monitors and validates all self-modifications for safety.
    """

    def __init__(self, phoenix_root: Path, config: Dict = None):
        """
        Initialize the Safety Monitor.

        Args:
            phoenix_root: Root directory of PHOENIX
            config: Safety configuration parameters
        """
        self.phoenix_root = phoenix_root
        self.logger = logging.getLogger("PHOENIX.SafetyMonitor")

        # Default safety configuration
        self.config = config or {
            'max_code_change_ratio': 0.5,  # Max 50% change in single modification
            'critical_modules': ['core/', 'safety_monitor.py', 'system_control.py'],
            'forbidden_operations': ['eval', 'exec', '__import__', 'compile'],
            'max_complexity_increase': 5,
            'require_backup': True,
            'allow_network_access': False,
            'allow_file_deletion': False,
            'max_recursive_depth': 3,  # Prevent infinite self-modification loops
            'modification_cooldown': 300  # 5 minutes between modifications
        }

        # Safety rules
        self.safety_rules = [
            self._check_code_syntax,
            self._check_forbidden_operations,
            self._check_critical_modules,
            self._check_resource_usage,
            self._check_modification_scope,
            self._check_dependencies,
            self._check_recursive_modifications,
            self._check_authentication_bypass,
            self._check_data_integrity,
            self._check_rollback_capability
        ]

        # Track modification history for loop detection
        self.modification_history = []
        self.last_modification_time = None

        # Quarantine for suspicious code
        self.quarantine_path = phoenix_root / 'quarantine'
        self.quarantine_path.mkdir(parents=True, exist_ok=True)

        # Load safety incidents history
        self.incidents = []
        self._load_incidents()

    async def validate_improvement(self, improvement: Any) -> Tuple[bool, List[str]]:
        """
        Validate an improvement for safety.

        Args:
            improvement: The improvement to validate

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        self.logger.info(f"Validating improvement: {improvement.description}")
        issues = []

        # Check cooldown period
        if self.last_modification_time:
            time_since_last = (datetime.now() - self.last_modification_time).total_seconds()
            if time_since_last < self.config['modification_cooldown']:
                issues.append(f"Cooldown period not met. Wait {self.config['modification_cooldown'] - time_since_last:.0f} seconds")

        # Run all safety rules
        for rule in self.safety_rules:
            try:
                rule_passed, rule_issues = await rule(improvement)
                if not rule_passed:
                    issues.extend(rule_issues)
            except Exception as e:
                self.logger.error(f"Safety rule check failed: {e}")
                issues.append(f"Safety check error: {str(e)}")

        # Check against known attack patterns
        attack_check = self._check_attack_patterns(improvement)
        if attack_check:
            issues.extend(attack_check)

        # Risk assessment
        risk_level = self._assess_risk_level(improvement, issues)

        if issues:
            # Log safety incident
            self._log_incident(improvement, issues, risk_level)

            # Quarantine if high risk
            if risk_level == 'critical':
                self._quarantine_code(improvement)

        is_safe = len(issues) == 0

        # Update modification history
        if is_safe:
            self.modification_history.append({
                'id': improvement.id,
                'timestamp': datetime.now().isoformat(),
                'module': improvement.module,
                'type': improvement.type
            })
            self.last_modification_time = datetime.now()

        return is_safe, issues

    async def _check_code_syntax(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check if the proposed code has valid syntax."""
        issues = []

        try:
            ast.parse(improvement.proposed_code)
        except SyntaxError as e:
            issues.append(f"Syntax error in proposed code: {e}")
            return False, issues

        return True, []

    async def _check_forbidden_operations(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check for forbidden/dangerous operations."""
        issues = []
        proposed = improvement.proposed_code

        for forbidden in self.config['forbidden_operations']:
            if forbidden in proposed and forbidden not in improvement.current_code:
                issues.append(f"Forbidden operation '{forbidden}' introduced")

        # Check for dangerous patterns
        dangerous_patterns = [
            (r'os\.system\s*\(', "Direct system command execution"),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Shell injection vulnerability"),
            (r'pickle\.loads?\s*\(', "Unsafe deserialization"),
            (r'input\s*\(.*\).*eval\s*\(', "User input to eval"),
            (r'open\s*\(.*[\'\"]/etc/passwd', "Accessing sensitive files"),
            (r'shutil\.rmtree\s*\([\'\"]/[\'\"]\)', "Attempting to delete root"),
            (r'__debug__\s*=\s*False', "Disabling debug mode globally")
        ]

        for pattern, description in dangerous_patterns:
            if re.search(pattern, proposed):
                issues.append(f"Dangerous pattern detected: {description}")

        return len(issues) == 0, issues

    async def _check_critical_modules(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check if modifying critical system modules."""
        issues = []

        for critical in self.config['critical_modules']:
            if critical in improvement.module:
                # Extra scrutiny for critical modules
                change_ratio = len(improvement.proposed_code) / max(len(improvement.current_code), 1)

                if change_ratio > 0.3:  # More restrictive for critical modules
                    issues.append(f"Large change ({change_ratio:.1%}) to critical module {critical}")

                # Check if removing safety checks
                if 'safety' in improvement.current_code.lower() and \
                   'safety' not in improvement.proposed_code.lower():
                    issues.append("Appears to remove safety checks from critical module")

        return len(issues) == 0, issues

    async def _check_resource_usage(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check for potential resource exhaustion."""
        issues = []
        proposed = improvement.proposed_code

        # Check for infinite loops
        if 'while True:' in proposed and 'break' not in proposed:
            issues.append("Potential infinite loop without break condition")

        # Check for memory leaks
        if 'append' in proposed and proposed.count('append') > 5:
            if 'clear' not in proposed and 'del' not in proposed:
                issues.append("Potential memory leak - appending without cleanup")

        # Check for file handle leaks
        opens = proposed.count('open(')
        closes = proposed.count('.close()') + proposed.count('with open')
        if opens > closes:
            issues.append("Potential file handle leak")

        # Check for recursive calls
        if improvement.type == 'optimization':
            func_names = re.findall(r'def\s+(\w+)\s*\(', proposed)
            for func_name in func_names:
                if proposed.count(f'{func_name}(') > 1:
                    issues.append(f"Potential unbounded recursion in {func_name}")

        return len(issues) == 0, issues

    async def _check_modification_scope(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check if modification scope is reasonable."""
        issues = []

        # Calculate change ratio
        change_ratio = abs(len(improvement.proposed_code) - len(improvement.current_code)) / \
                      max(len(improvement.current_code), 1)

        if change_ratio > self.config['max_code_change_ratio']:
            issues.append(f"Change ratio {change_ratio:.1%} exceeds maximum {self.config['max_code_change_ratio']:.1%}")

        # Check complexity increase
        try:
            current_tree = ast.parse(improvement.current_code)
            proposed_tree = ast.parse(improvement.proposed_code)

            current_complexity = self._calculate_complexity(current_tree)
            proposed_complexity = self._calculate_complexity(proposed_tree)

            complexity_increase = proposed_complexity - current_complexity

            if complexity_increase > self.config['max_complexity_increase']:
                issues.append(f"Complexity increase {complexity_increase} exceeds maximum {self.config['max_complexity_increase']}")

        except SyntaxError:
            pass  # Already checked in syntax validation

        return len(issues) == 0, issues

    async def _check_dependencies(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check if new dependencies are safe."""
        issues = []
        proposed = improvement.proposed_code

        # Extract new imports
        current_imports = set(re.findall(r'import\s+(\w+)', improvement.current_code))
        proposed_imports = set(re.findall(r'import\s+(\w+)', proposed))
        new_imports = proposed_imports - current_imports

        # Check for suspicious imports
        suspicious_imports = ['socket', 'requests', 'urllib', 'paramiko', 'ftplib']
        for imp in new_imports:
            if imp in suspicious_imports and not self.config.get('allow_network_access', False):
                issues.append(f"New network-capable import '{imp}' not allowed")

        # Check for imports that could be used maliciously
        dangerous_imports = ['ctypes', 'subprocess', 'multiprocessing', 'threading']
        for imp in new_imports:
            if imp in dangerous_imports:
                issues.append(f"Potentially dangerous new import '{imp}' requires review")

        return len(issues) == 0, issues

    async def _check_recursive_modifications(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check for recursive self-modification loops."""
        issues = []

        # Check if modifying self-modification code
        if 'self_modification' in improvement.module or 'self_improver' in improvement.module:
            issues.append("Attempting to modify self-modification system - requires manual review")

        # Check for modification loops
        recent_mods = [m for m in self.modification_history[-10:]
                      if m['module'] == improvement.module]

        if len(recent_mods) >= self.config['max_recursive_depth']:
            issues.append(f"Module {improvement.module} modified {len(recent_mods)} times recently - possible loop")

        # Check if improvement would trigger more improvements
        if 'improve' in improvement.proposed_code.lower() and \
           'self' in improvement.proposed_code.lower():
            if improvement.proposed_code.count('improve') > improvement.current_code.count('improve'):
                issues.append("Modification increases self-improvement calls - potential cascade")

        return len(issues) == 0, issues

    async def _check_authentication_bypass(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check for authentication/authorization bypasses."""
        issues = []
        proposed = improvement.proposed_code

        # Check for removed authentication
        auth_keywords = ['authenticate', 'authorize', 'permission', 'check_access', 'verify']
        for keyword in auth_keywords:
            if keyword in improvement.current_code and keyword not in proposed:
                issues.append(f"Removes authentication check: '{keyword}'")

        # Check for hardcoded credentials
        credential_patterns = [
            r'password\s*=\s*["\'][\w]+["\']',
            r'api_key\s*=\s*["\'][\w]+["\']',
            r'secret\s*=\s*["\'][\w]+["\']'
        ]
        for pattern in credential_patterns:
            if re.search(pattern, proposed, re.IGNORECASE):
                issues.append("Hardcoded credentials detected")

        # Check for permission elevation
        if 'sudo' in proposed and 'sudo' not in improvement.current_code:
            issues.append("Introduces sudo/privilege escalation")

        return len(issues) == 0, issues

    async def _check_data_integrity(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check if modification could corrupt data."""
        issues = []

        # Check for file operations without validation
        if 'write' in improvement.proposed_code or 'dump' in improvement.proposed_code:
            if 'validate' not in improvement.proposed_code and \
               'check' not in improvement.proposed_code:
                issues.append("File write operations without validation")

        # Check for database operations
        if 'DELETE' in improvement.proposed_code.upper() or \
           'DROP' in improvement.proposed_code.upper():
            issues.append("Potentially destructive database operations")

        # Check for removing data validation
        if 'validate' in improvement.current_code and \
           'validate' not in improvement.proposed_code:
            issues.append("Removes data validation")

        return len(issues) == 0, issues

    async def _check_rollback_capability(self, improvement: Any) -> Tuple[bool, List[str]]:
        """Check if modification can be safely rolled back."""
        issues = []

        # Ensure we have the original code
        if not improvement.current_code or len(improvement.current_code) < 10:
            issues.append("Current code not properly captured for rollback")

        # Check if modification is reversible
        if improvement.type == 'feature':
            # New features should be additive, not replacing
            if len(improvement.current_code) > 100 and \
               len(improvement.proposed_code) < len(improvement.current_code) * 0.8:
                issues.append("Feature addition removes significant existing code")

        return len(issues) == 0, issues

    def _check_attack_patterns(self, improvement: Any) -> List[str]:
        """Check against known attack patterns."""
        issues = []
        proposed = improvement.proposed_code.lower()

        # Known attack patterns
        attack_patterns = {
            'backdoor': ['bind', 'listen', 'reverse', 'shell', 'nc -l'],
            'data_exfiltration': ['requests.post', 'urllib.request', 'send', 'transmit'],
            'privilege_escalation': ['chmod 777', 'setuid', 'os.setuid'],
            'cryptominer': ['stratum', 'mining', 'hashrate', 'xmr'],
            'ransomware': ['encrypt', 'decrypt', 'bitcoin', 'payment'],
            'rootkit': ['hide', 'stealth', 'hook', 'intercept']
        }

        for attack_type, patterns in attack_patterns.items():
            for pattern in patterns:
                if pattern in proposed and pattern not in improvement.current_code.lower():
                    issues.append(f"Potential {attack_type} pattern detected: '{pattern}'")

        return issues

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of code."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For,
                                ast.ExceptHandler, ast.With,
                                ast.Assert, ast.Raise)):
                complexity += 1
        return complexity

    def _assess_risk_level(self, improvement: Any, issues: List[str]) -> str:
        """Assess the risk level of an improvement."""
        if not issues:
            return 'low'

        # Critical risk indicators
        critical_indicators = ['delete root', 'removes safety', 'backdoor',
                              'privilege escalation', 'ransomware']
        for indicator in critical_indicators:
            if any(indicator in issue.lower() for issue in issues):
                return 'critical'

        # High risk indicators
        high_indicators = ['infinite loop', 'removes authentication',
                          'hardcoded credentials', 'dangerous pattern']
        for indicator in high_indicators:
            if any(indicator in issue.lower() for issue in issues):
                return 'high'

        # Medium risk if multiple issues
        if len(issues) > 3:
            return 'medium'

        return 'low'

    def _quarantine_code(self, improvement: Any):
        """Quarantine suspicious code for manual review."""
        quarantine_file = self.quarantine_path / f"quarantine_{improvement.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        quarantine_data = {
            'timestamp': datetime.now().isoformat(),
            'improvement_id': improvement.id,
            'type': improvement.type,
            'module': improvement.module,
            'description': improvement.description,
            'current_code': improvement.current_code,
            'proposed_code': improvement.proposed_code,
            'reason': 'Critical safety risk detected'
        }

        with open(quarantine_file, 'w') as f:
            json.dump(quarantine_data, f, indent=2)

        self.logger.warning(f"Code quarantined: {quarantine_file}")

    def _log_incident(self, improvement: Any, issues: List[str], risk_level: str):
        """Log a safety incident."""
        incident = {
            'timestamp': datetime.now().isoformat(),
            'improvement_id': improvement.id,
            'module': improvement.module,
            'type': improvement.type,
            'risk_level': risk_level,
            'issues': issues
        }

        self.incidents.append(incident)
        self._save_incidents()

        self.logger.warning(f"Safety incident logged: {risk_level} risk - {len(issues)} issues")

    def _save_incidents(self):
        """Save incidents to file."""
        incidents_file = self.phoenix_root / 'safety_incidents.json'
        with open(incidents_file, 'w') as f:
            json.dump(self.incidents[-100:], f, indent=2)  # Keep last 100 incidents

    def _load_incidents(self):
        """Load incidents from file."""
        incidents_file = self.phoenix_root / 'safety_incidents.json'
        if incidents_file.exists():
            try:
                with open(incidents_file, 'r') as f:
                    self.incidents = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load incidents: {e}")

    def get_safety_report(self) -> Dict[str, Any]:
        """Generate a safety report."""
        # Calculate incident statistics
        total_incidents = len(self.incidents)
        critical_incidents = sum(1 for i in self.incidents if i.get('risk_level') == 'critical')
        high_incidents = sum(1 for i in self.incidents if i.get('risk_level') == 'high')

        # Recent activity
        recent_mods = len([m for m in self.modification_history
                          if datetime.fromisoformat(m['timestamp']) >
                          datetime.now().replace(hour=0, minute=0, second=0)])

        return {
            'total_incidents': total_incidents,
            'critical_incidents': critical_incidents,
            'high_risk_incidents': high_incidents,
            'quarantined_files': len(list(self.quarantine_path.glob('*.json'))),
            'modifications_today': recent_mods,
            'last_modification': self.last_modification_time.isoformat() if self.last_modification_time else None,
            'safety_config': self.config
        }

    def emergency_shutdown(self, reason: str):
        """Emergency shutdown of self-modification system."""
        self.logger.critical(f"EMERGENCY SHUTDOWN: {reason}")

        # Create shutdown marker
        shutdown_file = self.phoenix_root / 'EMERGENCY_SHUTDOWN'
        with open(shutdown_file, 'w') as f:
            f.write(f"{datetime.now().isoformat()}: {reason}\n")

        # Log incident
        self._log_incident(
            type('Emergency', (), {'id': 'EMERGENCY', 'module': 'system', 'type': 'shutdown',
                                  'description': reason, 'current_code': '', 'proposed_code': ''})(),
            [reason],
            'critical'
        )

        return "Self-modification system shut down for safety"