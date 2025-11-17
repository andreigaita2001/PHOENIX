#!/usr/bin/env python3
"""
Capability Manager - Ensures PHOENIX is aware of its actual capabilities.
Prevents hallucination by tracking what the system can and cannot do.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import importlib
import inspect

class CapabilityManager:
    """
    Manages and tracks PHOENIX's actual capabilities to prevent hallucination.
    """

    def __init__(self, modules: Dict = None, module_creator=None):
        """
        Initialize the Capability Manager.

        Args:
            modules: Dictionary of loaded modules
            module_creator: Reference to AutonomousModuleCreator
        """
        self.modules = modules or {}
        self.module_creator = module_creator
        self.logger = logging.getLogger("PHOENIX.Capabilities")

        # Track all capabilities
        self.capabilities = {
            'core': self._get_core_capabilities(),
            'modules': {},
            'pending': {},  # Capabilities that could be added
            'impossible': []  # Things we definitively cannot do
        }

        # Scan all modules for capabilities
        self._scan_module_capabilities()

        self.logger.info("Capability Manager initialized")

    def _get_core_capabilities(self) -> Dict[str, bool]:
        """Define core system capabilities."""
        return {
            # What we CAN do
            'execute_system_commands': True,
            'file_operations': True,
            'memory_persistence': True,
            'pattern_recognition': True,
            'conversation_history': True,
            'code_generation': True,
            'module_creation': True,

            # What we CANNOT do (yet)
            'web_browser_control': False,
            'gui_window_creation': False,
            'email_sending': False,
            'direct_api_calls': False,
            'real_time_notifications': False,
            'voice_interaction': False
        }

    def _scan_module_capabilities(self):
        """Scan all loaded modules for their capabilities."""
        for name, module in self.modules.items():
            # Check if module has get_capabilities method
            if hasattr(module, 'get_capabilities'):
                try:
                    caps = module.get_capabilities()
                    self.capabilities['modules'][name] = caps
                    self.logger.info(f"Loaded capabilities for {name}: {caps}")
                except Exception as e:
                    self.logger.error(f"Failed to get capabilities for {name}: {e}")

    def can_do(self, action: str) -> Tuple[bool, str]:
        """
        Check if PHOENIX can perform a specific action.

        Args:
            action: Description of the action

        Returns:
            Tuple of (can_do, explanation)
        """
        action_lower = action.lower()

        # Check for web search
        if any(term in action_lower for term in ['search web', 'google', 'internet', 'online']):
            if 'web_search' in self.capabilities.get('modules', {}):
                if self.capabilities['modules']['web_search'].get('web_search', False):
                    return (True, "I can search the web using the WebSearch module")
            return (False, "I don't have web search capability yet, but I can create a module for it")

        # Check for GUI creation
        if any(term in action_lower for term in ['gui', 'window', 'graphical', 'interface']):
            if self.capabilities['core'].get('gui_window_creation', False):
                return (True, "I can create GUI windows")
            return (False, "I cannot directly create GUI windows yet. I need to create a GUI module first")

        # Check for scheduling
        if any(term in action_lower for term in ['schedule', 'calendar', 'appointment', 'lesson']):
            if 'tennis_scheduler' in self.capabilities.get('modules', {}):
                return (True, "I can manage schedules using the Tennis Scheduler module")
            return (False, "I need to load or create a scheduling module first")

        # Check for code execution
        if any(term in action_lower for term in ['run code', 'execute', 'python script']):
            if self.capabilities['core'].get('execute_system_commands', False):
                return (True, "I can execute Python code and system commands")
            return (False, "I cannot execute code in the current configuration")

        # Default response for unknown capabilities
        return (None, "I'm not sure if I can do that. Let me check my modules...")

    def identify_capability_gap(self, user_request: str) -> Dict[str, Any]:
        """
        Identify what capabilities are missing for a user request.

        Args:
            user_request: What the user wants

        Returns:
            Analysis of capability gaps
        """
        analysis = {
            'request': user_request,
            'can_fulfill': False,
            'missing_capabilities': [],
            'available_capabilities': [],
            'suggested_modules': [],
            'can_create_solution': False
        }

        # Analyze the request
        request_lower = user_request.lower()

        # Check web search needs
        if 'search' in request_lower or 'look up' in request_lower:
            if not self._has_capability('web_search'):
                analysis['missing_capabilities'].append('web_search')
                analysis['suggested_modules'].append('WebSearchModule')
                analysis['can_create_solution'] = True

        # Check GUI needs
        if 'gui' in request_lower or 'graphical' in request_lower or 'window' in request_lower:
            if not self.capabilities['core'].get('gui_window_creation', False):
                analysis['missing_capabilities'].append('gui_creation')
                analysis['suggested_modules'].append('GUIModule')
                analysis['can_create_solution'] = True

        # Check scheduling needs
        if 'schedule' in request_lower or 'calendar' in request_lower:
            if not self._has_capability('scheduling'):
                analysis['missing_capabilities'].append('scheduling')
                analysis['suggested_modules'].append('SchedulingModule')
                analysis['can_create_solution'] = True
            else:
                analysis['available_capabilities'].append('scheduling')

        # Check if we can fulfill the request
        analysis['can_fulfill'] = len(analysis['missing_capabilities']) == 0

        return analysis

    def _has_capability(self, capability: str) -> bool:
        """Check if a capability exists anywhere in the system."""
        # Check core
        if self.capabilities['core'].get(capability, False):
            return True

        # Check modules
        for module_caps in self.capabilities['modules'].values():
            if module_caps.get(capability, False):
                return True

        return False

    def request_capability(self, capability: str, user_context: str) -> Dict[str, Any]:
        """
        Request a new capability to be added.

        Args:
            capability: The capability needed
            user_context: Context about why it's needed

        Returns:
            Result of the request
        """
        result = {
            'capability': capability,
            'status': 'pending',
            'action_taken': None,
            'message': ''
        }

        # Check if we can create this capability
        if self.module_creator:
            self.logger.info(f"Requesting capability: {capability}")

            # Analyze if a new module is needed
            analysis = self.module_creator.analyze_need(
                f"User needs {capability}: {user_context}",
                {'existing_capabilities': list(self.capabilities['modules'].keys())}
            )

            if analysis['need_detected']:
                # Design the module
                design = self.module_creator.design_module(
                    analysis['module_type'],
                    analysis['requirements']
                )

                if design:
                    result['status'] = 'creating'
                    result['action_taken'] = 'designing_module'
                    result['message'] = f"Designing {capability} module: {design['description']}"
                else:
                    result['status'] = 'failed'
                    result['message'] = f"Could not design module for {capability}"
            else:
                result['status'] = 'not_needed'
                result['message'] = f"Capability {capability} might already exist or isn't needed"
        else:
            result['status'] = 'no_creator'
            result['message'] = "Module creator not available to add new capabilities"

        return result

    def generate_capability_report(self) -> str:
        """
        Generate a human-readable report of all capabilities.

        Returns:
            Formatted capability report
        """
        report = []
        report.append("=" * 60)
        report.append("🤖 PHOENIX CAPABILITY REPORT")
        report.append("=" * 60)

        # Core capabilities
        report.append("\n📍 CORE CAPABILITIES:")
        report.append("-" * 40)
        for cap, available in self.capabilities['core'].items():
            status = "✅" if available else "❌"
            report.append(f"  {status} {cap.replace('_', ' ').title()}")

        # Module capabilities
        if self.capabilities['modules']:
            report.append("\n📦 MODULE CAPABILITIES:")
            report.append("-" * 40)

            for module_name, caps in self.capabilities['modules'].items():
                report.append(f"\n  🔧 {module_name}:")
                for cap, available in caps.items():
                    status = "✅" if available else "❌"
                    report.append(f"    {status} {cap.replace('_', ' ').title()}")

        # Pending capabilities
        if self.capabilities['pending']:
            report.append("\n⏳ PENDING CAPABILITIES:")
            report.append("-" * 40)
            for cap, info in self.capabilities['pending'].items():
                report.append(f"  ⏸️ {cap}: {info}")

        return "\n".join(report)

    def handle_hallucination_check(self, claimed_action: str) -> Tuple[bool, str]:
        """
        Check if a claimed action would be a hallucination.

        Args:
            claimed_action: What PHOENIX is about to claim it can do

        Returns:
            Tuple of (is_hallucination, correction_message)
        """
        can_do, explanation = self.can_do(claimed_action)

        if can_do:
            return (False, "")  # Not a hallucination

        # It would be a hallucination
        if can_do is False:
            return (True, f"I cannot actually {claimed_action}. {explanation}")
        else:
            # Uncertain - better to be honest
            return (True, f"I'm not certain I can {claimed_action}. Let me verify my capabilities first.")

    def update_capability(self, module_name: str, capability: str, status: bool):
        """
        Update a capability status.

        Args:
            module_name: Name of the module
            capability: Capability name
            status: New status
        """
        if module_name not in self.capabilities['modules']:
            self.capabilities['modules'][module_name] = {}

        self.capabilities['modules'][module_name][capability] = status
        self.logger.info(f"Updated {module_name}.{capability} = {status}")