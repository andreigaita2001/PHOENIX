#!/usr/bin/env python3
"""
Safety Guard Module - Intelligent safety system with user choice.
Instead of just blocking, it explains risks and lets YOU decide.
"""

import os
import re
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class SafetyGuard:
    """
    Intelligent safety system that respects user autonomy.
    Warns about risks but lets the user make the final decision.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Safety Guard.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("PHOENIX.Safety")

        # Risk levels
        self.SAFE = 0
        self.LOW_RISK = 1
        self.MEDIUM_RISK = 2
        self.HIGH_RISK = 3
        self.CRITICAL_RISK = 4

        # User preferences
        self.user_risk_tolerance = config.get('risk_tolerance', 'medium')
        self.auto_approve_safe = config.get('auto_approve_safe', True)
        self.require_confirm_high_risk = config.get('require_confirm_high_risk', True)

        # Command risk database
        self.risk_patterns = {
            self.CRITICAL_RISK: [
                (r'rm -rf /', "Delete entire filesystem"),
                (r'rm -rf ~', "Delete entire home directory"),
                (r'dd if=/dev/zero of=/dev/[sh]d', "Wipe hard drive"),
                (r':(){ :|:& };:', "Fork bomb - crash system"),
                (r'mkfs\.', "Format filesystem"),
            ],
            self.HIGH_RISK: [
                (r'rm -rf', "Recursive force delete"),
                (r'shutdown', "System shutdown"),
                (r'reboot', "System reboot"),
                (r'chmod -R 777', "Make everything world-writable"),
                (r'kill -9', "Force kill process"),
            ],
            self.MEDIUM_RISK: [
                (r'sudo', "Elevated privileges"),
                (r'apt remove', "Uninstall software"),
                (r'pip uninstall', "Remove Python package"),
                (r'systemctl stop', "Stop system service"),
            ],
            self.LOW_RISK: [
                (r'apt install', "Install software"),
                (r'pip install', "Install Python package"),
                (r'wget|curl', "Download from internet"),
                (r'git clone', "Clone repository"),
            ],
        }

        self.logger.info("Safety Guard initialized")

    def assess_risk(self, command: str) -> Tuple[int, str]:
        """
        Assess the risk level of a command.

        Args:
            command: Command to assess

        Returns:
            Tuple of (risk_level, explanation)
        """
        command_lower = command.lower()

        # Check each risk level
        for risk_level in sorted(self.risk_patterns.keys(), reverse=True):
            for pattern, explanation in self.risk_patterns[risk_level]:
                if re.search(pattern, command_lower):
                    return risk_level, explanation

        # Default to safe
        return self.SAFE, "Command appears safe"

    def check_command(self, command: str) -> Dict[str, Any]:
        """
        Check a command and return safety information.

        Args:
            command: Command to check

        Returns:
            Dictionary with safety assessment
        """
        risk_level, explanation = self.assess_risk(command)

        # Determine action based on risk and user preferences
        if risk_level == self.SAFE:
            action = 'allow'
            message = "✅ Safe command"

        elif risk_level == self.LOW_RISK:
            if self.user_risk_tolerance in ['high', 'medium']:
                action = 'allow'
                message = f"✅ Low risk: {explanation}"
            else:
                action = 'confirm'
                message = f"⚠️ Low risk: {explanation}"

        elif risk_level == self.MEDIUM_RISK:
            if self.user_risk_tolerance == 'high':
                action = 'allow_with_warning'
                message = f"⚠️ Medium risk: {explanation}"
            else:
                action = 'confirm'
                message = f"⚠️ Medium risk: {explanation}"

        elif risk_level == self.HIGH_RISK:
            if self.require_confirm_high_risk:
                action = 'require_confirm'
                message = f"⛔ HIGH RISK: {explanation}"
            else:
                action = 'confirm'
                message = f"⛔ High risk: {explanation}"

        else:  # CRITICAL_RISK
            action = 'block'
            message = f"🚫 CRITICAL RISK: {explanation}"

        return {
            'command': command,
            'risk_level': risk_level,
            'risk_name': self._get_risk_name(risk_level),
            'explanation': explanation,
            'action': action,
            'message': message,
            'alternatives': self._suggest_alternatives(command, risk_level)
        }

    def _get_risk_name(self, risk_level: int) -> str:
        """Get human-readable risk level name."""
        names = {
            self.SAFE: "Safe",
            self.LOW_RISK: "Low Risk",
            self.MEDIUM_RISK: "Medium Risk",
            self.HIGH_RISK: "High Risk",
            self.CRITICAL_RISK: "Critical Risk"
        }
        return names.get(risk_level, "Unknown")

    def _suggest_alternatives(self, command: str, risk_level: int) -> list:
        """
        Suggest safer alternatives to risky commands.

        Args:
            command: The risky command
            risk_level: Risk level

        Returns:
            List of alternative suggestions
        """
        alternatives = []

        if 'rm -rf /' in command:
            alternatives.append("Use 'rm -rf /specific/path' to target specific directory")
            alternatives.append("Add '--preserve-root' for safety")

        elif 'rm -rf' in command:
            alternatives.append("Use 'rm -i' for interactive confirmation")
            alternatives.append("Use 'trash' command instead (reversible)")
            alternatives.append("Create backup first: 'cp -r target target.bak'")

        elif 'shutdown' in command or 'reboot' in command:
            alternatives.append("Save all work first")
            alternatives.append("Use 'shutdown +5' to give 5 minute warning")

        elif 'chmod -R 777' in command:
            alternatives.append("Use more restrictive permissions like 755")
            alternatives.append("Only change specific files, not recursively")

        elif 'sudo' in command:
            alternatives.append("Check if you really need root privileges")
            alternatives.append("Use minimal privileges when possible")

        return alternatives

    def format_safety_response(self, safety_check: Dict[str, Any]) -> str:
        """
        Format a safety check into a user-friendly response.

        Args:
            safety_check: Safety check result

        Returns:
            Formatted response string
        """
        response = f"\n{safety_check['message']}\n"

        if safety_check['action'] in ['confirm', 'require_confirm']:
            response += f"\n**Command:** `{safety_check['command']}`\n"
            response += f"**Risk Level:** {safety_check['risk_name']}\n"
            response += f"**Concern:** {safety_check['explanation']}\n"

            if safety_check['alternatives']:
                response += "\n**Safer alternatives:**\n"
                for alt in safety_check['alternatives']:
                    response += f"  • {alt}\n"

            if safety_check['action'] == 'require_confirm':
                response += "\n⚠️ **This action requires explicit confirmation.**\n"
                response += "Type 'yes I understand the risks' to proceed.\n"
            else:
                response += "\nProceed anyway? (y/n): "

        elif safety_check['action'] == 'block':
            response += f"\n**Blocked command:** `{safety_check['command']}`\n"
            response += f"**Reason:** {safety_check['explanation']}\n"

            if safety_check['alternatives']:
                response += "\n**Try these instead:**\n"
                for alt in safety_check['alternatives']:
                    response += f"  • {alt}\n"

            response += "\nIf you absolutely need this, you can:\n"
            response += "  1. Run it manually outside PHOENIX\n"
            response += "  2. Adjust safety settings in config\n"
            response += "  3. Use override mode (if enabled)\n"

        return response

    def update_risk_tolerance(self, level: str):
        """
        Update user's risk tolerance.

        Args:
            level: 'low', 'medium', or 'high'
        """
        if level in ['low', 'medium', 'high']:
            self.user_risk_tolerance = level
            self.logger.info(f"Risk tolerance updated to: {level}")

    def learn_from_decision(self, command: str, user_choice: str):
        """
        Learn from user's safety decisions.

        Args:
            command: The command that was checked
            user_choice: User's decision ('allow', 'deny')
        """
        # This could be expanded to build a personalized safety profile
        self.logger.info(f"User chose to {user_choice}: {command}")

        # Could store patterns of what the user typically allows/denies
        # and adjust future recommendations accordingly