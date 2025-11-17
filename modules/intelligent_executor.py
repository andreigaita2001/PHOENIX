#!/usr/bin/env python3
"""
Intelligent Executor Module - Enables PHOENIX to understand and execute ANY command.
No more pre-programmed patterns - true understanding and autonomous execution.
"""

import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, List


class IntelligentExecutor:
    """
    Gives Phoenix the ability to understand natural language and
    generate appropriate system commands autonomously.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the Intelligent Executor.

        Args:
            llm_client: Reference to the Ollama client for AI reasoning
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger("PHOENIX.IntelligentExecutor")

        # Command knowledge base (examples to learn from)
        self.command_knowledge = {
            'storage': ['df -h', 'du -sh ~', 'lsblk'],
            'network': ['ip addr', 'ifconfig', 'netstat -tuln', 'ss -tuln'],
            'usb': ['lsusb', 'dmesg | grep -i usb'],
            'battery': ['upower -i /org/freedesktop/UPower/devices/battery_BAT0', 'acpi -b'],
            'temperature': ['sensors', 'cat /sys/class/thermal/thermal_zone*/temp'],
            'services': ['systemctl list-units --type=service --state=running'],
            'ports': ['netstat -tuln', 'ss -tuln', 'lsof -i'],
            'wifi': ['nmcli dev wifi', 'iwconfig'],
            'bluetooth': ['bluetoothctl devices', 'hcitool dev'],
            'audio': ['pactl list sinks', 'amixer'],
            'kernel': ['uname -a', 'dmesg | tail'],
            'packages': ['dpkg -l | wc -l', 'apt list --installed 2>/dev/null | wc -l'],
        }

        self.logger.info("Intelligent Executor initialized")

    def understand_intent(self, user_command: str) -> Dict[str, Any]:
        """
        Understand what the user wants using AI reasoning.

        Args:
            user_command: Natural language command from user

        Returns:
            Dictionary with intent and parameters
        """
        # Keywords to intent mapping
        intent_keywords = {
            'storage': ['storage', 'disk', 'space', 'drive', 'ssd', 'hdd', 'capacity', 'full'],
            'network': ['network', 'internet', 'connection', 'ip', 'ethernet', 'lan'],
            'usb': ['usb', 'devices', 'connected', 'plugged'],
            'battery': ['battery', 'charge', 'power', 'charging'],
            'temperature': ['temperature', 'temp', 'hot', 'thermal', 'heat'],
            'services': ['services', 'running', 'daemons', 'systemd'],
            'ports': ['ports', 'listening', 'open ports', 'connections'],
            'wifi': ['wifi', 'wireless', 'ssid', 'signal'],
            'bluetooth': ['bluetooth', 'bt', 'paired'],
            'audio': ['audio', 'sound', 'volume', 'speakers', 'microphone'],
            'kernel': ['kernel', 'boot', 'dmesg', 'system log'],
            'packages': ['packages', 'installed', 'software', 'programs'],
            'files': ['files', 'directory', 'folder', 'list', 'show'],
            'process': ['process', 'running', 'pid', 'task'],
            'users': ['users', 'logged', 'who', 'sessions'],
            'time': ['time', 'date', 'clock', 'timezone'],
            'updates': ['updates', 'upgrade', 'patches'],
        }

        command_lower = user_command.lower()

        # Find matching intent
        detected_intent = None
        confidence = 0

        for intent, keywords in intent_keywords.items():
            matches = sum(1 for kw in keywords if kw in command_lower)
            if matches > 0:
                score = matches / len(keywords)
                if score > confidence:
                    detected_intent = intent
                    confidence = score

        # Extract action type
        action = 'check'  # default
        if any(word in command_lower for word in ['list', 'show', 'display']):
            action = 'list'
        elif any(word in command_lower for word in ['monitor', 'watch']):
            action = 'monitor'
        elif any(word in command_lower for word in ['clean', 'clear', 'free']):
            action = 'clean'
        elif any(word in command_lower for word in ['install', 'setup']):
            action = 'install'

        return {
            'original_command': user_command,
            'intent': detected_intent,
            'action': action,
            'confidence': confidence,
            'needs_ai': confidence < 0.3  # Use AI if low confidence
        }

    def generate_command(self, intent_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate the appropriate system command based on intent.

        Args:
            intent_data: Intent analysis from understand_intent

        Returns:
            Tuple of (command, explanation)
        """
        intent = intent_data.get('intent')
        action = intent_data.get('action')

        # If we're not confident, use AI to figure it out
        if intent_data.get('needs_ai') and self.llm_client:
            return self._generate_with_ai(intent_data['original_command'])

        # Generate based on known patterns
        if intent == 'storage':
            if action == 'clean':
                return "du -sh ~/.cache && find ~/.cache -type f -atime +7 -delete", "Clean old cache files"
            else:
                return "df -h && echo '\n=== Largest Directories ===' && du -h ~ 2>/dev/null | sort -rh | head -10", "Check disk usage and largest directories"

        elif intent == 'network':
            return "ip addr show && echo '\n=== Network Stats ===' && ss -s", "Show network interfaces and statistics"

        elif intent == 'usb':
            return "lsusb -v 2>/dev/null | grep -E '^Bus|iProduct'", "List USB devices"

        elif intent == 'battery':
            return "upower -i $(upower -e | grep 'BAT') | grep -E 'state|percentage|time to'", "Check battery status"

        elif intent == 'temperature':
            return "sensors 2>/dev/null || (echo 'Installing sensors...' && sudo apt-get install -y lm-sensors && sensors-detect --auto && sensors)", "Check system temperatures"

        elif intent == 'services':
            return "systemctl list-units --type=service --state=running --no-pager | head -20", "List running services"

        elif intent == 'ports':
            return "sudo netstat -tuln | grep LISTEN", "Show listening ports"

        elif intent == 'wifi':
            return "nmcli dev wifi list", "List WiFi networks"

        elif intent == 'bluetooth':
            return "bluetoothctl devices", "List Bluetooth devices"

        elif intent == 'audio':
            return "pactl list short sinks && echo '\n=== Audio Inputs ===' && pactl list short sources", "List audio devices"

        elif intent == 'kernel':
            return "uname -a && echo '\n=== Recent Kernel Messages ===' && dmesg | tail -20", "Show kernel information"

        elif intent == 'packages':
            return "echo 'Installed packages:' && dpkg -l | wc -l && echo '\nRecently installed:' && grep ' install ' /var/log/dpkg.log 2>/dev/null | tail -5", "Count installed packages"

        elif intent == 'users':
            return "who && echo '\n=== Last Logins ===' && last -5", "Show logged in users"

        elif intent == 'time':
            return "date && echo '\nTimezone:' && timedatectl | grep 'Time zone'", "Show date and time"

        elif intent == 'updates':
            return "sudo apt update && apt list --upgradable 2>/dev/null | head -20", "Check for updates"

        else:
            # Unknown intent - try AI or return a generic help
            if self.llm_client:
                return self._generate_with_ai(intent_data['original_command'])
            else:
                return "", f"I don't understand '{intent_data['original_command']}' yet"

    def _generate_with_ai(self, user_command: str) -> Tuple[str, str]:
        """
        Use AI to generate a command when patterns don't match.

        Args:
            user_command: The user's natural language command

        Returns:
            Tuple of (command, explanation)
        """
        if not self.llm_client:
            return "", "AI not available to interpret this command"

        try:
            # Ask the AI to generate the appropriate command
            prompt = f"""You are a Linux system expert. The user wants to: "{user_command}"

Generate the EXACT Linux command(s) to accomplish this.
Rules:
1. Output ONLY the command, nothing else
2. Use common Linux tools (ls, df, ps, etc)
3. Chain commands with && if needed
4. Make it safe (no rm -rf, no sudo unless necessary)
5. If you don't know, output: UNKNOWN

Examples:
User: "check my storage" -> df -h
User: "show running programs" -> ps aux | head -20
User: "check internet speed" -> ping -c 4 google.com

Now generate for: "{user_command}"
Command:"""

            response = self.llm_client.generate(
                model='qwen2.5:14b-instruct',
                prompt=prompt,
                options={'temperature': 0.1}  # Low temperature for consistency
            )

            command = response['response'].strip()

            # Validate the response
            if 'UNKNOWN' in command or not command:
                return "", f"Couldn't understand how to: {user_command}"

            # Clean up the command
            command = command.replace('```bash', '').replace('```', '').strip()
            command = command.split('\n')[0]  # Take first line only

            # Safety check
            dangerous = ['rm -rf /', 'dd if=/dev/zero', ':(){ :|:&', 'mkfs']
            if any(d in command for d in dangerous):
                return "", "Generated command seems dangerous, refusing to execute"

            return command, f"AI-generated command for: {user_command}"

        except Exception as e:
            self.logger.error(f"AI generation failed: {e}")
            return "", f"Failed to generate command: {e}"

    def execute_intelligent_command(self, user_command: str, system_control) -> str:
        """
        Main entry point: understand and execute any command intelligently.

        Args:
            user_command: Natural language command from user
            system_control: System control module for execution

        Returns:
            Formatted result
        """
        self.logger.info(f"Processing intelligent command: {user_command}")

        # Understand intent
        intent_data = self.understand_intent(user_command)
        self.logger.info(f"Detected intent: {intent_data}")

        # Generate command
        command, explanation = self.generate_command(intent_data)

        if not command:
            return f"❌ {explanation}"

        # Execute command
        self.logger.info(f"Executing: {command}")
        success, stdout, stderr = system_control.run_command(command)

        if success:
            result = f"✅ **{explanation}**\n\n"
            result += "```\n" + stdout[:2000] + "\n```"  # Limit output
            if len(stdout) > 2000:
                result += f"\n... (output truncated, {len(stdout)} chars total)"
        else:
            result = f"❌ **Failed to {explanation}**\n\n"
            result += f"Error: {stderr}"

        return result

    def learn_command_mapping(self, user_command: str, system_command: str, success: bool):
        """
        Learn new command mappings from successful executions.

        Args:
            user_command: What the user said
            system_command: What system command was run
            success: Whether it worked
        """
        if success:
            # Store successful mappings for future use
            # In a full implementation, this would persist to disk
            self.logger.info(f"Learned: '{user_command}' -> '{system_command}'")

    def suggest_commands(self, partial_command: str) -> List[str]:
        """
        Suggest completions for partial commands.

        Args:
            partial_command: Partial command typed by user

        Returns:
            List of suggestions
        """
        suggestions = []

        # Common command starters
        starters = [
            "check my", "show me", "list all", "monitor",
            "clean", "free up", "install", "update"
        ]

        # Common targets
        targets = [
            "storage", "disk space", "network", "usb devices",
            "battery", "temperature", "running services", "open ports",
            "wifi networks", "bluetooth devices", "audio", "kernel",
            "installed packages", "system info", "memory usage"
        ]

        # Generate suggestions
        for starter in starters:
            if starter.startswith(partial_command.lower()):
                for target in targets[:3]:  # Limit suggestions
                    suggestions.append(f"{starter} {target}")

        return suggestions[:5]  # Return top 5