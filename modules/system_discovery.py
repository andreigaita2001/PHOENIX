#!/usr/bin/env python3
"""
System Discovery Module - Enables PHOENIX to autonomously explore and learn your system.
This module allows PHOENIX to familiarize itself with your entire system configuration.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import asyncio


class SystemDiscovery:
    """
    Enables Phoenix to autonomously discover and learn about the system.
    """

    def __init__(self, system_control=None, memory_manager=None, learning=None):
        """
        Initialize System Discovery.

        Args:
            system_control: System control module
            memory_manager: Memory module for storing discoveries
            learning: Learning module for pattern recognition
        """
        self.system_control = system_control
        self.memory = memory_manager
        self.learning = learning
        self.logger = logging.getLogger("PHOENIX.Discovery")

        self.discoveries = {
            'hardware': {},
            'software': {},
            'network': {},
            'user_environment': {},
            'services': {},
            'security': {}
        }

        self.discovery_commands = {
            'hardware': {
                'cpu': 'lscpu | head -20',
                'memory': 'free -h',
                'disks': 'lsblk -o NAME,SIZE,TYPE,MOUNTPOINT',
                'gpu': 'lspci | grep -i vga',
                'usb': 'lsusb',
                'pci': 'lspci | head -20',
                'sensors': 'sensors 2>/dev/null || echo "sensors not available"'
            },
            'software': {
                'os': 'lsb_release -a 2>/dev/null || cat /etc/os-release | head -5',
                'kernel': 'uname -a',
                'packages': 'dpkg -l | wc -l',
                'python': 'python3 --version',
                'shells': 'cat /etc/shells',
                'desktop': 'echo $XDG_CURRENT_DESKTOP'
            },
            'network': {
                'interfaces': 'ip -brief link',
                'addresses': 'ip -brief addr',
                'routes': 'ip route',
                'dns': 'cat /etc/resolv.conf',
                'hostname': 'hostname -f',
                'ports': 'ss -tuln | head -20'
            },
            'user_environment': {
                'username': 'whoami',
                'home': 'echo $HOME',
                'shell': 'echo $SHELL',
                'path': 'echo $PATH',
                'groups': 'groups',
                'important_dirs': 'ls -la ~ | head -20'
            },
            'services': {
                'systemd': 'systemctl list-units --type=service --state=running | head -15',
                'cron': 'crontab -l 2>/dev/null || echo "No cron jobs"',
                'listening': 'ss -tuln | grep LISTEN | head -10'
            },
            'security': {
                'sudo': 'sudo -n -l 2>/dev/null || echo "Requires password"',
                'firewall': 'sudo iptables -L -n 2>/dev/null | head -20 || echo "No access"',
                'selinux': 'getenforce 2>/dev/null || echo "SELinux not found"',
                'apparmor': 'aa-status 2>/dev/null | head -5 || echo "AppArmor not found"'
            }
        }

        self.logger.info("System Discovery initialized")

    async def full_system_discovery(self, verbose: bool = True) -> str:
        """
        Perform a complete system discovery and learning process.

        Args:
            verbose: Whether to show progress

        Returns:
            Discovery report
        """
        report = "🔍 **Starting Full System Discovery**\n\n"

        if verbose:
            report += "I'm going to explore your entire system and learn about it...\n\n"

        total_discoveries = 0

        for category, commands in self.discovery_commands.items():
            if verbose:
                report += f"**Discovering {category.upper()}...**\n"

            self.discoveries[category] = {}

            for item, command in commands.items():
                if self.system_control:
                    success, stdout, stderr = self.system_control.run_command(command)
                    if success and stdout:
                        # Store the discovery
                        self.discoveries[category][item] = stdout.strip()

                        # Store in memory if available
                        if self.memory:
                            fact = f"System {category}: {item} = {stdout[:100]}..."
                            self.memory.learn_fact(fact, f'system_{category}')

                        total_discoveries += 1

                        if verbose:
                            report += f"  ✓ {item}\n"
                    else:
                        if verbose:
                            report += f"  ✗ {item} (failed)\n"

            report += "\n"

        # Generate insights
        report += self._generate_insights()

        # Store complete discovery
        if self.memory:
            self.memory.learn_fact(
                f"Completed system discovery with {total_discoveries} items",
                'system_discovery'
            )

        return report

    def _generate_insights(self) -> str:
        """
        Generate insights from discoveries.

        Returns:
            Insights string
        """
        insights = "**📊 System Insights:**\n\n"

        # Hardware insights
        if 'cpu' in self.discoveries['hardware']:
            cpu_info = self.discoveries['hardware']['cpu']
            if 'AMD' in cpu_info:
                insights += "• AMD processor detected (excellent for multi-threading)\n"
            elif 'Intel' in cpu_info:
                insights += "• Intel processor detected\n"

        if 'gpu' in self.discoveries['hardware']:
            gpu_info = self.discoveries['hardware']['gpu']
            if 'NVIDIA' in gpu_info:
                insights += "• NVIDIA GPU detected (good for CUDA/AI workloads)\n"
            elif 'AMD' in gpu_info:
                insights += "• AMD GPU detected\n"

        # Software insights
        if 'os' in self.discoveries['software']:
            os_info = self.discoveries['software']['os']
            if 'Ubuntu' in os_info:
                insights += "• Ubuntu Linux detected (APT package manager)\n"
            elif 'Fedora' in os_info:
                insights += "• Fedora Linux detected (DNF package manager)\n"
            elif 'Arch' in os_info:
                insights += "• Arch Linux detected (Pacman package manager)\n"

        # Network insights
        if 'interfaces' in self.discoveries['network']:
            interfaces = self.discoveries['network']['interfaces']
            if 'wlan' in interfaces or 'wlp' in interfaces:
                insights += "• WiFi interface available\n"
            if 'docker' in interfaces:
                insights += "• Docker networking detected\n"

        # Service insights
        if 'systemd' in self.discoveries['services']:
            services = self.discoveries['services']['systemd']
            if 'docker' in services:
                insights += "• Docker service running\n"
            if 'ssh' in services:
                insights += "• SSH server running\n"
            if 'mysql' in services or 'postgres' in services:
                insights += "• Database server running\n"

        return insights

    async def quick_discovery(self, category: str) -> str:
        """
        Perform quick discovery of a specific category.

        Args:
            category: Category to discover

        Returns:
            Discovery results
        """
        if category not in self.discovery_commands:
            return f"Unknown category: {category}"

        report = f"**Discovering {category.upper()}:**\n\n"

        for item, command in self.discovery_commands[category].items():
            if self.system_control:
                success, stdout, stderr = self.system_control.run_command(command)
                if success and stdout:
                    report += f"**{item}:**\n```\n{stdout[:500]}\n```\n\n"

        return report

    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get a summary of discovered system information.

        Returns:
            System summary dictionary
        """
        summary = {
            'discovered': len([item for cat in self.discoveries.values() for item in cat]),
            'categories': list(self.discoveries.keys()),
            'key_facts': []
        }

        # Extract key facts
        if 'username' in self.discoveries.get('user_environment', {}):
            summary['key_facts'].append(f"User: {self.discoveries['user_environment']['username']}")

        if 'os' in self.discoveries.get('software', {}):
            os_line = self.discoveries['software']['os'].split('\n')[0]
            summary['key_facts'].append(f"OS: {os_line}")

        if 'cpu' in self.discoveries.get('hardware', {}):
            cpu_line = self.discoveries['hardware']['cpu'].split('\n')[0]
            summary['key_facts'].append(f"CPU: {cpu_line}")

        return summary

    def export_discoveries(self, filepath: str = None) -> str:
        """
        Export all discoveries to a file.

        Args:
            filepath: Where to save (default: ~/phoenix_system_discovery.json)

        Returns:
            Path where saved
        """
        if not filepath:
            filepath = str(Path.home() / 'phoenix_system_discovery.json')

        with open(filepath, 'w') as f:
            json.dump(self.discoveries, f, indent=2)

        self.logger.info(f"Exported discoveries to {filepath}")
        return filepath