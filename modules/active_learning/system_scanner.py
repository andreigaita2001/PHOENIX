#!/usr/bin/env python3
"""
System Scanner Module - Autonomously explores and learns about the system.
Maps system architecture, tools, projects, and resources.
"""

import os
import json
import subprocess
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import hashlib
import psutil
import platform


class SystemScanner:
    """
    Autonomously scans and learns about the system environment.
    """

    def __init__(self, memory_manager=None, multi_model=None):
        """
        Initialize the System Scanner.

        Args:
            memory_manager: Memory manager for storing discoveries
            multi_model: Multi-model coordinator for intelligent analysis
        """
        self.memory_manager = memory_manager
        self.multi_model = multi_model
        self.logger = logging.getLogger("PHOENIX.SystemScanner")

        # System knowledge base
        self.system_map = {
            'os_info': {},
            'hardware': {},
            'filesystem': {},
            'installed_software': {},
            'development_tools': {},
            'projects': {},
            'scripts': {},
            'configurations': {},
            'services': {},
            'network': {},
            'user_preferences': {}
        }

        # Scan configuration
        self.scan_config = {
            'max_depth': 5,
            'exclude_dirs': ['/proc', '/sys', '/dev', '/tmp', '/.snapshots'],
            'project_indicators': ['.git', 'package.json', 'Cargo.toml', 'pom.xml',
                                 'requirements.txt', 'setup.py', 'Makefile'],
            'config_files': ['.bashrc', '.zshrc', '.vimrc', '.gitconfig', '.ssh/config'],
            'interesting_extensions': ['.py', '.js', '.sh', '.rs', '.go', '.cpp', '.java']
        }

        # Scan history to avoid redundancy
        self.scan_history = set()
        self.discoveries = []

    async def full_system_scan(self) -> Dict[str, Any]:
        """
        Perform a comprehensive system scan.

        Returns:
            Complete system map
        """
        self.logger.info("Starting full system scan...")

        # Scan different aspects in parallel
        tasks = [
            self._scan_os_info(),
            self._scan_hardware(),
            self._scan_filesystem(),
            self._scan_installed_software(),
            self._scan_development_tools(),
            self._scan_user_environment(),
            self._scan_services(),
            self._scan_network()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Scan task {i} failed: {result}")

        # Intelligent analysis of findings
        if self.multi_model:
            await self._analyze_findings()

        # Store in memory if available
        if self.memory_manager:
            self._store_discoveries()

        self.logger.info(f"System scan complete. Found {len(self.discoveries)} interesting items")

        return self.system_map

    async def _scan_os_info(self):
        """Scan operating system information."""
        self.logger.info("Scanning OS information...")

        self.system_map['os_info'] = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'hostname': platform.node()
        }

        # Get distribution info for Linux
        if platform.system() == 'Linux':
            try:
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if line.startswith('PRETTY_NAME='):
                            self.system_map['os_info']['distribution'] = line.split('=')[1].strip().strip('"')
                            break
            except:
                pass

        self._add_discovery('os_info', 'System information collected', self.system_map['os_info'])

    async def _scan_hardware(self):
        """Scan hardware configuration."""
        self.logger.info("Scanning hardware...")

        hardware = {
            'cpu': {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'usage': psutil.cpu_percent(interval=1)
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'swap_total': psutil.swap_memory().total,
                'swap_used': psutil.swap_memory().used
            },
            'disks': []
        }

        # Disk information
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                hardware['disks'].append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent
                })
            except:
                pass

        # GPU information
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                hardware['gpu'] = result.stdout.strip()
                self._add_discovery('hardware', 'GPU detected', {'gpu': hardware['gpu']})
        except:
            pass

        self.system_map['hardware'] = hardware
        self._add_discovery('hardware', 'Hardware configuration scanned', hardware)

    async def _scan_filesystem(self):
        """Scan filesystem structure and find interesting locations."""
        self.logger.info("Scanning filesystem structure...")

        home_dir = Path.home()
        filesystem_info = {
            'home_directory': str(home_dir),
            'important_directories': {},
            'project_directories': [],
            'script_collections': [],
            'config_locations': []
        }

        # Scan home directory for important locations
        important_dirs = ['Documents', 'Projects', 'Code', 'Scripts', 'Work',
                         'Development', 'GitHub', 'GitLab', 'Workspace']

        for dir_name in important_dirs:
            dir_path = home_dir / dir_name
            if dir_path.exists() and dir_path.is_dir():
                filesystem_info['important_directories'][dir_name] = str(dir_path)
                # Scan for projects
                await self._scan_for_projects(dir_path, filesystem_info['project_directories'])

        # Find configuration files
        for config_file in self.scan_config['config_files']:
            config_path = home_dir / config_file
            if config_path.exists():
                filesystem_info['config_locations'].append(str(config_path))
                self._add_discovery('configuration', f'Found {config_file}', {'path': str(config_path)})

        self.system_map['filesystem'] = filesystem_info

    async def _scan_for_projects(self, directory: Path, project_list: List, depth: int = 0):
        """
        Recursively scan for project directories.

        Args:
            directory: Directory to scan
            project_list: List to append projects to
            depth: Current recursion depth
        """
        if depth > self.scan_config['max_depth']:
            return

        if str(directory) in self.scan_history:
            return

        self.scan_history.add(str(directory))

        try:
            for item in directory.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Check for project indicators
                    for indicator in self.scan_config['project_indicators']:
                        if (item / indicator).exists():
                            project_info = {
                                'path': str(item),
                                'name': item.name,
                                'type': self._detect_project_type(item),
                                'last_modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                            }
                            project_list.append(project_info)
                            self._add_discovery('project', f'Found project: {item.name}', project_info)
                            break
                    else:
                        # Recurse if not a project
                        await self._scan_for_projects(item, project_list, depth + 1)
        except PermissionError:
            pass

    def _detect_project_type(self, project_path: Path) -> str:
        """Detect the type of project."""
        if (project_path / 'package.json').exists():
            return 'nodejs'
        elif (project_path / 'requirements.txt').exists() or (project_path / 'setup.py').exists():
            return 'python'
        elif (project_path / 'Cargo.toml').exists():
            return 'rust'
        elif (project_path / 'go.mod').exists():
            return 'go'
        elif (project_path / 'pom.xml').exists():
            return 'java'
        elif (project_path / '.git').exists():
            return 'git'
        return 'unknown'

    async def _scan_installed_software(self):
        """Scan for installed software and packages."""
        self.logger.info("Scanning installed software...")

        software = {
            'package_managers': {},
            'development_tools': {},
            'system_packages': []
        }

        # Check for package managers
        package_managers = {
            'apt': 'dpkg -l',
            'yum': 'yum list installed',
            'pacman': 'pacman -Q',
            'brew': 'brew list',
            'snap': 'snap list',
            'flatpak': 'flatpak list'
        }

        for pm, cmd in package_managers.items():
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    software['package_managers'][pm] = 'available'
                    # Parse first few packages as sample
                    lines = result.stdout.split('\n')[:10]
                    software['system_packages'].extend(lines)
            except:
                pass

        # Check for development tools
        dev_tools = {
            'python': 'python3 --version',
            'node': 'node --version',
            'npm': 'npm --version',
            'cargo': 'cargo --version',
            'go': 'go version',
            'docker': 'docker --version',
            'git': 'git --version',
            'vim': 'vim --version',
            'code': 'code --version'
        }

        for tool, cmd in dev_tools.items():
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    software['development_tools'][tool] = version
                    self._add_discovery('tool', f'{tool} available', {'version': version})
            except:
                pass

        self.system_map['installed_software'] = software

    async def _scan_development_tools(self):
        """Deep scan of development environment."""
        self.logger.info("Scanning development environment...")

        dev_env = {
            'python': {},
            'node': {},
            'databases': {},
            'containers': {},
            'editors': {}
        }

        # Python environment
        try:
            # Check for virtual environments
            result = subprocess.run(['pip3', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                packages = result.stdout.split('\n')[:20]  # First 20 packages
                dev_env['python']['packages'] = packages

            # Check for conda
            result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                dev_env['python']['conda_envs'] = result.stdout
        except:
            pass

        # Node environment
        try:
            result = subprocess.run(['npm', 'list', '-g', '--depth=0'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                dev_env['node']['global_packages'] = result.stdout
        except:
            pass

        # Docker
        try:
            result = subprocess.run(['docker', 'ps', '-a'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                dev_env['containers']['docker'] = 'active'
                # Count containers
                containers = len(result.stdout.split('\n')) - 1
                if containers > 0:
                    self._add_discovery('docker', f'Found {containers} Docker containers', {})
        except:
            pass

        self.system_map['development_tools'] = dev_env

    async def _scan_user_environment(self):
        """Scan user environment and preferences."""
        self.logger.info("Scanning user environment...")

        env_info = {
            'shell': os.environ.get('SHELL', 'unknown'),
            'user': os.environ.get('USER', 'unknown'),
            'home': os.environ.get('HOME', 'unknown'),
            'path': os.environ.get('PATH', '').split(':')[:10],  # First 10 PATH entries
            'important_vars': {}
        }

        # Important environment variables
        important_vars = ['EDITOR', 'BROWSER', 'TERM', 'LANG', 'DISPLAY',
                         'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV']

        for var in important_vars:
            if var in os.environ:
                env_info['important_vars'][var] = os.environ[var]

        # Check shell history for patterns
        shell_history = await self._analyze_shell_history()
        if shell_history:
            env_info['common_commands'] = shell_history

        self.system_map['user_preferences'] = env_info

    async def _analyze_shell_history(self) -> Optional[List[str]]:
        """Analyze shell history for common patterns."""
        history_files = ['.bash_history', '.zsh_history', '.history']
        home = Path.home()

        for hist_file in history_files:
            hist_path = home / hist_file
            if hist_path.exists():
                try:
                    with open(hist_path, 'r', errors='ignore') as f:
                        lines = f.readlines()[-100:]  # Last 100 commands

                    # Find common commands (simplified)
                    from collections import Counter
                    commands = [line.split()[0] if line.split() else '' for line in lines]
                    common = Counter(commands).most_common(10)

                    return [cmd for cmd, _ in common if cmd]
                except:
                    pass

        return None

    async def _scan_services(self):
        """Scan running services and processes."""
        self.logger.info("Scanning services...")

        services = {
            'running_services': [],
            'listening_ports': [],
            'top_processes': []
        }

        # Get top processes by CPU and memory
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except:
                pass

        # Sort by CPU usage
        processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
        services['top_processes'] = processes[:10]

        # Check systemd services (Linux)
        if platform.system() == 'Linux':
            try:
                result = subprocess.run(['systemctl', 'list-units', '--state=running', '--no-legend'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    services['running_services'] = result.stdout.split('\n')[:20]
            except:
                pass

        # Check listening ports
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'LISTEN':
                services['listening_ports'].append({
                    'port': conn.laddr.port,
                    'address': conn.laddr.ip
                })

        self.system_map['services'] = services

    async def _scan_network(self):
        """Scan network configuration."""
        self.logger.info("Scanning network...")

        network = {
            'interfaces': [],
            'hostname': platform.node(),
            'connections': []
        }

        # Network interfaces
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == 2:  # IPv4
                    network['interfaces'].append({
                        'name': interface,
                        'ip': addr.address,
                        'netmask': addr.netmask
                    })

        self.system_map['network'] = network

    async def _analyze_findings(self):
        """Use multi-model AI to analyze findings."""
        if not self.multi_model:
            return

        self.logger.info("Analyzing findings with AI...")

        # Prepare summary for analysis
        summary = {
            'os': self.system_map['os_info'].get('distribution', 'Unknown'),
            'projects_found': len(self.system_map['filesystem'].get('project_directories', [])),
            'dev_tools': list(self.system_map['installed_software'].get('development_tools', {}).keys()),
            'services': len(self.system_map['services'].get('running_services', []))
        }

        prompt = f"""Analyze this system scan summary and provide insights:

System: {summary['os']}
Projects found: {summary['projects_found']}
Development tools: {', '.join(summary['dev_tools'])}
Running services: {summary['services']}

Provide:
1. What type of developer/user this appears to be
2. Suggested optimizations for their workflow
3. Potential areas PHOENIX should focus on learning"""

        try:
            response = await self.multi_model.query(
                prompt,
                task_type='analysis'
            )

            if response.get('success'):
                self._add_discovery('ai_analysis', 'System profile', {'analysis': response['response']})
        except:
            pass

    def _add_discovery(self, category: str, description: str, data: Dict):
        """Add a discovery to the list."""
        discovery = {
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'description': description,
            'data': data,
            'id': hashlib.md5(f"{category}{description}{datetime.now()}".encode()).hexdigest()[:8]
        }
        self.discoveries.append(discovery)

    def _store_discoveries(self):
        """Store discoveries in memory manager."""
        if not self.memory_manager:
            return

        for discovery in self.discoveries:
            self.memory_manager.learn_fact(
                discovery['description'],
                f"system_{discovery['category']}"
            )

    def get_system_summary(self) -> str:
        """Get a human-readable system summary."""
        summary = f"""
System Summary
==============
OS: {self.system_map['os_info'].get('distribution', 'Unknown')}
CPU: {self.system_map['hardware'].get('cpu', {}).get('logical_cores', 'Unknown')} cores
RAM: {self.system_map['hardware'].get('memory', {}).get('total', 0) / (1024**3):.1f} GB
Projects: {len(self.system_map['filesystem'].get('project_directories', []))}
Dev Tools: {', '.join(self.system_map['installed_software'].get('development_tools', {}).keys())}

Key Discoveries:
"""
        for discovery in self.discoveries[:10]:
            summary += f"- {discovery['description']}\n"

        return summary

    async def incremental_scan(self, focus_area: str = None) -> Dict:
        """
        Perform incremental scan of specific area.

        Args:
            focus_area: Specific area to scan

        Returns:
            Scan results
        """
        self.logger.info(f"Incremental scan: {focus_area or 'general'}")

        if focus_area == 'projects':
            await self._scan_filesystem()
        elif focus_area == 'tools':
            await self._scan_development_tools()
        elif focus_area == 'services':
            await self._scan_services()
        else:
            # Quick scan of changes
            await self._scan_os_info()
            await self._scan_services()

        return self.system_map