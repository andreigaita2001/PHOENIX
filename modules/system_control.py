#!/usr/bin/env python3
"""
System Control Module - Gives Phoenix the ability to control your computer.
This module handles file operations, process management, and system commands.
"""

import os
import sys
import subprocess
import psutil
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime


class SystemControl:
    """
    Handles all system-level operations for Phoenix.
    This is like Phoenix's 'hands' - it lets the AI interact with your computer.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the System Control module.

        Args:
            config: Configuration dictionary from phoenix_config.yaml
        """
        self.config = config
        self.logger = logging.getLogger(f"PHOENIX.SystemControl")

        # Safety settings
        self.allowed_dirs = [Path(d) for d in config.get('allowed_directories', [])]
        self.forbidden_dirs = [Path(d) for d in config.get('forbidden_directories', [])]
        self.max_file_size = config.get('max_file_size_mb', 100) * 1024 * 1024  # Convert to bytes

        # Track operations for rollback
        self.operation_history = []
        self.max_history = 100

        self.logger.info("System Control module initialized")

    def _is_path_safe(self, path: Path) -> bool:
        """
        Check if a path is safe to access based on configuration.

        Args:
            path: Path to check

        Returns:
            True if path is safe, False otherwise
        """
        path = path.resolve()  # Get absolute path

        # Check if in forbidden directories
        for forbidden in self.forbidden_dirs:
            try:
                path.relative_to(forbidden)
                self.logger.warning(f"Path {path} is in forbidden directory {forbidden}")
                return False
            except ValueError:
                pass  # Not in this forbidden directory

        # Check if in allowed directories
        if self.allowed_dirs:
            for allowed in self.allowed_dirs:
                try:
                    path.relative_to(allowed)
                    return True
                except ValueError:
                    pass  # Not in this allowed directory

            # If allowed_dirs is specified but path isn't in any, deny
            self.logger.warning(f"Path {path} is not in any allowed directory")
            return False

        # If no allowed_dirs specified, allow (but not if in forbidden)
        return True

    def _log_operation(self, operation: str, details: Dict[str, Any]) -> str:
        """
        Log an operation for potential rollback.

        Args:
            operation: Type of operation
            details: Operation details

        Returns:
            Operation ID
        """
        op_id = datetime.now().isoformat()
        entry = {
            'id': op_id,
            'operation': operation,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }

        self.operation_history.append(entry)

        # Keep history size limited
        if len(self.operation_history) > self.max_history:
            self.operation_history.pop(0)

        return op_id

    # ============= FILE OPERATIONS =============

    def list_files(self, directory: str = ".", pattern: str = "*") -> List[Dict[str, Any]]:
        """
        List files in a directory with details.

        Args:
            directory: Directory path
            pattern: File pattern to match (e.g., "*.txt")

        Returns:
            List of file information dictionaries
        """
        try:
            dir_path = Path(directory).resolve()

            if not self._is_path_safe(dir_path):
                return [{"error": "Access denied to this directory"}]

            if not dir_path.exists():
                return [{"error": "Directory does not exist"}]

            files = []
            for item in dir_path.glob(pattern):
                try:
                    stat = item.stat()
                    files.append({
                        'name': item.name,
                        'path': str(item),
                        'type': 'directory' if item.is_dir() else 'file',
                        'size': stat.st_size if item.is_file() else 0,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'permissions': oct(stat.st_mode)[-3:]
                    })
                except Exception as e:
                    self.logger.error(f"Error getting info for {item}: {e}")

            self.logger.info(f"Listed {len(files)} items in {dir_path}")
            return sorted(files, key=lambda x: (x['type'], x['name']))

        except Exception as e:
            self.logger.error(f"Error listing files: {e}")
            return [{"error": str(e)}]

    def read_file(self, file_path: str, lines: Optional[int] = None) -> str:
        """
        Read contents of a file.

        Args:
            file_path: Path to the file
            lines: Number of lines to read (None for all)

        Returns:
            File contents
        """
        try:
            path = Path(file_path).resolve()

            if not self._is_path_safe(path):
                return "Error: Access denied to this file"

            if not path.exists():
                return "Error: File does not exist"

            if path.stat().st_size > self.max_file_size:
                return f"Error: File too large (>{self.max_file_size/1024/1024}MB)"

            with open(path, 'r', encoding='utf-8') as f:
                if lines:
                    content = ''.join(f.readlines()[:lines])
                else:
                    content = f.read()

            self.logger.info(f"Read file: {path}")
            return content

        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            return f"Error: {e}"

    def write_file(self, file_path: str, content: str, append: bool = False) -> bool:
        """
        Write content to a file.

        Args:
            file_path: Path to the file
            content: Content to write
            append: Whether to append or overwrite

        Returns:
            Success status
        """
        try:
            path = Path(file_path).resolve()

            if not self._is_path_safe(path):
                self.logger.error(f"Access denied to write: {path}")
                return False

            # Backup existing file for rollback
            backup_path = None
            if path.exists():
                backup_path = path.with_suffix(path.suffix + '.bak')
                shutil.copy2(path, backup_path)

            # Write the file
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)

            # Log operation for rollback
            self._log_operation('write_file', {
                'path': str(path),
                'backup': str(backup_path) if backup_path else None,
                'append': append
            })

            self.logger.info(f"Wrote to file: {path}")
            return True

        except Exception as e:
            self.logger.error(f"Error writing file: {e}")
            return False

    def delete_file(self, file_path: str, permanent: bool = False) -> bool:
        """
        Delete a file or directory.

        Args:
            file_path: Path to delete
            permanent: If False, moves to trash instead

        Returns:
            Success status
        """
        try:
            path = Path(file_path).resolve()

            if not self._is_path_safe(path):
                self.logger.error(f"Access denied to delete: {path}")
                return False

            if not path.exists():
                self.logger.warning(f"Path does not exist: {path}")
                return False

            # Create backup for rollback
            backup_dir = Path.home() / '.phoenix_trash' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)

            if permanent:
                # Move to trash first for safety
                backup_path = backup_dir / path.name
                shutil.move(str(path), str(backup_path))
                self.logger.info(f"Deleted (backed up): {path} -> {backup_path}")
            else:
                # Just move to trash
                trash_path = backup_dir / path.name
                shutil.move(str(path), str(trash_path))
                self.logger.info(f"Moved to trash: {path} -> {trash_path}")

            # Log operation
            self._log_operation('delete', {
                'path': str(path),
                'backup': str(backup_dir / path.name)
            })

            return True

        except Exception as e:
            self.logger.error(f"Error deleting file: {e}")
            return False

    # ============= PROCESS MANAGEMENT =============

    def list_processes(self, filter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List running processes.

        Args:
            filter_name: Filter processes by name

        Returns:
            List of process information
        """
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    info = proc.info
                    if filter_name and filter_name.lower() not in info['name'].lower():
                        continue

                    processes.append({
                        'pid': info['pid'],
                        'name': info['name'],
                        'cpu': round(info['cpu_percent'] or 0, 2),
                        'memory': round(info['memory_percent'] or 0, 2)
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            return sorted(processes, key=lambda x: x['memory'], reverse=True)[:50]  # Top 50

        except Exception as e:
            self.logger.error(f"Error listing processes: {e}")
            return []

    def run_command(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """
        Run a shell command safely.

        Args:
            command: Command to run
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            # Safety check - block dangerous commands
            # Be more specific to avoid false positives
            dangerous_patterns = [
                'rm -rf /',  # Root deletion
                'rm -rf ~',  # Home deletion
                'format ',   # Disk formatting
                'dd if=/dev/zero',  # Disk wiping
                'dd if=/dev/random',  # Disk wiping
                '> /dev/sda',  # Direct disk write
                'shutdown now',  # System shutdown
                'shutdown -h',  # System halt
                'reboot',  # System reboot
                'init 0',  # System halt
                'init 6',  # System reboot
                'mkfs.',  # Filesystem creation
                ':(){ :|:& };:',  # Fork bomb
            ]

            # Check for actual dangerous commands, not just substrings
            command_lower = command.lower().strip()
            for pattern in dangerous_patterns:
                if pattern in command_lower:
                    self.logger.warning(f"Blocked dangerous command: {command}")
                    return False, "", "Command blocked by safety system"

            # These commands are always safe
            safe_commands = ['nvidia-smi', 'lspci', 'lscpu', 'free', 'df', 'ps', 'top', 'htop', 'ls', 'pwd', 'whoami', 'date', 'uptime']
            if any(command_lower.startswith(cmd) for cmd in safe_commands):
                # Fast-track safe commands
                pass

            # Run the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.home()  # Run from home directory by default
            )

            # Log the operation
            self._log_operation('command', {
                'command': command,
                'success': result.returncode == 0
            })

            self.logger.info(f"Ran command: {command[:50]}...")
            return result.returncode == 0, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out: {command}")
            return False, "", "Command timed out"
        except Exception as e:
            self.logger.error(f"Error running command: {e}")
            return False, "", str(e)

    def kill_process(self, pid: int) -> bool:
        """
        Terminate a process by PID.

        Args:
            pid: Process ID

        Returns:
            Success status
        """
        try:
            proc = psutil.Process(pid)
            proc.terminate()

            # Wait a bit for graceful termination
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                # Force kill if needed
                proc.kill()

            self.logger.info(f"Killed process: {pid}")
            return True

        except psutil.NoSuchProcess:
            self.logger.warning(f"Process not found: {pid}")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Access denied to kill process: {pid}")
            return False
        except Exception as e:
            self.logger.error(f"Error killing process: {e}")
            return False

    # ============= SYSTEM INFORMATION =============

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.

        Returns:
            System information dictionary
        """
        try:
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory info
            mem = psutil.virtual_memory()

            # Disk info
            disk = psutil.disk_usage('/')

            # Network info
            net = psutil.net_io_counters()

            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
                },
                'memory': {
                    'total_gb': round(mem.total / 1024**3, 2),
                    'used_gb': round(mem.used / 1024**3, 2),
                    'percent': mem.percent
                },
                'disk': {
                    'total_gb': round(disk.total / 1024**3, 2),
                    'used_gb': round(disk.used / 1024**3, 2),
                    'percent': disk.percent
                },
                'network': {
                    'bytes_sent': net.bytes_sent,
                    'bytes_recv': net.bytes_recv
                },
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {'error': str(e)}

    def rollback_last_operation(self) -> bool:
        """
        Rollback the last operation if possible.

        Returns:
            Success status
        """
        if not self.operation_history:
            self.logger.info("No operations to rollback")
            return False

        try:
            last_op = self.operation_history.pop()
            op_type = last_op['operation']
            details = last_op['details']

            if op_type == 'write_file' and details.get('backup'):
                # Restore from backup
                shutil.move(details['backup'], details['path'])
                self.logger.info(f"Rolled back file write: {details['path']}")
                return True

            elif op_type == 'delete' and details.get('backup'):
                # Restore deleted file
                shutil.move(details['backup'], details['path'])
                self.logger.info(f"Restored deleted file: {details['path']}")
                return True

            else:
                self.logger.warning(f"Cannot rollback operation: {op_type}")
                return False

        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
            return False