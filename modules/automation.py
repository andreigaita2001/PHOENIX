#!/usr/bin/env python3
"""
Automation Module - Enables PHOENIX to work autonomously in the background.
Monitors, schedules, and executes tasks without user intervention.
"""

import os
import asyncio
import schedule
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
import logging
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class AutomationEngine:
    """
    Enables Phoenix to autonomously monitor and manage your system.
    """

    def __init__(self, config: Dict[str, Any], system_control=None, memory_manager=None):
        """
        Initialize the Automation Engine.

        Args:
            config: Configuration dictionary
            system_control: System control module reference
            memory_manager: Memory manager reference
        """
        self.config = config
        self.system_control = system_control
        self.memory = memory_manager
        self.logger = logging.getLogger("PHOENIX.Automation")

        # Automation state
        self.is_running = False
        self.background_tasks = []
        self.monitors = {}
        self.scheduled_tasks = []
        self.file_watchers = {}

        # Thresholds for automatic actions
        self.thresholds = {
            'cpu_high': config.get('cpu_threshold', 80),
            'memory_high': config.get('memory_threshold', 85),
            'disk_high': config.get('disk_threshold', 90),
            'gpu_temp_high': config.get('gpu_temp_threshold', 80)
        }

        # Autonomous actions
        self.autonomous_actions = config.get('autonomous_actions', {
            'organize_downloads': True,
            'clean_temp_files': True,
            'monitor_resources': True,
            'backup_important': True,
            'optimize_system': True
        })

        self.logger.info("Automation Engine initialized")

    # ============= CORE AUTOMATION =============

    async def start(self):
        """Start all automation systems."""
        self.is_running = True
        self.logger.info("Starting automation engine...")

        # Start system monitors
        if self.autonomous_actions.get('monitor_resources', True):
            asyncio.create_task(self._monitor_system_resources())

        # Start file watchers
        if self.autonomous_actions.get('organize_downloads', True):
            self._start_downloads_organizer()

        # Start scheduled tasks
        self._setup_scheduled_tasks()

        # Start background processor
        asyncio.create_task(self._process_background_tasks())

        self.logger.info("Automation engine started")

    async def stop(self):
        """Stop all automation systems."""
        self.is_running = False

        # Stop file watchers
        for observer in self.file_watchers.values():
            observer.stop()
            observer.join()

        self.logger.info("Automation engine stopped")

    # ============= SYSTEM MONITORING =============

    async def _monitor_system_resources(self):
        """
        Continuously monitor system resources and take action when needed.
        """
        while self.is_running:
            try:
                if not self.system_control:
                    await asyncio.sleep(60)
                    continue

                # Get system info
                sys_info = self.system_control.get_system_info()

                # Check CPU
                cpu_usage = sys_info['cpu']['percent']
                if cpu_usage > self.thresholds['cpu_high']:
                    await self._handle_high_cpu(cpu_usage)

                # Check Memory
                mem_usage = sys_info['memory']['percent']
                if mem_usage > self.thresholds['memory_high']:
                    await self._handle_high_memory(mem_usage)

                # Check Disk
                disk_usage = sys_info['disk']['percent']
                if disk_usage > self.thresholds['disk_high']:
                    await self._handle_high_disk(disk_usage)

                # Check GPU temperature
                await self._check_gpu_temperature()

                # Store monitoring data
                if self.memory:
                    self.memory.learn_fact(
                        f"System resources at {datetime.now()}: CPU {cpu_usage}%, Memory {mem_usage}%",
                        'system_monitoring'
                    )

            except Exception as e:
                self.logger.error(f"Error monitoring system: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _handle_high_cpu(self, usage: float):
        """Handle high CPU usage autonomously."""
        self.logger.warning(f"High CPU usage detected: {usage}%")

        # Find top CPU consuming processes
        processes = self.system_control.list_processes()
        top_cpu = sorted(processes, key=lambda x: x['cpu'], reverse=True)[:5]

        # Log the information
        msg = f"⚠️ High CPU Alert: {usage}%\nTop processes:\n"
        for p in top_cpu:
            msg += f"  • {p['name']} (PID: {p['pid']}): {p['cpu']}%\n"

        # Store in memory for learning
        if self.memory:
            self.memory.learn_fact(msg, 'system_alerts')

        # Take action if configured
        if self.autonomous_actions.get('optimize_system'):
            # Could kill non-essential high CPU processes
            # For now, just log
            self.logger.info("Would optimize system but keeping safe for now")

    async def _handle_high_memory(self, usage: float):
        """Handle high memory usage autonomously."""
        self.logger.warning(f"High memory usage detected: {usage}%")

        if self.autonomous_actions.get('optimize_system'):
            # Clear system caches
            success, _, _ = self.system_control.run_command("sync && echo 1 > /proc/sys/vm/drop_caches")
            if success:
                self.logger.info("Cleared system caches to free memory")

    async def _handle_high_disk(self, usage: float):
        """Handle high disk usage autonomously."""
        self.logger.warning(f"High disk usage detected: {usage}%")

        if self.autonomous_actions.get('clean_temp_files'):
            # Clean temporary files
            temp_dirs = ['/tmp', '/var/tmp', '~/.cache']
            for temp_dir in temp_dirs:
                self._clean_old_files(temp_dir, days=7)

    async def _check_gpu_temperature(self):
        """Check GPU temperature and alert if high."""
        if not self.system_control:
            return

        success, stdout, _ = self.system_control.run_command(
            "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        )

        if success and stdout:
            try:
                temp = float(stdout.strip())
                if temp > self.thresholds['gpu_temp_high']:
                    self.logger.warning(f"High GPU temperature: {temp}°C")
                    # Could adjust fan speeds or reduce GPU load
            except ValueError:
                pass

    # ============= FILE ORGANIZATION =============

    def _start_downloads_organizer(self):
        """Start watching Downloads folder for automatic organization."""
        downloads_path = Path.home() / 'Downloads'
        if not downloads_path.exists():
            return

        class DownloadsHandler(FileSystemEventHandler):
            def __init__(self, automation):
                self.automation = automation

            def on_created(self, event):
                if not event.is_directory:
                    self.automation._organize_file(event.src_path)

        handler = DownloadsHandler(self)
        observer = Observer()
        observer.schedule(handler, str(downloads_path), recursive=False)
        observer.start()

        self.file_watchers['downloads'] = observer
        self.logger.info("Started Downloads folder organizer")

    def _organize_file(self, file_path: str):
        """
        Organize a file based on its type.

        Args:
            file_path: Path to the file to organize
        """
        file = Path(file_path)
        if not file.exists():
            return

        # Determine file category
        extension = file.suffix.lower()
        categories = {
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
            'videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'],
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.odt', '.rtf', '.tex'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'],
            'code': ['.py', '.js', '.html', '.css', '.cpp', '.java', '.go', '.rs']
        }

        target_dir = None
        for category, extensions in categories.items():
            if extension in extensions:
                target_dir = file.parent / category.capitalize()
                break

        if target_dir:
            target_dir.mkdir(exist_ok=True)
            new_path = target_dir / file.name

            # Handle duplicates
            if new_path.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_path = target_dir / f"{file.stem}_{timestamp}{file.suffix}"

            try:
                file.rename(new_path)
                self.logger.info(f"Organized: {file.name} -> {target_dir.name}/")

                # Learn from this
                if self.memory:
                    self.memory.learn_fact(
                        f"Auto-organized {extension} file to {target_dir.name}",
                        'file_organization'
                    )
            except Exception as e:
                self.logger.error(f"Failed to organize {file.name}: {e}")

    # ============= SCHEDULED TASKS =============

    def _setup_scheduled_tasks(self):
        """Set up scheduled automation tasks."""
        # Daily cleanup
        if self.autonomous_actions.get('clean_temp_files'):
            schedule.every().day.at("03:00").do(self._daily_cleanup)

        # Hourly system check
        schedule.every().hour.do(self._hourly_system_check)

        # Weekly backup reminder
        if self.autonomous_actions.get('backup_important'):
            schedule.every().week.do(self._weekly_backup_check)

        # Start scheduler thread
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                threading.Event().wait(60)

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

        self.logger.info("Scheduled tasks configured")

    def _daily_cleanup(self):
        """Perform daily system cleanup."""
        self.logger.info("Running daily cleanup...")

        # Clean old log files
        self._clean_old_files('/var/log', days=30, pattern='*.log.*')

        # Clean package caches
        if self.system_control:
            self.system_control.run_command("apt-get autoclean -y")

        # Clear thumbnail caches
        cache_dir = Path.home() / '.cache' / 'thumbnails'
        if cache_dir.exists():
            self._clean_old_files(str(cache_dir), days=30)

        self.logger.info("Daily cleanup completed")

    def _hourly_system_check(self):
        """Perform hourly system health check."""
        if not self.system_control:
            return

        sys_info = self.system_control.get_system_info()

        # Create health report
        health = {
            'timestamp': datetime.now().isoformat(),
            'cpu': sys_info['cpu']['percent'],
            'memory': sys_info['memory']['percent'],
            'disk': sys_info['disk']['percent'],
            'status': 'healthy'
        }

        # Determine overall health
        if health['cpu'] > 90 or health['memory'] > 90 or health['disk'] > 95:
            health['status'] = 'critical'
        elif health['cpu'] > 70 or health['memory'] > 80 or health['disk'] > 85:
            health['status'] = 'warning'

        # Store health data
        if self.memory:
            self.memory.learn_fact(
                f"System health: {health['status']} (CPU: {health['cpu']}%, Mem: {health['memory']}%)",
                'system_health'
            )

    def _weekly_backup_check(self):
        """Check if important files need backing up."""
        important_dirs = [
            Path.home() / 'Documents',
            Path.home() / 'Projects',
            Path.home() / '.config'
        ]

        for dir_path in important_dirs:
            if dir_path.exists():
                # Check last modification time
                # In real implementation, would trigger actual backup
                self.logger.info(f"Backup reminder: Check {dir_path}")

    def _clean_old_files(self, directory: str, days: int = 7, pattern: str = '*'):
        """
        Clean files older than specified days.

        Args:
            directory: Directory to clean
            days: Age threshold in days
            pattern: File pattern to match
        """
        dir_path = Path(directory).expanduser()
        if not dir_path.exists():
            return

        cutoff_time = datetime.now() - timedelta(days=days)

        for file in dir_path.glob(pattern):
            if file.is_file():
                try:
                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    if mtime < cutoff_time:
                        file.unlink()
                        self.logger.debug(f"Deleted old file: {file}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {file}: {e}")

    # ============= BACKGROUND TASKS =============

    async def _process_background_tasks(self):
        """Process queued background tasks."""
        while self.is_running:
            if self.background_tasks:
                task = self.background_tasks.pop(0)
                try:
                    await task()
                except Exception as e:
                    self.logger.error(f"Background task failed: {e}")

            await asyncio.sleep(5)

    def add_background_task(self, task: Callable):
        """
        Add a task to run in the background.

        Args:
            task: Async function to run
        """
        self.background_tasks.append(task)

    # ============= AUTOMATION RULES =============

    def add_automation_rule(self, trigger: str, action: Callable, conditions: Dict = None):
        """
        Add a custom automation rule.

        Args:
            trigger: Event that triggers the action
            action: Function to execute
            conditions: Optional conditions for the trigger
        """
        rule = {
            'trigger': trigger,
            'action': action,
            'conditions': conditions or {},
            'created': datetime.now().isoformat()
        }

        # Store rule for persistence
        # In full implementation, would save to disk

        self.logger.info(f"Added automation rule: {trigger}")

    def get_automation_status(self) -> Dict[str, Any]:
        """
        Get current automation status.

        Returns:
            Status dictionary
        """
        return {
            'is_running': self.is_running,
            'active_monitors': list(self.monitors.keys()),
            'file_watchers': list(self.file_watchers.keys()),
            'scheduled_tasks': len(self.scheduled_tasks),
            'background_tasks': len(self.background_tasks),
            'autonomous_actions': self.autonomous_actions
        }