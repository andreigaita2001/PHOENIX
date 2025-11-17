#!/usr/bin/env python3
"""
PHOENIX Trainer - Uses actual working PHOENIX SystemControl
Teaches PHOENIX terminal mastery through real command execution
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add PHOENIX to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.system_control import SystemControl


class PhoenixTrainer:
    """
    Trainer that uses PHOENIX's working SystemControl
    Actually executes real terminal commands
    """

    def __init__(self):
        # Initialize PHOENIX's SystemControl
        config = {
            'allowed_directories': ['/home/bone'],
            'forbidden_directories': ['/etc', '/sys', '/proc'],
            'max_file_size_mb': 100
        }
        self.system = SystemControl(config)

        self.progress_file = Path(__file__).parent / 'phoenix_progress.json'
        self.progress = self._load_progress()

        print("🔥 PHOENIX Trainer Initialized")
        print(f"   Using PHOENIX's working SystemControl")
        print(f"   Level: {self.progress['current_level']}")
        print(f"   Success Rate: {self.progress['success_rate']:.1%}")

    def _load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "current_level": 1,
            "total_attempts": 0,
            "total_successes": 0,
            "success_rate": 0.0,
            "levels_completed": [],
            "started_at": datetime.now().isoformat()
        }

    def _save_progress(self):
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def teach_exercise(self, task: str, command: str, validation_fn=None):
        """
        Teach one exercise using PHOENIX's real command execution

        Args:
            task: Description of what to do
            command: Correct command to execute
            validation_fn: Function to validate results

        Returns:
            bool: Success status
        """
        print(f"\n{'='*60}")
        print(f"📚 TEACHING: {task}")
        print(f"{'='*60}")

        # Demonstrate correct approach
        print(f"\n💡 Correct command: {command}")

        # Execute using PHOENIX's SystemControl - ACTUAL EXECUTION
        print(f"\n🔧 Executing with PHOENIX SystemControl...")

        try:
            success, stdout, stderr = self.system.run_command(command, timeout=30)

            if success:
                print(f"✅ Success! Output:")
                print(f"   {stdout[:200]}")
            else:
                print(f"❌ Error:")
                print(f"   {stderr[:200]}")
                return False

            # Validate if function provided
            if validation_fn:
                is_valid = validation_fn(stdout)
                if not is_valid:
                    print(f"⚠️  Validation failed")
                    return False

            # Update progress
            self.progress['total_attempts'] += 1
            self.progress['total_successes'] += 1
            self.progress['success_rate'] = (
                self.progress['total_successes'] / self.progress['total_attempts']
            )
            self._save_progress()

            print(f"\n🎉 Exercise completed successfully!")
            return True

        except Exception as e:
            print(f"❌ Exception: {e}")
            self.progress['total_attempts'] += 1
            self._save_progress()
            return False

    def run_session(self, num_exercises=5):
        """Run a training session"""

        print(f"\n{'🔥'*30}")
        print(f"PHOENIX TRAINING SESSION - Level {self.progress['current_level']}")
        print(f"{'🔥'*30}\n")

        exercises = self._get_level_exercises(self.progress['current_level'])
        exercises = exercises[:num_exercises]

        results = []
        for i, ex in enumerate(exercises, 1):
            print(f"\n--- Exercise {i}/{len(exercises)} ---")

            success = self.teach_exercise(
                task=ex['task'],
                command=ex['command'],
                validation_fn=ex.get('validate')
            )
            results.append(success)

            time.sleep(1)

        # Session summary
        success_count = sum(results)
        success_rate = success_count / len(results) if results else 0

        print(f"\n{'='*60}")
        print(f"📊 SESSION RESULTS")
        print(f"{'='*60}")
        print(f"   Exercises: {len(results)}")
        print(f"   Successful: {success_count}/{len(results)} ({success_rate:.1%})")
        print(f"   Overall Success Rate: {self.progress['success_rate']:.1%}")
        print(f"   Total Attempts: {self.progress['total_attempts']}")

        # Level advancement
        if success_rate >= 0.8 and self.progress['success_rate'] >= 0.85:
            print(f"\n🎊 LEVEL {self.progress['current_level']} MASTERED!")
            self.progress['levels_completed'].append(self.progress['current_level'])
            self.progress['current_level'] += 1
            self._save_progress()
            print(f"   Advancing to Level {self.progress['current_level']}")

    def _get_level_exercises(self, level: int):
        """Get exercises for a level"""

        exercises = {
            1: [  # Basic Commands
                {
                    'task': 'List all files in current directory',
                    'command': 'ls -la',
                    'validate': lambda out: 'total' in out or 'drwx' in out
                },
                {
                    'task': 'Show current directory path',
                    'command': 'pwd',
                    'validate': lambda out: '/home/bone' in out
                },
                {
                    'task': 'Find all Python files',
                    'command': 'find /home/bone/PHOENIX -name "*.py" -type f | head -10',
                    'validate': lambda out: '.py' in out
                },
                {
                    'task': 'Count lines in README',
                    'command': 'wc -l /home/bone/PHOENIX/README.md',
                    'validate': lambda out: out.strip().split()[0].isdigit()
                },
                {
                    'task': 'List only directories',
                    'command': 'ls -d /home/bone/PHOENIX/*/',
                    'validate': lambda out: '/' in out
                },
            ],
            2: [  # Text Processing
                {
                    'task': 'Search for "import" in Python files',
                    'command': 'grep -r "import" /home/bone/PHOENIX --include="*.py" | head -5',
                    'validate': lambda out: 'import' in out
                },
                {
                    'task': 'Count Python files',
                    'command': 'find /home/bone/PHOENIX -name "*.py" | wc -l',
                    'validate': lambda out: out.strip().isdigit() and int(out.strip()) > 0
                },
                {
                    'task': 'Find files modified today',
                    'command': 'find /home/bone/PHOENIX -mtime 0 -type f | head -5',
                    'validate': lambda out: True  # Any output is valid
                },
            ]
        }

        return exercises.get(level, exercises[1])


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    🔥 PHOENIX Training - Real Command Execution 🔥        ║
║                                                           ║
║  Uses PHOENIX's working SystemControl module              ║
║  Commands are ACTUALLY executed                           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    trainer = PhoenixTrainer()

    print("\nOptions:")
    print("  1. Run single session (5 exercises)")
    print("  2. Run 5 sessions continuously")
    print("  3. Run 10 sessions (full course)")
    print("  4. View progress")
    print("  5. Exit")

    choice = input("\nChoice: ").strip()

    if choice == '1':
        trainer.run_session(num_exercises=5)
    elif choice == '2':
        for i in range(5):
            print(f"\n{'='*70}")
            print(f"SESSION {i+1}/5")
            print(f"{'='*70}")
            trainer.run_session(num_exercises=5)
            if i < 4:
                time.sleep(5)
    elif choice == '3':
        for i in range(10):
            print(f"\n{'='*70}")
            print(f"SESSION {i+1}/10")
            print(f"{'='*70}")
            trainer.run_session(num_exercises=5)
            if i < 9:
                time.sleep(5)
    elif choice == '4':
        print("\n📊 Progress:")
        print(json.dumps(trainer.progress, indent=2))
    elif choice == '5':
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
