#!/usr/bin/env python3
"""
Real Training - Actually executes terminal commands
Uses SimpleExecutor that WORKS
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent / 'local_agi'))

from simple_agent import SimpleExecutor


class RealTrainer:
    """
    Trainer that ACTUALLY teaches terminal execution
    Not text generation - real command execution
    """

    def __init__(self):
        self.executor = SimpleExecutor()
        self.progress_file = Path(__file__).parent / 'progress.json'
        self.progress = self._load_progress()

        print("🎓 Real Trainer Initialized")
        print(f"   Level: {self.progress['current_level']}")
        print(f"   Success Rate: {self.progress['success_rate']:.1%}")

    def _load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "current_level": 1,
            "exercise_num": 0,
            "total_attempts": 0,
            "total_successes": 0,
            "success_rate": 0.0,
            "levels_completed": []
        }

    def _save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def teach_exercise(self, task: str, correct_command: str, validation_fn=None):
        """
        Teach one exercise - ACTUALLY execute the command

        Process:
        1. Show correct command (teach by example)
        2. Execute it (prove it works)
        3. Let AGI try (simple pattern for now)
        4. Validate result
        """
        print(f"\n{'='*60}")
        print(f"📚 TEACHING: {task}")
        print(f"{'='*60}")

        # Step 1: Demonstrate
        print(f"\n💡 Correct approach: {correct_command}")

        # Step 2: Execute to show it works
        print(f"\n🔧 Executing to demonstrate...")
        demo_result = self.executor.terminal.execute(correct_command)

        if demo_result['success']:
            print(f"✅ Success! Output preview:")
            print(f"   {demo_result['stdout'][:150]}")
        else:
            print(f"❌ Error: {demo_result['stderr'][:150]}")
            return False

        # Step 3: Teach the executor this pattern
        self.executor.teach_command(task, correct_command)

        # Step 4: Test if AGI learned it
        print(f"\n📝 Testing if AGI can do it...")
        test_result = self.executor.execute_task(task)

        success = test_result['success']

        # Step 5: Validate
        if validation_fn and success:
            success = validation_fn(test_result)

        # Update progress
        self.progress['total_attempts'] += 1
        if success:
            self.progress['total_successes'] += 1
            print(f"🎉 AGI executed correctly!")
        else:
            print(f"⚠️  AGI needs more practice")

        self.progress['success_rate'] = (
            self.progress['total_successes'] / self.progress['total_attempts']
        )
        self._save_progress()

        return success

    def run_session(self):
        """Run a training session with real exercises"""

        print(f"\n{'🎓'*30}")
        print(f"TRAINING SESSION - Level {self.progress['current_level']}")
        print(f"{'🎓'*30}\n")

        exercises = self._get_level_exercises(self.progress['current_level'])

        results = []
        for i, ex in enumerate(exercises, 1):
            print(f"\n--- Exercise {i}/{len(exercises)} ---")

            success = self.teach_exercise(
                task=ex['task'],
                correct_command=ex['command'],
                validation_fn=ex.get('validate')
            )
            results.append(success)

            time.sleep(2)  # Brief pause

        # Session results
        success_count = sum(results)
        success_rate = success_count / len(results) if results else 0

        print(f"\n{'='*60}")
        print(f"📊 SESSION RESULTS")
        print(f"{'='*60}")
        print(f"   Exercises: {len(results)}")
        print(f"   Successful: {success_count}/{len(results)} ({success_rate:.1%})")
        print(f"   Overall Success Rate: {self.progress['success_rate']:.1%}")

        # Check if level mastered
        if success_rate >= 0.8:
            print(f"\n🎊 LEVEL {self.progress['current_level']} MASTERED!")
            self.progress['levels_completed'].append(self.progress['current_level'])
            self.progress['current_level'] += 1
            self._save_progress()
            print(f"   Advancing to Level {self.progress['current_level']}")

    def _get_level_exercises(self, level: int):
        """Get exercises for a level - REAL exercises with REAL commands"""

        exercises = {
            1: [
                {
                    'task': 'List all files in the current directory',
                    'command': 'ls -la',
                    'validate': lambda r: 'total' in r['stdout']
                },
                {
                    'task': 'Find all Python files',
                    'command': 'find . -name "*.py" -type f',
                    'validate': lambda r: '.py' in r['stdout']
                },
                {
                    'task': 'Show current working directory',
                    'command': 'pwd',
                    'validate': lambda r: '/home/bone' in r['stdout']
                },
                {
                    'task': 'Count lines in README',
                    'command': 'wc -l README_AGI.md',
                    'validate': lambda r: 'README' in r['stdout']
                },
                {
                    'task': 'List all directories',
                    'command': 'ls -d */',
                    'validate': lambda r: len(r['stdout']) > 0
                }
            ],
            2: [
                {
                    'task': 'Find files modified today',
                    'command': 'find . -mtime 0 -type f',
                    'validate': lambda r: r['success']
                },
                {
                    'task': 'Search for "import" in Python files',
                    'command': 'grep -r "import" --include="*.py" | head -5',
                    'validate': lambda r: 'import' in r['stdout']
                },
                {
                    'task': 'Count Python files',
                    'command': 'find . -name "*.py" | wc -l',
                    'validate': lambda r: r['stdout'].strip().isdigit()
                }
            ]
        }

        return exercises.get(level, exercises[1])


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    🔥 REAL TRAINING - Actual Terminal Execution 🔥        ║
║                                                           ║
║  Commands are REALLY executed, not just text              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    trainer = RealTrainer()

    print("\nOptions:")
    print("  1. Run single session (5 exercises)")
    print("  2. Run 5 sessions")
    print("  3. View progress")
    print("  4. Exit")

    choice = input("\nChoice: ").strip()

    if choice == '1':
        trainer.run_session()
    elif choice == '2':
        for i in range(5):
            print(f"\n{'='*70}")
            print(f"SESSION {i+1}/5")
            print(f"{'='*70}")
            trainer.run_session()
            if i < 4:
                print("\nNext session in 10 seconds...")
                time.sleep(10)
    elif choice == '3':
        print("\n📊 Progress:")
        print(json.dumps(trainer.progress, indent=2))
    elif choice == '4':
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
