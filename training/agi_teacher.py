#!/usr/bin/env python3
"""
AGI Teacher - Teaches local AGI (Qwen) terminal mastery
Combines PHOENIX's working execution with local AGI learning
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Add PHOENIX to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.system_control import SystemControl


class AGITeacher:
    """
    Teaches local AGI by:
    1. Showing correct command (demonstrate)
    2. Having AGI attempt it
    3. Comparing results
    4. Providing feedback
    5. Storing successful patterns
    """

    def __init__(self, agi_model: str = "qwen2.5:32b"):
        # PHOENIX's working SystemControl
        config = {
            'allowed_directories': ['/home/bone'],
            'forbidden_directories': ['/etc', '/sys', '/proc'],
            'max_file_size_mb': 100
        }
        self.system = SystemControl(config)

        # Local AGI model
        self.agi_model = agi_model

        # Learning storage
        self.knowledge_file = Path(__file__).parent / 'agi_knowledge.json'
        self.knowledge = self._load_knowledge()

        # Progress tracking
        self.progress_file = Path(__file__).parent / 'agi_progress.json'
        self.progress = self._load_progress()

        print("🎓 AGI Teacher Initialized")
        print(f"   Using PHOENIX SystemControl (proven working)")
        print(f"   Teaching: {agi_model}")
        print(f"   Knowledge base: {len(self.knowledge.get('patterns', []))} patterns")
        print(f"   Autonomy: {self.progress['autonomy_rate']:.1%}")

    def _load_knowledge(self):
        if self.knowledge_file.exists():
            with open(self.knowledge_file, 'r') as f:
                return json.load(f)
        return {
            "patterns": [],  # Successful command patterns
            "task_mappings": {},  # Task description -> command
            "learned_at": datetime.now().isoformat()
        }

    def _save_knowledge(self):
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.knowledge, f, indent=2)

    def _load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "current_level": 1,
            "total_tasks": 0,
            "autonomous_successes": 0,
            "claude_assists": 0,
            "autonomy_rate": 0.0,
            "started_at": datetime.now().isoformat()
        }

    def _save_progress(self):
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def ask_agi(self, prompt: str) -> str:
        """
        Ask local AGI for a command
        Uses Ollama with the local model
        """
        try:
            # Call local AGI
            result = subprocess.run(
                ["ollama", "run", self.agi_model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"ERROR: {result.stderr}"

        except Exception as e:
            return f"ERROR: {str(e)}"

    def extract_command(self, agi_response: str) -> str:
        """
        Extract actual command from AGI's response
        AGI might explain things, we need just the command
        """
        lines = agi_response.strip().split('\n')

        # Look for code blocks
        if '```' in agi_response:
            in_code = False
            for line in lines:
                if '```' in line:
                    in_code = not in_code
                    continue
                if in_code and line.strip():
                    return line.strip()

        # Look for lines starting with common commands
        commands = ['ls', 'find', 'grep', 'cat', 'cd', 'pwd', 'wc', 'echo']
        for line in lines:
            line = line.strip()
            if any(line.startswith(cmd) for cmd in commands):
                return line

        # If single line, assume it's the command
        if len(lines) == 1:
            return lines[0].strip()

        return None

    def teach_task(self, task: str, correct_command: str = None):
        """
        Teach AGI to complete a task

        Process:
        1. Ask AGI how to do it
        2. Execute AGI's attempt
        3. Show correct way if AGI failed
        4. Store successful pattern
        """
        print(f"\n{'='*60}")
        print(f"📚 TASK: {task}")
        print(f"{'='*60}")

        self.progress['total_tasks'] += 1

        # Step 1: Ask AGI
        print(f"\n🤖 Asking local AGI...")
        agi_prompt = f"""Task: {task}

Respond with ONLY the terminal command to accomplish this task.
Do not explain, just give the command.

Command:"""

        agi_response = self.ask_agi(agi_prompt)
        print(f"   AGI responded: {agi_response[:200]}")

        # Step 2: Extract command
        agi_command = self.extract_command(agi_response)

        if agi_command:
            print(f"\n🔍 Extracted command: {agi_command}")

            # Step 3: Try AGI's command
            print(f"\n⚡ Testing AGI's command...")
            success, stdout, stderr = self.system.run_command(agi_command, timeout=30)

            if success and stdout:
                print(f"✅ AGI succeeded autonomously!")
                print(f"   Output: {stdout[:150]}")

                # AGI did it!
                self.progress['autonomous_successes'] += 1

                # Store this successful pattern
                self._store_pattern(task, agi_command, stdout)

                result = "autonomous_success"
            else:
                print(f"❌ AGI's command failed")
                if stderr:
                    print(f"   Error: {stderr[:150]}")
                result = "agi_failed"
        else:
            print(f"⚠️  Could not extract command from AGI response")
            result = "extraction_failed"

        # Step 4: If AGI failed, demonstrate correct way
        if result != "autonomous_success" and correct_command:
            print(f"\n💡 Demonstrating correct approach...")
            print(f"   Command: {correct_command}")

            success, stdout, stderr = self.system.run_command(correct_command, timeout=30)

            if success:
                print(f"✅ Correct execution succeeded")
                print(f"   Output: {stdout[:150]}")

                # Store for AGI to learn from
                self._store_pattern(task, correct_command, stdout)

                self.progress['claude_assists'] += 1
                result = "claude_demonstrated"
            else:
                print(f"❌ Error: {stderr[:150]}")
                result = "failed"

        # Update autonomy rate
        if self.progress['total_tasks'] > 0:
            self.progress['autonomy_rate'] = (
                self.progress['autonomous_successes'] /
                self.progress['total_tasks']
            )

        self._save_progress()

        print(f"\n📊 Autonomy Rate: {self.progress['autonomy_rate']:.1%} ({self.progress['autonomous_successes']}/{self.progress['total_tasks']})")

        return result

    def _store_pattern(self, task: str, command: str, output: str):
        """Store successful pattern for AGI to learn from"""
        pattern = {
            "task": task,
            "command": command,
            "output_sample": output[:200],
            "learned_at": datetime.now().isoformat()
        }

        self.knowledge['patterns'].append(pattern)
        self.knowledge['task_mappings'][task] = command
        self._save_knowledge()

        print(f"   💾 Pattern stored in knowledge base")

    def run_session(self, exercises: list):
        """Run a teaching session with multiple exercises"""
        print(f"\n{'🎓'*30}")
        print(f"AGI TEACHING SESSION")
        print(f"{'🎓'*30}\n")

        results = []
        for i, ex in enumerate(exercises, 1):
            print(f"\n--- Exercise {i}/{len(exercises)} ---")

            result = self.teach_task(
                task=ex['task'],
                correct_command=ex.get('command')
            )
            results.append(result)

            time.sleep(2)

        # Session summary
        autonomous = results.count('autonomous_success')
        assisted = results.count('claude_demonstrated')
        failed = len(results) - autonomous - assisted

        print(f"\n{'='*60}")
        print(f"📊 SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"   Total Tasks: {len(results)}")
        print(f"   AGI Autonomous: {autonomous} ({autonomous/len(results)*100:.1f}%)")
        print(f"   Claude Assisted: {assisted}")
        print(f"   Failed: {failed}")
        print(f"\n   Overall Autonomy: {self.progress['autonomy_rate']:.1%}")
        print(f"   Progress: {self.progress['autonomous_successes']}/{self.progress['total_tasks']} tasks")


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    🎓 AGI Teacher - Train Local AGI with PHOENIX 🎓       ║
║                                                           ║
║  Local AGI attempts tasks autonomously                    ║
║  PHOENIX demonstrates when AGI fails                      ║
║  Builds toward 100% autonomy                              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    teacher = AGITeacher()

    # Level 1 exercises - Basic Commands
    exercises = [
        {
            'task': 'List all files in the current directory with details',
            'command': 'ls -la'
        },
        {
            'task': 'Show the current working directory path',
            'command': 'pwd'
        },
        {
            'task': 'Find all Python files in /home/bone/PHOENIX',
            'command': 'find /home/bone/PHOENIX -name "*.py" -type f | head -10'
        },
        {
            'task': 'Count the number of lines in /home/bone/PHOENIX/README.md',
            'command': 'wc -l /home/bone/PHOENIX/README.md'
        },
        {
            'task': 'List only directories in /home/bone/PHOENIX',
            'command': 'ls -d /home/bone/PHOENIX/*/'
        }
    ]

    print("\nStarting teaching session...")
    print(f"AGI will attempt each task autonomously")
    print(f"PHOENIX will demonstrate correct approach if AGI fails\n")

    input("Press Enter to begin training...")

    teacher.run_session(exercises)

    print("\n✨ Training session complete!")
    print(f"Knowledge base now has {len(teacher.knowledge['patterns'])} patterns")


if __name__ == "__main__":
    main()
