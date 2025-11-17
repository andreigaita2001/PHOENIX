#!/usr/bin/env python3
"""
Direct AGI Chat Interface
Input your tasks, AGI attempts them autonomously
Claude steps in only when needed
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from training.agi_teacher import AGITeacher


class AGIChat:
    """
    Direct interface to your local AGI
    You input tasks, AGI executes them
    """

    def __init__(self):
        self.teacher = AGITeacher()
        print("\n🔥 Direct AGI Interface Ready")
        print(f"   Autonomy: {self.teacher.progress['autonomy_rate']:.1%}")
        print(f"   Knowledge: {len(self.teacher.knowledge['patterns'])} patterns learned")
        print(f"   Total tasks completed: {self.teacher.progress['total_tasks']}")

    def run(self):
        print("\n" + "="*60)
        print("💬 AGI CHAT - Input your terminal tasks")
        print("="*60)
        print("\nCommands:")
        print("  - Type your task and press Enter")
        print("  - 'status' - Show AGI stats")
        print("  - 'knowledge' - Show learned patterns")
        print("  - 'quit' - Exit\n")

        while True:
            try:
                task = input("📝 Task: ").strip()

                if not task:
                    continue

                if task.lower() == 'quit':
                    print("\n👋 Goodbye!")
                    break

                if task.lower() == 'status':
                    self._show_status()
                    continue

                if task.lower() == 'knowledge':
                    self._show_knowledge()
                    continue

                # Execute the task
                print()
                result = self.teacher.teach_task(task)

                if result == 'autonomous_success':
                    print("\n✨ Your AGI is getting smarter!")
                elif result == 'claude_demonstrated':
                    print("\n📚 AGI learned from demonstration")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def _show_status(self):
        print(f"\n📊 AGI STATUS:")
        print(f"   Autonomy Rate: {self.teacher.progress['autonomy_rate']:.1%}")
        print(f"   Autonomous Tasks: {self.teacher.progress['autonomous_successes']}")
        print(f"   Claude Assists: {self.teacher.progress['claude_assists']}")
        print(f"   Total Tasks: {self.teacher.progress['total_tasks']}")
        print(f"   Knowledge Base: {len(self.teacher.knowledge['patterns'])} patterns")

    def _show_knowledge(self):
        print(f"\n💡 LEARNED PATTERNS ({len(self.teacher.knowledge['patterns'])}):")
        for i, pattern in enumerate(self.teacher.knowledge['patterns'][-10:], 1):
            print(f"\n{i}. {pattern['task']}")
            print(f"   Command: {pattern['command']}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        🔥 PHOENIX - Direct AGI Interface 🔥               ║
║                                                           ║
║  Your local AGI executes tasks autonomously               ║
║  Claude Code assists only when needed                     ║
║  Working toward 100% autonomy                             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    chat = AGIChat()
    chat.run()


if __name__ == "__main__":
    main()
