#!/usr/bin/env python3
"""
Claude as Teacher - Active Training System

I (Claude) actively teach the local AGI through:
- Demonstrating tasks
- Evaluating attempts
- Providing feedback
- Correcting mistakes
- Tracking progress
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.append(str(Path(__file__).parent.parent / 'local_agi'))

from core.agent import AutonomousAgent
from learning.pattern_learner import PatternLearner
from knowledge.vector_store import VectorKnowledgeBase
from monitoring.logger import ActivityLogger


class TeachingSession:
    """A single teaching session with the AGI"""

    def __init__(self, level: int, exercise: dict):
        self.level = level
        self.exercise = exercise
        self.attempts = []
        self.start_time = time.time()
        self.completed = False
        self.success = False

    def record_attempt(self, attempt: dict):
        """Record an attempt by the AGI"""
        self.attempts.append({
            'timestamp': time.time(),
            'command': attempt.get('command'),
            'output': attempt.get('output'),
            'success': attempt.get('success'),
            'feedback': attempt.get('feedback')
        })

    def mark_complete(self, success: bool):
        """Mark session as complete"""
        self.completed = True
        self.success = success
        self.end_time = time.time()

    def get_duration(self):
        """Get session duration"""
        if hasattr(self, 'end_time'):
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self):
        """Export to dict"""
        return {
            'level': self.level,
            'exercise': self.exercise,
            'attempts': self.attempts,
            'completed': self.completed,
            'success': self.success,
            'duration': self.get_duration(),
            'num_attempts': len(self.attempts)
        }


class AITeacher:
    """
    Claude acting as teacher for the local AGI

    Responsibilities:
    - Load curriculum
    - Run teaching sessions
    - Demonstrate correct approaches
    - Evaluate AGI attempts
    - Provide detailed feedback
    - Track progress over time
    """

    def __init__(self):
        self.curriculum = self._load_curriculum()
        self.agent = AutonomousAgent()
        self.learner = PatternLearner()
        self.knowledge = VectorKnowledgeBase()
        self.activity_logger = ActivityLogger()

        # Training state
        self.current_level = 1
        self.sessions = []
        self.progress = self._load_progress()

        print("🎓 Claude Teacher Initialized")
        print(f"   Current Level: {self.current_level}")
        print(f"   Total Levels: {len(self.curriculum['levels'])}")

    def _load_curriculum(self):
        """Load training curriculum"""
        curriculum_file = Path(__file__).parent / 'curriculum.json'
        with open(curriculum_file, 'r') as f:
            return json.load(f)

    def _load_progress(self):
        """Load training progress"""
        progress_file = Path(__file__).parent / 'progress.json'
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return json.load(f)
        return {
            'current_level': 1,
            'completed_levels': [],
            'total_sessions': 0,
            'total_attempts': 0,
            'success_rate': 0.0,
            'skills_mastered': []
        }

    def _save_progress(self):
        """Save training progress"""
        progress_file = Path(__file__).parent / 'progress.json'
        with open(progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def _save_session(self, session: TeachingSession):
        """Save session to history"""
        sessions_file = Path(__file__).parent / 'sessions.jsonl'
        with open(sessions_file, 'a') as f:
            f.write(json.dumps(session.to_dict()) + '\n')

    def demonstrate(self, task: str, correct_command: str, explanation: str):
        """
        Demonstrate the correct way to do a task

        This is Claude showing the AGI how it should be done
        """
        print(f"\n📚 DEMONSTRATION")
        print(f"   Task: {task}")
        print(f"   Correct Command: {correct_command}")
        print(f"   Explanation: {explanation}")

        # Store demonstration in knowledge base
        self.knowledge.add(
            text=f"Task: {task}\nCommand: {correct_command}\nExplanation: {explanation}",
            metadata={
                'type': 'demonstration',
                'task': task,
                'teacher': 'claude'
            }
        )

        return {
            'task': task,
            'command': correct_command,
            'explanation': explanation
        }

    def evaluate_attempt(self, task: str, agi_command: str, agi_output: dict) -> dict:
        """
        Evaluate AGI's attempt at a task

        Returns detailed feedback
        """
        print(f"\n🔍 EVALUATING AGI ATTEMPT")
        print(f"   Task: {task}")
        print(f"   AGI's Command: {agi_command}")
        print(f"   Success: {agi_output.get('success', False)}")

        # Search for similar demonstrations
        similar = self.knowledge.find_similar_tasks(task, limit=1)

        feedback = {
            'success': agi_output.get('success', False),
            'command': agi_command,
            'output': agi_output,
            'feedback': '',
            'suggestions': []
        }

        if feedback['success']:
            feedback['feedback'] = "✅ Excellent! Command executed correctly."

            # Check if optimal
            if similar and 'command' in str(similar[0]):
                # Compare with demonstration
                feedback['suggestions'].append("Consider the demonstrated approach for future reference")
        else:
            feedback['feedback'] = "❌ Command failed. Let me help you understand why."

            if similar:
                feedback['suggestions'].append(f"Review the demonstration for this task")
                feedback['suggestions'].append(f"Common mistakes to avoid...")

            # Specific error analysis
            if 'stderr' in agi_output:
                feedback['suggestions'].append(f"Error: {agi_output['stderr'][:100]}")

        return feedback

    def teach_exercise(self, exercise: dict) -> TeachingSession:
        """
        Teach a single exercise through active feedback loop

        Process:
        1. Demonstrate correct approach
        2. AGI attempts task
        3. Evaluate attempt
        4. Provide feedback
        5. Repeat until mastery (or max attempts)
        """
        print(f"\n" + "="*60)
        print(f"🎯 TEACHING SESSION: {exercise['description']}")
        print(f"="*60)

        session = TeachingSession(self.current_level, exercise)

        # Step 1: Demonstrate
        demo = self.demonstrate(
            task=exercise['description'],
            correct_command=exercise['correct_command'],
            explanation=exercise.get('explanation', '')
        )

        max_attempts = 3
        attempt_num = 0

        while attempt_num < max_attempts:
            attempt_num += 1
            print(f"\n📝 Attempt {attempt_num}/{max_attempts}")

            # Step 2: AGI attempts the task
            print("   AGI is attempting the task...")

            # Create prompt for AGI
            prompt = f"""
Task: {exercise['description']}

I demonstrated: {demo['command']}
Explanation: {demo['explanation']}

Now you try. What command would you use?
"""

            result = self.agent.process_task(prompt)

            # Extract command from result (simplified)
            # In production, parse properly
            agi_command = exercise['correct_command']  # Placeholder
            agi_output = result

            # Step 3: Evaluate
            evaluation = self.evaluate_attempt(
                task=exercise['description'],
                agi_command=agi_command,
                agi_output=agi_output
            )

            # Step 4: Record attempt
            session.record_attempt(evaluation)

            # Step 5: Provide feedback
            print(f"\n💬 FEEDBACK:")
            print(f"   {evaluation['feedback']}")
            for suggestion in evaluation['suggestions']:
                print(f"   💡 {suggestion}")

            # Check if successful
            if evaluation['success']:
                print(f"\n🎉 SUCCESS! Mastered in {attempt_num} attempts")
                session.mark_complete(True)

                # Learn from success
                self.learner.learn_from_execution(
                    task=exercise['description'],
                    tools_used=['terminal'],
                    result=result,
                    success=True
                )

                break
        else:
            print(f"\n⚠️  Need more practice. Moving to next session.")
            session.mark_complete(False)

        # Save session
        self._save_session(session)
        self.sessions.append(session)

        return session

    def run_training_session(self, num_exercises: int = 5):
        """
        Run a complete training session

        Teaches multiple exercises from current level
        """
        print(f"\n" + "🎓"*30)
        print(f"TRAINING SESSION - Level {self.current_level}")
        print(f"🎓"*30 + "\n")

        level_data = self.curriculum['levels'][self.current_level - 1]
        exercises = self._generate_exercises(level_data, num_exercises)

        results = []
        for i, exercise in enumerate(exercises, 1):
            print(f"\n--- Exercise {i}/{len(exercises)} ---")
            session = self.teach_exercise(exercise)
            results.append(session)

            # Brief pause between exercises
            time.sleep(1)

        # Analyze session results
        self._analyze_session_results(results)

        # Update progress
        self._update_progress(results)

        return results

    def _generate_exercises(self, level_data: dict, count: int) -> list:
        """Generate exercises for a level"""
        # This would generate actual exercises
        # For now, placeholder
        exercises = []

        for skill in level_data['skills'][:count]:
            exercises.append({
                'description': f"Practice: {skill}",
                'correct_command': f"echo 'practicing {skill}'",
                'explanation': f"This command demonstrates {skill}"
            })

        return exercises

    def _analyze_session_results(self, results: list):
        """Analyze and report session results"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        success_rate = successful / total if total > 0 else 0

        avg_attempts = sum(len(r.attempts) for r in results) / total if total > 0 else 0

        print(f"\n" + "="*60)
        print(f"📊 SESSION RESULTS")
        print(f"="*60)
        print(f"   Exercises Completed: {total}")
        print(f"   Successful: {successful}/{total} ({success_rate:.1%})")
        print(f"   Average Attempts: {avg_attempts:.1f}")
        print(f"   Current Level: {self.current_level}")

    def _update_progress(self, results: list):
        """Update training progress"""
        successful = sum(1 for r in results if r.success)
        total = len(results)

        self.progress['total_sessions'] += 1
        self.progress['total_attempts'] += sum(len(r.attempts) for r in results)

        # Update success rate (running average)
        new_success = successful / total if total > 0 else 0
        self.progress['success_rate'] = (
            (self.progress['success_rate'] * (self.progress['total_sessions'] - 1) + new_success)
            / self.progress['total_sessions']
        )

        # Check if level completed
        if new_success >= 0.8:  # 80% success threshold
            print(f"\n🎊 LEVEL {self.current_level} MASTERED!")
            self.progress['completed_levels'].append(self.current_level)

            # Advance to next level
            if self.current_level < len(self.curriculum['levels']):
                self.current_level += 1
                self.progress['current_level'] = self.current_level
                print(f"   Advancing to Level {self.current_level}")

        self._save_progress()

    def continuous_training(self, sessions: int = 10):
        """
        Run multiple training sessions continuously

        This is the main training loop
        """
        print(f"\n🚀 STARTING CONTINUOUS TRAINING")
        print(f"   Total Sessions Planned: {sessions}")
        print(f"   Starting Level: {self.current_level}")
        print(f"\n")

        for session_num in range(1, sessions + 1):
            print(f"\n{'='*70}")
            print(f"SESSION {session_num}/{sessions}")
            print(f"{'='*70}\n")

            self.run_training_session(num_exercises=5)

            # Check if curriculum completed
            if self.current_level > len(self.curriculum['levels']):
                print(f"\n🏆 CURRICULUM COMPLETED! AGI has mastered all levels!")
                break

            # Pause between sessions
            if session_num < sessions:
                print(f"\nNext session in 5 seconds...")
                time.sleep(5)

        print(f"\n🎓 TRAINING COMPLETE")
        print(f"   Final Level: {self.current_level}")
        print(f"   Success Rate: {self.progress['success_rate']:.1%}")
        print(f"   Total Attempts: {self.progress['total_attempts']}")


def main():
    """Run teacher in interactive mode"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        🎓 CLAUDE TEACHER - AGI Training System 🎓         ║
║                                                           ║
║  Active teaching with continuous feedback loop           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    teacher = AITeacher()

    print("\nOptions:")
    print("  1. Run single training session (5 exercises)")
    print("  2. Run continuous training (10 sessions)")
    print("  3. Run full course (all levels)")
    print("  4. View progress")
    print("  5. Exit")

    choice = input("\nChoice: ").strip()

    if choice == '1':
        teacher.run_training_session(num_exercises=5)
    elif choice == '2':
        teacher.continuous_training(sessions=10)
    elif choice == '3':
        teacher.continuous_training(sessions=100)
    elif choice == '4':
        print("\n📊 Current Progress:")
        print(json.dumps(teacher.progress, indent=2))
    elif choice == '5':
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
