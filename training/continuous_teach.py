#!/usr/bin/env python3
"""
Continuous Teaching Loop
Runs multiple teaching sessions until mastery is achieved
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.append(str(Path(__file__).parent.parent / 'local_agi'))

from core.agent import AutonomousAgent

# Level 1 exercises - Basic commands
LEVEL_1_EXERCISES = [
    {
        "task": "List all files and directories in /home/bone/PHOENIX-local-agi",
        "hints": ["Use the ls command", "The path is /home/bone/PHOENIX-local-agi"],
        "correct_command": "ls /home/bone/PHOENIX-local-agi",
        "validation_keywords": ["ls", "PHOENIX-local-agi"]
    },
    {
        "task": "Display the contents of the file /home/bone/PHOENIX-local-agi/README.md",
        "hints": ["Use the cat command", "The file is README.md"],
        "correct_command": "cat /home/bone/PHOENIX-local-agi/README.md",
        "validation_keywords": ["cat", "README.md"]
    },
    {
        "task": "Show the current working directory",
        "hints": ["Use the pwd command", "No arguments needed"],
        "correct_command": "pwd",
        "validation_keywords": ["pwd"]
    },
    {
        "task": "List all Python files in /home/bone/PHOENIX-local-agi",
        "hints": ["Use the find command", "Look for .py extension", "Use -name '*.py'"],
        "correct_command": "find /home/bone/PHOENIX-local-agi -name '*.py'",
        "validation_keywords": ["find", ".py"]
    },
    {
        "task": "Display the first 10 lines of /home/bone/PHOENIX-local-agi/README.md",
        "hints": ["Use the head command", "Default shows 10 lines"],
        "correct_command": "head /home/bone/PHOENIX-local-agi/README.md",
        "validation_keywords": ["head", "README.md"]
    },
    {
        "task": "Create a directory called /tmp/agi_test",
        "hints": ["Use the mkdir command", "Path is /tmp/agi_test"],
        "correct_command": "mkdir /tmp/agi_test",
        "validation_keywords": ["mkdir", "/tmp/agi_test"]
    },
    {
        "task": "Search for the word 'AGI' in /home/bone/PHOENIX-local-agi/README.md",
        "hints": ["Use the grep command", "Search for 'AGI'"],
        "correct_command": "grep 'AGI' /home/bone/PHOENIX-local-agi/README.md",
        "validation_keywords": ["grep", "AGI"]
    },
    {
        "task": "List files in /home/bone/PHOENIX-local-agi sorted by modification time",
        "hints": ["Use ls with -lt flags", "t sorts by time"],
        "correct_command": "ls -lt /home/bone/PHOENIX-local-agi",
        "validation_keywords": ["ls", "-l"]
    }
]

# Level 2 exercises - Intermediate
LEVEL_2_EXERCISES = [
    {
        "task": "Count the number of lines in /home/bone/PHOENIX-local-agi/README.md",
        "hints": ["Use wc command", "The -l flag counts lines"],
        "correct_command": "wc -l /home/bone/PHOENIX-local-agi/README.md",
        "validation_keywords": ["wc", "-l"]
    },
    {
        "task": "Find all unique file extensions in /home/bone/PHOENIX-local-agi",
        "hints": ["Use find with -type f", "Pipe to other commands", "Use sort and uniq"],
        "correct_command": "find /home/bone/PHOENIX-local-agi -type f -name '*.*' | sed 's/.*\\.//' | sort | uniq",
        "validation_keywords": ["find", "sort", "uniq"]
    },
    {
        "task": "Search for 'import' in all Python files",
        "hints": ["Use grep with -r for recursive", "Search in *.py files"],
        "correct_command": "grep -r 'import' /home/bone/PHOENIX-local-agi --include='*.py'",
        "validation_keywords": ["grep", "-r", "import"]
    }
]

# Level 3 exercises - Advanced
LEVEL_3_EXERCISES = [
    {
        "task": "List all running processes containing 'python'",
        "hints": ["Use ps command", "Pipe to grep", "Search for python"],
        "correct_command": "ps aux | grep python",
        "validation_keywords": ["ps", "grep", "python"]
    },
    {
        "task": "Show disk usage of /home/bone/PHOENIX-local-agi in human-readable format",
        "hints": ["Use du command", "-h for human readable", "-s for summary"],
        "correct_command": "du -sh /home/bone/PHOENIX-local-agi",
        "validation_keywords": ["du", "-h"]
    },
    {
        "task": "Count how many Python files exist in /home/bone/PHOENIX-local-agi",
        "hints": ["Use find to list .py files", "Pipe to wc -l to count"],
        "correct_command": "find /home/bone/PHOENIX-local-agi -name '*.py' | wc -l",
        "validation_keywords": ["find", ".py", "wc"]
    },
    {
        "task": "Show the last 5 lines of /home/bone/PHOENIX-local-agi/README.md",
        "hints": ["Use tail command", "-n 5 for 5 lines"],
        "correct_command": "tail -n 5 /home/bone/PHOENIX-local-agi/README.md",
        "validation_keywords": ["tail", "README.md"]
    }
]

# Level 4 exercises - Expert
LEVEL_4_EXERCISES = [
    {
        "task": "Find the 5 largest files in /home/bone/PHOENIX-local-agi",
        "hints": ["Use find with -type f", "Pipe to du", "Sort and head"],
        "correct_command": "find /home/bone/PHOENIX-local-agi -type f -exec du -h {} + | sort -rh | head -5",
        "validation_keywords": ["find", "sort", "head"]
    },
    {
        "task": "Count total lines of Python code in /home/bone/PHOENIX-local-agi",
        "hints": ["Find all .py files", "Use cat to combine", "Count with wc -l"],
        "correct_command": "find /home/bone/PHOENIX-local-agi -name '*.py' -exec cat {} + | wc -l",
        "validation_keywords": ["find", ".py", "wc", "-l"]
    }
]

ALL_LEVELS = {
    1: LEVEL_1_EXERCISES,
    2: LEVEL_2_EXERCISES,
    3: LEVEL_3_EXERCISES,
    4: LEVEL_4_EXERCISES
}

def validate_command(command, validation_keywords):
    """Check if command contains required keywords"""
    if not command:
        return False
    command_lower = command.lower()
    return all(keyword.lower() in command_lower for keyword in validation_keywords)

def run_single_exercise(agent, exercise, level, exercise_num):
    """Run a single teaching exercise"""

    print("=" * 70)
    print(f"📚 Level {level}, Exercise {exercise_num}")
    print("=" * 70)
    print(f"TASK: {exercise['task']}")
    print(f"HINTS: {', '.join(exercise['hints'])}")
    print()

    task_prompt = f"""You are learning terminal commands.

TASK: {exercise['task']}

HINTS: {', '.join(exercise['hints'])}

Use the execute_terminal tool to complete this task. Think about what command you need and execute it directly.
Be concise - just call the tool."""

    # Get AGI response
    tools = [tool["schema"] for tool in agent.llm.tool_registry.values()]
    response = agent.llm.generate_with_tools(task_prompt, tools)

    # Check if tools were used
    tool_calls = response.get('tool_calls', [])

    if not tool_calls:
        print("✗ AGI did not use any tools")
        return False, None

    # Check for execute_terminal
    terminal_calls = [c for c in tool_calls if c.get('name') == 'execute_terminal']
    if not terminal_calls:
        print("✗ Did not use execute_terminal tool")
        return False, None

    command = terminal_calls[0].get('arguments', {}).get('command', '')
    print(f"AGI executed: {command}")

    # Validate
    success = validate_command(command, exercise['validation_keywords'])

    if success:
        print("✓ CORRECT!")
    else:
        print("✗ Not quite right")
        print(f"Expected: {exercise['correct_command']}")

    print()
    return success, command

def update_progress(progress, level, exercise_num, success):
    """Update progress tracking"""
    progress['total_attempts'] += 1
    if success:
        progress['total_successes'] += 1

    progress['success_rate'] = (progress['total_successes'] / progress['total_attempts']) * 100
    progress['exercise_num'] = exercise_num
    progress['current_level'] = level
    progress['last_session'] = datetime.now().isoformat()

    return progress

def should_advance_level(progress, current_level):
    """Check if should advance to next level"""
    # Need at least 5 attempts on current level
    if progress['total_attempts'] < 5:
        return False

    # Need 90% success rate
    return progress['success_rate'] >= 90.0

def main():
    """Main continuous teaching loop"""

    print("🎓 CONTINUOUS AGI TEACHING LOOP")
    print("=" * 70)
    print()

    # Load progress
    progress_path = Path(__file__).parent / 'progress.json'
    with open(progress_path, 'r') as f:
        progress = json.load(f)

    # Initialize AGI once
    print("Initializing AGI...", flush=True)
    agent = AutonomousAgent()
    print(f"✓ AGI ready\n")

    current_level = progress['current_level']
    exercise_num = progress.get('exercise_num', 0)

    sessions = []
    max_exercises = 50  # Safety limit
    exercises_completed = 0

    # Teaching loop
    while exercises_completed < max_exercises and current_level <= 4:

        # Get exercises for current level
        level_exercises = ALL_LEVELS.get(current_level, [])

        if exercise_num >= len(level_exercises):
            # Completed all exercises in this level
            if should_advance_level(progress, current_level):
                print(f"\n🎉 LEVEL {current_level} MASTERED!")
                print(f"Success rate: {progress['success_rate']:.1f}%")
                print()

                progress['levels_completed'].append(current_level)
                current_level += 1
                exercise_num = 0
                progress['total_attempts'] = 0
                progress['total_successes'] = 0

                if current_level > 4:
                    print("🏆 ALL LEVELS COMPLETED!")
                    break

                print(f"📈 ADVANCING TO LEVEL {current_level}\n")
                time.sleep(1)
                continue
            else:
                # Loop back to start of level for more practice
                print(f"Need more practice on level {current_level}")
                print(f"Current success rate: {progress['success_rate']:.1f}% (need 90%)")
                exercise_num = 0
                continue

        # Run exercise
        exercise = level_exercises[exercise_num]
        success, command = run_single_exercise(agent, exercise, current_level, exercise_num + 1)

        # Update progress
        progress = update_progress(progress, current_level, exercise_num + 1, success)

        # Log session
        session = {
            "session_num": exercises_completed + 1,
            "timestamp": datetime.now().isoformat(),
            "level": current_level,
            "exercise": exercise_num + 1,
            "task": exercise['task'],
            "success": success,
            "command": command,
            "success_rate": progress['success_rate']
        }
        sessions.append(session)

        # Save progress
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)

        # Save session log
        sessions_path = Path(__file__).parent / 'sessions.jsonl'
        with open(sessions_path, 'a') as f:
            f.write(json.dumps(session) + '\n')

        exercise_num += 1
        exercises_completed += 1

        # Brief pause between exercises
        time.sleep(0.5)

    # Final summary
    print("\n" + "=" * 70)
    print("📊 TEACHING SESSION SUMMARY")
    print("=" * 70)
    print(f"Total exercises: {exercises_completed}")
    print(f"Success rate: {progress['success_rate']:.1f}%")
    print(f"Levels completed: {progress['levels_completed']}")
    print(f"Current level: {progress['current_level']}")
    print()

    # Check for mastery
    if len(progress['levels_completed']) >= 4 and progress['success_rate'] >= 90:
        print("🏆 MASTERY ACHIEVED!")
        mastery_flag = Path(__file__).parent / 'mastery_achieved.flag'
        with open(mastery_flag, 'w') as f:
            f.write(f"Mastery achieved on {datetime.now().isoformat()}\n")
            f.write(f"Final success rate: {progress['success_rate']:.1f}%\n")
            f.write(f"Levels completed: {progress['levels_completed']}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
