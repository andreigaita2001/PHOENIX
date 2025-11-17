#!/usr/bin/env python3
"""
Single Teaching Session Script
Claude uses this to teach one exercise to the AGI
"""

import sys
import json
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent.parent / 'local_agi'))

from core.agent import AutonomousAgent

def run_teaching_session():
    """Run a single teaching session"""

    # Load current exercise
    exercise_path = Path(__file__).parent / 'current_exercise.json'
    with open(exercise_path, 'r') as f:
        exercise = json.load(f)

    print("=" * 70)
    print(f"📚 TEACHING SESSION - Level {exercise['level']}, Exercise {exercise['exercise_num']}")
    print("=" * 70)
    print()
    print(f"TASK: {exercise['task']}")
    print()

    # Initialize AGI
    print("Initializing AGI...", flush=True)
    agent = AutonomousAgent()
    print(f"✓ AGI ready (Model: {agent.llm.model_name})")
    print()

    # Demonstrate first
    print("🎓 DEMONSTRATION:")
    print(f"   Command: {exercise['correct_command']}")
    print(f"   Hints: {', '.join(exercise['hints'])}")
    print()

    # Have AGI attempt the task
    print("🤖 AGI ATTEMPTING TASK...")
    print("-" * 70)

    task_prompt = f"""You are learning terminal commands.

TASK: {exercise['task']}

HINTS: {', '.join(exercise['hints'])}

Use the execute_terminal tool to complete this task. Think step by step:
1. What command do I need?
2. What are the arguments?
3. Execute the command

Be concise and execute the command."""

    # Get AGI response
    tools = [tool["schema"] for tool in agent.llm.tool_registry.values()]
    response = agent.llm.generate_with_tools(task_prompt, tools)

    print()
    print("AGI Response:")
    print(response.get('text', 'No text response'))
    print()

    # Check if tools were used
    tool_calls = response.get('tool_calls', [])

    if tool_calls:
        print(f"\n✓ AGI requested {len(tool_calls)} tool call(s):")
        for call in tool_calls:
            print(f"   - {call.get('name', 'unknown')}")
            print(f"     Arguments: {call.get('arguments', {})}")

        # Check if terminal was used correctly
        terminal_calls = [c for c in tool_calls if c.get('name') == 'execute_terminal']
        if terminal_calls:
            command = terminal_calls[0].get('arguments', {}).get('command', '')
            print(f"\n   Command requested: {command}")

            # Validate
            success = False
            if 'ls' in command and 'PHOENIX-local-agi' in command:
                success = True
                print("   ✓ CORRECT! AGI understands the ls command!")
            else:
                print("   ✗ Not quite right.")
                print(f"   Expected something like: {exercise['correct_command']}")
                print(f"   Hint: You need to list files in a directory")

            return success, command
        else:
            print("   ✗ Did not use execute_terminal tool")
            print(f"   Hint: Use the 'execute_terminal' tool to run bash commands")
            return False, None
    else:
        print("✗ AGI did not use any tools")
        print("   Hint: You should use the execute_terminal tool")
        return False, None

if __name__ == "__main__":
    success, command = run_teaching_session()

    # Save result
    result = {
        'success': success,
        'command': command,
        'timestamp': str(Path(__file__).stat().st_mtime)
    }

    result_path = Path(__file__).parent / 'last_attempt.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 70)
    print(f"SESSION RESULT: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print("=" * 70)

    sys.exit(0 if success else 1)
