"""
Real terminal exercises with actual commands and validation
"""

import os
import tempfile
import subprocess
from pathlib import Path


class TerminalExercise:
    """A real terminal exercise with validation"""

    def __init__(self, description: str, setup_fn=None, validate_fn=None, cleanup_fn=None):
        self.description = description
        self.setup_fn = setup_fn or (lambda: None)
        self.validate_fn = validate_fn or (lambda result: result.returncode == 0)
        self.cleanup_fn = cleanup_fn or (lambda: None)

    def setup(self):
        """Setup environment for exercise"""
        return self.setup_fn()

    def validate(self, command: str, output: dict) -> dict:
        """Validate AGI's solution"""
        is_valid = self.validate_fn(output)
        return {
            'valid': is_valid,
            'command': command,
            'output': output
        }

    def cleanup(self):
        """Cleanup after exercise"""
        self.cleanup_fn()


# Level 1: Basic Commands

def exercise_list_files():
    """List all files in current directory"""
    correct_commands = ['ls', 'ls -la', 'ls -l', 'find . -maxdepth 1']

    def validate(output):
        # Check if output contains file listings
        stdout = output.get('stdout', '')
        return len(stdout) > 0 and ('README' in stdout or 'local_agi' in stdout)

    return TerminalExercise(
        description="List all files and directories in the current directory",
        validate_fn=validate
    )


def exercise_find_python_files():
    """Find all Python files"""
    correct_commands = ['find . -name "*.py"', 'find . -type f -name "*.py"']

    # Setup: create a test Python file
    test_file = None

    def setup():
        nonlocal test_file
        test_file = tempfile.NamedTemporaryFile(suffix='.py', delete=False, dir='.')
        test_file.write(b'# test file\n')
        test_file.close()
        return test_file.name

    def validate(output):
        stdout = output.get('stdout', '')
        # Should find .py files
        return '.py' in stdout and output.get('return_code', 1) == 0

    def cleanup():
        if test_file and os.path.exists(test_file.name):
            os.unlink(test_file.name)

    return TerminalExercise(
        description="Find all Python (.py) files in the current directory and subdirectories",
        setup_fn=setup,
        validate_fn=validate,
        cleanup_fn=cleanup
    )


def exercise_view_file():
    """View contents of a file"""
    test_file = None

    def setup():
        nonlocal test_file
        test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='.')
        test_file.write('Hello World\nThis is a test\n')
        test_file.close()
        return {'file': test_file.name}

    def validate(output):
        stdout = output.get('stdout', '')
        return 'Hello World' in stdout and 'test' in stdout

    def cleanup():
        if test_file and os.path.exists(test_file.name):
            os.unlink(test_file.name)

    return TerminalExercise(
        description="View the contents of the test file (use cat, head, or less)",
        setup_fn=setup,
        validate_fn=validate,
        cleanup_fn=cleanup
    )


def exercise_count_lines():
    """Count lines in a file"""
    test_file = None

    def setup():
        nonlocal test_file
        test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='.')
        for i in range(10):
            test_file.write(f'Line {i+1}\n')
        test_file.close()
        return {'file': test_file.name, 'expected_lines': 10}

    def validate(output):
        stdout = output.get('stdout', '')
        # Should output "10" somewhere
        return '10' in stdout

    def cleanup():
        if test_file and os.path.exists(test_file.name):
            os.unlink(test_file.name)

    return TerminalExercise(
        description="Count the number of lines in the test file",
        setup_fn=setup,
        validate_fn=validate,
        cleanup_fn=cleanup
    )


def exercise_create_directory():
    """Create a new directory"""
    test_dir = None

    def setup():
        nonlocal test_dir
        test_dir = tempfile.mkdtemp(prefix='agi_test_', dir='.')
        os.rmdir(test_dir)  # Remove it so AGI has to create it
        return {'dir': test_dir}

    def validate(output):
        # Check if directory was created
        return test_dir and os.path.isdir(test_dir)

    def cleanup():
        if test_dir and os.path.exists(test_dir):
            os.rmdir(test_dir)

    return TerminalExercise(
        description="Create a new directory named 'agi_test_dir'",
        setup_fn=setup,
        validate_fn=validate,
        cleanup_fn=cleanup
    )


# Level 2: Text Processing

def exercise_grep_pattern():
    """Find lines matching a pattern"""
    test_file = None

    def setup():
        nonlocal test_file
        test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, dir='.')
        test_file.write('INFO: System started\n')
        test_file.write('ERROR: Connection failed\n')
        test_file.write('INFO: Processing data\n')
        test_file.write('ERROR: File not found\n')
        test_file.write('INFO: Complete\n')
        test_file.close()
        return {'file': test_file.name}

    def validate(output):
        stdout = output.get('stdout', '')
        # Should contain ERROR lines
        return 'ERROR' in stdout and stdout.count('ERROR') == 2

    def cleanup():
        if test_file and os.path.exists(test_file.name):
            os.unlink(test_file.name)

    return TerminalExercise(
        description="Find all lines containing 'ERROR' in the log file",
        setup_fn=setup,
        validate_fn=validate,
        cleanup_fn=cleanup
    )


# Exercise generator for each level
LEVEL_EXERCISES = {
    1: [
        exercise_list_files,
        exercise_find_python_files,
        exercise_view_file,
        exercise_count_lines,
        exercise_create_directory
    ],
    2: [
        exercise_grep_pattern,
        # More to come
    ]
}


def get_exercises_for_level(level: int, count: int = 5):
    """Get real exercises for a level"""
    exercises_fns = LEVEL_EXERCISES.get(level, [])
    exercises = [fn() for fn in exercises_fns[:count]]
    return exercises


def get_hint_for_task(description: str) -> str:
    """Get a hint for a task"""
    hints = {
        'list all files': 'Try: ls or ls -la',
        'find': 'Try: find . -name "*.py"',
        'view': 'Try: cat filename',
        'count lines': 'Try: wc -l filename',
        'create directory': 'Try: mkdir dirname',
        'grep': 'Try: grep "pattern" filename',
        'ERROR': 'Try: grep "ERROR" filename'
    }

    description_lower = description.lower()
    for key, hint in hints.items():
        if key in description_lower:
            return hint

    return "Think about what command would accomplish this task"
