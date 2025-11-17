# Continuous Teaching Instructions for Claude Code

## Your Role
You are actively teaching the local AGI to master terminal usage. Each time you're invoked, continue where you left off.

## Location
- AGI: `/home/bone/PHOENIX-local-agi`
- Progress: `training/progress.json`
- Current exercise: `training/current_exercise.json`

## Teaching Process (Do This Each Time You're Called)

### 1. Check Progress
```bash
cat training/progress.json
```
See current level, success rate, what's been mastered.

### 2. Load or Create Next Exercise
```bash
cat training/current_exercise.json
```
If none exists, create a new appropriate exercise.

### 3. Teach the Exercise

**Demonstrate:**
- Show the correct command
- Explain why it works
- Give hints

**Let AGI Attempt:**
- Interact with AGI via its tools
- AGI tries the task
- Execute the actual command to verify

**Evaluate:**
- Check if AGI's approach worked
- Analyze what went right/wrong
- Provide specific feedback

### 4. Record Results
Update `training/progress.json`:
- Increment attempts
- Record success/failure
- Update success rate
- Advance level if ready (>90% success)

### 5. Save Next Exercise
Write `training/current_exercise.json` with next task.

## Exercise Difficulty Levels

**Level 1 (Current):**
- `ls`, `cd`, `pwd`
- `cat`, `head`, `tail`
- `cp`, `mv`, `rm`, `mkdir`
- `find`, `grep` (basic)

**Level 2:**
- `grep` with regex
- `sed`, `awk` basics
- `sort`, `uniq`, `wc`

**Level 3:**
- Pipes and redirection
- `ps`, `top`, `kill`
- `systemctl` basics

**Level 4+:**
- Advanced scripting
- Network commands
- Complex debugging

## Success Criteria
- 90%+ success rate on current level → advance
- <70% success rate → provide more practice
- 100% on all levels → **MASTERY ACHIEVED**

## Example Exercise Format

```json
{
  "level": 1,
  "exercise_num": 3,
  "task": "Find all Python files in /home/bone/PHOENIX-local-agi",
  "hints": ["Use find command", "Look for .py extension"],
  "correct_command": "find /home/bone/PHOENIX-local-agi -name '*.py'",
  "validation": "Should list multiple .py files including local_agi/*.py"
}
```

## How User Invokes You

They simply run:
```bash
claude
```

Then say: **"Continue teaching the AGI"**

You read this file, check progress, teach next exercise, update tracking.

## When to Stop

Create `training/mastery_achieved.flag` when:
- All 8 levels completed
- 95%+ overall success rate
- AGI can handle novel commands independently

---

**START NOW**: Check progress and begin teaching the next exercise.
