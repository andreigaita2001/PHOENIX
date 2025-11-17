# AGI Training System - Claude as Active Teacher

## Overview

This is a **continuous active teaching system** where Claude works directly with the local AGI to achieve terminal mastery.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                  TEACHING CYCLE                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Claude demonstrates correct approach                │
│           ↓                                             │
│  2. AGI attempts the task                               │
│           ↓                                             │
│  3. Claude evaluates the attempt                        │
│           ↓                                             │
│  4. Claude provides detailed feedback                   │
│           ↓                                             │
│  5. AGI tries again (if needed)                         │
│           ↓                                             │
│  6. Repeat until mastery                                │
│           ↓                                             │
│  7. Move to next exercise                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Training Curriculum

**8 Levels of Terminal Mastery:**

1. **Basic Commands** (ls, cd, cat, cp, mv, find)
2. **Text Processing** (grep, sed, awk, sort)
3. **Process Management** (ps, kill, systemctl)
4. **Networking** (ping, curl, wget, ssh)
5. **Scripting** (bash, variables, loops, functions)
6. **Advanced Operations** (pipes, jobs, archives, git)
7. **System Administration** (logs, cron, permissions)
8. **Performance & Debugging** (profiling, troubleshooting)

Each level has multiple skills and exercises.

## Usage

### Quick Start

```bash
cd /home/bone/PHOENIX-local-agi
python3 training/teacher.py
```

Choose:
- **Option 1**: Single session (5 exercises, ~15 minutes)
- **Option 2**: Continuous training (10 sessions, ~2 hours)
- **Option 3**: Full course (all levels, runs until mastery)

### What Happens

**During Each Exercise:**

1. **Demonstration**
   ```
   📚 DEMONSTRATION
      Task: List all Python files
      Command: find . -name "*.py"
      Explanation: Uses find to search recursively
   ```

2. **AGI Attempts**
   ```
   📝 Attempt 1/3
      AGI is attempting...
   ```

3. **Evaluation**
   ```
   🔍 EVALUATING
      Success: ❌
      Feedback: Command syntax incorrect
   ```

4. **Feedback**
   ```
   💬 FEEDBACK:
      ❌ Command failed. Here's why...
      💡 Try using the -name flag
      💡 Remember to quote the pattern
   ```

5. **Retry or Next**
   - Success → Move to next exercise
   - Failed → Try again (max 3 attempts)

### Session Results

After each session:
```
📊 SESSION RESULTS
   Exercises Completed: 5
   Successful: 4/5 (80%)
   Average Attempts: 1.8
   Current Level: 2
```

### Progress Tracking

```bash
# View current progress
python3 training/teacher.py
# Choose option 4
```

Shows:
- Current level
- Completed levels
- Total sessions run
- Overall success rate
- Skills mastered

## Files

```
training/
├── curriculum.json       # 8-level course structure
├── teacher.py           # Main teaching system
├── progress.json        # AGI's training progress
├── sessions.jsonl       # History of all sessions
└── README.md            # This file
```

## Integration with Main AGI

The teaching system:
- ✅ Uses the same agent instance
- ✅ Stores learnings in knowledge base
- ✅ Updates pattern library
- ✅ Logs all activities
- ✅ Tracks autonomy progression

**Result**: Skills learned in training transfer to regular AGI usage!

## Continuous Training Mode

For maximum learning speed, run continuously:

```bash
# Run in background
nohup python3 training/teacher.py --continuous &

# Or with screen/tmux
screen -S agi-training
python3 training/teacher.py --continuous
# Detach with Ctrl+A, D
```

## Expected Timeline

**With continuous training:**

- **Week 1**: Levels 1-3 (Basic commands, text processing, processes)
- **Week 2**: Levels 4-5 (Networking, scripting)
- **Week 3**: Levels 6-7 (Advanced ops, sys admin)
- **Week 4**: Level 8 (Performance, debugging)

**End Result**: AGI capable of autonomous terminal usage at 95%+ success rate

## Monitoring Training

Watch in real-time:

```bash
# Terminal 1: Run training
python3 training/teacher.py

# Terminal 2: Monitor logs
tail -f /home/bone/PHOENIX-local-agi/logs/phoenix_agi.log

# Terminal 3: Watch sessions
tail -f training/sessions.jsonl
```

## Success Metrics

AGI has mastered terminal usage when:
- ✅ 95%+ success rate across all levels
- ✅ Average 1.2 attempts per exercise
- ✅ Can handle novel commands independently
- ✅ Completes tasks faster than 80th percentile

## Customization

Edit `curriculum.json` to:
- Add new exercises
- Adjust difficulty
- Focus on specific skills
- Add domain-specific tasks

## Example Session Output

```
🎓 TRAINING SESSION - Level 2
═══════════════════════════════

--- Exercise 1/5 ---

📚 DEMONSTRATION
   Task: Find all error lines in log files
   Command: grep -r "ERROR" /var/log/
   Explanation: grep with -r for recursive search

📝 Attempt 1/3
   AGI is attempting...

🔍 EVALUATING AGI ATTEMPT
   Success: ✅

💬 FEEDBACK:
   ✅ Excellent! Command executed correctly.

🎉 SUCCESS! Mastered in 1 attempt

[... 4 more exercises ...]

📊 SESSION RESULTS
   Exercises: 5
   Successful: 5/5 (100%)
   Avg Attempts: 1.2
   Level: 2

🎊 LEVEL 2 MASTERED!
   Advancing to Level 3
```

## Tips

1. **Run regularly**: Daily sessions for fastest progress
2. **Review failures**: Check sessions.jsonl for patterns
3. **Increase difficulty**: Once success rate > 90%, move up
4. **Track metrics**: Monitor progress.json weekly
5. **Be patient**: Mastery takes time and repetition

---

**This is active, continuous teaching.** Claude works directly with your AGI until it masters terminal usage. The feedback loop ensures rapid learning through demonstration, practice, evaluation, and correction.

🔥 **Let the training begin!**
