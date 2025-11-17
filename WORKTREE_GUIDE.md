# Git Worktrees Setup for PHOENIX

## What You Have Now

```
/home/bone/PHOENIX/         → main branch (stable, production-ready)
/home/bone/PHOENIX-dev/     → dev/feature-work (active development)
/home/bone/PHOENIX-hotfix/  → hotfix/ready (urgent bug fixes)
```

## How to Use This

### When I (Claude) Work on Features
```bash
# I work in PHOENIX-dev
cd /home/bone/PHOENIX-dev
# Make changes, commit, push
```

### When You Need an Urgent Bug Fix
```bash
# I switch to PHOENIX-hotfix (or you can)
cd /home/bone/PHOENIX-hotfix
# Fix bug, test, commit
git push origin hotfix/ready
# Create PR from hotfix/ready to main
```

### Your Main Branch Stays Clean
```bash
# PHOENIX is always on main, always stable
cd /home/bone/PHOENIX
# You can test production code here anytime
# Or I can pull latest changes
git pull
```

## Helper Scripts

### Create a New Worktree
```bash
cd /home/bone/PHOENIX
./worktree-new.sh my-feature-name
# Creates /home/bone/PHOENIX-my-feature-name
```

### List All Worktrees
```bash
cd /home/bone/PHOENIX
./worktree-list.sh
```

### Remove a Worktree (when done)
```bash
cd /home/bone/PHOENIX
./worktree-remove.sh my-feature-name
```

## Common Workflows

### Scenario 1: Normal Development
1. You tell me to work on a feature
2. I work in `/home/bone/PHOENIX-dev`
3. When done, I push and create a PR
4. You can test in PHOENIX-dev while I keep working

### Scenario 2: Urgent Bug in Production
1. You find a critical bug
2. I quickly switch to `/home/bone/PHOENIX-hotfix`
3. Fix bug, commit, push
4. Meanwhile, `/home/bone/PHOENIX-dev` has my unfinished feature work untouched
5. After fix is merged, I go back to PHOENIX-dev

### Scenario 3: You Want to Test Something
1. While I'm working in PHOENIX-dev
2. You can test stable code in `/home/bone/PHOENIX`
3. Or create your own worktree: `./worktree-new.sh test-idea`

## Quick Commands

```bash
# See all worktrees
git worktree list

# Jump between them
cd /home/bone/PHOENIX        # stable
cd /home/bone/PHOENIX-dev    # my work
cd /home/bone/PHOENIX-hotfix # urgent fixes

# Each one is independent!
# Changes in one don't affect the others
```

## Pro Tips

1. **Never commit directly to main** - use PHOENIX-dev or create new worktrees
2. **Keep PHOENIX on main branch** - it's your stable reference
3. **Create temporary worktrees** for experiments - easy to delete later
4. **Each worktree shares the same .git** - saves disk space!

## When Working with Claude

**Just tell me which worktree to use:**
- "Work on this feature in PHOENIX-dev"
- "Fix this bug in PHOENIX-hotfix"
- "Create a new worktree for testing X"

I'll handle the rest!
