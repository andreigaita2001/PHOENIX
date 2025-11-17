#!/bin/bash
# List all worktrees for PHOENIX project

cd /home/bone/PHOENIX
echo "PHOENIX Worktrees:"
echo "=================="
git worktree list
echo ""
echo "Quick access:"
git worktree list | awk '{print "  cd " $1}'
