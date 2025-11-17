#!/bin/bash
# Remove a worktree for PHOENIX project
# Usage: ./worktree-remove.sh <name>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <worktree-name>"
    echo ""
    echo "Available worktrees:"
    cd /home/bone/PHOENIX
    git worktree list
    exit 1
fi

WORKTREE_NAME="$1"
WORKTREE_PATH="/home/bone/PHOENIX-${WORKTREE_NAME}"

if [ ! -d "${WORKTREE_PATH}" ]; then
    echo "Error: Worktree ${WORKTREE_PATH} does not exist"
    exit 1
fi

echo "Removing worktree: ${WORKTREE_PATH}"
read -p "Are you sure? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd /home/bone/PHOENIX
    git worktree remove "${WORKTREE_PATH}"
    echo "✓ Worktree removed!"
else
    echo "Cancelled."
fi
