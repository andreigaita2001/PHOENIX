#!/bin/bash
# Create a new worktree for PHOENIX project
# Usage: ./worktree-new.sh <name> [base-branch]

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <worktree-name> [base-branch]"
    echo ""
    echo "Examples:"
    echo "  $0 my-feature          # Creates PHOENIX-my-feature from main"
    echo "  $0 my-fix dev/current  # Creates PHOENIX-my-fix from dev/current"
    exit 1
fi

WORKTREE_NAME="$1"
BASE_BRANCH="${2:-main}"
WORKTREE_PATH="/home/bone/PHOENIX-${WORKTREE_NAME}"
BRANCH_NAME="work/${WORKTREE_NAME}"

echo "Creating new worktree..."
echo "  Path: ${WORKTREE_PATH}"
echo "  Branch: ${BRANCH_NAME}"
echo "  Based on: ${BASE_BRANCH}"
echo ""

cd /home/bone/PHOENIX
git worktree add "${WORKTREE_PATH}" -b "${BRANCH_NAME}" "${BASE_BRANCH}"

echo ""
echo "✓ Worktree created!"
echo ""
echo "To use it:"
echo "  cd ${WORKTREE_PATH}"
echo ""
echo "To remove it later:"
echo "  ./worktree-remove.sh ${WORKTREE_NAME}"
