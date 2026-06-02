#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/wangyd97/visual-servoing.git"
BRANCH="main"
COMMIT_MSG="${1:-Update by Wang}"

cd "$(dirname "$0")"

echo "Working directory: $(pwd)"

if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    {
        echo "__pycache__/"
        echo "*.pyc"
        echo "exp_figures/"
    } > .gitignore
fi

if git remote get-url origin >/dev/null 2>&1; then
    echo "Remote origin already exists: $(git remote get-url origin)"
else
    echo "Adding remote origin: $REPO_URL"
    git remote add origin "$REPO_URL"
fi

echo "Switching branch to $BRANCH..."
git branch -M "$BRANCH"

echo "Adding files..."
git add .

if git diff --cached --quiet; then
    echo "No local changes to commit."
else
    echo "Committing changes..."
    git commit -m "$COMMIT_MSG"
fi

if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    echo "Remote branch origin/$BRANCH exists. Pulling before push..."
    git pull origin "$BRANCH" --allow-unrelated-histories --no-rebase
fi

echo "Pushing to GitHub..."
git push -u origin "$BRANCH"

echo "Done."
