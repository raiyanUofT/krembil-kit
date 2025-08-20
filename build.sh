#!/usr/bin/env bash
# build.sh — Bump version, commit+tag, build, and upload

set -e  # Exit on error
set -o pipefail

############################################
# 1. Pre-flight checks
############################################
echo "Running pre-flight checks..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Check if we're on main/master branch
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "main" && "$CURRENT_BRANCH" != "master" ]]; then
    echo "Warning: You're not on main/master branch (current: $CURRENT_BRANCH)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

############################################
# 2. Check package can be built
############################################
echo "Testing package build..."
python -m build --wheel --outdir temp_dist/ || {
    echo "Package build failed! Aborting."
    exit 1
}
rm -rf temp_dist/

############################################
# 3. Bump the version (PATCH by default)
############################################
# Usage examples:
#   ./build.sh            → bump patch  (1.6.0 → 1.6.1)
#   ./build.sh minor      → bump minor  (1.6.1 → 1.7.0)
#   ./build.sh major      → bump major  (1.6.1 → 2.0.0)
#############################################

BUMP_PART=${1:-patch}   # patch / minor / major
echo "Bumping $BUMP_PART version with bump2version ..."
bump2version "$BUMP_PART"

############################################
# 2. Push commit and tag (optional skip with SKIP_PUSH=1)
############################################
if [[ "$SKIP_PUSH" != "1" ]]; then
  TAG=$(git describe --tags --abbrev=0)
  echo "Pushing commit and tag $TAG ..."
  git push origin HEAD
  git push origin "$TAG"
else
  echo "Skipping git push (SKIP_PUSH=1)"
fi

############################################
# 3. Clean previous builds
############################################
echo "Cleaning previous builds..."
rm -rf dist/

echo "Cleaning egg-info..."
rm -rf src/krembil_kit.egg-info/

############################################
# 4. Build distribution
############################################
echo "Building new distributions..."
python -m build

############################################
# 5. Upload to PyPI
############################################
echo "Uploading to PyPI..."
python -m twine upload dist/*

echo "Done!"
