#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: ./tools/align_release.sh <new_version>"
    echo "Example: ./tools/align_release.sh 0.6.0"
    exit 1
fi

VERSION=$1

# 1. Update pyproject.toml
sed -i '' -e "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# 2. Update __init__.py
sed -i '' -e "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" src/prefgraph/__init__.py

# 3. Update docs/conf.py
sed -i '' -e "s/^release = \".*\"/release = \"$VERSION\"/" docs/conf.py

echo "✅ Bumped version to $VERSION in pyproject.toml, src/prefgraph/__init__.py, and docs/conf.py"
echo ""
echo "Next steps:"
echo "1. Update CHANGELOG.md"
echo "2. git commit -am \"release: v$VERSION\""
echo "3. git tag v$VERSION"
echo "4. git push && git push --tags"
