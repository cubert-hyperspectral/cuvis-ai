#!/bin/bash
set -euo pipefail

cd /app

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv sync --locked --extra docs

echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
uv run python -c "import cuvis; import cuvis_ai"
echo "======================="
echo "Generate CUVIS.AI documentation"
echo "======================="
mkdir -p docs/_build
uv run sphinx-build -M html docs docs/_build
touch docs/_build/html/.nojekyll
