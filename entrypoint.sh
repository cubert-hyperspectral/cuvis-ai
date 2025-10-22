#!/bin/bash
set -euo pipefail

cd /app

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv sync --locked

echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
uv run python -c "import cuvis; import cuvis_ai; print(cuvis.General.version())"
echo "======================="
echo "OUTPUT OF CUVIS.AI pytest suite"
echo "======================="
uv run pytest --maxfail=1 --disable-warnings
