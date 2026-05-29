#!/usr/bin/env bash
set -e

echo "Removing virtual environment..."
rm -rf .venv

echo "Removing run artifacts..."
rm -rf runs/

echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
rm -rf .pytest_cache .mypy_cache .ruff_cache

echo "Teardown complete."
