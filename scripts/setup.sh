#!/usr/bin/env bash
set -e

echo "Creating virtual environment..."
uv venv

echo "Installing dependencies..."
uv sync --all-groups

if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env created from .env.example — fill in your GCP project ID."
else
    echo ".env already exists, skipping."
fi

echo "Setup complete. Activate with: source .venv/bin/activate"
