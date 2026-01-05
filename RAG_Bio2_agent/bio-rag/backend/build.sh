#!/usr/bin/env bash
# Render Build Script for Bio-RAG Backend
# This script is executed during Render deployment

set -o errexit  # Exit on error

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Running database migrations ==="
# Check if DATABASE_URL is set
if [ -n "$DATABASE_URL" ]; then
    echo "Database URL found, running Alembic migrations..."

    # Convert DATABASE_URL format if needed (Render uses postgres:// but SQLAlchemy needs postgresql://)
    export DATABASE_URL="${DATABASE_URL/postgres:\/\//postgresql:\/\/}"

    # Run migrations
    alembic upgrade head

    echo "Migrations completed successfully!"
else
    echo "WARNING: DATABASE_URL not set, skipping migrations"
fi

echo "=== Build completed ==="
