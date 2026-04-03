#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r /workspace/Ivatar/requirements.txt

echo "Installing SMPLicit package..."
cd /workspace/Ivatar/lib && pip install -e .

echo "Starting server on port 8888..."
cd /workspace && python3 -m uvicorn Ivatar.main:app --host 0.0.0.0 --port 8000
