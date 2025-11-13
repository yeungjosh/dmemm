#!/bin/bash

# Sentiment Flow Visualizer - Quick Start Script

echo "======================================"
echo "  Sentiment Flow Visualizer"
echo "  Deep MEMM for NLP"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "Error: pip3 is not installed"
    exit 1
fi

echo "Installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "Starting Flask server..."
echo "The app will be available at http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd app/backend
python3 app.py
