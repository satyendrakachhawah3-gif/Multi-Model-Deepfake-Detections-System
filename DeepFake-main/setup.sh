#!/bin/bash
# Deepfake Detection Project Setup Script
# This script sets up the environment and downloads required datasets

echo "========================================="
echo "Deepfake Detection Project Setup"
echo "========================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install kaggle datasets huggingface-hub pandas

# Check if Kaggle API credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Warning: Kaggle API credentials not found at ~/.kaggle/kaggle.json"
    echo "Please set up Kaggle API credentials to download datasets."
    echo "Visit: https://www.kaggle.com/docs/api"
fi

# Run Python script to download datasets
echo "Downloading datasets..."
python download_datasets.py

echo "========================================="
echo "Setup complete!"
echo "========================================="
