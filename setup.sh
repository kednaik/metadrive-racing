#!/bin/bash

# Setup script for MetaDrive Autonomous Driving project

# Exit on error
set -e

echo "🚀 Starting setup..."

# Create project directories
echo "📁 Creating directories..."
mkdir -p model/models
mkdir -p model/logs

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# Activate virtual environment
echo "🔗 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️ Warning: requirements.txt not found. Installing defaults..."
    pip install metadrive-simulator stable-baselines3[extra] tensorboard
fi

echo "✅ Setup complete! You can now start training."
echo "💡 To activate the environment manually, run: source venv/bin/activate"
