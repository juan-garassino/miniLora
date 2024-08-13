#!/bin/bash

# Create main project directory
mkdir -p miniLora
cd miniLora

# Create subdirectories
mkdir -p data src models utils visualizations results

# Create empty files in src directory
touch src/__init__.py
touch src/model.py
touch src/train.py
touch src/data.py

# Create empty files in utils directory
touch utils/__init__.py
touch utils/visualization.py
touch utils/analysis.py

# Create main script
touch main.py

# Create requirements.txt
touch requirements.txt

# Create README.md
touch README.md

echo "miniLora project structure created successfully!"
echo "You can now populate the files with your existing implementations."