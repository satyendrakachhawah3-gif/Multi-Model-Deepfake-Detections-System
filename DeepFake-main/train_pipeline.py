#!/usr/bin/env python3
"""Train all models sequentially"""
import subprocess
import sys

notebooks = [
    "notebooks/02_text_detection.ipynb",
    "notebooks/03_image_detection.ipynb",
    "notebooks/04_video_detection.ipynb",
]

for nb in notebooks:
    print(f"Training {nb}...")
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", nb])
print("✅ All models trained!")
