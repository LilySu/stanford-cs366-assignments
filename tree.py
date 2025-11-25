#!/usr/bin/env python3
"""
Simple one-line project tree visualizer
Usage: python tree.py [depth]
Example: python tree.py 3
"""

import os
import sys
from pathlib import Path


def tree(directory=".", prefix="", depth=None, current_depth=0, ignore=['.git', '__pycache__', '.venv', 'venv', '.pytest_cache', 'node_modules', '.DS_Store', '*.pyc', '.idea', '.vscode', 'dist', 'build', '*.egg-info']):
    """Print directory tree"""
    if depth and current_depth >= depth:
        return
    
    try:
        items = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))
        items = [i for i in items if not any(p in i.name or (i.name.startswith('.') and p.startswith('.')) for p in ignore)]
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}{'/' if item.is_dir() else ''}")
            if item.is_dir():
                tree(item, prefix + ('    ' if is_last else '│   '), depth, current_depth + 1, ignore)
    except PermissionError:
        pass

if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else None
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"{Path(directory).resolve().name}/")
    tree(directory, depth=depth)