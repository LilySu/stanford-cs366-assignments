#!/usr/bin/env python3
"""
Visualize project folder structure in markdown format
Usage: python visualize_structure.py [directory]
"""

import os
import sys
from pathlib import Path


def get_tree_structure(directory, prefix="", max_depth=None, current_depth=0, ignore_patterns=None):
    """
    Generate tree structure for a directory.
    
    Args:
        directory: Path to directory
        prefix: Prefix for tree lines
        max_depth: Maximum depth to traverse (None for unlimited)
        current_depth: Current depth in tree
        ignore_patterns: List of patterns to ignore
    
    Returns:
        List of strings representing the tree
    """
    if ignore_patterns is None:
        ignore_patterns = [
            '.git', '__pycache__', '.pytest_cache', 
            '.venv', 'venv', '.env', 'node_modules',
            '.DS_Store', '*.pyc', '.idea', '.vscode',
            '*.egg-info', 'dist', 'build'
        ]
    
    lines = []
    directory = Path(directory)
    
    # Check if we've reached max depth
    if max_depth is not None and current_depth >= max_depth:
        return lines
    
    try:
        # Get all items in directory
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        
        # Filter out ignored patterns
        items = [
            item for item in items 
            if not any(
                pattern in item.name or item.name.startswith('.') and pattern.startswith('.')
                for pattern in ignore_patterns
            )
        ]
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            
            # Add item to tree
            if item.is_dir():
                lines.append(f"{prefix}{current_prefix}{item.name}/")
                # Recursively add subdirectory contents
                lines.extend(
                    get_tree_structure(
                        item, 
                        prefix + next_prefix, 
                        max_depth, 
                        current_depth + 1,
                        ignore_patterns
                    )
                )
            else:
                # Get file size
                try:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024**2:
                        size_str = f"{size/1024:.1f}KB"
                    elif size < 1024**3:
                        size_str = f"{size/(1024**2):.1f}MB"
                    else:
                        size_str = f"{size/(1024**3):.1f}GB"
                    
                    lines.append(f"{prefix}{current_prefix}{item.name} ({size_str})")
                except:
                    lines.append(f"{prefix}{current_prefix}{item.name}")
    
    except PermissionError:
        pass
    
    return lines


def generate_markdown(directory, max_depth=None, title=None):
    """Generate markdown formatted project structure"""
    directory = Path(directory).resolve()
    
    if title is None:
        title = f"Project Structure: {directory.name}"
    
    markdown = []
    markdown.append(f"# {title}\n")
    markdown.append(f"**Root:** `{directory}`\n")
    markdown.append("```")
    markdown.append(f"{directory.name}/")
    
    tree_lines = get_tree_structure(directory, max_depth=max_depth)
    markdown.extend(tree_lines)
    
    markdown.append("```\n")
    
    # Add statistics
    total_files = sum(1 for line in tree_lines if not line.strip().endswith('/'))
    total_dirs = sum(1 for line in tree_lines if line.strip().endswith('/'))
    
    markdown.append(f"**Statistics:**")
    markdown.append(f"- Total directories: {total_dirs}")
    markdown.append(f"- Total files: {total_files}")
    
    return "\n".join(markdown)


def main():
    """Main function"""
    # Parse arguments
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."
    
    # Options
    max_depth = None  # Set to a number to limit depth (e.g., 3)
    output_file = "PROJECT_STRUCTURE.md"
    
    # Check if directory exists
    if not Path(directory).exists():
        print(f"Error: Directory '{directory}' does not exist!")
        sys.exit(1)
    
    # Generate markdown
    print(f"Generating project structure for: {Path(directory).resolve()}")
    markdown = generate_markdown(directory, max_depth=max_depth)
    
    # Print to console
    print("\n" + "="*60)
    print(markdown)
    print("="*60 + "\n")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"✓ Saved to {output_file}")
    
    # Also create a detailed version with descriptions
    create_detailed_structure(directory, "PROJECT_STRUCTURE_DETAILED.md")


def create_detailed_structure(directory, output_file):
    """Create detailed structure with file descriptions"""
    directory = Path(directory).resolve()
    
    markdown = []
    markdown.append(f"# Detailed Project Structure\n")
    markdown.append(f"**Root:** `{directory}`\n")
    
    # Python files
    markdown.append("## Python Files\n")
    for py_file in sorted(directory.rglob("*.py")):
        if '.venv' not in str(py_file) and '__pycache__' not in str(py_file):
            rel_path = py_file.relative_to(directory)
            size = py_file.stat().st_size
            markdown.append(f"- **{rel_path}** ({size} bytes)")
            
            # Try to get docstring
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content:
                        # Extract first docstring
                        start = content.find('"""') + 3
                        end = content.find('"""', start)
                        if end > start:
                            docstring = content[start:end].strip().split('\n')[0]
                            markdown.append(f"  - {docstring}")
            except:
                pass
            
            markdown.append("")
    
    # Text/markdown files
    markdown.append("## Documentation Files\n")
    for ext in ['*.md', '*.txt', '*.rst']:
        for doc_file in sorted(directory.rglob(ext)):
            if '.venv' not in str(doc_file):
                rel_path = doc_file.relative_to(directory)
                size = doc_file.stat().st_size
                markdown.append(f"- **{rel_path}** ({size} bytes)")
    
    markdown.append("")
    
    # Data files
    markdown.append("## Data Files\n")
    data_extensions = ['*.json', '*.pkl', '*.csv', '*.txt']
    for ext in data_extensions:
        for data_file in sorted(directory.rglob(ext)):
            if 'data' in str(data_file) or 'tokenizers' in str(data_file):
                if '.venv' not in str(data_file):
                    rel_path = data_file.relative_to(directory)
                    size = data_file.stat().st_size
                    if size < 1024**2:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/(1024**2):.1f}MB"
                    markdown.append(f"- **{rel_path}** ({size_str})")
    
    markdown.append("")
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown))
    
    print(f"✓ Saved detailed structure to {output_file}")


if __name__ == "__main__":
    main()