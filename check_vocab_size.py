import os
import time

def check_dataset(name, path):
    print(f"--- Analyzing {name} ---")
    
    # Handle paths whether running from root or cs336_basics
    if not os.path.exists(path):
        if os.path.exists("../" + path):
            path = "../" + path
        else:
            print(f"❌ File not found: {path}")
            return

    print(f"  Path: {path}")
    
    # 1. Check Disk Size
    file_size_gb = os.path.getsize(path) / (1024**3)
    print(f"  Disk Size: {file_size_gb:.2f} GB")

    # 2. Count Characters (Streaming)
    print("  Counting characters (this may take a minute for large files)...")
    start_time = time.time()
    char_count = 0
    
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            # Read in 1MB chunks to avoid memory overflow
            for chunk in iter(lambda: f.read(1_000_000), ''):
                char_count += len(chunk)
                
        elapsed = time.time() - start_time
        print(f"  ✅ Total Characters: {char_count:,}")
        
        # Estimate Token Count (Roughly 1 token ~= 4 chars for English text)
        est_tokens = char_count / 4
        est_bin_size_gb = (est_tokens * 2) / (1024**3) # 2 bytes per token (uint16)
        
        print(f"  ℹ️  Estimated Tokens: ~{int(est_tokens):,}")
        print(f"  ℹ️  Estimated Binary File Size Needed: ~{est_bin_size_gb:.2f} GB")
        
    except Exception as e:
        print(f"  Error reading file: {e}")
    print("")

# Checking the files requested
# Note: I corrected the mapping based on the filenames provided in your prompt.
check_dataset("OpenWebText", "data/owt_train.txt")
check_dataset("TinyStories", "data/TinyStoriesV2-GPT4-train.txt")