#!/usr/bin/env python3
"""
Quick verification script for BPE implementation
Run this before running the actual pytest tests
"""

import os
import tempfile
import time
from pathlib import Path


def test_basic_functionality():
    """Test basic BPE training on a tiny corpus"""
    print("="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    # Create a simple test corpus
    test_corpus = """Once upon a time there was a little girl.
She loved to play in the garden.
The garden had many flowers.
The flowers were red and blue.
She was very happy."""
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(test_corpus)
        temp_path = f.name
    
    try:
        from bpe import train_bpe
        
        print(f"Training on {len(test_corpus)} characters...")
        start_time = time.time()
        
        vocab, merges = train_bpe(
            input_path=temp_path,
            vocab_size=300,
            special_tokens=[]
        )
        
        end_time = time.time()
        
        print(f"✓ Training completed in {end_time - start_time:.3f} seconds")
        print(f"✓ Vocabulary size: {len(vocab)}")
        print(f"✓ Number of merges: {len(merges)}")
        
        # Check basic properties
        assert len(vocab) == 256 + len(merges), "Vocab size mismatch"
        assert all(isinstance(k, int) for k in vocab.keys()), "Vocab keys should be int"
        assert all(isinstance(v, bytes) for v in vocab.values()), "Vocab values should be bytes"
        assert all(isinstance(m, tuple) and len(m) == 2 for m in merges), "Merges should be 2-tuples"
        
        # Print first few merges
        print("\nFirst 5 merges:")
        for i, (b1, b2) in enumerate(merges[:5]):
            merged = b1 + b2
            try:
                print(f"  {i+1}. {b1.decode('utf-8')!r} + {b2.decode('utf-8')!r} -> {merged.decode('utf-8')!r}")
            except:
                print(f"  {i+1}. {b1!r} + {b2!r} -> {merged!r}")
        
        print("\n✅ Basic functionality test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(temp_path)


def test_special_tokens():
    """Test that special tokens are handled correctly"""
    print("\n" + "="*60)
    print("TEST 2: Special Token Handling")
    print("="*60)
    
    # Create corpus with special tokens
    test_corpus = "Hello world<|endoftext|>Goodbye world<|endoftext|>Test test test"
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(test_corpus)
        temp_path = f.name
    
    try:
        from bpe import train_bpe
        
        print(f"Training with special token '<|endoftext|>'...")
        
        vocab, merges = train_bpe(
            input_path=temp_path,
            vocab_size=300,
            special_tokens=["<|endoftext|>"]
        )
        
        print(f"✓ Vocabulary size: {len(vocab)}")
        print(f"✓ Number of merges: {len(merges)}")
        
        # Check special token is in vocab
        special_bytes = b"<|endoftext|>"
        if special_bytes in vocab.values():
            print(f"✓ Special token {special_bytes!r} found in vocab")
        else:
            print(f"❌ Special token {special_bytes!r} NOT in vocab")
            return False
        
        # Check no other tokens contain "<|"
        bad_tokens = []
        for token_id, token_bytes in vocab.items():
            if token_bytes != special_bytes and b"<|" in token_bytes:
                bad_tokens.append((token_id, token_bytes))
        
        if bad_tokens:
            print(f"❌ Found {len(bad_tokens)} tokens containing '<|':")
            for tid, tb in bad_tokens[:5]:
                print(f"  {tid}: {tb!r}")
            return False
        else:
            print("✓ No other tokens contain '<|'")
        
        print("\n✅ Special token test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Special token test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(temp_path)


def test_adapter():
    """Test that the adapter function works"""
    print("\n" + "="*60)
    print("TEST 3: Adapter Function")
    print("="*60)
    
    test_corpus = "test test test hello hello world"
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(test_corpus)
        temp_path = f.name
    
    try:
        from adapters import run_train_bpe
        
        print("Testing run_train_bpe adapter...")
        
        vocab, merges = run_train_bpe(
            input_path=temp_path,
            vocab_size=280,
            special_tokens=[]
        )
        
        print(f"✓ Adapter function works")
        print(f"✓ Vocabulary size: {len(vocab)}")
        print(f"✓ Number of merges: {len(merges)}")
        
        print("\n✅ Adapter test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Adapter test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(temp_path)


def test_data_structures():
    """Verify correct data structures are used"""
    print("\n" + "="*60)
    print("TEST 4: Data Structure Verification")
    print("="*60)
    
    test_corpus = "low low low lower lower"
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(test_corpus)
        temp_path = f.name
    
    try:
        from bpe import train_bpe
        
        vocab, merges = train_bpe(
            input_path=temp_path,
            vocab_size=270,
            special_tokens=[]
        )
        
        # Check vocab structure
        print("Checking vocab structure...")
        assert isinstance(vocab, dict), "vocab should be dict"
        for k, v in list(vocab.items())[:5]:
            assert isinstance(k, int), f"vocab key {k} should be int"
            assert isinstance(v, bytes), f"vocab value {v} should be bytes"
        print("✓ vocab is dict[int, bytes]")
        
        # Check merges structure
        print("Checking merges structure...")
        assert isinstance(merges, list), "merges should be list"
        for m in merges[:5]:
            assert isinstance(m, tuple), f"merge {m} should be tuple"
            assert len(m) == 2, f"merge {m} should have length 2"
            assert isinstance(m[0], bytes), f"merge[0] {m[0]} should be bytes"
            assert isinstance(m[1], bytes), f"merge[1] {m[1]} should be bytes"
        print("✓ merges is list[tuple[bytes, bytes]]")
        
        # Check first 256 entries are single bytes
        print("Checking initial byte vocabulary...")
        for i in range(256):
            assert i in vocab, f"Missing byte {i}"
            assert vocab[i] == bytes([i]), f"vocab[{i}] should be bytes([{i}])"
        print("✓ First 256 entries are single bytes")
        
        print("\n✅ Data structure test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Data structure test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(temp_path)


def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("BPE IMPLEMENTATION VERIFICATION")
    print("="*60)
    print("\nThis script verifies your BPE implementation before running pytest")
    print("It checks:")
    print("  1. Basic functionality")
    print("  2. Special token handling")
    print("  3. Adapter function")
    print("  4. Data structures")
    print()
    
    results = []
    
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Special Tokens", test_special_tokens()))
    results.append(("Adapter Function", test_adapter()))
    results.append(("Data Structures", test_data_structures()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("🎉 ALL VERIFICATION TESTS PASSED!")
        print("="*60)
        print("\nYou can now run the actual pytest tests:")
        print("  uv run pytest tests/test_train_bpe.py -v")
    else:
        print("\n" + "="*60)
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues before running pytest tests.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)