#!/usr/bin/env python3
"""
Setup script to download sentence transformer models
Run this after installing requirements.txt
"""

import os
import sys

def setup_sentence_transformers():
    """Download and cache sentence transformer model"""
    try:
        print("Setting up sentence transformers...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        from sentence_transformers import SentenceTransformer
        
        print("Downloading sentence transformer model (all-MiniLM-L6-v2)...")
        print("This is a one-time download (~90MB)...")
        
        # Download and cache the model
        model = SentenceTransformer('all-MiniLM-L6-v2', 
                                   cache_folder=os.path.expanduser('~/.cache/sentence_transformers'))
        
        # Test the model with a simple sentence
        test_embedding = model.encode(["test sentence"])
        
        print("✅ Sentence transformer model downloaded and cached successfully!")
        print(f"Model cached at: {os.path.expanduser('~/.cache/sentence_transformers')}")
        return True
        
    except ImportError:
        print("❌ sentence-transformers not installed. Please install requirements.txt first.")
        return False
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def main():
    print("🧠 Smart Text Categorizer - Model Setup")
    print("=" * 50)
    
    success = setup_sentence_transformers()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("You can now run the text categorizer with full offline capability.")
    else:
        print("\n⚠️  Setup failed. The system will still work with LLM and rule-based fallbacks.")
        sys.exit(1)

if __name__ == "__main__":
    main() 