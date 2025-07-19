#!/usr/bin/env python3
"""
Test script to verify ARC system setup.
"""

import sys
import os
from pathlib import Path

# Add tools directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import neo4j
        print("✓ Neo4j imported successfully")
    except ImportError as e:
        print(f"✗ Neo4j import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✓ ChromaDB imported successfully")
    except ImportError as e:
        print(f"✗ ChromaDB import failed: {e}")
        return False
    
    try:
        import spacy
        print("✓ spaCy imported successfully")
    except ImportError as e:
        print(f"✗ spaCy import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✓ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Sentence Transformers import failed: {e}")
        return False
    
    return True


def test_spacy_model():
    """Test that the spaCy model can be loaded."""
    print("\nTesting spaCy model...")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_lg')
        
        # Test with a simple sentence
        doc = nlp("David Curlewis works at Canva and met with Scott from Anyscale.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"✓ spaCy model loaded successfully")
        print(f"  Entities found: {entities}")
        return True
    except Exception as e:
        print(f"✗ spaCy model test failed: {e}")
        return False


def test_sentence_transformer():
    """Test that the sentence transformer model can be loaded."""
    print("\nTesting Sentence Transformer model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test with a simple sentence
        embedding = model.encode(["This is a test sentence."])
        print(f"✓ Sentence Transformer model loaded successfully")
        print(f"  Embedding shape: {embedding.shape}")
        return True
    except Exception as e:
        print(f"✗ Sentence Transformer test failed: {e}")
        return False


def test_chromadb():
    """Test ChromaDB client creation."""
    print("\nTesting ChromaDB...")
    
    try:
        import chromadb
        
        # Create a temporary client
        client = chromadb.PersistentClient(path="./data/chromadb")
        print("✓ ChromaDB client created successfully")
        return True
    except Exception as e:
        print(f"✗ ChromaDB test failed: {e}")
        return False


def test_arc_core():
    """Test our ARC core module."""
    print("\nTesting ARC core module...")
    
    try:
        from arc_core import ARCConfig, get_config, FileProcessor
        
        # Test configuration
        config = get_config()
        print(f"✓ ARC config loaded successfully")
        print(f"  ChromaDB path: {config.get('chromadb.path')}")
        print(f"  spaCy model: {config.get('spacy.model')}")
        
        # Test file processor
        processor = FileProcessor(config)
        import_dir = Path(config.get('import.source_dir', './import'))
        if import_dir.exists():
            files = processor.list_markdown_files()
            print(f"  Found {len(files)} markdown files in import directory")
        else:
            print(f"  Import directory not found: {import_dir}")
        
        return True
    except Exception as e:
        print(f"✗ ARC core test failed: {e}")
        return False


def test_file_structure():
    """Test that required directories exist."""
    print("\nTesting file structure...")
    
    required_dirs = [
        "tools",
        "data",
        "import",
        "memory",
        "prompts",
        "commands",
        "backups"
    ]
    
    all_good = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            all_good = False
    
    return all_good


def main():
    """Run all tests."""
    print("ARC System Setup Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_spacy_model,
        test_sentence_transformer,
        test_chromadb,
        test_arc_core
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ARC setup is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 