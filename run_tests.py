#!/usr/bin/env python3
"""
ARC Test Runner

Convenient script for running different types of tests for the ARC system.

Usage:
    python run_tests.py [command] [options]

Commands:
    all          - Run all tests
    unit         - Run unit tests only
    integration  - Run integration tests only
    coverage     - Run tests with coverage report
    quick        - Run quick smoke tests
    setup        - Test system setup and dependencies
    
Options:
    -v, --verbose    - Verbose output
    -q, --quiet      - Quiet output
    --no-deps        - Skip dependency tests
    --parallel       - Run tests in parallel (if supported)
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add tools and tests to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'tools'))
sys.path.insert(0, str(project_root / 'tests'))


def run_command(cmd, capture_output=True, check=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True if isinstance(cmd, str) else False,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return None


def test_dependencies():
    """Test that all required dependencies are available."""
    print("🔍 Testing dependencies...")
    
    dependencies = [
        'pytest',
        'pytest-mock',
        'pytest-cov',
        'neo4j',
        'chromadb',
        'spacy',
        'sentence_transformers',
        'pandas',
        'numpy'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} - MISSING")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True


def test_spacy_model():
    """Test that spaCy model is available."""
    print("🔍 Testing spaCy model...")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_lg')
        print("  ✓ en_core_web_lg model loaded")
        
        # Quick test
        doc = nlp("David works at Canva")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"  ✓ Entity extraction working: {entities}")
        return True
    except OSError:
        print("  ✗ en_core_web_lg model not found")
        print("Run: python -m spacy download en_core_web_lg")
        return False
    except Exception as e:
        print(f"  ✗ spaCy test failed: {e}")
        return False


def test_setup():
    """Test overall system setup."""
    print("🧪 Testing ARC System Setup")
    print("=" * 50)
    
    success = True
    
    # Test dependencies
    if not test_dependencies():
        success = False
    
    print()
    
    # Test spaCy model
    if not test_spacy_model():
        success = False
    
    print()
    
    # Test basic imports
    print("🔍 Testing core imports...")
    try:
        from arc_core import get_config, DatabaseManager, ARCConfig
        from arc_query import query_person, semantic_search
        print("  ✓ Core modules import successfully")
    except Exception as e:
        print(f"  ✗ Core import failed: {e}")
        success = False
    
    print()
    
    # Test configuration
    print("🔍 Testing configuration...")
    try:
        from arc_core import get_config
        config = get_config()
        print("  ✓ Configuration loads successfully")
        print(f"  ✓ ChromaDB path: {config.get('chromadb.path')}")
        print(f"  ✓ Neo4j URI: {config.get('neo4j.uri')}")
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 System setup complete! Ready for testing.")
        return True
    else:
        print("❌ System setup incomplete. Please fix the errors above.")
        return False


def run_quick_tests():
    """Run quick smoke tests."""
    print("🏃 Running quick smoke tests...")
    
    try:
        from tests.utils.test_helpers import quick_test
        return quick_test()
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False


def run_unit_tests(verbose=False, parallel=False):
    """Run unit tests."""
    print("🧪 Running unit tests...")
    
    cmd = ["python", "-m", "pytest", "tests/unit/", "-m", "unit"]
    
    if verbose:
        cmd.append("-v")
    if parallel:
        cmd.extend(["-n", "auto"])
    
    result = run_command(cmd, capture_output=False)
    return result is not None and result.returncode == 0


def run_integration_tests(verbose=False, parallel=False):
    """Run integration tests."""
    print("🔗 Running integration tests...")
    
    cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    if parallel:
        cmd.extend(["-n", "auto"])
    
    result = run_command(cmd, capture_output=False)
    return result is not None and result.returncode == 0


def run_coverage_tests():
    """Run tests with coverage report."""
    print("📊 Running tests with coverage...")
    
    cmd = [
        "python", "-m", "pytest",
        "--cov=tools",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml"
    ]
    
    result = run_command(cmd, capture_output=False)
    
    if result and result.returncode == 0:
        print("\n📈 Coverage report generated:")
        print("  - HTML: htmlcov/index.html")
        print("  - XML: coverage.xml")
        print("  - Terminal: see output above")
        return True
    
    return False


def run_all_tests(verbose=False, parallel=False):
    """Run all tests."""
    print("🎯 Running all tests...")
    
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    if parallel:
        cmd.extend(["-n", "auto"])
    
    result = run_command(cmd, capture_output=False)
    return result is not None and result.returncode == 0


def install_test_dependencies():
    """Install additional test dependencies if needed."""
    print("📦 Installing test dependencies...")
    
    test_deps = [
        "pytest-xdist",  # for parallel testing
        "pytest-html",   # for HTML reports
        "pytest-mock",   # for mocking
        "pytest-cov",    # for coverage
    ]
    
    for dep in test_deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            print(f"Installing {dep}...")
            result = run_command([sys.executable, "-m", "pip", "install", dep])
            if not result or result.returncode != 0:
                print(f"Failed to install {dep}")
                return False
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ARC Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'command',
        choices=['all', 'unit', 'integration', 'coverage', 'quick', 'setup', 'deps'],
        help='Test command to run'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet output'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel (requires pytest-xdist)'
    )
    
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install missing test dependencies'
    )
    
    args = parser.parse_args()
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            sys.exit(1)
    
    # Change to project directory
    os.chdir(project_root)
    
    # Run the requested command
    success = False
    
    if args.command == 'setup':
        success = test_setup()
    elif args.command == 'deps':
        success = test_dependencies()
    elif args.command == 'quick':
        success = run_quick_tests()
    elif args.command == 'unit':
        success = run_unit_tests(verbose=verbose, parallel=args.parallel)
    elif args.command == 'integration':
        success = run_integration_tests(verbose=verbose, parallel=args.parallel)
    elif args.command == 'coverage':
        success = run_coverage_tests()
    elif args.command == 'all':
        success = run_all_tests(verbose=verbose, parallel=args.parallel)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 