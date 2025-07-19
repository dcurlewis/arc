#!/usr/bin/env python3
"""
Comprehensive test runner for the Enhanced ARC System.
Runs all test suites and provides detailed reporting.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status}")
        
        return success, result.stdout + result.stderr
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, str(e)


def main():
    """Run comprehensive test suite."""
    print("🚀 ARC Enhanced System Comprehensive Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    results = {}
    
    # Test suites to run
    test_suites = [
        {
            'name': 'Enhanced Entity Extractor Tests',
            'cmd': ['python', '-m', 'pytest', 'tests/unit/test_enhanced_entity_extractor.py', '-v'],
            'critical': True
        },
        {
            'name': 'Enhanced Embeddings Tests',
            'cmd': ['python', '-m', 'pytest', 'tests/unit/test_enhanced_embeddings.py', '-v'],
            'critical': True
        },
        {
            'name': 'MCP Server Integration Tests',
            'cmd': ['python', '-m', 'pytest', 'tests/integration/test_mcp_server_integration.py', '-v'],
            'critical': True
        },
        {
            'name': 'Existing Unit Tests',
            'cmd': ['python', '-m', 'pytest', 'tests/unit/', '-v', '--tb=short'],
            'critical': False
        },
        {
            'name': 'Existing Integration Tests',
            'cmd': ['python', '-m', 'pytest', 'tests/integration/', '-v', '--tb=short'],
            'critical': False
        },
        {
            'name': 'Enhanced Entity Extractor (Standalone)',
            'cmd': ['python', 'test_enhanced_extractor.py'],
            'critical': False
        },
        {
            'name': 'Enhanced Embeddings (Standalone)',
            'cmd': ['python', 'test_enhanced_embeddings.py'],
            'critical': False
        }
    ]
    
    # Run each test suite
    for suite in test_suites:
        success, output = run_command(suite['cmd'], suite['name'])
        results[suite['name']] = {
            'success': success,
            'output': output,
            'critical': suite['critical']
        }
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    critical_failures = 0
    
    for name, result in results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        critical = " (CRITICAL)" if result['critical'] else ""
        print(f"{status} {name}{critical}")
        
        if result['success']:
            passed += 1
        else:
            failed += 1
            if result['critical']:
                critical_failures += 1
    
    print(f"\n📈 Total: {len(results)} suites")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"🚨 Critical Failures: {critical_failures}")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    
    # Overall status
    if critical_failures > 0:
        print(f"\n🚨 CRITICAL FAILURES DETECTED!")
        print("The enhanced ARC system has critical issues that need to be addressed.")
        return_code = 2
    elif failed > 0:
        print(f"\n⚠️  Some tests failed, but core functionality is working.")
        return_code = 1
    else:
        print(f"\n🎉 ALL TESTS PASSED! Enhanced ARC system is fully functional.")
        return_code = 0
    
    # Recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if critical_failures > 0:
        print("1. Fix critical test failures before using the system")
        print("2. Review error messages above for specific issues")
        print("3. Check dependencies and configuration files")
    elif failed > 0:
        print("1. Review non-critical test failures when convenient")
        print("2. Core enhanced functionality is working correctly")
        print("3. System is ready for use")
    else:
        print("1. ✅ All enhanced features are working correctly")
        print("2. ✅ Entity extraction with disambiguation is functional")
        print("3. ✅ Enhanced embeddings and search are operational") 
        print("4. ✅ MCP server integration is complete")
        print("5. 🚀 System is ready for production use!")
    
    # Next steps
    print(f"\n🔮 NEXT STEPS")
    print("-" * 20)
    if return_code == 0:
        print("• Test the enhanced system with real data queries")
        print("• Explore the new hybrid search capabilities")
        print("• Try entity-centric and relationship searches")
        print("• Experiment with temporal filtering")
    else:
        print("• Address test failures shown above")
        print("• Re-run tests after fixes: python tools/run_comprehensive_tests.py")
        print("• Check system logs for additional debugging info")
    
    sys.exit(return_code)


if __name__ == "__main__":
    main() 