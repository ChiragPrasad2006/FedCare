#!/usr/bin/env python3
"""
Run all FedProx tests and generate report

This script:
1. Validates FedProx implementation
2. Runs full test suite
3. Generates validation report
4. Provides summary and recommendations

Usage:
  python run_tests.py
"""

import sys
import os
import subprocess
import json
from datetime import datetime

def run_command(cmd, description):
    """Run command and return result"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        print(f"ERROR: {e}")
        return False, "", str(e)

def main():
    """Main test runner"""
    print("\n" + "="*70)
    print("FedCare FedProx Test Suite")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Step 1: Validation
    print("\n[1/2] Running FedProx Implementation Validation...")
    success, output, error = run_command(
        "python tests/validate_fedprox_alt.py",
        "FedProx Implementation Validation"
    )
    results['validation'] = {
        'success': success,
        'output': output[:500],  # Store first 500 chars
        'error': error[:500] if error else None
    }
    
    if not success:
        print("\n[!] Validation failed. Review errors above.")
        print("Fix issues before running full test suite.")
    else:
        print("\n[OK] Validation passed!")
    
    # Step 2: Full test suite (only if validation passed)
    if success:
        print("\n[2/2] Running Full Test Suite...")
        success, output, error = run_command(
            "python tests/test_fedprox.py",
            "Full FedProx Test Suite"
        )
        results['tests'] = {
            'success': success,
            'output': output[-1000:],  # Store last 1000 chars (summary)
            'error': error[-500:] if error else None
        }
    else:
        results['tests'] = {
            'success': False,
            'output': 'Skipped due to validation failure',
            'error': None
        }
    
    # Summary
    print("\n" + "="*70)
    print("TEST EXECUTION SUMMARY")
    print("="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Validation: {'[PASSED]' if results['validation']['success'] else '[FAILED]'}")
    print(f"Test Suite: {'[PASSED]' if results['tests']['success'] else '[FAILED/SKIPPED]'}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if results['validation']['success'] and results['tests']['success']:
        print("""
[OK] All tests passed! FedProx implementation is complete and working.

Next Steps:
1. Review test output above for details
2. Run examples to test functionality:
   python examples/fedprox_personalization_example.py
   python examples/main_server_training_example.py
3. Start local servers:
   python main_server/app.py
   python hospital_server/app.py
4. Run orchestrator:
   python orchestrator.py
5. Ready for production deployment
        """)
        return 0
    elif results['validation']['success']:
        print("""
[!] Validation passed but some tests failed.

Review test failures above for details. Common issues:
- TensorFlow/NumPy version mismatches
- Missing test dependencies
- Model weight shape incompatibilities

Next Steps:
1. Install test dependencies: pip install pytest coverage
2. Run with more verbosity: pytest tests/ -v --tb=long
3. Check system TensorFlow/NumPy versions
        """)
        return 1
    else:
        print("""
[FAIL] Validation failed. FedProx implementation has issues.

Review validation output above. Common issues:
- Missing config parameters
- Endpoints not registered
- Import failures
- Corrupted files

Next Steps:
1. Check error messages in validation output
2. Verify all files in FedCare directory
3. Reinstall dependencies: pip install -r requirements.txt
4. Review documentation: docs/FEDPROX_IMPLEMENTATION.md
        """)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
