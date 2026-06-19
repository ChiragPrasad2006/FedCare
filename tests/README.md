# FedCare Testing Guide

## Overview

This directory contains comprehensive tests for the FedProx personalization implementation.

## Test Files

### 1. `validate_fedprox.py` - Quick Validation Script

**Purpose**: Fast validation that all FedProx components are properly integrated

**Usage**:
```bash
python tests/validate_fedprox.py
```

**What It Checks**:
- ✅ All imports work correctly
- ✅ Configuration parameters load properly
- ✅ FedProx functions execute without errors
- ✅ Hospital server endpoints are registered
- ✅ Main server endpoints are registered
- ✅ Orchestrator integration is correct
- ✅ Documentation files exist

**Expected Output**:
```
✅ All imports successful
✅ Configuration loaded
✅ Functions working
✅ Hospital endpoints registered
✅ Main server endpoints registered
✅ Orchestrator compatible
✅ Documentation complete

🎉 ALL VALIDATIONS PASSED!
```

### 2. `test_fedprox.py` - Comprehensive Test Suite

**Purpose**: Full unit and integration tests for FedProx

**Usage** (with unittest):
```bash
python tests/test_fedprox.py
```

**Usage** (with pytest):
```bash
pip install pytest
pytest tests/test_fedprox.py -v
```

**Test Classes**:

1. **TestFedProxFunctions** - Core FedProx functions
   - `test_create_simple_model()` - Model creation
   - `test_create_federated_model()` - Federated model creation
   - `test_get_set_model_weights()` - Weight management
   - `test_compute_proximal_term()` - Proximal term computation
   - `test_train_with_fedprox_basic()` - FedProx without proximal term
   - `test_train_with_fedprox_proximal()` - FedProx with proximal term
   - `test_personalize_model_standard()` - Standard personalization
   - `test_average_weights()` - Weight averaging

2. **TestConfiguration** - Config parameter validation
   - `test_personalization_config_exists()` - Parameters exist
   - `test_personalization_config_types()` - Correct types
   - `test_personalization_config_values()` - Valid ranges

3. **TestHospitalServerEndpoints** - Hospital server validation
   - `test_personalize_endpoint_exists()` - /personalize endpoint
   - `test_personalize_fedprox_endpoint_exists()` - /personalize_fedprox endpoint
   - `test_personalization_history_endpoint_exists()` - /personalization_history endpoint

4. **TestMainServerEndpoints** - Main server validation
   - `test_trigger_personalization_endpoint_exists()` - /trigger_personalization endpoint
   - `test_personalization_metrics_endpoint_exists()` - /personalization_metrics endpoint

5. **TestOrchestratorIntegration** - Orchestrator validation
   - `test_orchestrator_run_federated_learning_signature()` - Method signature

6. **TestEndToEndValidation** - Full pipeline tests
   - `test_fedprox_training_pipeline()` - Complete training workflow
   - `test_personalization_workflow()` - Complete personalization workflow

7. **TestMetricsCollection** - Metrics validation
   - `test_fedprox_metrics_structure()` - Metrics structure validation

**Expected Output**:
```
test_create_simple_model ... ok
test_compute_proximal_term ... ok
test_train_with_fedprox_proximal ... ok
...
======================== TEST SUMMARY ========================
Tests run: 28
Successes: 28
Failures: 0
Errors: 0

✅ ALL TESTS PASSED!
```

## Running Tests

### Quick Validation (30 seconds)
```bash
python tests/validate_fedprox.py
```

### Full Test Suite (2-3 minutes)
```bash
python tests/test_fedprox.py
```

### With pytest (more details)
```bash
pytest tests/ -v --tb=short
```

### Generate coverage report
```bash
pip install coverage
coverage run -m pytest tests/
coverage report
```

## Test Requirements

**Python Packages**:
- tensorflow >= 2.13.0
- numpy >= 1.24.3
- flask >= 2.3.2 (for endpoint tests)

**Optional**:
- pytest (for pytest runner)
- coverage (for coverage reports)

Install with:
```bash
pip install -r requirements.txt
```

## Interpreting Results

### ✅ All Validations Pass
- FedProx is properly implemented
- Ready for local testing
- Ready for production deployment

### ❌ Validation Failures
- Check error message for specific issue
- Review corresponding documentation
- Verify all files were created
- Check Python environment

### Test Failures
- Review assertion details
- Check test logs for stack traces
- Ensure TensorFlow/NumPy versions compatible
- Verify test data generation

## Quick Troubleshooting

### Import Errors
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Function Test Failures
```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Check NumPy version
python -c "import numpy as np; print(np.__version__)"
```

### Endpoint Test Failures
```bash
# Check Flask installation
python -c "import flask; print(flask.__version__)"

# Verify app.py files not modified
ls -la hospital_server/app.py
ls -la main_server/app.py
```

## Performance Expectations

| Test | Time |
|------|------|
| Validation Script | 30-60s |
| Unit Tests (28 total) | 2-3 min |
| Full Test Suite | 5-10 min |
| Coverage Report | 2-3 min |

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Test FedProx

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - run: pip install -r requirements.txt
      - run: python tests/validate_fedprox.py
      - run: python tests/test_fedprox.py
```

## Next Steps

1. **Run Validation**: `python tests/validate_fedprox.py`
2. **Review Results**: Check for any failures
3. **Run Full Tests**: `python tests/test_fedprox.py` (if validation passes)
4. **Manual Testing**: Run examples from `examples/` directory
5. **Deployment**: If all tests pass, ready for production

## Support

For issues or questions:
1. Review test output carefully
2. Check logs in `./logs/`
3. Review documentation in `docs/`
4. Run specific test class: `python -m pytest tests/test_fedprox.py::TestFedProxFunctions -v`

