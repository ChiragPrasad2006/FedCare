#!/usr/bin/env python3
"""
FedProx Implementation Validation Script

Quick validation that all FedProx components are properly integrated.
Checks:
- Imports work correctly
- Configuration loads
- Functions are accessible
- Endpoints are registered
- State management works

Usage:
  python tests/validate_fedprox.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Validator:
    """FedProx Implementation Validator"""
    
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
    
    def log(self, status, message):
        """Log validation result"""
        symbol = "✅" if status else "❌"
        print(f"{symbol} {message}")
        self.checks.append((status, message))
        
        if status:
            self.passed += 1
        else:
            self.failed += 1
    
    def section(self, title):
        """Print section header"""
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
    
    def validate_imports(self):
        """Validate all imports work"""
        self.section("1. IMPORT VALIDATION")
        
        try:
            from shared import (
                create_federated_model,
                create_simple_model,
                compile_model,
                get_model_weights,
                set_model_weights,
                average_weights,
                create_fedprox_model,
                compute_proximal_term,
                train_with_fedprox,
                personalize_model,
                setup_logger
            )
            self.log(True, "All FedProx functions imported successfully")
        except ImportError as e:
            self.log(False, f"Failed to import FedProx functions: {e}")
            return False
        
        try:
            from shared.config import (
                PERSONALIZATION_ENABLED,
                PROXIMAL_MU,
                PERSONALIZATION_ROUNDS,
                PERSONAL_EPOCHS_PER_ROUND
            )
            self.log(True, "All FedProx config parameters imported successfully")
        except ImportError as e:
            self.log(False, f"Failed to import config: {e}")
            return False
        
        return True
    
    def validate_config(self):
        """Validate configuration"""
        self.section("2. CONFIGURATION VALIDATION")
        
        try:
            from shared.config import (
                PERSONALIZATION_ENABLED,
                PROXIMAL_MU,
                PERSONALIZATION_ROUNDS,
                PERSONAL_EPOCHS_PER_ROUND
            )
            
            # Check types
            if not isinstance(PERSONALIZATION_ENABLED, bool):
                self.log(False, f"PERSONALIZATION_ENABLED should be bool, got {type(PERSONALIZATION_ENABLED)}")
            else:
                self.log(True, f"PERSONALIZATION_ENABLED: {PERSONALIZATION_ENABLED}")
            
            if not isinstance(PROXIMAL_MU, float):
                self.log(False, f"PROXIMAL_MU should be float, got {type(PROXIMAL_MU)}")
            else:
                self.log(True, f"PROXIMAL_MU: {PROXIMAL_MU}")
            
            if not isinstance(PERSONALIZATION_ROUNDS, int):
                self.log(False, f"PERSONALIZATION_ROUNDS should be int, got {type(PERSONALIZATION_ROUNDS)}")
            else:
                self.log(True, f"PERSONALIZATION_ROUNDS: {PERSONALIZATION_ROUNDS}")
            
            if not isinstance(PERSONAL_EPOCHS_PER_ROUND, int):
                self.log(False, f"PERSONAL_EPOCHS_PER_ROUND should be int, got {type(PERSONAL_EPOCHS_PER_ROUND)}")
            else:
                self.log(True, f"PERSONAL_EPOCHS_PER_ROUND: {PERSONAL_EPOCHS_PER_ROUND}")
            
            # Check value ranges
            if not (0.0 <= PROXIMAL_MU <= 1.0):
                self.log(False, f"PROXIMAL_MU out of range: {PROXIMAL_MU}")
            else:
                self.log(True, "PROXIMAL_MU in valid range [0.0, 1.0]")
            
            if not (1 <= PERSONAL_EPOCHS_PER_ROUND <= 50):
                self.log(False, f"PERSONAL_EPOCHS_PER_ROUND out of range: {PERSONAL_EPOCHS_PER_ROUND}")
            else:
                self.log(True, "PERSONAL_EPOCHS_PER_ROUND in valid range [1, 50]")
            
        except Exception as e:
            self.log(False, f"Configuration validation failed: {e}")
            return False
        
        return True
    
    def validate_functions(self):
        """Validate FedProx functions"""
        self.section("3. FUNCTION VALIDATION")
        
        try:
            import tensorflow as tf
            import numpy as np
            from shared import (
                create_simple_model,
                get_model_weights,
                set_model_weights,
                compute_proximal_term,
                train_with_fedprox,
                personalize_model
            )
            from shared.config import LEARNING_RATE, INPUT_SHAPE, NUM_CLASSES
            
            # Create model
            model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            self.log(True, "Model created successfully")
            
            # Get weights
            weights = get_model_weights(model)
            self.log(True, f"get_model_weights works (got {len(weights)} layers)")
            
            # Set weights
            set_model_weights(model, weights)
            self.log(True, "set_model_weights works")
            
            # Compute proximal term
            proximal = compute_proximal_term(weights, weights)
            if proximal == 0.0:
                self.log(True, "compute_proximal_term works (identical weights = 0)")
            else:
                self.log(False, f"compute_proximal_term should return 0 for identical weights, got {proximal}")
            
            # Test data
            X_test = np.random.randn(50, 28, 28, 1).astype('float32')
            y_test = np.random.randint(0, 10, 50)
            
            # Train with FedProx
            metrics = train_with_fedprox(
                model=model,
                X_train=X_test,
                y_train=y_test,
                global_weights=weights,
                epochs=1,
                batch_size=32,
                verbose=0
            )
            
            if all(k in metrics for k in ['loss', 'accuracy', 'proximal_term', 'is_personalized']):
                self.log(True, "train_with_fedprox returns proper metrics")
            else:
                self.log(False, "train_with_fedprox missing required metrics")
            
            # Personalize model
            model2 = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
            model2.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            metrics2 = personalize_model(
                model=model2,
                X_train=X_test,
                y_train=y_test,
                global_weights=weights,
                epochs=1,
                verbose=0
            )
            
            if all(k in metrics2 for k in ['loss', 'accuracy']):
                self.log(True, "personalize_model returns proper metrics")
            else:
                self.log(False, "personalize_model missing required metrics")
            
        except Exception as e:
            self.log(False, f"Function validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def validate_hospital_endpoints(self):
        """Validate hospital server endpoints"""
        self.section("4. HOSPITAL SERVER ENDPOINTS")
        
        try:
            from hospital_server.app import app
            
            endpoints = [rule.rule for rule in app.url_map.iter_rules() if 'personali' in rule.rule]
            
            if '/personalize' in str(endpoints):
                self.log(True, "/personalize endpoint registered")
            else:
                self.log(False, "/personalize endpoint not found")
            
            if '/personalize_fedprox' in str(endpoints):
                self.log(True, "/personalize_fedprox endpoint registered")
            else:
                self.log(False, "/personalize_fedprox endpoint not found")
            
            if '/personalization_history' in str(endpoints):
                self.log(True, "/personalization_history endpoint registered")
            else:
                self.log(False, "/personalization_history endpoint not found")
            
            # Check for state variables
            import hospital_server.app as app_module
            if hasattr(app_module, 'global_model_reference'):
                self.log(True, "global_model_reference variable exists")
            else:
                self.log(False, "global_model_reference variable missing")
            
            if hasattr(app_module, 'personalization_history'):
                self.log(True, "personalization_history variable exists")
            else:
                self.log(False, "personalization_history variable missing")
            
        except Exception as e:
            self.log(False, f"Hospital endpoint validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def validate_main_endpoints(self):
        """Validate main server endpoints"""
        self.section("5. MAIN SERVER ENDPOINTS")
        
        try:
            from main_server.app import app
            
            endpoints = [rule.rule for rule in app.url_map.iter_rules() if 'personali' in rule.rule]
            
            if '/trigger_personalization' in str(endpoints):
                self.log(True, "/trigger_personalization endpoint registered")
            else:
                self.log(False, "/trigger_personalization endpoint not found")
            
            if '/personalization_metrics' in str(endpoints):
                self.log(True, "/personalization_metrics endpoint registered")
            else:
                self.log(False, "/personalization_metrics endpoint not found")
            
            # Check for state variables
            import main_server.app as app_module
            if hasattr(app_module, 'personalization_metrics'):
                self.log(True, "personalization_metrics variable exists")
            else:
                self.log(False, "personalization_metrics variable missing")
            
        except Exception as e:
            self.log(False, f"Main endpoint validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def validate_orchestrator(self):
        """Validate orchestrator integration"""
        self.section("6. ORCHESTRATOR INTEGRATION")
        
        try:
            from orchestrator import FederatedLearningOrchestrator
            import inspect
            
            # Check method signature
            sig = inspect.signature(FederatedLearningOrchestrator.run_federated_learning)
            
            if 'enable_personalization' in sig.parameters:
                self.log(True, "run_federated_learning has enable_personalization parameter")
            else:
                self.log(False, "enable_personalization parameter missing")
            
            # Check default value
            param = sig.parameters.get('enable_personalization')
            if param and param.default is False:
                self.log(True, "enable_personalization defaults to False (backward compatible)")
            else:
                self.log(False, "enable_personalization default value incorrect")
            
        except Exception as e:
            self.log(False, f"Orchestrator validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def validate_documentation(self):
        """Validate documentation exists"""
        self.section("7. DOCUMENTATION VALIDATION")
        
        import os
        
        docs_to_check = [
            'docs/FEDPROX_IMPLEMENTATION.md',
            'docs/TRAINING_DATA_SOURCING.md',
            'docs/FEDPROX_IMPLEMENTATION_SUMMARY.md',
            'examples/fedprox_personalization_example.py',
            'examples/main_server_training_example.py'
        ]
        
        for doc_path in docs_to_check:
            full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), doc_path)
            if os.path.exists(full_path):
                size_kb = os.path.getsize(full_path) / 1024
                self.log(True, f"{doc_path} ({size_kb:.1f} KB)")
            else:
                self.log(False, f"{doc_path} not found")
    
    def run_all(self):
        """Run all validations"""
        print("\n" + "="*70)
        print("FedProx Implementation Validation")
        print("="*70)
        
        self.validate_imports()
        self.validate_config()
        self.validate_functions()
        self.validate_hospital_endpoints()
        self.validate_main_endpoints()
        self.validate_orchestrator()
        self.validate_documentation()
        
        # Summary
        self.section("VALIDATION SUMMARY")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Total: {self.passed + self.failed}")
        
        if self.failed == 0:
            print("\n🎉 ALL VALIDATIONS PASSED!")
            print("FedProx implementation is complete and ready to use.")
            return 0
        else:
            print(f"\n⚠️  {self.failed} validation(s) failed. Please review the errors above.")
            return 1


if __name__ == "__main__":
    validator = Validator()
    exit_code = validator.run_all()
    sys.exit(exit_code)
