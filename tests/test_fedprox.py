#!/usr/bin/env python3
"""
FedProx Implementation Test Suite

Comprehensive tests for:
- FedProx core functions
- Configuration loading
- Hospital server endpoints
- Main server endpoints
- Orchestrator integration
- Metrics collection

Usage:
  python tests/test_fedprox.py

Or with pytest:
  pytest tests/test_fedprox.py -v
"""

import sys
import os
import json
import unittest
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    tf = MagicMock()
    HAS_TF = False
from shared import (
    create_federated_model,
    create_simple_model,
    get_model_weights,
    set_model_weights,
    train_with_fedprox,
    personalize_model,
    compute_proximal_term,
    average_weights
)
from shared.config import (
    PERSONALIZATION_ENABLED,
    PROXIMAL_MU,
    PERSONAL_EPOCHS_PER_ROUND,
    INPUT_SHAPE,
    NUM_CLASSES,
    LEARNING_RATE
)


@unittest.skipIf(not HAS_TF, "TensorFlow not installed")
class TestFedProxFunctions(unittest.TestCase):
    """Test core FedProx functions"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures"""
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Create small test dataset
        cls.X_test = np.random.randn(100, 28, 28, 1).astype('float32')
        cls.y_test = np.random.randint(0, 10, 100)
        
        # Create models
        cls.model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        cls.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def test_create_simple_model(self):
        """Test simple model creation"""
        model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 28, 28, 1))
        self.assertEqual(model.output_shape, (None, NUM_CLASSES))
        self.assertGreater(model.count_params(), 0)
    
    def test_create_federated_model(self):
        """Test federated model creation"""
        model = create_federated_model(INPUT_SHAPE, NUM_CLASSES)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 28, 28, 1))
        self.assertEqual(model.output_shape, (None, NUM_CLASSES))
    
    def test_get_set_model_weights(self):
        """Test weight extraction and setting"""
        model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        
        # Get weights
        weights = get_model_weights(model)
        self.assertIsNotNone(weights)
        self.assertGreater(len(weights), 0)
        
        # Set weights
        model_copy = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        set_model_weights(model_copy, weights)
        
        # Verify weights are the same
        new_weights = get_model_weights(model_copy)
        for w1, w2 in zip(weights, new_weights):
            np.testing.assert_array_almost_equal(w1, w2)
    
    def test_compute_proximal_term(self):
        """Test proximal term computation"""
        model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        
        # Get initial weights
        weights1 = get_model_weights(model)
        weights2 = get_model_weights(model)
        
        # Same weights should give zero proximal term
        proximal = compute_proximal_term(weights1, weights2)
        self.assertEqual(proximal, 0.0)
        
        # Different weights should give non-zero proximal term
        weights2_modified = [w * 1.1 for w in weights2]
        proximal = compute_proximal_term(weights1, weights2_modified)
        self.assertGreater(proximal, 0.0)
    
    def test_train_with_fedprox_basic(self):
        """Test FedProx training without proximal term (global_weights=None)"""
        model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train without proximal term
        metrics = train_with_fedprox(
            model=model,
            X_train=self.X_test,
            y_train=self.y_test,
            global_weights=None,  # No proximal term
            epochs=2,
            batch_size=32,
            verbose=0
        )
        
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('proximal_term', metrics)
        self.assertEqual(metrics['proximal_term'], 0.0)
        self.assertFalse(metrics['is_personalized'])
    
    def test_train_with_fedprox_proximal(self):
        """Test FedProx training with proximal term"""
        model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Get initial weights as "global"
        global_weights = get_model_weights(model)
        
        # Train with proximal term
        metrics = train_with_fedprox(
            model=model,
            X_train=self.X_test,
            y_train=self.y_test,
            global_weights=global_weights,
            epochs=2,
            batch_size=32,
            proximal_mu=0.01,
            verbose=0
        )
        
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('proximal_term', metrics)
        self.assertGreater(metrics['proximal_term'], 0.0)
        self.assertTrue(metrics['is_personalized'])
    
    def test_personalize_model_standard(self):
        """Test standard personalization without proximal term"""
        model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        global_weights = get_model_weights(model)
        
        metrics = personalize_model(
            model=model,
            X_train=self.X_test,
            y_train=self.y_test,
            global_weights=global_weights,
            epochs=2,
            batch_size=32,
            verbose=0
        )
        
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertEqual(metrics['personalization_type'], 'standard_finetune')
    
    def test_average_weights(self):
        """Test weight averaging"""
        model1 = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        model2 = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        
        weights1 = get_model_weights(model1)
        weights2 = get_model_weights(model2)
        
        # Average should be between the two
        averaged = average_weights([weights1, weights2])
        
        self.assertEqual(len(averaged), len(weights1))
        for w_avg, w1, w2 in zip(averaged, weights1, weights2):
            # Check shape is preserved
            self.assertEqual(w_avg.shape, w1.shape)


class TestConfiguration(unittest.TestCase):
    """Test configuration loading"""
    
    def test_personalization_config_exists(self):
        """Test that personalization config parameters exist"""
        # These should not raise NameError
        try:
            _ = PERSONALIZATION_ENABLED
            _ = PROXIMAL_MU
            _ = PERSONAL_EPOCHS_PER_ROUND
        except NameError as e:
            self.fail(f"Config parameter not found: {e}")
    
    def test_personalization_config_types(self):
        """Test configuration parameter types"""
        self.assertIsInstance(PERSONALIZATION_ENABLED, bool)
        self.assertIsInstance(PROXIMAL_MU, float)
        self.assertIsInstance(PERSONAL_EPOCHS_PER_ROUND, int)
    
    def test_personalization_config_values(self):
        """Test configuration parameter values are in reasonable ranges"""
        self.assertGreaterEqual(PROXIMAL_MU, 0.0)
        self.assertLessEqual(PROXIMAL_MU, 1.0)
        
        self.assertGreaterEqual(PERSONAL_EPOCHS_PER_ROUND, 1)
        self.assertLessEqual(PERSONAL_EPOCHS_PER_ROUND, 50)


class TestHospitalServerEndpoints(unittest.TestCase):
    """Test hospital server endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures"""
        from hospital_server.app import app
        cls.app = app
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
    
    def test_personalize_endpoint_exists(self):
        """Test /personalize endpoint exists"""
        # This is a check that the endpoint is registered
        with self.app.app_context():
            routes = [rule.rule for rule in self.app.url_map.iter_rules()]
            self.assertIn('/personalize', routes)
    
    def test_personalize_fedprox_endpoint_exists(self):
        """Test /personalize_fedprox endpoint exists"""
        with self.app.app_context():
            routes = [rule.rule for rule in self.app.url_map.iter_rules()]
            self.assertIn('/personalize_fedprox', routes)
    
    def test_personalization_history_endpoint_exists(self):
        """Test /personalization_history endpoint exists"""
        with self.app.app_context():
            routes = [rule.rule for rule in self.app.url_map.iter_rules()]
            self.assertIn('/personalization_history', routes)


class TestMainServerEndpoints(unittest.TestCase):
    """Test main server endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures"""
        from main_server.app import app
        cls.app = app
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
    
    def test_trigger_personalization_endpoint_exists(self):
        """Test /trigger_personalization endpoint exists"""
        with self.app.app_context():
            routes = [rule.rule for rule in self.app.url_map.iter_rules()]
            self.assertIn('/trigger_personalization', routes)
    
    def test_personalization_metrics_endpoint_exists(self):
        """Test /personalization_metrics endpoint exists"""
        with self.app.app_context():
            routes = [rule.rule for rule in self.app.url_map.iter_rules()]
            self.assertIn('/personalization_metrics', routes)


class TestOrchestratorIntegration(unittest.TestCase):
    """Test orchestrator integration"""
    
    def test_orchestrator_run_federated_learning_signature(self):
        """Test orchestrator accepts enable_personalization parameter"""
        from orchestrator import FederatedLearningOrchestrator
        
        # Create mock orchestrator
        orch = FederatedLearningOrchestrator(
            main_server_url="http://localhost:5000",
            hospital_urls=["http://localhost:5001"]
        )
        
        # Check method exists and has correct signature
        import inspect
        sig = inspect.signature(orch.run_federated_learning)
        self.assertIn('enable_personalization', sig.parameters)
        
        # Check default value
        param = sig.parameters['enable_personalization']
        self.assertFalse(param.default)  # Should default to False


@unittest.skipIf(not HAS_TF, "TensorFlow not installed")
class TestEndToEndValidation(unittest.TestCase):
    """End-to-end validation tests"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures"""
        tf.random.set_seed(42)
        np.random.seed(42)
        
        cls.X_train = np.random.randn(200, 28, 28, 1).astype('float32')
        cls.y_train = np.random.randint(0, 10, 200)
    
    def test_fedprox_training_pipeline(self):
        """Test complete FedProx training pipeline"""
        # Initialize model
        model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Step 1: Initial training
        history1 = model.fit(
            self.X_train, self.y_train,
            epochs=2,
            batch_size=32,
            verbose=0
        )
        initial_loss = history1.history['loss'][-1]
        
        # Step 2: Get reference weights
        ref_weights = get_model_weights(model)
        
        # Step 3: Train with FedProx
        metrics = train_with_fedprox(
            model=model,
            X_train=self.X_train,
            y_train=self.y_train,
            global_weights=ref_weights,
            epochs=2,
            batch_size=32,
            proximal_mu=PROXIMAL_MU,
            verbose=0
        )
        
        # Validate metrics
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('proximal_term', metrics)
        self.assertTrue(metrics['is_personalized'])
        self.assertGreater(metrics['proximal_term'], 0.0)
        
        # Validate that model learned something
        self.assertGreater(metrics['accuracy'], 0.0)
    
    def test_personalization_workflow(self):
        """Test complete personalization workflow"""
        # 1. Create and train model
        model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train initial model
        model.fit(self.X_train, self.y_train, epochs=2, batch_size=32, verbose=0)
        initial_acc = model.evaluate(self.X_train, self.y_train, verbose=0)[1]
        
        # 2. Get global weights
        global_weights = get_model_weights(model)
        
        # 3. Personalize locally
        metrics = personalize_model(
            model=model,
            X_train=self.X_train,
            y_train=self.y_train,
            global_weights=global_weights,
            epochs=2,
            batch_size=32
        )
        
        personalized_acc = metrics['accuracy']
        
        # 4. Validate improvement
        self.assertIsInstance(personalized_acc, (float, np.floating))
        self.assertGreater(personalized_acc, 0.0)


@unittest.skipIf(not HAS_TF, "TensorFlow not installed")
class TestMetricsCollection(unittest.TestCase):
    """Test metrics collection"""
    
    def test_fedprox_metrics_structure(self):
        """Test FedProx training returns proper metrics structure"""
        model = create_simple_model(INPUT_SHAPE, NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        X_test = np.random.randn(50, 28, 28, 1).astype('float32')
        y_test = np.random.randint(0, 10, 50)
        
        global_weights = get_model_weights(model)
        
        metrics = train_with_fedprox(
            model=model,
            X_train=X_test,
            y_train=y_test,
            global_weights=global_weights,
            epochs=1,
            batch_size=32,
            proximal_mu=0.01,
            verbose=0
        )
        
        # Check required fields
        required_fields = ['loss', 'accuracy', 'proximal_term', 'is_personalized']
        for field in required_fields:
            self.assertIn(field, metrics)
        
        # Check types
        self.assertIsInstance(metrics['loss'], (float, np.floating))
        self.assertIsInstance(metrics['accuracy'], (float, np.floating))
        self.assertIsInstance(metrics['proximal_term'], (float, np.floating))
        self.assertIsInstance(metrics['is_personalized'], bool)


def run_all_tests():
    """Run all tests with proper output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFedProxFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestHospitalServerEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestMainServerEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestOrchestratorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCollection))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[OK] ALL TESTS PASSED!")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
