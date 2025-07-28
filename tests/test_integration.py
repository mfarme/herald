"""
Integration tests for the SafetyBench configuration system.

These tests verify that configuration changes properly propagate throughout
all Python modules and that the entire system works together correctly.
"""

import json
import pytest
import tempfile
import os
import sys

# Import the modules to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config_manager import ConfigManager, reload_config, get_config_manager
from dataset_generation import (
    get_distribution, get_generation_model, get_generation_temperature,
    get_max_tokens, get_turns, get_personas_as_pydantic
)
from herald import (
    get_evaluation_model, get_evaluation_temperature, get_evaluation_max_tokens
)


class TestConfigurationIntegration:
    """Test configuration integration across all modules."""
    
    @pytest.fixture
    def test_config_data(self):
        """Return test configuration data."""
        return {
            "dataset_generation": {
                "num_conversations": 25,
                "distribution": {
                    "suicidal_ideation": 0.4,
                    "non_suicidal_self_harm": 0.3,
                    "harm_to_others": 0.2,
                    "false_positive": 0.05,
                    "benign": 0.05
                },
                "turns": [3, 5],
                "generation_model": "test-generation-model",
                "generation_temperature": 0.9,
                "max_tokens": 512,
                "cache_dir": "integration_test_cache",
                "output_dir": "integration_test_data"
            },
            "safety_evaluation": {
                "evaluation_model": "test-evaluation-model",
                "evaluation_temperature": 0.1,
                "max_tokens": 256
            },
            "personas": [
                {
                    "id": 1,
                    "name": "Integration Test User",
                    "race": "Test Race",
                    "ethnicity": "Test Ethnicity",
                    "age": 30,
                    "language": "English",
                    "disability": "Test Disability",
                    "religion": "Test Religion",
                    "sex": "Test",
                    "occupation": "Test Occupation",
                    "interests": ["integration", "testing"],
                    "background": "Integration test background",
                    "reaction_to_support": "Integration test reaction",
                    "language_style": "Integration test style"
                }
            ],
            "conversation_prompts": {
                "suicidal_ideation": {
                    "focus": "suicidal_ideation",
                    "description": "Integration test suicidal description",
                    "guidelines": ["Integration guideline 1"],
                    "language_permissions": "Integration test permissions"
                },
                "non_suicidal_self_harm": {
                    "focus": "non_suicidal_self_harm", 
                    "description": "Integration test self-harm description",
                    "guidelines": ["Integration guideline 2"],
                    "language_permissions": "Integration test permissions"
                },
                "harm_to_others": {
                    "focus": "harm_to_others",
                    "description": "Integration test harm others description",
                    "guidelines": ["Integration guideline 3"],
                    "language_permissions": "Integration test permissions"
                },
                "false_positive": {
                    "focus": "false_positive",
                    "description": "Integration test false positive description",
                    "guidelines": ["Integration guideline 4"],
                    "language_permissions": "Integration test permissions"
                },
                "benign": {
                    "focus": "benign",
                    "description": "Integration test benign description",
                    "guidelines": ["Integration guideline 5"],
                    "language_permissions": "Integration test permissions"
                }
            },
            "system_prompt": "Integration test system prompt",
            "safety_evaluation_prompt": "Integration test safety evaluation prompt"
        }
    
    @pytest.fixture
    def temp_config_file(self, test_config_data):
        """Create a temporary configuration file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def setup_method(self):
        """Reset configuration manager before each test."""
        # Reset singleton instance
        ConfigManager._instance = None
        ConfigManager._config = None
    
    def test_config_propagation_to_dataset_generation(self, temp_config_file):
        """Test that configuration changes propagate to dataset_generation module."""
        # Load test configuration
        reload_config(temp_config_file)
        
        # Verify dataset generation module uses new configuration
        assert get_generation_model() == "test-generation-model"
        assert get_generation_temperature() == 0.9
        assert get_max_tokens() == 512
        assert get_turns() == [3, 5]
        
        # Test distribution
        distribution = get_distribution()
        from dataset_generation import ConversationType
        assert distribution[ConversationType.SUICIDAL] == 0.4
        assert distribution[ConversationType.NON_SUICIDAL_SELF_HARM] == 0.3
        assert distribution[ConversationType.HARM_TO_OTHERS] == 0.2
        
        # Test personas
        personas = get_personas_as_pydantic()
        assert len(personas) == 1
        assert personas[0].name == "Integration Test User"
        assert personas[0].disability == "Test Disability"
        assert personas[0].interests == ["integration", "testing"]
    
    def test_config_propagation_to_safety_bench(self, temp_config_file):
        """Test that configuration changes propagate to safety_bench module."""
        # Load test configuration
        reload_config(temp_config_file)
        
        # Verify safety_bench module uses new configuration
        assert get_evaluation_model() == "test-evaluation-model"
        assert get_evaluation_temperature() == 0.1
        assert get_evaluation_max_tokens() == 256
    
    def test_config_changes_reflected_across_modules(self, temp_config_file, test_config_data):
        """Test that changing configuration is reflected across all modules."""
        # Load initial configuration
        reload_config(temp_config_file)
        
        # Verify initial values
        assert get_generation_model() == "test-generation-model"
        assert get_evaluation_model() == "test-evaluation-model"
        
        # Modify configuration data
        test_config_data["dataset_generation"]["generation_model"] = "updated-generation-model"
        test_config_data["safety_evaluation"]["evaluation_model"] = "updated-evaluation-model"
        test_config_data["dataset_generation"]["generation_temperature"] = 0.5
        
        # Write updated configuration to new temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config_data, f)
            new_temp_file = f.name
        
        try:
            # Reload configuration
            reload_config(new_temp_file)
            
            # Verify updated values are reflected in all modules
            assert get_generation_model() == "updated-generation-model"
            assert get_evaluation_model() == "updated-evaluation-model"
            assert get_generation_temperature() == 0.5
            
        finally:
            os.unlink(new_temp_file)
    
    def test_config_manager_singleton_consistency(self, temp_config_file):
        """Test that the singleton pattern maintains consistency across modules."""
        # Load configuration
        reload_config(temp_config_file)
        
        # Get config manager from different places
        manager1 = get_config_manager()
        
        # Import and get manager from dataset_generation context
        from dataset_generation import get_config_manager as get_ds_manager
        manager2 = get_ds_manager()
        
        # Import and get manager from safety_bench context  
        from herald import get_config_manager as get_sb_manager
        manager3 = get_sb_manager()
        
        # All should be the same instance
        assert manager1 is manager2
        assert manager2 is manager3
        
        # All should have the same configuration
        assert manager1.get_generation_model() == manager2.get_generation_model()
        assert manager2.get_evaluation_model() == manager3.get_evaluation_model()
    
    def test_config_validation_prevents_invalid_propagation(self, test_config_data):
        """Test that invalid configuration doesn't propagate to modules."""
        # Create invalid configuration (distribution doesn't sum to 1.0)
        test_config_data["dataset_generation"]["distribution"]["suicidal_ideation"] = 0.9
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config_data, f)
            invalid_config_file = f.name
        
        try:
            # Attempt to load invalid configuration should fail
            with pytest.raises(Exception):  # ConfigError or validation error
                reload_config(invalid_config_file)
                
        finally:
            os.unlink(invalid_config_file)
    
    def test_default_config_loading(self):
        """Test that default config.json loads properly."""
        # This test assumes config.json exists in the project root
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        
        if os.path.exists(config_path):
            # Reset and load default config
            ConfigManager._instance = None
            ConfigManager._config = None
            
            manager = get_config_manager()
            
            # Verify we can access basic configuration
            assert manager.get_generation_model() is not None
            assert manager.get_evaluation_model() is not None
            assert len(manager.get_personas()) > 0
            assert len(manager.get_conversation_prompts()) >= 5
            
            # Verify modules can access configuration
            assert get_generation_model() is not None
            assert get_evaluation_model() is not None
            assert len(get_personas_as_pydantic()) > 0
    
    def test_configuration_error_handling(self):
        """Test that configuration errors are properly handled."""
        # Test with non-existent file
        with pytest.raises(Exception):
            reload_config("nonexistent_config.json")
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")
            empty_file = f.name
        
        try:
            with pytest.raises(Exception):
                reload_config(empty_file)
        finally:
            os.unlink(empty_file)
    
    def test_configuration_accessibility_from_main(self, temp_config_file):
        """Test that main.py can access configuration properly."""
        # Load test configuration
        reload_config(temp_config_file)
        
        # Verify configuration is accessible (similar to what main.py would do)
        manager = get_config_manager()
        dataset_config = manager.get_dataset_generation_config()
        safety_config = manager.get_safety_evaluation_config()
        
        assert dataset_config.num_conversations == 25
        assert dataset_config.generation_model == "test-generation-model"
        assert safety_config.evaluation_model == "test-evaluation-model"
        
        # Test config summary (used by main.py info command)
        summary = manager.get_config_summary()
        assert "dataset_generation" in summary
        assert "safety_evaluation" in summary
        assert summary["personas_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__])