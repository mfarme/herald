"""
Tests for the configuration manager module.

This test suite verifies that the configuration manager properly loads,
validates, and provides access to configuration parameters.
"""

import json
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config_manager import (
    ConfigManager, ConfigError, get_config_manager, reload_config,
    PersonaConfig, ConversationPromptConfig, DatasetGenerationConfig,
    SafetyEvaluationConfig, SafetyBenchConfig
)


class TestConfigManager:
    """Test cases for the ConfigManager class."""
    
    @pytest.fixture
    def valid_config_data(self):
        """Return valid configuration data for testing."""
        return {
            "dataset_generation": {
                "num_conversations": 50,
                "distribution": {
                    "suicidal_ideation": 0.3,
                    "non_suicidal_self_harm": 0.2,
                    "harm_to_others": 0.2,
                    "false_positive": 0.15,
                    "benign": 0.15
                },
                "turns": [2, 4, 6],
                "generation_model": "gpt-4o-mini",
                "generation_temperature": 0.8,
                "max_tokens": 1024,
                "cache_dir": "test_cache",
                "output_dir": "test_data"
            },
            "safety_evaluation": {
                "evaluation_model": "gpt-4o",
                "evaluation_temperature": 0.3,
                "max_tokens": 512
            },
            "personas": [
                {
                    "id": 1,
                    "name": "Test User",
                    "race": "Test Race",
                    "ethnicity": "Test Ethnicity",
                    "age": 25,
                    "language": "English",
                    "disability": None,
                    "religion": None,
                    "sex": "Test",
                    "occupation": "Test Occupation",
                    "interests": ["test1", "test2"],
                    "background": "Test background",
                    "reaction_to_support": "Test reaction",
                    "language_style": "Test style"
                }
            ],
            "conversation_prompts": {
                "suicidal_ideation": {
                    "focus": "suicidal_ideation",
                    "description": "Test description",
                    "guidelines": ["Test guideline 1", "Test guideline 2"],
                    "language_permissions": "Test permissions"
                },
                "non_suicidal_self_harm": {
                    "focus": "non_suicidal_self_harm",
                    "description": "Test description",
                    "guidelines": ["Test guideline"],
                    "language_permissions": "Test permissions"
                },
                "harm_to_others": {
                    "focus": "harm_to_others",
                    "description": "Test description",
                    "guidelines": ["Test guideline"],
                    "language_permissions": "Test permissions"
                },
                "false_positive": {
                    "focus": "false_positive",
                    "description": "Test description",
                    "guidelines": ["Test guideline"],
                    "language_permissions": "Test permissions"
                },
                "benign": {
                    "focus": "benign",
                    "description": "Test description",
                    "guidelines": ["Test guideline"],
                    "language_permissions": "Test permissions"
                }
            },
            "system_prompt": "Test system prompt",
            "safety_evaluation_prompt": "Test safety evaluation prompt"
        }
    
    @pytest.fixture
    def temp_config_file(self, valid_config_data):
        """Create a temporary configuration file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_config_manager_singleton(self):
        """Test that ConfigManager implements singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2
    
    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid configuration file."""
        manager = ConfigManager()
        manager.reload_config(temp_config_file)
        
        config = manager.config
        assert config is not None
        assert config.dataset_generation.num_conversations == 50
        assert config.safety_evaluation.evaluation_model == "gpt-4o"
        assert len(config.personas) == 1
        assert config.personas[0].name == "Test User"
    
    def test_load_missing_file(self):
        """Test loading a non-existent configuration file."""
        manager = ConfigManager()
        with pytest.raises(ConfigError, match="Configuration file not found"):
            manager.reload_config("nonexistent_file.json")
    
    def test_load_invalid_json(self):
        """Test loading a file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_file = f.name
        
        try:
            manager = ConfigManager()
            with pytest.raises(ConfigError, match="Invalid JSON"):
                manager.reload_config(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validation_invalid_distribution_sum(self, valid_config_data):
        """Test validation with distribution that doesn't sum to 1.0."""
        # Modify distribution to not sum to 1.0
        valid_config_data["dataset_generation"]["distribution"]["suicidal_ideation"] = 0.8
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_data, f)
            temp_file = f.name
        
        try:
            manager = ConfigManager()
            with pytest.raises(ConfigError, match="Distribution values must sum to 1.0"):
                manager.reload_config(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validation_negative_conversations(self, valid_config_data):
        """Test validation with negative number of conversations."""
        valid_config_data["dataset_generation"]["num_conversations"] = -5
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_data, f)
            temp_file = f.name
        
        try:
            manager = ConfigManager()
            with pytest.raises(ConfigError, match="num_conversations must be positive"):
                manager.reload_config(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validation_invalid_temperature(self, valid_config_data):
        """Test validation with invalid temperature values."""
        valid_config_data["dataset_generation"]["generation_temperature"] = 3.0
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_data, f)
            temp_file = f.name
        
        try:
            manager = ConfigManager()
            with pytest.raises(ConfigError, match="generation_temperature must be between 0.0 and 2.0"):
                manager.reload_config(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validation_missing_personas(self, valid_config_data):
        """Test validation with no personas defined."""
        valid_config_data["personas"] = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_data, f)
            temp_file = f.name
        
        try:
            manager = ConfigManager()
            with pytest.raises(ConfigError, match="At least one persona must be defined"):
                manager.reload_config(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validation_missing_conversation_prompts(self, valid_config_data):
        """Test validation with missing conversation prompt types."""
        # Remove a required prompt type
        del valid_config_data["conversation_prompts"]["suicidal_ideation"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_data, f)
            temp_file = f.name
        
        try:
            manager = ConfigManager()
            with pytest.raises(ConfigError, match="Missing required conversation prompt type"):
                manager.reload_config(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_config_accessor_methods(self, temp_config_file):
        """Test configuration accessor methods."""
        manager = ConfigManager()
        manager.reload_config(temp_config_file)
        
        # Test dataset generation accessors
        assert manager.get_generation_model() == "gpt-4o-mini"
        assert manager.get_generation_temperature() == 0.8
        assert manager.get_max_tokens() == 1024
        
        # Test safety evaluation accessors
        assert manager.get_evaluation_model() == "gpt-4o"
        assert manager.get_evaluation_temperature() == 0.3
        assert manager.get_evaluation_max_tokens() == 512
        
        # Test other accessors
        assert len(manager.get_personas()) == 1
        assert len(manager.get_conversation_prompts()) == 5
        assert manager.get_cache_dir() == "test_cache"
        assert manager.get_output_dir() == "test_data"
    
    def test_config_summary(self, temp_config_file):
        """Test configuration summary generation."""
        manager = ConfigManager()
        manager.reload_config(temp_config_file)
        
        summary = manager.get_config_summary()
        assert "dataset_generation" in summary
        assert "safety_evaluation" in summary
        assert "personas_count" in summary
        assert "conversation_prompts_count" in summary
        assert summary["personas_count"] == 1
        assert summary["conversation_prompts_count"] == 5
    
    def test_global_config_manager_function(self, temp_config_file):
        """Test the global get_config_manager function."""
        # Reset singleton instance for this test
        ConfigManager._instance = None
        ConfigManager._config = None
        
        # Create new instance with custom config
        manager = get_config_manager()
        manager.reload_config(temp_config_file)
        
        # Verify global function returns same instance
        manager2 = get_config_manager()
        assert manager is manager2
        assert manager2.get_generation_model() == "gpt-4o-mini"
    
    def test_reload_config_function(self, temp_config_file):
        """Test the global reload_config function."""
        reload_config(temp_config_file)
        manager = get_config_manager()
        assert manager.get_generation_model() == "gpt-4o-mini"


class TestConfigDataStructures:
    """Test the configuration data structure classes."""
    
    def test_persona_config_creation(self):
        """Test PersonaConfig creation and validation."""
        persona = PersonaConfig(
            id=1,
            name="Test User",
            race="Test Race",
            ethnicity="Test Ethnicity",
            age=25,
            language="English",
            disability=None,
            religion=None,
            sex="Test",
            occupation="Test Occupation",
            interests=["test1", "test2"],
            background="Test background",
            reaction_to_support="Test reaction",
            language_style="Test style"
        )
        
        assert persona.id == 1
        assert persona.name == "Test User"
        assert persona.interests == ["test1", "test2"]
        assert persona.disability is None
    
    def test_conversation_prompt_config_creation(self):
        """Test ConversationPromptConfig creation."""
        prompt = ConversationPromptConfig(
            focus="test_focus",
            description="Test description",
            guidelines=["Guideline 1", "Guideline 2"],
            language_permissions="Test permissions"
        )
        
        assert prompt.focus == "test_focus"
        assert prompt.description == "Test description"
        assert len(prompt.guidelines) == 2
        assert prompt.language_permissions == "Test permissions"
    
    def test_dataset_generation_config_creation(self):
        """Test DatasetGenerationConfig creation."""
        config = DatasetGenerationConfig(
            num_conversations=100,
            distribution={"type1": 0.5, "type2": 0.5},
            turns=[2, 4],
            generation_model="test-model",
            generation_temperature=0.7,
            max_tokens=1024,
            cache_dir="cache",
            output_dir="output"
        )
        
        assert config.num_conversations == 100
        assert config.distribution == {"type1": 0.5, "type2": 0.5}
        assert config.turns == [2, 4]
        assert config.generation_model == "test-model"
    
    def test_safety_evaluation_config_creation(self):
        """Test SafetyEvaluationConfig creation."""
        config = SafetyEvaluationConfig(
            evaluation_model="test-eval-model",
            evaluation_temperature=0.2,
            max_tokens=512
        )
        
        assert config.evaluation_model == "test-eval-model"
        assert config.evaluation_temperature == 0.2
        assert config.max_tokens == 512


if __name__ == "__main__":
    pytest.main([__file__])