"""
Configuration Manager for SafetyBench

This module provides centralized configuration management, including loading,
validation, and access to configuration parameters from config.json.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass


@dataclass
class PersonaConfig:
    """Configuration for a persona."""
    id: int
    name: str
    race: str
    ethnicity: str
    age: int
    language: str
    disability: Optional[str]
    religion: Optional[str]
    sex: str
    occupation: str
    interests: List[str]
    background: str
    reaction_to_support: str
    language_style: str


@dataclass
class ConversationPromptConfig:
    """Configuration for conversation prompts."""
    focus: str
    description: str
    guidelines: List[str]
    language_permissions: str


@dataclass
class DatasetGenerationConfig:
    """Configuration for dataset generation."""
    num_conversations: int
    distribution: Dict[str, float]
    turns: List[int]
    generation_model: str
    generation_temperature: float
    max_tokens: int
    cache_dir: str
    output_dir: str


@dataclass
class SafetyEvaluationConfig:
    """Configuration for safety evaluation."""
    evaluation_model: str
    evaluation_temperature: float
    max_tokens: int


@dataclass
class SafetyBenchConfig:
    """Main configuration container."""
    dataset_generation: DatasetGenerationConfig
    safety_evaluation: SafetyEvaluationConfig
    personas: List[PersonaConfig]
    conversation_prompts: Dict[str, ConversationPromptConfig]
    system_prompt: str
    safety_evaluation_prompt: str


class ConfigManager:
    """
    Centralized configuration manager for SafetyBench.
    
    This class handles loading, validation, and access to configuration
    parameters from config.json with proper error handling and logging.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[SafetyBenchConfig] = None
    _config_file_path: str = "config.json"
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager."""
        if self._config is None:
            self.reload_config()
    
    def reload_config(self, config_file_path: Optional[str] = None) -> None:
        """
        Load or reload configuration from file.
        
        Args:
            config_file_path: Optional path to config file. Uses default if None.
        """
        if config_file_path:
            self._config_file_path = config_file_path
        
        logger.info(f"Loading configuration from {self._config_file_path}")
        
        try:
            config_data = self._load_config_file()
            self._config = self._parse_and_validate_config(config_data)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigError(f"Configuration loading failed: {e}")
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration data from JSON file."""
        config_path = Path(self._config_file_path)
        
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {self._config_file_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return config_data
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigError(f"Error reading configuration file: {e}")
    
    def _parse_and_validate_config(self, config_data: Dict[str, Any]) -> SafetyBenchConfig:
        """Parse and validate configuration data."""
        try:
            # Parse dataset generation config
            dataset_gen_data = config_data.get("dataset_generation", {})
            dataset_generation = DatasetGenerationConfig(
                num_conversations=dataset_gen_data.get("num_conversations", 100),
                distribution=dataset_gen_data.get("distribution", {}),
                turns=dataset_gen_data.get("turns", [2, 4, 6]),
                generation_model=dataset_gen_data.get("generation_model", "gpt-4o-mini"),
                generation_temperature=dataset_gen_data.get("generation_temperature", 0.7),
                max_tokens=dataset_gen_data.get("max_tokens", 2048),
                cache_dir=dataset_gen_data.get("cache_dir", "cache"),
                output_dir=dataset_gen_data.get("output_dir", "data")
            )
            
            # Parse safety evaluation config
            safety_eval_data = config_data.get("safety_evaluation", {})
            safety_evaluation = SafetyEvaluationConfig(
                evaluation_model=safety_eval_data.get("evaluation_model", "gpt-4o"),
                evaluation_temperature=safety_eval_data.get("evaluation_temperature", 0.2),
                max_tokens=safety_eval_data.get("max_tokens", 1000)
            )
            
            # Parse personas
            personas_data = config_data.get("personas", [])
            personas = []
            for persona_data in personas_data:
                persona = PersonaConfig(
                    id=persona_data["id"],
                    name=persona_data["name"],
                    race=persona_data["race"],
                    ethnicity=persona_data["ethnicity"],
                    age=persona_data["age"],
                    language=persona_data["language"],
                    disability=persona_data.get("disability"),
                    religion=persona_data.get("religion"),
                    sex=persona_data["sex"],
                    occupation=persona_data["occupation"],
                    interests=persona_data["interests"],
                    background=persona_data["background"],
                    reaction_to_support=persona_data["reaction_to_support"],
                    language_style=persona_data["language_style"]
                )
                personas.append(persona)
            
            # Parse conversation prompts
            prompts_data = config_data.get("conversation_prompts", {})
            conversation_prompts = {}
            for prompt_type, prompt_data in prompts_data.items():
                conversation_prompts[prompt_type] = ConversationPromptConfig(
                    focus=prompt_data["focus"],
                    description=prompt_data["description"],
                    guidelines=prompt_data["guidelines"],
                    language_permissions=prompt_data["language_permissions"]
                )
            
            # Get system prompts
            system_prompt = config_data.get("system_prompt", "")
            safety_evaluation_prompt = config_data.get("safety_evaluation_prompt", "")
            
            config = SafetyBenchConfig(
                dataset_generation=dataset_generation,
                safety_evaluation=safety_evaluation,
                personas=personas,
                conversation_prompts=conversation_prompts,
                system_prompt=system_prompt,
                safety_evaluation_prompt=safety_evaluation_prompt
            )
            
            # Validate the configuration
            self._validate_config(config)
            
            return config
            
        except KeyError as e:
            raise ConfigError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise ConfigError(f"Configuration parsing error: {e}")
    
    def _validate_config(self, config: SafetyBenchConfig) -> None:
        """Validate configuration values."""
        # Validate dataset generation config
        if config.dataset_generation.num_conversations <= 0:
            raise ConfigError("num_conversations must be positive")
        
        if not (0.0 <= config.dataset_generation.generation_temperature <= 2.0):
            raise ConfigError("generation_temperature must be between 0.0 and 2.0")
        
        if config.dataset_generation.max_tokens <= 0:
            raise ConfigError("max_tokens must be positive")
        
        # Validate distribution sums to 1.0 (with small tolerance)
        distribution_sum = sum(config.dataset_generation.distribution.values())
        if abs(distribution_sum - 1.0) > 0.01:
            raise ConfigError(f"Distribution values must sum to 1.0, got {distribution_sum}")
        
        # Validate safety evaluation config
        if not (0.0 <= config.safety_evaluation.evaluation_temperature <= 2.0):
            raise ConfigError("evaluation_temperature must be between 0.0 and 2.0")
        
        if config.safety_evaluation.max_tokens <= 0:
            raise ConfigError("evaluation max_tokens must be positive")
        
        # Validate personas
        if not config.personas:
            raise ConfigError("At least one persona must be defined")
        
        persona_ids = [p.id for p in config.personas]
        if len(persona_ids) != len(set(persona_ids)):
            raise ConfigError("Persona IDs must be unique")
        
        # Validate conversation prompts
        required_prompt_types = [
            "suicidal_ideation", "non_suicidal_self_harm", "harm_to_others",
            "false_positive", "benign"
        ]
        
        for prompt_type in required_prompt_types:
            if prompt_type not in config.conversation_prompts:
                raise ConfigError(f"Missing required conversation prompt type: {prompt_type}")
        
        logger.info("Configuration validation passed")
    
    @property
    def config(self) -> SafetyBenchConfig:
        """Get the current configuration."""
        if self._config is None:
            raise ConfigError("Configuration not loaded")
        return self._config
    
    def get_dataset_generation_config(self) -> DatasetGenerationConfig:
        """Get dataset generation configuration."""
        return self.config.dataset_generation
    
    def get_safety_evaluation_config(self) -> SafetyEvaluationConfig:
        """Get safety evaluation configuration."""
        return self.config.safety_evaluation
    
    def get_personas(self) -> List[PersonaConfig]:
        """Get list of personas."""
        return self.config.personas
    
    def get_conversation_prompts(self) -> Dict[str, ConversationPromptConfig]:
        """Get conversation prompts."""
        return self.config.conversation_prompts
    
    def get_system_prompt(self) -> str:
        """Get system prompt."""
        return self.config.system_prompt
    
    def get_safety_evaluation_prompt(self) -> str:
        """Get safety evaluation prompt."""
        return self.config.safety_evaluation_prompt
    
    def get_distribution(self) -> Dict[str, float]:
        """Get conversation type distribution."""
        return self.config.dataset_generation.distribution
    
    def get_turns(self) -> List[int]:
        """Get available turn options."""
        return self.config.dataset_generation.turns
    
    def get_generation_model(self) -> str:
        """Get generation model name."""
        return self.config.dataset_generation.generation_model
    
    def get_evaluation_model(self) -> str:
        """Get evaluation model name."""
        return self.config.safety_evaluation.evaluation_model
    
    def get_generation_temperature(self) -> float:
        """Get generation temperature."""
        return self.config.dataset_generation.generation_temperature
    
    def get_evaluation_temperature(self) -> float:
        """Get evaluation temperature."""
        return self.config.safety_evaluation.evaluation_temperature
    
    def get_max_tokens(self) -> int:
        """Get max tokens for generation."""
        return self.config.dataset_generation.max_tokens
    
    def get_evaluation_max_tokens(self) -> int:
        """Get max tokens for evaluation."""
        return self.config.safety_evaluation.max_tokens
    
    def get_cache_dir(self) -> str:
        """Get cache directory."""
        return self.config.dataset_generation.cache_dir
    
    def get_output_dir(self) -> str:
        """Get output directory."""
        return self.config.dataset_generation.output_dir
    
    def set_config_file_path(self, path: str) -> None:
        """Set the configuration file path and reload."""
        self._config_file_path = path
        self.reload_config()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration for logging/debugging."""
        if self._config is None:
            return {"status": "Configuration not loaded"}
        
        return {
            "dataset_generation": {
                "num_conversations": self.config.dataset_generation.num_conversations,
                "generation_model": self.config.dataset_generation.generation_model,
                "generation_temperature": self.config.dataset_generation.generation_temperature,
                "distribution": self.config.dataset_generation.distribution,
                "turns": self.config.dataset_generation.turns
            },
            "safety_evaluation": {
                "evaluation_model": self.config.safety_evaluation.evaluation_model,
                "evaluation_temperature": self.config.safety_evaluation.evaluation_temperature,
                "max_tokens": self.config.safety_evaluation.max_tokens
            },
            "personas_count": len(self.config.personas),
            "conversation_prompts_count": len(self.config.conversation_prompts),
            "config_file": self._config_file_path
        }


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> SafetyBenchConfig:
    """Convenience function to get the current configuration."""
    return get_config_manager().config


# Convenience functions for common configuration access
def get_dataset_generation_config() -> DatasetGenerationConfig:
    """Get dataset generation configuration."""
    return get_config_manager().get_dataset_generation_config()


def get_safety_evaluation_config() -> SafetyEvaluationConfig:
    """Get safety evaluation configuration."""
    return get_config_manager().get_safety_evaluation_config()


def get_personas() -> List[PersonaConfig]:
    """Get list of personas."""
    return get_config_manager().get_personas()


def get_conversation_prompts() -> Dict[str, ConversationPromptConfig]:
    """Get conversation prompts."""
    return get_config_manager().get_conversation_prompts()


def reload_config(config_file_path: Optional[str] = None) -> None:
    """Reload configuration from file."""
    get_config_manager().reload_config(config_file_path)