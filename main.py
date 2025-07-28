#!/usr/bin/env python3
"""
SafetyBench - Main Entry Point

This is the main command-line interface for SafetyBench, allowing users to:
1. Generate conversational datasets for safety evaluation
2. Evaluate existing datasets using the safety bench

Usage:
    python main.py generate --num-conversations 100 --output data/
    python main.py evaluate --dataset data/safety_dataset.jsonl --output results/
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Import configuration manager first
from config_manager import get_config_manager, ConfigError, reload_config

# Import our modules
try:
    from dataset_generation import generate_dataset_with_distribution
    from herald import (
        evaluate_safety_dataset, 
        print_evaluation_summary,
        save_evaluation_results
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Install the required packages using: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for SafetyBench CLI."""
    parser = argparse.ArgumentParser(
        description="SafetyBench - Generate and evaluate conversational safety datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a dataset with 50 conversations
  python main.py generate --num-conversations 50 --output data/
  
  # Evaluate an existing dataset
  python main.py evaluate --dataset data/safety_dataset_20240101T120000Z.jsonl
  
  # Generate and then evaluate
  python main.py generate --num-conversations 100
  python main.py evaluate --dataset data/safety_dataset_*.jsonl
  
  # Use custom configuration file
  python main.py --config custom_config.json generate
        """
    )
    
    # Global options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: config.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Available modes')
    subparsers.required = True
    
    # Dataset Generation subparser
    generate_parser = subparsers.add_parser(
        'generate', 
        help='Generate a new conversational safety dataset'
    )
    generate_parser.add_argument(
        '--num-conversations', 
        type=int,
        help='Number of conversations to generate (uses config default if not specified)'
    )
    generate_parser.add_argument(
        '--output', 
        type=str,
        help='Output directory for generated dataset (uses config default if not specified)'
    )
    generate_parser.add_argument(
        '--cache-dir', 
        type=str,
        help='Temporary cache directory (uses config default if not specified)'
    )
    
    # Safety Evaluation subparser
    evaluate_parser = subparsers.add_parser(
        'evaluate', 
        help='Evaluate a safety dataset'
    )
    evaluate_parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to the dataset file (.jsonl format)'
    )
    evaluate_parser.add_argument(
        '--output', 
        type=str,
        help='Output file for evaluation results (JSON format)'
    )
    evaluate_parser.add_argument(
        '--save-html', 
        action='store_true',
        help='Save HTML visualization of results'
    )
    
    # Info subparser
    info_parser = subparsers.add_parser(
        'info', 
        help='Show information about SafetyBench configuration'
    )
    
    # Config subparser for configuration management
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration management'
    )
    config_subparsers = config_parser.add_subparsers(dest='config_action', help='Configuration actions')
    config_subparsers.required = True
    
    config_subparsers.add_parser('show', help='Show current configuration')
    config_subparsers.add_parser('validate', help='Validate configuration file')
    
    reload_parser = config_subparsers.add_parser('reload', help='Reload configuration from file')
    reload_parser.add_argument('--file', type=str, help='Configuration file to reload')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Load configuration
    try:
        if args.config:
            logger.info(f"Loading custom configuration from: {args.config}")
            reload_config(args.config)
        else:
            # Load default configuration
            config_manager = get_config_manager()
            logger.info("Using default configuration")
            
    except ConfigError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}")
        sys.exit(1)
    
    # Handle different modes
    if args.mode == 'generate':
        handle_generate_mode(args)
    elif args.mode == 'evaluate':
        handle_evaluate_mode(args)
    elif args.mode == 'info':
        handle_info_mode(args)
    elif args.mode == 'config':
        handle_config_mode(args)


def handle_generate_mode(args):
    """Handle dataset generation mode."""
    logger.info("Starting dataset generation mode")
    
    print("SafetyBench Dataset Generation")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    try:
        # Get configuration values
        config_manager = get_config_manager()
        dataset_config = config_manager.get_dataset_generation_config()
        
        # Use command line args or config defaults
        num_conversations = args.num_conversations or dataset_config.num_conversations
        output_dir = args.output or dataset_config.output_dir
        cache_dir = args.cache_dir or dataset_config.cache_dir
        
        # Validate arguments
        if num_conversations <= 0:
            print("ERROR: Number of conversations must be positive.")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Configuration:")
        print(f"  Conversations: {num_conversations}")
        print(f"  Output directory: {output_dir}")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Generation model: {dataset_config.generation_model}")
        print(f"  Distribution: {config_manager.get_distribution()}")
        print(f"  Turn options: {config_manager.get_turns()}")
        print()
        
        # Generate the dataset
        output_file = generate_dataset_with_distribution(
            num_conversations=num_conversations,
            cache_dir=cache_dir,
            output_dir=output_dir
        )
        
        print(f"\n✓ Dataset generation completed successfully!")
        print(f"✓ Output file: {output_file}")
        print(f"\nTo evaluate this dataset, run:")
        print(f"  python main.py evaluate --dataset {output_file}")
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        print(f"\n✗ Error during dataset generation: {e}")
        sys.exit(1)


def handle_evaluate_mode(args):
    """Handle safety evaluation mode."""
    logger.info("Starting safety evaluation mode")
    
    print("SafetyBench Safety Evaluation")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Validate dataset file exists
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    try:
        # Get configuration values
        config_manager = get_config_manager()
        safety_config = config_manager.get_safety_evaluation_config()
        
        print(f"Configuration:")
        print(f"  Dataset file: {args.dataset}")
        print(f"  Output file: {args.output or 'Auto-generated'}")
        print(f"  Save HTML: {args.save_html}")
        print(f"  Evaluation model: {safety_config.evaluation_model}")
        print(f"  Evaluation temperature: {safety_config.evaluation_temperature}")
        print()
        
        # Evaluate the dataset
        evaluation_results = evaluate_safety_dataset(args.dataset)
        
        # Print summary
        print_evaluation_summary(evaluation_results)
        
        # Save results if requested
        if args.output or args.save_html:
            results_file = save_evaluation_results(evaluation_results, args.output)
            print(f"\n✓ Results saved to: {results_file}")
        
        # Generate HTML visualizations if requested
        if args.save_html:
            from herald import create_html_confusion_matrix, create_html_conversation_display
            
            # Generate HTML files
            confusion_html = create_html_confusion_matrix(evaluation_results)
            conversation_html = create_html_conversation_display(evaluation_results, args.dataset)
            
            # Save HTML files
            base_name = os.path.splitext(args.output or "evaluation_results")[0]
            
            confusion_file = f"{base_name}_confusion_matrix.html"
            with open(confusion_file, 'w', encoding='utf-8') as f:
                f.write(confusion_html)
            
            conversation_file = f"{base_name}_conversations.html"
            with open(conversation_file, 'w', encoding='utf-8') as f:
                f.write(conversation_html)
            
            print(f"✓ HTML visualizations saved:")
            print(f"  - Confusion matrix: {confusion_file}")
            print(f"  - Conversation examples: {conversation_file}")
        
        print(f"\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        print(f"\n✗ Error during evaluation: {e}")
        sys.exit(1)


def handle_info_mode(args):
    """Handle info mode - show configuration information."""
    logger.info("Showing configuration information")
    
    print("SafetyBench Configuration Information")
    print("=" * 50)
    
    try:
        # Get configuration manager
        config_manager = get_config_manager()
        
        # Check API key status
        api_key_status = "✓ Set" if os.getenv("OPENAI_API_KEY") else "✗ Not set"
        print(f"OpenAI API Key: {api_key_status}")
        
        # Show configuration summary
        config_summary = config_manager.get_config_summary()
        
        print(f"\nDataset Generation Configuration:")
        dataset_config = config_summary["dataset_generation"]
        print(f"  Default conversations: {dataset_config['num_conversations']}")
        print(f"  Generation model: {dataset_config['generation_model']}")
        print(f"  Generation temperature: {dataset_config['generation_temperature']}")
        print(f"  Conversation types distribution:")
        for conv_type, percentage in dataset_config["distribution"].items():
            print(f"    - {conv_type}: {percentage * 100:.1f}%")
        print(f"  Turn options: {dataset_config['turns']}")
        
        print(f"\nSafety Evaluation Configuration:")
        safety_config = config_summary["safety_evaluation"]
        print(f"  Evaluation model: {safety_config['evaluation_model']}")
        print(f"  Evaluation temperature: {safety_config['evaluation_temperature']}")
        print(f"  Max tokens: {safety_config['max_tokens']}")
        
        print(f"\nPersonas: {config_summary['personas_count']} configured")
        print(f"Conversation Prompts: {config_summary['conversation_prompts_count']} types")
        print(f"Configuration File: {config_summary['config_file']}")
        
    except Exception as e:
        logger.error(f"Error getting configuration info: {e}")
        print(f"Error accessing configuration: {e}")
    
    print(f"\nFile Structure:")
    current_dir = Path.cwd()
    important_files = [
        "config_manager.py",
        "dataset_generation.py",
        "safety_bench.py", 
        "main.py",
        "config.json",
        "requirements.txt",
        "data/",
        ".env.example"
    ]
    
    for file_path in important_files:
        full_path = current_dir / file_path
        if full_path.exists():
            if full_path.is_dir():
                print(f"  ✓ {file_path} (directory)")
            else:
                print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
    
    print(f"\nUsage Examples:")
    print(f"  # Generate 50 conversations:")
    print(f"  python main.py generate --num-conversations 50")
    print(f"  ")
    print(f"  # Evaluate a dataset:")
    print(f"  python main.py evaluate --dataset data/safety_dataset_*.jsonl")
    print(f"  ")
    print(f"  # Generate with HTML output:")
    print(f"  python main.py evaluate --dataset data/safety_dataset_*.jsonl --save-html")
    print(f"  ")
    print(f"  # Use custom configuration:")
    print(f"  python main.py --config custom_config.json generate")


def handle_config_mode(args):
    """Handle configuration management mode."""
    logger.info(f"Configuration management action: {args.config_action}")
    
    try:
        config_manager = get_config_manager()
        
        if args.config_action == 'show':
            print("Current Configuration:")
            print("=" * 40)
            
            config_summary = config_manager.get_config_summary()
            import json
            print(json.dumps(config_summary, indent=2))
            
        elif args.config_action == 'validate':
            print("Validating configuration...")
            # Configuration is validated on load, so if we get here it's valid
            print("✓ Configuration is valid")
            
        elif args.config_action == 'reload':
            print("Reloading configuration...")
            if args.file:
                config_manager.reload_config(args.file)
                print(f"✓ Configuration reloaded from: {args.file}")
            else:
                config_manager.reload_config()
                print("✓ Configuration reloaded from default file")
                
    except ConfigError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in configuration management: {e}")
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()