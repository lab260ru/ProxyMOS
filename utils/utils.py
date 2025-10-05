import argparse
import yaml
import torch
import importlib
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich import box
from models.base_model import BaseModelWrapper

# Initialize Rich console
console = Console()


# Supported models registry
MODEL_REGISTRY = {
    "nisqa": ("models.nisqa_wrapper", "NISQAWrapper"),
    "dnsmos": ("models.dnsmos_wrapper", "DNSMOSWrapper"),
    "whisqa": ("models.whisqa_wrapper", "WhiSQAWrapper"),
}


def get_model_wrapper(model_name: str, config: Dict[str, Any], device: str = "cuda") -> BaseModelWrapper:
    """
    Factory function to create model wrapper instances.
    
    Args:
        model_name: Name of the model ("nisqa", "dnsmos", "whisqa")
        config: Model configuration dictionary
        device: Device to run model on (default: "cuda")
    
    Returns:
        Initialized model wrapper instance
        
    Raises:
        ValueError: If model_name is not supported
        ImportError: If model module cannot be imported
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {available}")
    
    module_name, class_name = MODEL_REGISTRY[model_name]
    
    try:
        module = importlib.import_module(module_name)
        wrapper_class = getattr(module, class_name)
        return wrapper_class(config, device)
    except ImportError as e:
        raise ImportError(f"Failed to import {module_name}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in {module_name}: {e}")



# Configuration utilities

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge configuration with command line arguments.
    
    CLI arguments take precedence over config file values.
    
    Args:
        config: Base configuration dictionary
        args: Command line arguments from argparse
    
    Returns:
        Updated configuration dictionary
    """
    # Override dataset parameters
    if hasattr(args, 'audio_dir') and args.audio_dir:
        config['dataset']['audio_dir'] = args.audio_dir
    
    # Override inference parameters
    if hasattr(args, 'output_dir') and args.output_dir:
        config['inference']['output_dir'] = args.output_dir
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config['inference']['batch_size'] = args.batch_size
    
    if hasattr(args, 'num_workers') and args.num_workers is not None:
        config['inference']['num_workers'] = args.num_workers
    
    # Override model parameters
    if hasattr(args, 'device') and args.device:
        config['model']['device'] = args.device
    
    return config


# Command line argument parsing

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for audio quality assessment inference.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Audio quality assessment model inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # Dataset parameter overrides
    parser.add_argument(
        "--audio_dir",
        type=str,
        help="Directory containing audio files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory for saving results"
    )
    
    # Inference parameter overrides
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of DataLoader workers"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device for computation"
    )
    
    # Additional parameters
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="inference",
        help="Experiment name (for result files)"
    )
    
    parser.add_argument(
        "--no_async_save",
        action="store_true",
        help="Disable asynchronous saving"
    )
    
    return parser


# Setup utilities

def setup_device(device: str = "cuda") -> str:
    """
    Check device availability and return appropriate device.
    
    Args:
        device: Desired device ("cuda" or "cpu")
    
    Returns:
        Available device ("cuda" or "cpu")
    """
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]⚠️  CUDA not available, using CPU[/yellow]")
        return "cpu"
    return device


def print_gpu_info():
    """
    Print GPU information if available using Rich console.
    
    Displays GPU name and memory information for debugging and monitoring.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        gpu_info = Table(title="GPU Information", box=box.ROUNDED)
        gpu_info.add_column("Property", style="cyan", no_wrap=True)
        gpu_info.add_column("Value", style="green")
        
        gpu_info.add_row("GPU Name", gpu_name)
        gpu_info.add_row("Memory", f"{memory_gb:.2f} GB")
        gpu_info.add_row("CUDA Version", torch.version.cuda)
        
        console.print(gpu_info)
    else:
        console.print("[yellow]ℹ️  GPU not available, using CPU[/yellow]")


def validate_paths(config: Dict[str, Any]):
    """
    Validate paths in configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        FileNotFoundError: If required paths don't exist
    """
    if 'audio_dir' in config['dataset']:
        audio_dir = Path(config['dataset']['audio_dir'])
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    elif 'manifest' in config['dataset']:
        manifest = config['dataset']['manifest']
        for item in manifest:
            if not Path(item['audio_path']).exists():
                raise FileNotFoundError(f"Audio file not found: {item['audio_path']}")
    else:
        raise ValueError("No audio directory or manifest found in configuration")
    
    # Create output directory if it doesn't exist
    output_dir = Path(config['inference']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)


def print_config_summary(config: Dict[str, Any]):
    """
    Print configuration summary for verification using Rich console.
    
    Args:
        config: Configuration dictionary
    """
    # Create configuration table
    config_table = Table(title="🔧 Configuration Summary", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="green")
    
    # Model configuration
    config_table.add_row("Model", config['model']['name'])
    config_table.add_row("Device", config['model']['device'])
    
    # Dataset configuration
    if 'audio_dir' in config['dataset']:
        config_table.add_row("Audio Directory", str(config['dataset']['audio_dir']))
    elif 'manifest' in config['dataset']:
        manifest_size = len(config['dataset']['manifest'])
        config_table.add_row("Manifest Files", str(manifest_size))
    
    config_table.add_row("File Extension", config['dataset'].get('file_extension', '.wav'))
    config_table.add_row("Sample Rate", str(config['dataset'].get('sample_rate', 'N/A')))
    
    # Inference configuration
    config_table.add_row("Output Directory", str(config['inference']['output_dir']))
    config_table.add_row("Batch Size", str(config['inference']['batch_size']))
    config_table.add_row("Num Workers", str(config['inference']['num_workers']))
    config_table.add_row("Save Every N Batches", str(config['inference'].get('save_every_n_batches', 10)))
    
    console.print(config_table)
    console.print()