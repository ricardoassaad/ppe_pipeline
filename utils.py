"""
Utility functions for YOLO fine-tuning pipeline
"""

import os
import logging
import torch
import yaml
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil


def setup_logging(log_dir: str, experiment_name: str, level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        level: Logging level
        
    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(
    model: Any,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    config: Optional[Dict] = None
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        path: Path to save checkpoint
        config: Configuration dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save model state
    if hasattr(model, 'state_dict'):
        checkpoint['model_state_dict'] = model.state_dict()
    elif hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
        checkpoint['model_state_dict'] = model.model.state_dict()
    
    # Save optimizer state
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Save config
    if config is not None:
        checkpoint['config'] = config
    
    # Save checkpoint
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    if model is not None and 'model_state_dict' in checkpoint:
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(checkpoint['model_state_dict'])
        elif hasattr(model, 'model') and hasattr(model.model, 'load_state_dict'):
            model.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logging.info(f"Checkpoint loaded from {path} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return checkpoint


def save_config(config: Dict, path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        path: Path to save config
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logging.info(f"Configuration saved to {path}")


def load_config(path: str) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Configuration loaded from {path}")
    
    return config


def get_device(use_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    """
    Get computation device
    
    Args:
        use_gpu: Whether to use GPU if available
        gpu_id: GPU device ID
        
    Returns:
        PyTorch device
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_yolo_dataset_yaml(
    train_path: str,
    val_path: str,
    test_path: Optional[str],
    class_names: List[str],
    output_path: str
):
    """
    Create YOLO dataset YAML file
    
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        test_path: Path to test images (optional)
        class_names: List of class names
        output_path: Path to save YAML file
    """
    data = {
        'path': str(Path(train_path).parent),
        'train': str(Path(train_path).name),
        'val': str(Path(val_path).name),
        'nc': len(class_names),
        'names': class_names
    }
    
    if test_path:
        data['test'] = str(Path(test_path).name)
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    logging.info(f"Dataset YAML created: {output_path}")


def split_dataset_kfold(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    n_splits: int = 5,
    seed: int = 42
):
    """
    Split dataset into K-Folds
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        output_dir: Output directory for splits
        n_splits: Number of folds
        seed: Random seed
    """
    from sklearn.model_selection import KFold
    
    # Get all image files
    image_files = sorted(list(Path(image_dir).glob('*.jpg')) + 
                        list(Path(image_dir).glob('*.png')))
    
    # Create K-Fold splitter
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split and save
    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_files)):
        fold_dir = output_path / f'fold_{fold}'
        
        # Create directories
        (fold_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (fold_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (fold_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (fold_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy training files
        for idx in train_idx:
            img_file = image_files[idx]
            label_file = Path(label_dir) / (img_file.stem + '.txt')
            
            shutil.copy(img_file, fold_dir / 'train' / 'images' / img_file.name)
            if label_file.exists():
                shutil.copy(label_file, fold_dir / 'train' / 'labels' / label_file.name)
        
        # Copy validation files
        for idx in val_idx:
            img_file = image_files[idx]
            label_file = Path(label_dir) / (img_file.stem + '.txt')
            
            shutil.copy(img_file, fold_dir / 'val' / 'images' / img_file.name)
            if label_file.exists():
                shutil.copy(label_file, fold_dir / 'val' / 'labels' / label_file.name)
        
        logging.info(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")


def calculate_model_size(model_path: str) -> Dict[str, float]:
    """
    Calculate model file size
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary with size information
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    size_gb = size_mb / 1024
    
    return {
        'bytes': size_bytes,
        'kb': size_kb,
        'mb': size_mb,
        'gb': size_gb
    }


def compare_model_performance(
    model_paths: List[str],
    metric_paths: List[str],
    output_path: str
):
    """
    Compare performance of multiple models
    
    Args:
        model_paths: List of model file paths
        metric_paths: List of metric JSON file paths
        output_path: Path to save comparison report
    """
    comparison = []
    
    for model_path, metric_path in zip(model_paths, metric_paths):
        # Load metrics
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        
        # Get model size
        size_info = calculate_model_size(model_path)
        
        comparison.append({
            'model': Path(model_path).name,
            'size_mb': size_info['mb'],
            'metrics': metrics
        })
    
    # Save comparison
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logging.info(f"Model comparison saved to {output_path}")
    
    return comparison


def verify_dataset_structure(data_yaml_path: str) -> bool:
    """
    Verify YOLO dataset structure
    
    Args:
        data_yaml_path: Path to dataset YAML file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data:
                logging.error(f"Missing required key: {key}")
                return False
        
        # Check paths exist
        base_path = Path(data['path'])
        train_path = base_path / data['train']
        val_path = base_path / data['val']
        
        if not train_path.exists():
            logging.error(f"Training path does not exist: {train_path}")
            return False
        
        if not val_path.exists():
            logging.error(f"Validation path does not exist: {val_path}")
            return False
        
        # Check number of classes matches names
        if len(data['names']) != data['nc']:
            logging.error(f"Number of classes ({data['nc']}) doesn't match names ({len(data['names'])})")
            return False
        
        logging.info("Dataset structure verified successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error verifying dataset structure: {e}")
        return False


def export_metrics_to_csv(metrics_list: List[Dict], output_path: str):
    """
    Export metrics to CSV file
    
    Args:
        metrics_list: List of metric dictionaries
        output_path: Path to save CSV file
    """
    import csv
    
    if not metrics_list:
        logging.warning("No metrics to export")
        return
    
    # Get all unique keys
    keys = set()
    for metrics in metrics_list:
        keys.update(metrics.keys())
    keys = sorted(list(keys))
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics_list)
    
    logging.info(f"Metrics exported to {output_path}")


def create_inference_config(
    model_path: str,
    class_names: List[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = 'cpu',
    output_path: str = 'inference_config.yaml'
):
    """
    Create inference configuration file
    
    Args:
        model_path: Path to model file
        class_names: List of class names
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold for NMS
        device: Inference device
        output_path: Path to save config
    """
    config = {
        'model_path': model_path,
        'class_names': class_names,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'device': device,
        'input_size': 640,
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Inference config saved to {output_path}")


def benchmark_model(
    model_path: str,
    input_size: tuple = (1, 3, 640, 640),
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Benchmark model inference speed
    
    Args:
        model_path: Path to model file
        input_size: Input tensor size (batch, channels, height, width)
        num_iterations: Number of inference iterations
        warmup_iterations: Number of warmup iterations
        device: Device to run inference on
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    import onnxruntime as ort
    
    # Load model
    if model_path.endswith('.onnx'):
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # Create dummy input
        dummy_input = np.random.randn(*input_size).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = session.run(None, {input_name: dummy_input})
            times.append(time.time() - start)
        
    else:
        # PyTorch model
        device = torch.device(device)
        model = torch.load(model_path, map_location=device)
        model.eval()
        
        dummy_input = torch.randn(*input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
    
    # Calculate statistics
    times = np.array(times)
    results = {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'fps': float(1.0 / np.mean(times)),
        'latency_ms': float(np.mean(times) * 1000),
    }
    
    logging.info(f"Benchmark results: {results['fps']:.2f} FPS, {results['latency_ms']:.2f} ms latency")
    
    return results


def print_system_info():
    """Print system information"""
    import platform
    
    logging.info("=" * 50)
    logging.info("System Information")
    logging.info("=" * 50)
    logging.info(f"Platform: {platform.platform()}")
    logging.info(f"Python: {platform.python_version()}")
    logging.info(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        logging.info(f"CUDA: {torch.version.cuda}")
        logging.info(f"cuDNN: {torch.backends.cudnn.version()}")
        logging.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logging.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        logging.info("CUDA: Not available")
    
    logging.info("=" * 50)


# Example usage
if __name__ == '__main__':
    # Test utilities
    logger = setup_logging('test_logs', 'test_experiment')
    set_seed(42)
    print_system_info()
    
    # Test config save/load
    test_config = {'test': 'value', 'nested': {'key': 123}}
    save_config(test_config, 'test_config.yaml')
    loaded_config = load_config('test_config.yaml')
    print(f"Config loaded: {loaded_config}")
    
    print("Utilities test completed!")