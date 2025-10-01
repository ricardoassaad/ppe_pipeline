"""
Training callbacks for monitoring and control
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from torch.utils.tensorboard import SummaryWriter


class TensorBoardCallback:
    """
    TensorBoard logging callback
    """
    
    def __init__(self, log_dir: str, experiment_name: str = 'experiment'):
        """
        Initialize TensorBoard callback
        
        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.logger = logging.getLogger(__name__)
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value"""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalar values"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log histogram of values"""
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int):
        """Log an image"""
        self.writer.add_image(tag, image, step, dataformats='HWC')
    
    def log_images(self, tag: str, images: np.ndarray, step: int):
        """Log multiple images"""
        self.writer.add_images(tag, images, step, dataformats='NHWC')
    
    def log_model_graph(self, model: torch.nn.Module, input_size: tuple):
        """Log model graph"""
        try:
            dummy_input = torch.randn(input_size)
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            self.logger.warning(f"Could not log model graph: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and metrics"""
        self.writer.add_hparams(hparams, metrics)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        self.writer.add_text(tag, text, step)
    
    def log_pr_curve(self, tag: str, labels: np.ndarray, predictions: np.ndarray, step: int):
        """Log precision-recall curve"""
        self.writer.add_pr_curve(tag, labels, predictions, step)
    
    def close(self):
        """Close the writer"""
        self.writer.close()


class EarlyStoppingCallback:
    """
    Early stopping callback to stop training when validation metric stops improving
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping callback
        
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_path = None
        self.logger = logging.getLogger(__name__)
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def __call__(
        self,
        current_score: float,
        model: Any,
        checkpoint_path: Optional[str] = None
    ):
        """
        Check if early stopping criteria is met
        
        Args:
            current_score: Current validation metric value
            model: Model to save if improvement
            checkpoint_path: Path to save best model checkpoint
        """
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model, checkpoint_path)
        elif self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.save_checkpoint(model, checkpoint_path)
            self.logger.info(f"Validation metric improved to {current_score:.4f}")
        else:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter}/{self.patience} "
                f"(best: {self.best_score:.4f}, current: {current_score:.4f})"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info("Early stopping triggered!")
                
                if self.restore_best_weights and self.best_model_path:
                    self.logger.info(f"Restoring best model from {self.best_model_path}")
    
    def save_checkpoint(self, model: Any, checkpoint_path: Optional[str]):
        """Save model checkpoint"""
        if checkpoint_path:
            try:
                # For Ultralytics YOLO model
                if hasattr(model, 'save'):
                    model.save(checkpoint_path)
                # For PyTorch model
                elif hasattr(model, 'state_dict'):
                    torch.save(model.state_dict(), checkpoint_path)
                
                self.best_model_path = checkpoint_path
                self.logger.info(f"Model checkpoint saved to {checkpoint_path}")
            except Exception as e:
                self.logger.warning(f"Could not save checkpoint: {e}")


class LearningRateSchedulerCallback:
    """
    Custom learning rate scheduler callback
    """
    
    def __init__(
        self,
        scheduler_type: str = 'cosine',
        initial_lr: float = 0.001,
        min_lr: float = 0.00001,
        warmup_epochs: int = 3,
        total_epochs: int = 100
    ):
        """
        Initialize LR scheduler callback
        
        Args:
            scheduler_type: Type of scheduler ('cosine', 'linear', 'step')
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
        """
        self.scheduler_type = scheduler_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def get_lr(self) -> float:
        """
        Calculate learning rate for current epoch
        
        Returns:
            Current learning rate
        """
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            return self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        
        # After warmup
        progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = max(0.0, min(1.0, progress))
        
        if self.scheduler_type == 'cosine':
            # Cosine annealing
            return self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        elif self.scheduler_type == 'linear':
            # Linear decay
            return self.initial_lr - (self.initial_lr - self.min_lr) * progress
        
        elif self.scheduler_type == 'step':
            # Step decay
            decay_factor = 0.1
            step_size = self.total_epochs // 3
            steps = self.current_epoch // step_size
            return self.initial_lr * (decay_factor ** steps)
        
        else:
            return self.initial_lr
    
    def step(self):
        """Increment epoch counter"""
        self.current_epoch += 1


class ModelCheckpointCallback:
    """
    Save model checkpoints during training
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = False,
        save_frequency: int = 1,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Initialize checkpoint callback
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save best model
            save_frequency: Save every N epochs
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for monitored metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.logger = logging.getLogger(__name__)
    
    def __call__(
        self,
        epoch: int,
        model: Any,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Save checkpoint if conditions are met
        
        Args:
            epoch: Current epoch
            model: Model to save
            metrics: Dictionary of metrics
            optimizer: Optimizer to save (optional)
        """
        monitored_value = metrics.get(self.monitor, None)
        
        # Check if this is the best model
        is_best = False
        if monitored_value is not None:
            if self.mode == 'min':
                is_best = monitored_value < self.best_score
            else:
                is_best = monitored_value > self.best_score
            
            if is_best:
                self.best_score = monitored_value
        
        # Save checkpoint
        should_save = (
            (not self.save_best_only or is_best) and
            (epoch % self.save_frequency == 0)
        )
        
        if should_save:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}"
            if is_best:
                checkpoint_name += "_best"
            checkpoint_name += ".pt"
            
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'metrics': metrics,
                'best_score': self.best_score,
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model separately
            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                self.logger.info(f"Saved best model: {best_path}")


class MetricsLoggerCallback:
    """
    Log metrics to file and console
    """
    
    def __init__(self, log_file: str):
        """
        Initialize metrics logger
        
        Args:
            log_file: Path to log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        # Add epoch to metrics
        metrics['epoch'] = epoch
        self.metrics_history.append(metrics)
        
        # Log to console
        metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} - {metrics_str}")
        
        # Save to file
        with open(self.log_file, 'a') as f:
            if epoch == 0:
                # Write header
                f.write(','.join(metrics.keys()) + '\n')
            f.write(','.join([str(v) for v in metrics.values()]) + '\n')
    
    def get_history(self) -> list:
        """Get metrics history"""
        return self.metrics_history


# Example usage
if __name__ == '__main__':
    # Test TensorBoard callback
    tb_callback = TensorBoardCallback('logs/test', 'test_experiment')
    for i in range(10):
        tb_callback.log_scalar('loss', np.random.random(), i)
    tb_callback.close()
    
    # Test early stopping
    early_stop = EarlyStoppingCallback(patience=3, mode='min')
    for i in range(20):
        score = np.random.random()
        early_stop(score, None, None)
        if early_stop.early_stop:
            print(f"Early stopping at iteration {i}")
            break
    
    print("Callbacks test completed!")