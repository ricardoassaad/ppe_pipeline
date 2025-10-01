"""
YOLO11n Fine-tuning Pipeline
Main training script with full configurability
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json

from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Custom imports (other modules)
from augmentations import YOLOAugmentation
from callbacks import TensorBoardCallback, EarlyStoppingCallback
from utils import setup_logging, save_checkpoint, load_checkpoint


class YOLOFineTuner:
    """
    Complete YOLO11n fine-tuning pipeline with advanced features
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the fine-tuner with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.logger = setup_logging(
            self.config['paths']['log_dir'],
            self.config['training']['experiment_name']
        )
        
        # Initialize device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config['training']['use_gpu'] 
            else 'cpu'
        )
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize augmentation
        self.augmentation = YOLOAugmentation(self.config['augmentation'])
        
        # Initialize model
        self.model = None
        self.best_metrics = {}
        
    def setup_directories(self):
        """Create necessary directories"""
        for key, path in self.config['paths'].items():
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def load_model(self, weights: Optional[str] = None):
        """
        Load YOLO model
        
        Args:
            weights: Path to weights file, or None to use pretrained
        """
        weights = weights or self.config['model']['pretrained_weights']
        self.logger.info(f"Loading model from: {weights}")
        self.model = YOLO(weights)
        
    def freeze_layers(self, freeze: bool = True, freeze_until: int = None):
        """
        Freeze/unfreeze model layers
        
        Args:
            freeze: Whether to freeze or unfreeze
            freeze_until: Layer index to freeze until (None = all except last)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        model_params = list(self.model.model.parameters())
        
        if freeze_until is None:
            # Freeze all except last layer
            freeze_until = len(model_params) - 1
            
        for i, param in enumerate(model_params):
            if i < freeze_until:
                param.requires_grad = not freeze
            else:
                param.requires_grad = True
                
        frozen_count = sum(1 for p in model_params if not p.requires_grad)
        self.logger.info(
            f"{'Frozen' if freeze else 'Unfrozen'} {frozen_count}/{len(model_params)} layers"
        )
        
    def train_fold(
        self,
        fold: int,
        train_data: str,
        val_data: str,
        freeze_epochs: int,
        unfreeze_epochs: int
    ) -> Dict:
        """
        Train a single fold with two-phase training
        
        Args:
            fold: Fold number
            train_data: Path to training data config
            val_data: Path to validation data config
            freeze_epochs: Epochs to train with frozen backbone
            unfreeze_epochs: Epochs to train with unfrozen model
            
        Returns:
            Dictionary of final metrics
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Training Fold {fold}")
        self.logger.info(f"{'='*50}\n")
        
        # Create fold-specific directories
        fold_dir = Path(self.config['paths']['checkpoint_dir']) / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        
        # TensorBoard writer
        tb_writer = SummaryWriter(
            log_dir=str(Path(self.config['paths']['tensorboard_dir']) / f"fold_{fold}")
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            patience=self.config['training']['early_stopping_patience'],
            min_delta=self.config['training']['early_stopping_min_delta'],
            mode='max'  # Assuming we're maximizing mAP
        )
        
        # Phase 1: Train with frozen backbone
        if freeze_epochs > 0:
            self.logger.info("Phase 1: Training with frozen backbone")
            self.freeze_layers(freeze=True)
            
            phase1_metrics = self._train_phase(
                data_config=train_data,
                epochs=freeze_epochs,
                phase_name="frozen",
                fold=fold,
                fold_dir=fold_dir,
                tb_writer=tb_writer,
                early_stopping=early_stopping
            )
            
            if early_stopping.early_stop:
                self.logger.info("Early stopping triggered in Phase 1")
                return phase1_metrics
        
        # Phase 2: Fine-tune entire model
        if unfreeze_epochs > 0:
            self.logger.info("Phase 2: Fine-tuning entire model")

            self.logger.info("Reloading best frozen model")
            best_frozen_model_path = fold_dir / "frozen_best.pt"
            if best_frozen_model_path.exists():
                self.logger.info(f"Loading best model from Phase 1: {best_frozen_model_path}")
                self.load_model(weights=str(best_frozen_model_path))
            else:
                self.logger.warning(
                    "Could not find 'frozen_best.pt'. "
                    "Make sure your EarlyStoppingCallback is saving it. "
                    "Proceeding with the model's last state."
                )

            self.freeze_layers(freeze=False)
            
            # Reduce learning rate for fine-tuning
            phase2_lr = self.config['optimizer']['learning_rate'] * self.config['training']['finetune_lr_multiplier']
            
            phase2_metrics = self._train_phase(
                data_config=train_data,
                epochs=unfreeze_epochs,
                phase_name="unfrozen",
                fold=fold,
                fold_dir=fold_dir,
                tb_writer=tb_writer,
                early_stopping=early_stopping,
                learning_rate=phase2_lr
            )
            
        tb_writer.close()
        
        # Return final metrics
        final_metrics = phase2_metrics if unfreeze_epochs > 0 else phase1_metrics
        return final_metrics
    
    def _train_phase(
        self,
        data_config: str,
        epochs: int,
        phase_name: str,
        fold: int,
        fold_dir: Path,
        tb_writer: SummaryWriter,
        early_stopping: EarlyStoppingCallback,
        learning_rate: Optional[float] = None
    ) -> Dict:
        """
        Train a single phase
        
        Args:
            data_config: Path to data YAML
            epochs: Number of epochs
            phase_name: Name of training phase
            fold: Fold number
            fold_dir: Directory for fold checkpoints
            tb_writer: TensorBoard writer
            early_stopping: Early stopping callback
            learning_rate: Override learning rate
            
        Returns:
            Final metrics dictionary
        """
        # Training arguments
        train_args = {
            'data': data_config,
            'epochs': epochs,
            'imgsz': self.config['model']['input_size'],
            'batch': self.config['training']['batch_size'],
            'device': self.device,
            'workers': self.config['training']['num_workers'],
            'project': str(fold_dir),
            'name': phase_name,
            'optimizer': self.config['optimizer']['type'],
            'lr0': learning_rate or self.config['optimizer']['learning_rate'],
            'lrf': self.config['optimizer']['final_lr_ratio'],
            'momentum': self.config['optimizer']['momentum'],
            'weight_decay': self.config['optimizer']['weight_decay'],
            'warmup_epochs': self.config['optimizer']['warmup_epochs'],
            'box': self.config['loss']['box_loss_weight'],
            'cls': self.config['loss']['cls_loss_weight'],
            'dfl': self.config['loss']['dfl_loss_weight'],
            'save': True,
            'save_period': self.config['training']['save_period'],
            'cache': self.config['training']['cache_images'],
            'verbose': True,
            'seed': self.config['training']['seed'],
            'deterministic': True,
            'amp': self.config['training']['use_amp'],
        }
        
        # Add augmentation parameters
        if self.config['augmentation']['enabled']:
            train_args.update(self.augmentation.get_ultralytics_params())
        
        # Train
        results = self.model.train(**train_args)
        
        # Validate
        val_results = self.model.val()
        
        # Extract metrics
        metrics = {
            'mAP50': val_results.box.map50,
            'mAP50-95': val_results.box.map,
            'precision': val_results.box.mp,
            'recall': val_results.box.mr,
        }
        
        # Log to TensorBoard
        for metric_name, metric_value in metrics.items():
            tb_writer.add_scalar(
                f'{phase_name}/{metric_name}',
                metric_value,
                fold
            )
        
        # Check early stopping
        early_stopping(metrics['mAP50-95'], self.model, fold_dir / f"{phase_name}_best.pt")
        
        # Save checkpoint
        checkpoint_path = fold_dir / f"{phase_name}_final.pt"
        save_checkpoint(
            model=self.model,
            optimizer=None,
            epoch=epochs,
            metrics=metrics,
            path=checkpoint_path,
            config=self.config
        )
        
        self.logger.info(f"Phase {phase_name} completed. Metrics: {metrics}")
        
        return metrics
    
    def train_kfold(self, n_splits: int = 5):
        """
        Train with K-Fold cross-validation
        
        Args:
            n_splits: Number of folds
        """
        self.logger.info(f"Starting {n_splits}-Fold Cross-Validation")
        
        # Load dataset information
        data_config = self.config['data']['data_yaml']
        
        # Initialize K-Fold
        kfold = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config['training']['seed']
        )
        
        fold_metrics = []
        
        # Note: You'll need to implement dataset splitting logic based on your data structure
        # This is a placeholder for the K-Fold iteration
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(100))):  # Placeholder
            # Load fresh model for each fold
            self.load_model()
            
            # Train fold
            metrics = self.train_fold(
                fold=fold,
                train_data=data_config,  # You'd create fold-specific configs here
                val_data=data_config,
                freeze_epochs=self.config['training']['freeze_epochs'],
                unfreeze_epochs=self.config['training']['unfreeze_epochs']
            )
            
            fold_metrics.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in fold_metrics])
            for metric in fold_metrics[0].keys()
        }
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"K-Fold Cross-Validation Results")
        self.logger.info(f"{'='*50}")
        for metric, value in avg_metrics.items():
            self.logger.info(f"{metric}: {value:.4f} (+/- {np.std([fold[metric] for fold in fold_metrics]):.4f})")
        
        # Save results
        results_path = Path(self.config['paths']['checkpoint_dir']) / 'kfold_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'fold_metrics': fold_metrics,
                'average_metrics': avg_metrics
            }, f, indent=2)
        
        return avg_metrics, fold_metrics
    
    def export_to_onnx(
        self,
        model_path: str,
        output_path: str,
        simplify: bool = True,
        dynamic: bool = False,
        opset: int = 12
    ):
        """
        Export model to ONNX format
        
        Args:
            model_path: Path to PyTorch model
            output_path: Path to save ONNX model
            simplify: Whether to simplify the model
            dynamic: Whether to use dynamic input shapes
            opset: ONNX opset version
        """
        self.logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Load model if not loaded
        if self.model is None or str(self.model.ckpt_path) != model_path:
            self.load_model(model_path)
        
        # Export to ONNX
        self.model.export(
            format='onnx',
            imgsz=self.config['model']['input_size'],
            simplify=simplify,
            dynamic=dynamic,
            opset=opset,
        )
        
        # Move to desired location
        source_onnx = Path(model_path).with_suffix('.onnx')
        if source_onnx.exists():
            source_onnx.rename(output_path)
        
        self.logger.info(f"ONNX model saved to: {output_path}")
        
        return output_path
    
    def quantize_onnx_model(
        self,
        onnx_model_path: str,
        quantized_output_path: str,
        quantization_mode: str = 'dynamic'
    ):
        """
        Quantize ONNX model to INT8 for faster CPU inference
        
        Args:
            onnx_model_path: Path to ONNX model
            quantized_output_path: Path to save quantized model
            quantization_mode: 'dynamic' or 'static'
        """
        self.logger.info(f"Quantizing ONNX model: {onnx_model_path}")
        
        if quantization_mode == 'dynamic':
            # Dynamic quantization (no calibration data needed)
            quantize_dynamic(
                model_input=onnx_model_path,
                model_output=quantized_output_path,
                weight_type=QuantType.QUInt8,
                optimize_model=True,
                extra_options={
                    'EnableSubgraph': True,
                    'EnableRewriter': True,
                    'ForceQuantizeNoInputCheck': False,
                }
            )
        else:
            raise NotImplementedError("Static quantization requires calibration data")
        
        self.logger.info(f"Quantized model saved to: {quantized_output_path}")
        
        # Compare model sizes
        original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(quantized_output_path) / (1024 * 1024)
        compression_ratio = (1 - quantized_size / original_size) * 100
        
        self.logger.info(f"Original size: {original_size:.2f} MB")
        self.logger.info(f"Quantized size: {quantized_size:.2f} MB")
        self.logger.info(f"Compression: {compression_ratio:.2f}%")
        
        return quantized_output_path
    
    def validate_onnx_model(self, onnx_path: str, test_image: Optional[np.ndarray] = None):
        """
        Validate ONNX model can run inference
        
        Args:
            onnx_path: Path to ONNX model
            test_image: Optional test image (will create random if None)
        """
        self.logger.info(f"Validating ONNX model: {onnx_path}")
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input shape
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        
        # Create test input
        if test_image is None:
            test_image = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        outputs = ort_session.run(None, {input_name: test_image})
        
        self.logger.info(f"ONNX inference successful. Output shapes: {[o.shape for o in outputs]}")
        
        return outputs


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO11n Fine-tuning Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'train_kfold', 'export', 'quantize'],
                       help='Operation mode')
    parser.add_argument('--model_path', type=str, help='Model path for export/quantize')
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    finetuner = YOLOFineTuner(args.config)
    
    if args.mode == 'train':
        # Single training run
        finetuner.load_model()
        finetuner.train_fold(
            fold=0,
            train_data=finetuner.config['data']['data_yaml'],
            val_data=finetuner.config['data']['data_yaml'],
            freeze_epochs=finetuner.config['training']['freeze_epochs'],
            unfreeze_epochs=finetuner.config['training']['unfreeze_epochs']
        )
        
    elif args.mode == 'train_kfold':
        # K-Fold cross-validation
        finetuner.train_kfold(n_splits=finetuner.config['training']['kfold_splits'])
        
    elif args.mode == 'export':
        # Export to ONNX
        if not args.model_path:
            raise ValueError("--model_path required for export mode")
        
        onnx_path = Path(finetuner.config['paths']['export_dir']) / 'model.onnx'
        finetuner.export_to_onnx(
            model_path=args.model_path,
            output_path=str(onnx_path),
            simplify=True,
            dynamic=finetuner.config['export']['dynamic_shapes'],
            opset=finetuner.config['export']['onnx_opset']
        )
        
        # Validate
        finetuner.validate_onnx_model(str(onnx_path))
        
    elif args.mode == 'quantize':
        # Quantize ONNX model
        if not args.model_path:
            raise ValueError("--model_path required for quantize mode")
        
        quantized_path = Path(args.model_path).with_name(
            Path(args.model_path).stem + '_int8.onnx'
        )
        finetuner.quantize_onnx_model(
            onnx_model_path=args.model_path,
            quantized_output_path=str(quantized_path),
            quantization_mode='dynamic'
        )
        
        # Validate
        finetuner.validate_onnx_model(str(quantized_path))


if __name__ == '__main__':
    main()
