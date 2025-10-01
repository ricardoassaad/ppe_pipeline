"""
Custom data augmentation pipeline for YOLO training
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import random


class YOLOAugmentation:
    """
    Custom augmentation pipeline for YOLO object detection
    """
    
    def __init__(self, config: Dict):
        """
        Initialize augmentation pipeline
        
        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        if self.enabled:
            self.transform = self._build_transform()
        else:
            self.transform = None
    
    def _build_transform(self) -> A.Compose:
        """
        Build Albumentations transformation pipeline
        
        Returns:
            Albumentations Compose object
        """
        transforms = []
        
        # Geometric transformations
        if self.config.get('degrees', 0) > 0:
            transforms.append(
                A.Rotate(
                    limit=self.config['degrees'],
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                )
            )
        
        if self.config.get('scale', 0) > 0:
            scale_limit = self.config['scale']
            transforms.append(
                A.RandomScale(
                    scale_limit=scale_limit,
                    p=0.5
                )
            )
        
        if self.config.get('translate', 0) > 0:
            translate = self.config['translate']
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=translate,
                    scale_limit=0,
                    rotate_limit=0,
                    p=0.5
                )
            )
        
        if self.config.get('shear', 0) > 0:
            transforms.append(
                A.Affine(
                    shear=self.config['shear'],
                    p=0.3
                )
            )
        
        # Flip augmentations
        if self.config.get('fliplr', 0) > 0:
            transforms.append(
                A.HorizontalFlip(p=self.config['fliplr'])
            )
        
        if self.config.get('flipud', 0) > 0:
            transforms.append(
                A.VerticalFlip(p=self.config['flipud'])
            )
        
        # Color augmentations
        if any([self.config.get('hsv_h', 0), 
                self.config.get('hsv_s', 0), 
                self.config.get('hsv_v', 0)]):
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=int(self.config.get('hsv_h', 0) * 255),
                    sat_shift_limit=int(self.config.get('hsv_s', 0) * 255),
                    val_shift_limit=int(self.config.get('hsv_v', 0) * 255),
                    p=0.7
                )
            )
        
        # Advanced augmentations
        transforms.extend([
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.OpticalDistortion(p=1.0),
                A.GridDistortion(p=1.0),
            ], p=0.2),
            
            A.OneOf([
                A.CLAHE(clip_limit=2, p=1.0),
                A.Sharpen(p=1.0),
                A.Emboss(p=1.0),
            ], p=0.2),
            
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            A.GaussNoise(std_range=(10/255, 50/255), p=0.2),
        ])
        
        # Random erasing
        if self.config.get('erasing', 0) > 0:
            num_holes_range: Tuple[int, int] = (1, 8)
            hole_height_range: Tuple[int, int] = (8, 32)
            hole_width_range: Tuple[int, int] = (8, 32)
            transforms.append(
                A.CoarseDropout(
                    num_holes_range=num_holes_range,
                    hole_height_range=hole_height_range,
                    hole_width_range=hole_width_range,
                    p=self.config['erasing']
                )
            )
        
        # Weather augmentations (optional, advanced)
        transforms.extend([
            A.OneOf([
                A.RandomRain(p=1.0),
                A.RandomSnow(p=1.0),
                A.RandomFog(p=1.0),
                A.RandomShadow(p=1.0),
            ], p=0.1),
        ])
        
        # Compose all transforms
        # For YOLO, we need to use bbox_params with 'yolo' format
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3,
                min_area=0.0
            )
        )
    
    def __call__(
        self, 
        image: np.ndarray, 
        bboxes: np.ndarray, 
        class_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentations to image and bounding boxes
        
        Args:
            image: Input image (H, W, C)
            bboxes: Bounding boxes in YOLO format (N, 4) [x_center, y_center, width, height]
            class_labels: Class labels (N,)
            
        Returns:
            Augmented image, bboxes, and class_labels
        """
        if not self.enabled or self.transform is None:
            return image, bboxes, class_labels
        
        # Apply transformations
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        return (
            transformed['image'],
            np.array(transformed['bboxes']),
            np.array(transformed['class_labels'])
        )
    
    def get_ultralytics_params(self) -> Dict:
        """
        Get augmentation parameters in Ultralytics format
        
        Returns:
            Dictionary of augmentation parameters for Ultralytics YOLO
        """
        params = {}
        
        if self.enabled:
            # Map config to Ultralytics parameters
            params['hsv_h'] = self.config.get('hsv_h', 0.015)
            params['hsv_s'] = self.config.get('hsv_s', 0.7)
            params['hsv_v'] = self.config.get('hsv_v', 0.4)
            params['degrees'] = self.config.get('degrees', 0.0)
            params['translate'] = self.config.get('translate', 0.1)
            params['scale'] = self.config.get('scale', 0.5)
            params['shear'] = self.config.get('shear', 0.0)
            params['perspective'] = self.config.get('perspective', 0.0)
            params['flipud'] = self.config.get('flipud', 0.0)
            params['fliplr'] = self.config.get('fliplr', 0.5)
            params['mosaic'] = self.config.get('mosaic', 1.0)
            params['mixup'] = self.config.get('mixup', 0.0)
            params['copy_paste'] = self.config.get('copy_paste', 0.0)
            params['erasing'] = self.config.get('erasing', 0.4)
        
        return params


# class CustomAugmentationExamples:
#     """
#     Additional custom augmentation examples
#     You can add your own augmentations here
#     """
    
#     @staticmethod
#     def create_cutout_augmentation(n_holes: int = 1, length: int = 50):
#         """
#         Create Cutout augmentation
        
#         Args:
#             n_holes: Number of holes to cut
#             length: Length of the square hole
            
#         Returns:
#             Albumentations transform
#         """
#         return A.CoarseDropout(
#             max_holes=n_holes,
#             max_height=length,
#             max_width=length,
#             min_holes=n_holes,
#             min_height=length,
#             min_width=length,
#             fill_value=0,
#             p=1.0
#         )
    
#     @staticmethod
#     def create_gridmask_augmentation():
#         """
#         Create GridMask-style augmentation
        
#         Returns:
#             Albumentations transform
#         """
#         return A.GridDistortion(
#             num_steps=5,
#             distort_limit=0.3,
#             p=1.0
#         )
    
#     @staticmethod
#     def create_jpeg_compression_augmentation():
#         """
#         Create JPEG compression augmentation for robustness
        
#         Returns:
#             Albumentations transform
#         """
#         return A.ImageCompression(
#             quality_lower=50,
#             quality_upper=100,
#             p=0.5
#         )
    
#     @staticmethod
#     def create_advanced_color_jitter():
#         """
#         Advanced color jittering
        
#         Returns:
#             Albumentations transform
#         """
#         return A.Compose([
#             A.ColorJitter(
#                 brightness=0.2,
#                 contrast=0.2,
#                 saturation=0.2,
#                 hue=0.1,
#                 p=0.8
#             ),
#             A.RGBShift(
#                 r_shift_limit=20,
#                 g_shift_limit=20,
#                 b_shift_limit=20,
#                 p=0.5
#             ),
#             A.ChannelShuffle(p=0.2),
#         ])
    
#     @staticmethod
#     def create_domain_adaptation_augmentation():
#         """
#         Create augmentations for domain adaptation
#         (e.g., synthetic to real)
        
#         Returns:
#             Albumentations transform
#         """
#         return A.Compose([
#             A.OneOf([
#                 A.ToGray(p=1.0),
#                 A.ToSepia(p=1.0),
#             ], p=0.2),
            
#             A.OneOf([
#                 A.RandomToneCurve(scale=0.3, p=1.0),
#                 A.RandomGamma(gamma_limit=(80, 120), p=1.0),
#             ], p=0.5),
            
#             A.OneOf([
#                 A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
#                 A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
#                 A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
#             ], p=0.3),
            
#             A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),
#         ])
    
#     @staticmethod
#     def create_test_time_augmentation():
#         """
#         Create Test-Time Augmentation (TTA) transforms
        
#         Returns:
#             List of Albumentations transforms for TTA
#         """
#         tta_transforms = [
#             A.Compose([]),  # Original
#             A.Compose([A.HorizontalFlip(p=1.0)]),
#             A.Compose([A.VerticalFlip(p=1.0)]),
#             A.Compose([A.Rotate(limit=90, p=1.0)]),
#             A.Compose([A.Rotate(limit=180, p=1.0)]),
#             A.Compose([A.Rotate(limit=270, p=1.0)]),
#         ]
#         return tta_transforms


# Example usage and testing
if __name__ == '__main__':
    # Test augmentation pipeline
    config = {
        'enabled': True,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'erasing': 0.4,
    }
    
    # Create augmentation
    aug = YOLOAugmentation(config)
    
    # Test image and bboxes
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_bboxes = np.array([[0.5, 0.5, 0.2, 0.2]])  # YOLO format
    test_labels = np.array([0])
    
    # Apply augmentation
    aug_image, aug_bboxes, aug_labels = aug(test_image, test_bboxes, test_labels)
    
    print(f"Original image shape: {test_image.shape}")
    print(f"Augmented image shape: {aug_image.shape}")
    print(f"Original bboxes: {test_bboxes}")
    print(f"Augmented bboxes: {aug_bboxes}")
    print(f"Augmentation successful!")