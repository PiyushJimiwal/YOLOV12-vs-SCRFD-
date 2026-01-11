"""
Face Recognition Dataset Handler
Supports dynamic number of students with face detection and recognition
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FaceRecognitionDataset(Dataset):
    """
    Dynamic face recognition dataset that can handle any number of students
    
    Dataset structure:
    data/
        train/
            student_001/
                img_001.jpg
                img_002.jpg
            student_002/
                img_001.jpg
        val/
            student_001/
                img_001.jpg
        annotations/
            train_annotations.json
            val_annotations.json
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        img_size: int = 640,
        augment: bool = True,
        max_students: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root directory containing the dataset
            split: 'train', 'val', or 'test'
            img_size: Input image size for the model
            augment: Whether to apply data augmentation
            max_students: Maximum number of students (None for unlimited)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.max_students = max_students
        
        # Student mapping (student_id -> label_index)
        self.student_to_idx = {}
        self.idx_to_student = {}
        
        # Load data
        self.samples = []
        self._load_dataset()
        
        # Setup augmentation
        self.transform = self._get_transforms()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Number of unique students: {len(self.student_to_idx)}")
    
    def _load_dataset(self):
        """Load dataset samples and build student mapping"""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Get all student directories
        student_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        if self.max_students:
            student_dirs = student_dirs[:self.max_students]
        
        # Build student mapping
        for idx, student_dir in enumerate(student_dirs):
            student_id = student_dir.name
            self.student_to_idx[student_id] = idx
            self.idx_to_student[idx] = student_id
        
        # Load samples
        for student_dir in student_dirs:
            student_id = student_dir.name
            label = self.student_to_idx[student_id]
            
            # Get all images for this student
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(student_dir.glob(ext))
            
            for img_path in image_files:
                self.samples.append({
                    'image_path': str(img_path),
                    'student_id': student_id,
                    'label': label
                })
        
        # Load annotations if available
        self._load_annotations()
    
    def _load_annotations(self):
        """Load bounding box and landmark annotations if available"""
        annotation_file = self.data_dir / 'annotations' / f'{self.split}_annotations.json'
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            # Update samples with annotations
            for sample in self.samples:
                img_name = Path(sample['image_path']).name
                if img_name in annotations:
                    sample.update(annotations[img_name])
        else:
            print(f"No annotations found for {self.split} split. Using full image.")
    
    def _get_transforms(self):
        """Get image transformations"""
        if self.augment:
            # Training augmentations
            transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.ColorJitter(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ) if 'bbox' in str(self.samples[0]) else None)
        else:
            # Validation/Test transformations
            transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        return transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label
        label = sample['label']
        
        # Prepare bounding box if available
        if 'bbox' in sample:
            bbox = sample['bbox']  # [x1, y1, x2, y2]
            class_labels = [1]  # Face class
        else:
            # Use full image as bbox
            h, w = image.shape[:2]
            bbox = [0, 0, w, h]
            class_labels = [1]
        
        # Apply transformations
        if self.augment and 'bbox' in sample:
            transformed = self.transform(
                image=image,
                bboxes=[bbox],
                class_labels=class_labels
            )
        else:
            transformed = self.transform(image=image)
        
        image_tensor = transformed['image']
        
        # Prepare output
        output = {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'student_id': sample['student_id']
        }
        
        # Add bbox and landmarks if available
        if 'bbox' in sample and self.augment:
            if len(transformed['bboxes']) > 0:
                output['bbox'] = torch.tensor(transformed['bboxes'][0], dtype=torch.float32)
            else:
                output['bbox'] = torch.tensor([0, 0, self.img_size, self.img_size], dtype=torch.float32)
        elif 'bbox' in sample:
            # Normalize bbox to image size
            h, w = image.shape[:2]
            bbox = sample['bbox']
            bbox_norm = [
                bbox[0] / w * self.img_size,
                bbox[1] / h * self.img_size,
                bbox[2] / w * self.img_size,
                bbox[3] / h * self.img_size
            ]
            output['bbox'] = torch.tensor(bbox_norm, dtype=torch.float32)
        
        if 'landmarks' in sample:
            output['landmarks'] = torch.tensor(sample['landmarks'], dtype=torch.float32)
        
        return output
    
    def add_new_student(self, student_id: str, image_paths: List[str]):
        """
        Dynamically add a new student to the dataset
        
        Args:
            student_id: Unique student identifier
            image_paths: List of image paths for this student
        """
        if student_id in self.student_to_idx:
            print(f"Student {student_id} already exists. Skipping.")
            return
        
        # Add to mapping
        new_idx = len(self.student_to_idx)
        self.student_to_idx[student_id] = new_idx
        self.idx_to_student[new_idx] = student_id
        
        # Add samples
        for img_path in image_paths:
            self.samples.append({
                'image_path': img_path,
                'student_id': student_id,
                'label': new_idx
            })
        
        print(f"Added student {student_id} with {len(image_paths)} images")
        print(f"Total students: {len(self.student_to_idx)}")
    
    def get_student_count(self):
        """Return the current number of students"""
        return len(self.student_to_idx)
    
    def get_samples_per_student(self):
        """Return distribution of samples per student"""
        distribution = {}
        for sample in self.samples:
            student_id = sample['student_id']
            distribution[student_id] = distribution.get(student_id, 0) + 1
        return distribution


class FaceDataModule:
    """Data module for managing train/val/test dataloaders"""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        img_size: int = 640,
        max_students: Optional[int] = None
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.max_students = max_students
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        self.train_dataset = FaceRecognitionDataset(
            data_dir=self.data_dir,
            split='train',
            img_size=self.img_size,
            augment=True,
            max_students=self.max_students
        )
        
        self.val_dataset = FaceRecognitionDataset(
            data_dir=self.data_dir,
            split='val',
            img_size=self.img_size,
            augment=False,
            max_students=self.max_students
        )
        
        # Test dataset is optional
        test_dir = Path(self.data_dir) / 'test'
        if test_dir.exists():
            self.test_dataset = FaceRecognitionDataset(
                data_dir=self.data_dir,
                split='test',
                img_size=self.img_size,
                augment=False,
                max_students=self.max_students
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None
    
    def get_num_classes(self):
        """Return number of unique students/classes"""
        if self.train_dataset:
            return self.train_dataset.get_student_count()
        return 0


if __name__ == "__main__":
    # Example usage
    data_dir = "data/faces"
    
    # Create data module
    data_module = FaceDataModule(
        data_dir=data_dir,
        batch_size=8,
        num_workers=2,
        img_size=640
    )
    
    # Setup datasets
    try:
        data_module.setup()
        
        # Get dataloaders
        train_loader = data_module.train_dataloader()
        
        # Test loading a batch
        batch = next(iter(train_loader))
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch labels shape: {batch['label'].shape}")
        print(f"Number of classes: {data_module.get_num_classes()}")
    except Exception as e:
        print(f"Error: {e}")
        print("Please create the dataset structure first.")
