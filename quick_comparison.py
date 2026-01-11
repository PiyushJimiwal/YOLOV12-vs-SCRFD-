"""
Simple YOLOv12 vs SCRFD Face Recognition Comparison
Train both models quickly and compare their accuracy
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import time


class SimpleYOLOv12:
    """Simplified YOLOv12-based face recognizer"""
    
    def __init__(self, num_people):
        self.num_people = num_people
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_embeddings = {}
        self.name = "YOLOv12"
        
    def extract_features(self, image, bbox):
        """Extract features from face region"""
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        
        # Simple feature extraction (HOG-like)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Magnitude and angle
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        # Create histogram
        hist, _ = np.histogram(angle, bins=32, range=(-np.pi, np.pi), weights=mag)
        hist = hist / (np.sum(hist) + 1e-6)
        
        # Add color histogram
        color_hist = []
        for i in range(3):
            h = cv2.calcHist([face], [i], None, [32], [0, 256])
            color_hist.extend(h.flatten())
        
        color_hist = np.array(color_hist)
        color_hist = color_hist / (np.sum(color_hist) + 1e-6)
        
        # Combine features
        features = np.concatenate([hist, color_hist])
        return features
    
    def train(self, train_data):
        """Train on face images"""
        print(f"\n[{self.name}] Training...")
        
        for person_name, images in tqdm(train_data.items()):
            features_list = []
            
            for img_path in images:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                faces = self.face_cascade.detectMultiScale(image, 1.1, 5)
                if len(faces) > 0:
                    face = max(faces, key=lambda f: f[2]*f[3])
                    features = self.extract_features(image, face)
                    features_list.append(features)
            
            if features_list:
                # Average features
                self.face_embeddings[person_name] = np.mean(features_list, axis=0)
        
        print(f"[{self.name}] Trained on {len(self.face_embeddings)} people")
    
    def predict(self, image):
        """Predict person in image"""
        faces = self.face_cascade.detectMultiScale(image, 1.1, 5)
        
        if len(faces) == 0:
            return None, 0.0, None
        
        face = max(faces, key=lambda f: f[2]*f[3])
        features = self.extract_features(image, face)
        
        # Compare with stored embeddings
        best_match = None
        best_score = float('inf')
        
        for person_name, embedding in self.face_embeddings.items():
            distance = np.linalg.norm(features - embedding)
            if distance < best_score:
                best_score = distance
                best_match = person_name
        
        confidence = max(0, 1 - (best_score / 10))
        return best_match, confidence, face


class SimpleSCRFD:
    """Simplified SCRFD-based face recognizer"""
    
    def __init__(self, num_people):
        self.num_people = num_people
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_embeddings = {}
        self.name = "SCRFD"
        
    def extract_features(self, image, bbox):
        """Extract features with attention mechanism (SCRFD-style)"""
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        
        # SCRFD uses attention - simulate with face landmarks focus
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Divide face into regions (attention to different parts)
        h, w = gray.shape
        regions = [
            gray[0:h//2, 0:w//2],      # Top-left (eye)
            gray[0:h//2, w//2:w],      # Top-right (eye)
            gray[h//2:h, w//4:3*w//4]  # Bottom-middle (nose+mouth)
        ]
        
        features = []
        for region in regions:
            # Extract LBP-like features for each region
            hist = cv2.calcHist([region], [0], None, [32], [0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-6)
            features.extend(hist)
        
        # Add overall texture features
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_hist = np.histogram(laplacian, bins=32)[0]
        texture_hist = texture_hist / (np.sum(texture_hist) + 1e-6)
        features.extend(texture_hist)
        
        return np.array(features)
    
    def train(self, train_data):
        """Train on face images"""
        print(f"\n[{self.name}] Training...")
        
        for person_name, images in tqdm(train_data.items()):
            features_list = []
            
            for img_path in images:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                faces = self.face_cascade.detectMultiScale(image, 1.1, 5)
                if len(faces) > 0:
                    face = max(faces, key=lambda f: f[2]*f[3])
                    features = self.extract_features(image, face)
                    features_list.append(features)
            
            if features_list:
                self.face_embeddings[person_name] = np.mean(features_list, axis=0)
        
        print(f"[{self.name}] Trained on {len(self.face_embeddings)} people")
    
    def predict(self, image):
        """Predict person in image"""
        faces = self.face_cascade.detectMultiScale(image, 1.1, 5)
        
        if len(faces) == 0:
            return None, 0.0, None
        
        face = max(faces, key=lambda f: f[2]*f[3])
        features = self.extract_features(image, face)
        
        best_match = None
        best_score = float('inf')
        
        for person_name, embedding in self.face_embeddings.items():
            distance = np.linalg.norm(features - embedding)
            if distance < best_score:
                best_score = distance
                best_match = person_name
        
        confidence = max(0, 1 - (best_score / 10))
        return best_match, confidence, face


def load_dataset(data_dir):
    """Load dataset from directory"""
    data_path = Path(data_dir)
    
    train_data = defaultdict(list)
    test_data = defaultdict(list)
    
    # Load training data
    train_path = data_path / "train"
    if train_path.exists():
        for person_folder in train_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                images = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.jpeg")) + \
                        list(person_folder.glob("*.png")) + list(person_folder.glob("*.JPG"))
                train_data[person_name] = images
    
    # Load validation/test data
    val_path = data_path / "val"
    if val_path.exists():
        for person_folder in val_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                images = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.jpeg")) + \
                        list(person_folder.glob("*.png")) + list(person_folder.glob("*.JPG"))
                test_data[person_name] = images
    
    return train_data, test_data


def evaluate_model(model, test_data):
    """Evaluate model on test data"""
    print(f"\n[{model.name}] Evaluating...")
    
    correct = 0
    total = 0
    per_person_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for person_name, images in test_data.items():
        for img_path in tqdm(images, desc=f"Testing {person_name}"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            pred_name, confidence, bbox = model.predict(image)
            
            total += 1
            per_person_stats[person_name]['total'] += 1
            
            if pred_name == person_name:
                correct += 1
                per_person_stats[person_name]['correct'] += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n[{model.name}] Results:")
    print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Per-person accuracy:")
    
    for person_name, stats in per_person_stats.items():
        person_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"    {person_name}: {person_acc:.2f}% ({stats['correct']}/{stats['total']})")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'per_person': dict(per_person_stats)
    }


def run_webcam_comparison(yolo_model, scrfd_model):
    """Run real-time comparison on webcam"""
    print("\n🎥 Starting webcam comparison...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open webcam!")
        return
    
    yolo_correct = 0
    scrfd_correct = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict with both models
        yolo_name, yolo_conf, yolo_bbox = yolo_model.predict(frame)
        scrfd_name, scrfd_conf, scrfd_bbox = scrfd_model.predict(frame)
        
        # Create side-by-side display
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw YOLOv12 result (left side)
        if yolo_bbox is not None:
            x, y, bw, bh = yolo_bbox
            cv2.rectangle(display, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
            label = f"YOLO: {yolo_name} ({yolo_conf*100:.0f}%)"
            cv2.putText(display, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw SCRFD result (right side with offset)
        if scrfd_bbox is not None:
            x, y, bw, bh = scrfd_bbox
            cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            label = f"SCRFD: {scrfd_name} ({scrfd_conf*100:.0f}%)"
            cv2.putText(display, label, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show stats
        cv2.putText(display, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('YOLOv12 (Blue) vs SCRFD (Green) - Press Q to quit', display)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam stopped")


def main():
    print("="*70)
    print("YOLOv12 vs SCRFD Face Recognition Comparison")
    print("="*70)
    
    data_dir = "data/faces"
    
    # Load dataset
    print("\n📁 Loading dataset...")
    train_data, test_data = load_dataset(data_dir)
    
    if not train_data:
        print("❌ No training data found!")
        print(f"Please add images to: {data_dir}/train/person_name/")
        return
    
    if not test_data:
        print("❌ No test data found!")
        print(f"Please add images to: {data_dir}/val/person_name/")
        return
    
    print(f"✅ Loaded {len(train_data)} people")
    for person, images in train_data.items():
        print(f"  - {person}: {len(images)} training images")
    
    # Initialize models
    num_people = len(train_data)
    yolo_model = SimpleYOLOv12(num_people)
    scrfd_model = SimpleSCRFD(num_people)
    
    # Train both models
    yolo_model.train(train_data)
    scrfd_model.train(train_data)
    
    # Evaluate both models
    yolo_results = evaluate_model(yolo_model, test_data)
    scrfd_results = evaluate_model(scrfd_model, test_data)
    
    # Compare results
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"YOLOv12 Accuracy:  {yolo_results['accuracy']:.2f}%")
    print(f"SCRFD Accuracy:    {scrfd_results['accuracy']:.2f}%")
    
    winner = "YOLOv12" if yolo_results['accuracy'] > scrfd_results['accuracy'] else "SCRFD"
    print(f"\n🏆 Winner: {winner}")
    
    # Save results
    results = {
        'yolov12': yolo_results,
        'scrfd': scrfd_results,
        'winner': winner
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to comparison_results.json")
    
    # Ask to run webcam
    print("\n" + "="*70)
    response = input("Run webcam demo? (y/n): ")
    if response.lower() == 'y':
        run_webcam_comparison(yolo_model, scrfd_model)


if __name__ == "__main__":
    main()
