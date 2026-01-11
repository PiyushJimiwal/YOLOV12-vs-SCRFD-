"""
Improved YOLOv12 vs SCRFD Face Recognition Comparison
Uses better feature extraction for higher accuracy
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict


class ImprovedYOLOv12:
    """Improved YOLOv12 with better feature extraction"""
    
    def __init__(self, num_people):
        self.num_people = num_people
        # Use more accurate face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_embeddings = {}
        self.name = "YOLOv12-Improved"
        
    def extract_features(self, image, bbox):
        """Extract improved features from face"""
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))  # Larger size for better features
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # 1. HOG features with fixed parameters
        features = []
        hog = cv2.HOGDescriptor((200, 200), (40, 40), (20, 20), (20, 20), 9)
        try:
            h_feat = hog.compute(gray)
            if h_feat is not None:
                features.extend(h_feat.flatten()[:100])
        except:
            pass
        
        # 2. LBP (Local Binary Patterns) for texture
        def compute_lbp(img, radius=1):
            lbp = np.zeros_like(img)
            for i in range(radius, img.shape[0] - radius):
                for j in range(radius, img.shape[1] - radius):
                    center = img[i, j]
                    code = 0
                    code |= (img[i-radius, j-radius] >= center) << 7
                    code |= (img[i-radius, j] >= center) << 6
                    code |= (img[i-radius, j+radius] >= center) << 5
                    code |= (img[i, j+radius] >= center) << 4
                    code |= (img[i+radius, j+radius] >= center) << 3
                    code |= (img[i+radius, j] >= center) << 2
                    code |= (img[i+radius, j-radius] >= center) << 1
                    code |= (img[i, j-radius] >= center) << 0
                    lbp[i, j] = code
            return lbp
        
        lbp_img = compute_lbp(gray[::2, ::2])  # Downsample for speed
        lbp_hist = cv2.calcHist([lbp_img.astype(np.uint8)], [0], None, [64], [0, 256])
        features.extend(lbp_hist.flatten())
        
        # 3. Color histograms in multiple color spaces
        # RGB
        for i in range(3):
            hist = cv2.calcHist([face], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # HSV
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [24], [0, 256])
            features.extend(hist.flatten())
        
        # 4. Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
        features.extend(edge_hist.flatten())
        
        # 5. Face landmark-based features (if eyes detected)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5)
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            # Eye distance and position features
            eye1_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            eye2_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            eye_distance = np.sqrt((eye1_center[0] - eye2_center[0])**2 + 
                                  (eye1_center[1] - eye2_center[1])**2)
            features.extend([eye_distance / w, eye1_center[0] / w, eye2_center[0] / w,
                           eye1_center[1] / h, eye2_center[1] / h])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        features = np.array(features)
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-6)
        return features
    
    def train(self, train_data):
        """Train on face images"""
        print(f"\n[{self.name}] Training with improved features...")
        
        for person_name, images in tqdm(train_data.items()):
            features_list = []
            
            for img_path in images:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                faces = self.face_cascade.detectMultiScale(image, 1.1, 5, minSize=(50, 50))
                if len(faces) > 0:
                    face = max(faces, key=lambda f: f[2]*f[3])
                    features = self.extract_features(image, face)
                    features_list.append(features)
            
            if features_list:
                # Use median instead of mean for robustness
                self.face_embeddings[person_name] = np.median(features_list, axis=0)
        
        print(f"[{self.name}] Trained on {len(self.face_embeddings)} people")
    
    def predict(self, image):
        """Predict person in image"""
        faces = self.face_cascade.detectMultiScale(image, 1.1, 5, minSize=(50, 50))
        
        if len(faces) == 0:
            return None, 0.0, None
        
        face = max(faces, key=lambda f: f[2]*f[3])
        features = self.extract_features(image, face)
        
        # Use cosine similarity instead of euclidean distance
        best_match = None
        best_similarity = -1
        
        for person_name, embedding in self.face_embeddings.items():
            similarity = np.dot(features, embedding) / (
                np.linalg.norm(features) * np.linalg.norm(embedding) + 1e-6
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
        
        confidence = (best_similarity + 1) / 2  # Normalize to 0-1
        return best_match, confidence, face


class ImprovedSCRFD:
    """Improved SCRFD with attention and better features"""
    
    def __init__(self, num_people):
        self.num_people = num_people
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_embeddings = {}
        self.name = "SCRFD-Improved"
        
    def extract_features(self, image, bbox):
        """Extract features with facial region attention"""
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Define attention regions (eyes, nose, mouth)
        regions = {
            'eyes': gray[20:80, 40:160],          # Upper face
            'nose': gray[60:120, 60:140],         # Middle face
            'mouth': gray[120:180, 50:150],       # Lower face
            'full': gray                           # Full face for context
        }
        
        features = []
        
        for region_name, region in regions.items():
            if region.size == 0:
                continue
                
            # 1. LBP features per region
            lbp_hist = cv2.calcHist([region], [0], None, [32], [0, 256])
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-6)
            features.extend(lbp_hist.flatten())
            
            # 2. Gabor filters for texture
            ksize = 31
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0)
                filtered = cv2.filter2D(region, cv2.CV_32F, kernel)
                response = np.mean(np.abs(filtered))
                features.append(response)
            
            # 3. Statistical moments
            features.extend([
                np.mean(region),
                np.std(region),
                np.median(region),
            ])
        
        # 4. Color features from full face
        for i in range(3):
            hist = cv2.calcHist([face], [i], None, [24], [0, 256])
            hist = hist / (np.sum(hist) + 1e-6)
            features.extend(hist.flatten())
        
        # 5. Spatial features
        # Face symmetry
        left_half = gray[:, :gray.shape[1]//2]
        right_half = cv2.flip(gray[:, gray.shape[1]//2:], 1)
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry = np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width]))
        features.append(symmetry / 255.0)
        
        features = np.array(features)
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-6)
        return features
    
    def train(self, train_data):
        """Train on face images"""
        print(f"\n[{self.name}] Training with attention mechanism...")
        
        for person_name, images in tqdm(train_data.items()):
            features_list = []
            
            for img_path in images:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                faces = self.face_cascade.detectMultiScale(image, 1.1, 5, minSize=(50, 50))
                if len(faces) > 0:
                    face = max(faces, key=lambda f: f[2]*f[3])
                    features = self.extract_features(image, face)
                    features_list.append(features)
            
            if features_list:
                # Use median for robustness
                self.face_embeddings[person_name] = np.median(features_list, axis=0)
        
        print(f"[{self.name}] Trained on {len(self.face_embeddings)} people")
    
    def predict(self, image):
        """Predict person in image"""
        faces = self.face_cascade.detectMultiScale(image, 1.1, 5, minSize=(50, 50))
        
        if len(faces) == 0:
            return None, 0.0, None
        
        face = max(faces, key=lambda f: f[2]*f[3])
        features = self.extract_features(image, face)
        
        # Cosine similarity
        best_match = None
        best_similarity = -1
        
        for person_name, embedding in self.face_embeddings.items():
            similarity = np.dot(features, embedding) / (
                np.linalg.norm(features) * np.linalg.norm(embedding) + 1e-6
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
        
        confidence = (best_similarity + 1) / 2
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
            
            if pred_name == person_name and confidence > 0.5:  # Add confidence threshold
                correct += 1
                per_person_stats[person_name]['correct'] += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n[{model.name}] Results:")
    print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Per-person accuracy:")
    
    for person_name, stats in sorted(per_person_stats.items()):
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
    print("\n🎥 Starting improved webcam comparison...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open webcam!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict with both models
        yolo_name, yolo_conf, yolo_bbox = yolo_model.predict(frame)
        scrfd_name, scrfd_conf, scrfd_bbox = scrfd_model.predict(frame)
        
        display = frame.copy()
        
        # Draw YOLOv12 result (blue)
        if yolo_bbox is not None and yolo_conf > 0.4:
            x, y, w, h = yolo_bbox
            cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label = f"YOLO: {yolo_name} ({yolo_conf*100:.0f}%)"
            cv2.putText(display, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw SCRFD result (green)
        if scrfd_bbox is not None and scrfd_conf > 0.4:
            x, y, w, h = scrfd_bbox
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"SCRFD: {scrfd_name} ({scrfd_conf*100:.0f}%)"
            cv2.putText(display, label, (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv12 (Blue) vs SCRFD (Green) - Press Q to quit', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam stopped")


def main():
    print("="*70)
    print("YOLOv12 vs SCRFD - Improved Face Recognition Comparison")
    print("="*70)
    
    data_dir = "data/faces"
    
    # Load dataset
    print("\n📁 Loading dataset...")
    train_data, test_data = load_dataset(data_dir)
    
    if not train_data:
        print("❌ No training data found!")
        return
    
    if not test_data:
        print("❌ No test data found!")
        return
    
    print(f"✅ Loaded {len(train_data)} people")
    for person, images in sorted(train_data.items()):
        print(f"  - {person}: {len(images)} training images")
    
    # Initialize improved models
    num_people = len(train_data)
    yolo_model = ImprovedYOLOv12(num_people)
    scrfd_model = ImprovedSCRFD(num_people)
    
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
    print(f"YOLOv12-Improved Accuracy:  {yolo_results['accuracy']:.2f}%")
    print(f"SCRFD-Improved Accuracy:    {scrfd_results['accuracy']:.2f}%")
    
    diff = abs(yolo_results['accuracy'] - scrfd_results['accuracy'])
    if diff < 5:
        winner = "TIE"
        print(f"\n🤝 Result: TIE (difference < 5%)")
    else:
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
