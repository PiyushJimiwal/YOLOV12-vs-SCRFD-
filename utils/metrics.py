"""
Metrics Module for Face Recognition System
Implements: Accuracy, False Acceptance Rate (FAR), False Rejection Rate (FRR), and related metrics
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class FaceRecognitionMetrics:
    """Calculate and track face recognition metrics"""
    
    def __init__(self, num_classes: int, threshold: float = 0.5):
        """
        Args:
            num_classes: Number of unique students/identities
            threshold: Distance threshold for face matching
        """
        self.num_classes = num_classes
        self.threshold = threshold
        
        # Tracking variables
        self.reset()
    
    def reset(self):
        """Reset all metric trackers"""
        self.all_predictions = []
        self.all_labels = []
        self.all_distances = []
        self.all_embeddings = []
        
        # For detection metrics
        self.tp_detection = 0  # True Positive (face detected correctly)
        self.fp_detection = 0  # False Positive (non-face detected as face)
        self.fn_detection = 0  # False Negative (face not detected)
        self.tn_detection = 0  # True Negative (non-face correctly rejected)
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with batch results
        
        Args:
            predictions: Predicted class labels [batch_size]
            labels: Ground truth labels [batch_size]
            embeddings: Face embeddings [batch_size, embedding_dim]
            distances: Pairwise distances [batch_size] or similarity scores
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)
        
        if embeddings is not None:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            self.all_embeddings.append(embeddings)
        
        if distances is not None:
            if isinstance(distances, torch.Tensor):
                distances = distances.detach().cpu().numpy()
            self.all_distances.extend(distances)
    
    def update_detection(self, tp: int = 0, fp: int = 0, fn: int = 0, tn: int = 0):
        """Update face detection metrics"""
        self.tp_detection += tp
        self.fp_detection += fp
        self.fn_detection += fn
        self.tn_detection += tn
    
    def compute_accuracy(self) -> float:
        """
        Compute overall recognition accuracy
        
        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(self.all_predictions) == 0:
            return 0.0
        
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        return accuracy
    
    def compute_far_frr(self, distances: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR)
        
        FAR: Probability that an unauthorized person is accepted (Type II error)
        FRR: Probability that an authorized person is rejected (Type I error)
        
        Args:
            distances: Distance/dissimilarity scores (optional, uses stored if None)
        
        Returns:
            Dictionary with FAR, FRR, and threshold
        """
        if distances is None:
            if len(self.all_distances) == 0:
                # Compute from predictions and labels
                return self._compute_far_frr_from_predictions()
            distances = np.array(self.all_distances)
        
        labels = np.array(self.all_labels)
        predictions = np.array(self.all_predictions)
        
        # Genuine pairs: same identity (correct match)
        genuine_mask = labels == predictions
        genuine_distances = distances[genuine_mask]
        
        # Impostor pairs: different identity (incorrect match)
        impostor_mask = labels != predictions
        impostor_distances = distances[impostor_mask]
        
        if len(genuine_distances) == 0 or len(impostor_distances) == 0:
            return {'FAR': 0.0, 'FRR': 0.0, 'threshold': self.threshold}
        
        # FRR: genuine pairs rejected (distance > threshold)
        frr = np.sum(genuine_distances > self.threshold) / len(genuine_distances)
        
        # FAR: impostor pairs accepted (distance < threshold)
        far = np.sum(impostor_distances < self.threshold) / len(impostor_distances)
        
        return {
            'FAR': float(far),
            'FRR': float(frr),
            'threshold': self.threshold
        }
    
    def _compute_far_frr_from_predictions(self) -> Dict[str, float]:
        """Compute FAR/FRR from confusion matrix when distances not available"""
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        # For multi-class scenario
        # False Acceptance: predicted as someone else (off-diagonal)
        # False Rejection: should be accepted but rejected
        
        total_samples = len(self.all_labels)
        
        # Calculate per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # FAR: False acceptance among all negative cases
        total_negatives = fp + tn
        far = fp.sum() / total_negatives.sum() if total_negatives.sum() > 0 else 0.0
        
        # FRR: False rejection among all positive cases  
        total_positives = tp + fn
        frr = fn.sum() / total_positives.sum() if total_positives.sum() > 0 else 0.0
        
        return {
            'FAR': float(far),
            'FRR': float(frr),
            'threshold': self.threshold
        }
    
    def compute_eer(self, num_thresholds: int = 100) -> Dict[str, float]:
        """
        Compute Equal Error Rate (EER) - point where FAR = FRR
        
        Args:
            num_thresholds: Number of thresholds to test
        
        Returns:
            Dictionary with EER value and corresponding threshold
        """
        if len(self.all_distances) == 0:
            return {'EER': 0.0, 'EER_threshold': self.threshold}
        
        distances = np.array(self.all_distances)
        labels = np.array(self.all_labels)
        predictions = np.array(self.all_predictions)
        
        # Generate thresholds
        min_dist = distances.min()
        max_dist = distances.max()
        thresholds = np.linspace(min_dist, max_dist, num_thresholds)
        
        far_list = []
        frr_list = []
        
        genuine_mask = labels == predictions
        impostor_mask = labels != predictions
        
        genuine_distances = distances[genuine_mask]
        impostor_distances = distances[impostor_mask]
        
        for thresh in thresholds:
            frr = np.sum(genuine_distances > thresh) / len(genuine_distances) if len(genuine_distances) > 0 else 0
            far = np.sum(impostor_distances < thresh) / len(impostor_distances) if len(impostor_distances) > 0 else 0
            
            far_list.append(far)
            frr_list.append(frr)
        
        # Find EER (where FAR and FRR intersect)
        far_array = np.array(far_list)
        frr_array = np.array(frr_list)
        
        diff = np.abs(far_array - frr_array)
        min_idx = np.argmin(diff)
        
        eer = (far_array[min_idx] + frr_array[min_idx]) / 2
        eer_threshold = thresholds[min_idx]
        
        return {
            'EER': float(eer),
            'EER_threshold': float(eer_threshold)
        }
    
    def compute_detection_metrics(self) -> Dict[str, float]:
        """
        Compute face detection metrics
        
        Returns:
            Dictionary with precision, recall, F1 score for detection
        """
        tp = self.tp_detection
        fp = self.fp_detection
        fn = self.fn_detection
        tn = self.tn_detection
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        detection_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            'detection_precision': float(precision),
            'detection_recall': float(recall),
            'detection_f1': float(f1),
            'detection_accuracy': float(detection_accuracy)
        }
    
    def compute_per_class_metrics(self) -> Dict[str, np.ndarray]:
        """
        Compute per-class (per-student) metrics
        
        Returns:
            Dictionary with per-class precision, recall, F1 scores
        """
        if len(self.all_predictions) == 0:
            return {}
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels,
            self.all_predictions,
            average=None,
            zero_division=0
        )
        
        return {
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support
        }
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all available metrics
        
        Returns:
            Dictionary with all computed metrics
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = self.compute_accuracy()
        
        # FAR and FRR
        far_frr = self.compute_far_frr()
        metrics.update(far_frr)
        
        # EER
        if len(self.all_distances) > 0:
            eer = self.compute_eer()
            metrics.update(eer)
        
        # Detection metrics
        detection_metrics = self.compute_detection_metrics()
        metrics.update(detection_metrics)
        
        # Per-class metrics
        per_class = self.compute_per_class_metrics()
        if per_class:
            metrics['avg_precision'] = float(per_class['per_class_precision'].mean())
            metrics['avg_recall'] = float(per_class['per_class_recall'].mean())
            metrics['avg_f1'] = float(per_class['per_class_f1'].mean())
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, normalize: bool = False):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
        """
        if len(self.all_predictions) == 0:
            print("No predictions available to plot confusion matrix")
            return
        
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        
        # For large number of classes, show heatmap
        if self.num_classes > 20:
            sns.heatmap(cm, annot=False, fmt='.2f' if normalize else 'd', cmap='Blues')
        else:
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, save_path: Optional[str] = None):
        """
        Plot ROC curve (FAR vs FRR)
        
        Args:
            save_path: Path to save the plot
        """
        if len(self.all_distances) == 0:
            print("No distance data available to plot ROC curve")
            return
        
        distances = np.array(self.all_distances)
        labels = np.array(self.all_labels)
        predictions = np.array(self.all_predictions)
        
        genuine_mask = labels == predictions
        impostor_mask = labels != predictions
        
        genuine_distances = distances[genuine_mask]
        impostor_distances = distances[impostor_mask]
        
        # Generate thresholds
        thresholds = np.linspace(distances.min(), distances.max(), 100)
        
        far_list = []
        frr_list = []
        
        for thresh in thresholds:
            frr = np.sum(genuine_distances > thresh) / len(genuine_distances) if len(genuine_distances) > 0 else 0
            far = np.sum(impostor_distances < thresh) / len(impostor_distances) if len(impostor_distances) > 0 else 0
            
            far_list.append(far)
            frr_list.append(frr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(far_list, frr_list, 'b-', linewidth=2, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        
        # Mark EER point
        eer_data = self.compute_eer()
        plt.plot(eer_data['EER'], eer_data['EER'], 'go', markersize=10, label=f"EER = {eer_data['EER']:.4f}")
        
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('False Rejection Rate (FRR)')
        plt.title('ROC Curve (FAR vs FRR)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_summary(self):
        """Print a summary of all metrics"""
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*60)
        print("FACE RECOGNITION METRICS SUMMARY")
        print("="*60)
        
        print("\n--- Recognition Metrics ---")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        print("\n--- Error Rates ---")
        print(f"False Acceptance Rate (FAR): {metrics['FAR']:.4f} ({metrics['FAR']*100:.2f}%)")
        print(f"False Rejection Rate (FRR): {metrics['FRR']:.4f} ({metrics['FRR']*100:.2f}%)")
        
        if 'EER' in metrics:
            print(f"Equal Error Rate (EER): {metrics['EER']:.4f} ({metrics['EER']*100:.2f}%)")
            print(f"EER Threshold: {metrics['EER_threshold']:.4f}")
        
        print("\n--- Detection Metrics ---")
        print(f"Detection Precision: {metrics['detection_precision']:.4f}")
        print(f"Detection Recall: {metrics['detection_recall']:.4f}")
        print(f"Detection F1 Score: {metrics['detection_f1']:.4f}")
        print(f"Detection Accuracy: {metrics['detection_accuracy']:.4f}")
        
        if 'avg_precision' in metrics:
            print("\n--- Average Per-Class Metrics ---")
            print(f"Average Precision: {metrics['avg_precision']:.4f}")
            print(f"Average Recall: {metrics['avg_recall']:.4f}")
            print(f"Average F1 Score: {metrics['avg_f1']:.4f}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Example usage
    num_classes = 10
    num_samples = 100
    
    # Simulate predictions and labels
    predictions = np.random.randint(0, num_classes, num_samples)
    labels = np.random.randint(0, num_classes, num_samples)
    distances = np.random.rand(num_samples)
    
    # Create metrics tracker
    metrics = FaceRecognitionMetrics(num_classes=num_classes, threshold=0.5)
    
    # Update metrics
    metrics.update(
        predictions=torch.tensor(predictions),
        labels=torch.tensor(labels),
        distances=torch.tensor(distances)
    )
    
    # Simulate detection results
    metrics.update_detection(tp=80, fp=10, fn=5, tn=5)
    
    # Print summary
    metrics.print_summary()
    
    # Plot confusion matrix
    metrics.plot_confusion_matrix()
    
    # Plot ROC curve
    metrics.plot_roc_curve()
