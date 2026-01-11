"""
Model Comparison Script
Compare YOLOv12 vs SCRFD vs Combined Model
Train all three models and compare their performance metrics
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from models.yolov12_model import create_yolov12_model
from models.scrfd_model import create_scrfd_model
from models.yolov12_scrfd_combined import create_model as create_combined_model
from dataset.face_dataset import FaceDataModule
from utils.metrics import FaceRecognitionMetrics


class SimpleLoss(nn.Module):
    """Simplified loss for comparison"""
    def __init__(self, num_classes):
        super(SimpleLoss, self).__init__()
        self.num_classes = num_classes
        self.cls_criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        embeddings = outputs['embeddings']
        labels = targets['label']
        
        # Simple classification using embeddings
        if not hasattr(self, 'fc_classifier'):
            self.fc_classifier = nn.Linear(embeddings.size(1), self.num_classes).to(embeddings.device)
        
        logits = self.fc_classifier(embeddings)
        loss = self.cls_criterion(logits, labels)
        
        return {'total_loss': loss, 'logits': logits}


class ModelComparator:
    """Compare multiple face recognition models"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(config['output_dir']) / 'comparison'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup data
        self.setup_data()
        
        # Results storage
        self.results = {
            'yolov12': {},
            'scrfd': {},
            'combined': {}
        }
        
        print(f"Device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
    
    def setup_data(self):
        """Setup datasets"""
        self.data_module = FaceDataModule(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            img_size=self.config['img_size']
        )
        
        self.data_module.setup()
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        self.num_classes = self.data_module.get_num_classes()
    
    def train_model(self, model, model_name, epochs):
        """Train a single model"""
        print(f"\n{'='*70}")
        print(f"Training {model_name}")
        print(f"{'='*70}")
        
        model = model.to(self.device)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # Loss
        criterion = SimpleLoss(self.num_classes).to(self.device)
        
        best_accuracy = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f'[{model_name}] Epoch {epoch}/{epochs}')
            num_batches = 0
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                targets = {'label': labels}
                losses = criterion(outputs, targets)
                loss = losses['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate accuracy
                logits = losses['logits']
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            # Handle case when no batches were processed
            if num_batches == 0:
                print(f"Warning: No batches processed in epoch {epoch}. Skipping...")
                continue
            
            avg_loss = total_loss / num_batches
            train_accuracy = 100. * correct / total if total > 0 else 0.0
            train_losses.append(avg_loss)
            
            # Validate
            val_metrics = self.validate_model(model, criterion, epoch)
            val_accuracies.append(val_metrics['accuracy'])
            
            print(f"[{model_name}] Epoch {epoch}: Loss={avg_loss:.4f}, "
                  f"Train Acc={train_accuracy:.2f}%, Val Acc={val_metrics['accuracy']*100:.2f}%")
            
            # Update scheduler
            scheduler.step()
            
            # Save best
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics
                }
                torch.save(checkpoint, self.output_dir / f'{model_name}_best.pth')
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_accuracy': best_accuracy
        }
    
    def validate_model(self, model, criterion, epoch):
        """Validate model"""
        model.eval()
        
        metrics_tracker = FaceRecognitionMetrics(
            num_classes=self.num_classes,
            threshold=self.config.get('threshold', 0.5)
        )
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(images)
                targets = {'label': labels}
                losses = criterion(outputs, targets)
                
                logits = losses['logits']
                _, predicted = logits.max(1)
                
                # Compute distances
                embeddings = outputs['embeddings']
                distances = torch.cdist(embeddings, embeddings, p=2).diagonal()
                
                metrics_tracker.update(
                    predictions=predicted,
                    labels=labels,
                    embeddings=embeddings,
                    distances=distances
                )
        
        return metrics_tracker.compute_all_metrics()
    
    def compare_models(self):
        """Train and compare all models"""
        epochs = self.config['epochs']
        
        # 1. Train YOLOv12
        print("\n" + "="*70)
        print("MODEL 1: YOLOv12")
        print("="*70)
        yolov12 = create_yolov12_model(
            num_classes=self.num_classes,
            embedding_dim=self.config['embedding_dim']
        )
        self.results['yolov12'] = self.train_model(yolov12, 'yolov12', epochs)
        
        # 2. Train SCRFD
        print("\n" + "="*70)
        print("MODEL 2: SCRFD")
        print("="*70)
        scrfd = create_scrfd_model(
            num_classes=self.num_classes,
            embedding_dim=self.config['embedding_dim']
        )
        self.results['scrfd'] = self.train_model(scrfd, 'scrfd', epochs)
        
        # 3. Train Combined
        print("\n" + "="*70)
        print("MODEL 3: Combined (YOLOv12 + SCRFD)")
        print("="*70)
        combined = create_combined_model(
            num_classes=self.num_classes,
            embedding_dim=self.config['embedding_dim']
        )
        self.results['combined'] = self.train_model(combined, 'combined', epochs)
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        
        # Load best models and evaluate
        final_results = {}
        
        for model_name in ['yolov12', 'scrfd', 'combined']:
            checkpoint_path = self.output_dir / f'{model_name}_best.pth'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                final_results[model_name] = checkpoint['metrics']
        
        # Print comparison table
        print("\n" + "="*70)
        print("FINAL METRICS COMPARISON")
        print("="*70)
        
        metrics_to_compare = ['accuracy', 'FAR', 'FRR']
        if 'EER' in final_results['yolov12']:
            metrics_to_compare.append('EER')
        
        comparison_data = []
        for metric in metrics_to_compare:
            row = {'Metric': metric}
            for model_name in ['yolov12', 'scrfd', 'combined']:
                if model_name in final_results:
                    value = final_results[model_name].get(metric, 0.0)
                    row[model_name.upper()] = f"{value:.4f}"
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
        
        # Determine winner
        print("\n" + "="*70)
        print("WINNER DETERMINATION")
        print("="*70)
        
        best_accuracy = max([final_results[m]['accuracy'] for m in final_results])
        best_far = min([final_results[m]['FAR'] for m in final_results])
        best_frr = min([final_results[m]['FRR'] for m in final_results])
        
        for model_name in ['yolov12', 'scrfd', 'combined']:
            metrics = final_results[model_name]
            print(f"\n{model_name.upper()}:")
            if metrics['accuracy'] == best_accuracy:
                print("  ✓ BEST ACCURACY")
            if metrics['FAR'] == best_far:
                print("  ✓ BEST FAR (Lowest)")
            if metrics['FRR'] == best_frr:
                print("  ✓ BEST FRR (Lowest)")
        
        # Overall winner (highest accuracy)
        winner = max(final_results.keys(), key=lambda k: final_results[k]['accuracy'])
        print(f"\n🏆 OVERALL WINNER: {winner.upper()}")
        print(f"   Accuracy: {final_results[winner]['accuracy']:.4f}")
        print(f"   FAR: {final_results[winner]['FAR']:.4f}")
        print(f"   FRR: {final_results[winner]['FRR']:.4f}")
        
        # Plot comparison
        self.plot_comparison(final_results)
        
        # Save results
        with open(self.output_dir / 'comparison_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {self.output_dir / 'comparison_results.json'}")
    
    def plot_comparison(self, final_results):
        """Plot comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(final_results.keys())
        model_labels = [m.upper() for m in models]
        
        # 1. Accuracy comparison
        accuracies = [final_results[m]['accuracy'] * 100 for m in models]
        axes[0, 0].bar(model_labels, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Recognition Accuracy Comparison')
        axes[0, 0].set_ylim([0, 100])
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # 2. FAR comparison
        fars = [final_results[m]['FAR'] * 100 for m in models]
        axes[0, 1].bar(model_labels, fars, color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[0, 1].set_ylabel('FAR (%)')
        axes[0, 1].set_title('False Acceptance Rate (Lower is Better)')
        for i, v in enumerate(fars):
            axes[0, 1].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # 3. FRR comparison
        frrs = [final_results[m]['FRR'] * 100 for m in models]
        axes[1, 0].bar(model_labels, frrs, color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[1, 0].set_ylabel('FRR (%)')
        axes[1, 0].set_title('False Rejection Rate (Lower is Better)')
        for i, v in enumerate(frrs):
            axes[1, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # 4. Overall comparison radar/table
        metrics_names = ['Accuracy', 'FAR', 'FRR']
        if 'EER' in final_results[models[0]]:
            metrics_names.append('EER')
        
        # Create comparison table
        table_data = []
        for model in models:
            row = [
                f"{final_results[model]['accuracy']*100:.2f}%",
                f"{final_results[model]['FAR']*100:.2f}%",
                f"{final_results[model]['FRR']*100:.2f}%"
            ]
            if 'EER' in final_results[model]:
                row.append(f"{final_results[model]['EER']*100:.2f}%")
            table_data.append(row)
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(
            cellText=table_data,
            rowLabels=model_labels,
            colLabels=metrics_names,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Detailed Metrics Comparison')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_chart.png', dpi=300, bbox_inches='tight')
        print(f"✓ Comparison chart saved to: {self.output_dir / 'comparison_chart.png'}")
        
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Compare YOLOv12 vs SCRFD vs Combined')
    parser.add_argument('--data_dir', type=str, default='data/faces', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_workers': 4,
        'img_size': 640,
        'embedding_dim': 512,
        'weight_decay': 0.0001,
        'threshold': 0.5
    }
    
    print("\n" + "="*70)
    print("MODEL COMPARISON: YOLOv12 vs SCRFD vs Combined")
    print("="*70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*70)
    
    comparator = ModelComparator(config)
    comparator.compare_models()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
