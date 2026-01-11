"""
SCRFD Model for Face Detection and Recognition
Standalone SCRFD architecture without YOLOv12 components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ConvBNReLU(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SCRFDAttentionBlock(nn.Module):
    """SCRFD Attention Block with Channel and Spatial Attention"""
    def __init__(self, in_channels):
        super(SCRFDAttentionBlock, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        
        return x


class SCRFDBlock(nn.Module):
    """SCRFD Feature Extraction Block"""
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super(SCRFDBlock, self).__init__()
        
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, 1, 1)
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ConvBNReLU(out_channels, out_channels, 3, 1, 1))
            blocks.append(SCRFDAttentionBlock(out_channels))
        
        self.blocks = nn.Sequential(*blocks)
        
        self.conv2 = ConvBNReLU(out_channels, out_channels, 1, 1, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        return x


class SCRFDBackbone(nn.Module):
    """SCRFD Backbone Network"""
    def __init__(self, in_channels=3):
        super(SCRFDBackbone, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, 2, 1),
            ConvBNReLU(32, 64, 3, 2, 1)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1),
            SCRFDBlock(128, 128, num_blocks=2)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNReLU(128, 256, 3, 2, 1),
            SCRFDBlock(256, 256, num_blocks=3)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNReLU(256, 512, 3, 2, 1),
            SCRFDBlock(512, 512, num_blocks=3)
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBNReLU(512, 1024, 3, 2, 1),
            SCRFDBlock(1024, 1024, num_blocks=2)
        )
    
    def forward(self, x):
        x = self.stem(x)
        
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        
        return c3, c4, c5


class SCRFDFPN(nn.Module):
    """SCRFD Feature Pyramid Network"""
    def __init__(self):
        super(SCRFDFPN, self).__init__()
        
        self.lateral_c5 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.lateral_c4 = nn.Conv2d(512, 256, 1, 1, 0)
        self.lateral_c3 = nn.Conv2d(256, 256, 1, 1, 0)
        
        self.smooth_p5 = SCRFDBlock(256, 256, num_blocks=1)
        self.smooth_p4 = SCRFDBlock(256, 256, num_blocks=1)
        self.smooth_p3 = SCRFDBlock(256, 256, num_blocks=1)
    
    def forward(self, c3, c4, c5):
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        
        return p3, p4, p5


class SCRFDDetectionHead(nn.Module):
    """SCRFD Detection Head with Landmark Support"""
    def __init__(self, in_channels=256, num_classes=2):
        super(SCRFDDetectionHead, self).__init__()
        
        # Classification
        self.cls_conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels),
            SCRFDAttentionBlock(in_channels),
            nn.Conv2d(in_channels, num_classes, 1, 1, 0)
        )
        
        # Bounding box
        self.bbox_conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels),
            SCRFDAttentionBlock(in_channels),
            nn.Conv2d(in_channels, 4, 1, 1, 0)
        )
        
        # Landmarks (5 facial landmarks * 2 coordinates)
        self.landmark_conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels),
            SCRFDAttentionBlock(in_channels),
            nn.Conv2d(in_channels, 10, 1, 1, 0)
        )
        
        # Embedding for recognition
        self.embed_conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels),
            SCRFDAttentionBlock(in_channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512)
        )
    
    def forward(self, x):
        cls_score = self.cls_conv(x)
        bbox_pred = self.bbox_conv(x)
        landmark_pred = self.landmark_conv(x)
        embeddings = self.embed_conv(x)
        
        return cls_score, bbox_pred, landmark_pred, embeddings


class SCRFD(nn.Module):
    """SCRFD Face Recognition Model"""
    
    def __init__(self, num_classes=2, embedding_dim=512):
        super(SCRFD, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Backbone
        self.backbone = SCRFDBackbone()
        
        # Neck (FPN)
        self.neck = SCRFDFPN()
        
        # Detection heads
        self.head_p3 = SCRFDDetectionHead(256, num_classes)
        self.head_p4 = SCRFDDetectionHead(256, num_classes)
        self.head_p5 = SCRFDDetectionHead(256, num_classes)
        
        # Embedding projection
        self.embedding_projector = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, x):
        # Backbone
        c3, c4, c5 = self.backbone(x)
        
        # FPN
        p3, p4, p5 = self.neck(c3, c4, c5)
        
        # Multi-scale detection
        cls_p3, bbox_p3, lm_p3, embed_p3 = self.head_p3(p3)
        cls_p4, bbox_p4, lm_p4, embed_p4 = self.head_p4(p4)
        cls_p5, bbox_p5, lm_p5, embed_p5 = self.head_p5(p5)
        
        # Combine embeddings
        combined_embedding = torch.cat([embed_p3, embed_p4, embed_p5], dim=1)
        final_embedding = self.embedding_projector(combined_embedding)
        final_embedding = F.normalize(final_embedding, p=2, dim=1)
        
        outputs = {
            'classifications': [cls_p3, cls_p4, cls_p5],
            'bboxes': [bbox_p3, bbox_p4, bbox_p5],
            'landmarks': [lm_p3, lm_p4, lm_p5],
            'embeddings': final_embedding
        }
        
        return outputs
    
    def extract_features(self, x):
        """Extract face embeddings"""
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['embeddings']


def create_scrfd_model(num_classes=2, embedding_dim=512):
    """Factory function to create SCRFD model"""
    model = SCRFD(num_classes=num_classes, embedding_dim=embedding_dim)
    return model


if __name__ == "__main__":
    # Test model
    model = create_scrfd_model(num_classes=10, embedding_dim=512)
    dummy_input = torch.randn(2, 3, 640, 640)
    
    outputs = model(dummy_input)
    print("SCRFD Model Output Shapes:")
    print(f"Classifications P3: {outputs['classifications'][0].shape}")
    print(f"Bounding boxes P3: {outputs['bboxes'][0].shape}")
    print(f"Landmarks P3: {outputs['landmarks'][0].shape}")
    print(f"Embeddings: {outputs['embeddings'].shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
