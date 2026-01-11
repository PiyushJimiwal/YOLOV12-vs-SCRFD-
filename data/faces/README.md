# Dataset Directory

This directory should contain your face recognition dataset organized by student.

## Directory Structure

```
faces/
├── train/
│   ├── student_001/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── student_002/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── student_N/
│       └── ...
├── val/
│   ├── student_001/
│   │   └── img_001.jpg
│   └── ...
├── test/  (optional)
│   ├── student_001/
│   │   └── img_001.jpg
│   └── ...
└── annotations/  (optional)
    ├── train_annotations.json
    ├── val_annotations.json
    └── test_annotations.json
```

## Dataset Guidelines

### Image Requirements
- **Format**: JPG, JPEG, PNG, or BMP
- **Resolution**: At least 640x640 pixels recommended
- **Quality**: Clear, well-lit face images
- **Face Size**: Face should occupy at least 30% of the image

### Recommended Images per Student
- **Minimum**: 5-10 images per student
- **Recommended**: 20-50 images per student
- **Optimal**: 50+ images with diverse conditions

### Image Diversity
Capture images with variations in:
- **Angles**: Front, slight left/right turns
- **Lighting**: Natural light, indoor lighting, different brightness
- **Expressions**: Neutral, smiling, various expressions
- **Accessories**: With/without glasses, hats (if applicable)
- **Background**: Different backgrounds

### Student ID Naming
- Use consistent naming: `student_001`, `student_002`, etc.
- Or use actual IDs: `john_doe`, `jane_smith`, etc.
- Avoid special characters and spaces

## Annotation Format (Optional)

If you have bounding box and landmark annotations:

### annotations/train_annotations.json
```json
{
  "img_001.jpg": {
    "bbox": [x1, y1, x2, y2],
    "landmarks": [
      x1, y1,  // Left eye
      x2, y2,  // Right eye
      x3, y3,  // Nose
      x4, y4,  // Left mouth corner
      x5, y5   // Right mouth corner
    ]
  },
  "img_002.jpg": {
    "bbox": [x1, y1, x2, y2],
    "landmarks": [...]
  }
}
```

Where:
- `bbox`: [x_min, y_min, x_max, y_max] in pixels
- `landmarks`: 10 values (5 points × 2 coordinates)

## Data Split Recommendations

### Standard Split
- **Training**: 70-80% of images
- **Validation**: 10-15% of images
- **Testing**: 10-15% of images

### For Small Datasets
- **Training**: 80% of images
- **Validation**: 20% of images
- **Testing**: Use validation set

## Quick Dataset Creation

### 1. Collect Images
```bash
# Create directories
mkdir -p train/student_001 val/student_001

# Add images (5-10 per student minimum)
```

### 2. Organize by Student
```
train/
  student_001/
    img_001.jpg  # Front view
    img_002.jpg  # Slight left
    img_003.jpg  # Slight right
    img_004.jpg  # Different lighting
    img_005.jpg  # With smile
    ...
```

### 3. Split Data
- Move 20% of images from train/ to val/ for each student
- Ensure each student appears in both splits

## Example Dataset Sizes

### Small Dataset
- 10-20 students
- 10-20 images per student
- Total: 100-400 images

### Medium Dataset
- 50-100 students
- 20-50 images per student
- Total: 1,000-5,000 images

### Large Dataset
- 100+ students
- 50+ images per student
- Total: 5,000+ images

## Data Augmentation

The training script automatically applies augmentation:
- Horizontal flips
- Brightness/contrast adjustments
- Rotation and scaling
- Color jittering
- Gaussian noise

This means fewer images per student are needed during training.

## Troubleshooting

### Issue: "No students found"
- Check directory structure matches the format above
- Ensure student folders contain image files
- Verify image file extensions (.jpg, .jpeg, .png, .bmp)

### Issue: "Insufficient images"
- Each student needs at least 5 images for training
- Add more diverse images per student

### Issue: "Poor recognition accuracy"
- Collect more images per student (aim for 20+)
- Ensure image quality is high
- Add more diversity in poses and lighting
- Balance dataset (similar images per student)

## Data Privacy

⚠️ **Important**: 
- Obtain proper consent before collecting face images
- Follow data protection regulations (GDPR, etc.)
- Securely store and handle personal data
- Implement appropriate access controls

## Contact

For dataset-related questions, refer to the main README.md file.
