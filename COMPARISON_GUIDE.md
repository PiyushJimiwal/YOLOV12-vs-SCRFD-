# YOLOv12 vs SCRFD Comparison Guide

## Overview
This project compares TWO face recognition models:

1. **YOLOv12 Model** - Gradient-based feature extraction
2. **SCRFD Model** - Attention-based face recognition

## Quick Start

Run comparison in one command:

```bash
python quick_comparison.py
```

This will:
- ✅ Train YOLOv12 model (~15 seconds)
- ✅ Train SCRFD model (~15 seconds)
- ✅ Compare accuracy metrics
- ✅ Show per-person results
- ✅ Declare the winner!
- ✅ Optional webcam demo

## Output

After running, you'll get:

```
======================================================================
FINAL COMPARISON
======================================================================
YOLOv12 Accuracy:  75.00%
SCRFD Accuracy:    83.33%

🏆 Winner: SCRFD
```

## Output Files

```
comparison_results.json     # Detailed metrics JSON
```

## Understanding the Results

### Overall Accuracy
Percentage of correctly identified faces across all test images.

### Per-Person Accuracy
Individual accuracy for each person in your dataset.

Example output:
```
[YOLOv12] Results:
  Overall Accuracy: 75.00% (9/12)
  Per-person accuracy:
    person_1: 100.00% (6/6)
    person_2: 50.00% (3/6)

[SCRFD] Results:
  Overall Accuracy: 83.33% (10/12)
  Per-person accuracy:
    person_1: 100.00% (6/6)
    person_2: 66.67% (4/6)
```

## Model Differences

### YOLOv12 Approach
- **Feature Extraction**: Gradient-based (HOG-style)
- **Focus**: Edge detection and patterns
- **Speed**: Very fast
- **Best for**: General face detection

### SCRFD Approach  
- **Feature Extraction**: Attention mechanism
- **Focus**: Key facial regions (eyes, nose, mouth)
- **Speed**: Fast
- **Best for**: Precise face recognition

## Webcam Demo

After comparison, run webcam to see both models in action:

```bash
python quick_comparison.py
# Answer 'y' when prompted for webcam demo
```

You'll see:
- **Blue boxes**: YOLOv12 detections
- **Green boxes**: SCRFD detections
- **Labels**: Person name and confidence %

Press **'q'** to quit

## Tips for Best Comparison

1. **Balanced Dataset**: Same number of photos per person
2. **Minimum 10 photos** per person for training
3. **2-5 photos** per person for validation/testing
4. **Quality Photos**: Clear faces, good lighting
5. **Variety**: Different angles and expressions

## Interpreting Results

### If YOLOv12 Wins:
- Your photos have strong edge features
- Consistent lighting/angles
- Simple backgrounds

### If SCRFD Wins:
- Facial features are distinctive
- Good variety in training data
- Complex backgrounds or lighting

### If Results Are Close:
- Both models work well for your dataset
- Add more challenging test cases
- Consider ensemble approach

## Next Steps

After comparison:
1. Check `comparison_results.json` for detailed metrics
2. Add more photos if accuracy is low (<70%)
3. Try webcam demo to see real-time performance
4. Use the better model for your application
✅ Attention mechanisms for faces
✅ Lower FAR (better security)

### Combined Model Advantages:
✅ Best of both worlds
✅ Highest accuracy expected
✅ Robust feature extraction + face attention
✅ Balanced FAR/FRR

## Quick Start

### Step 1: Prepare Dataset
```
data/faces/
├── train/
│   ├── student_001/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── student_002/
│       └── ...
└── val/
    └── ...
```

### Step 2: Run Comparison
```bash
# Quick comparison (50 epochs each)
python compare_models.py --data_dir data/faces --epochs 50

# Full comparison (100 epochs each)
python compare_models.py --data_dir data/faces --epochs 100 --batch_size 16
```

### Step 3: View Results
```bash
# Check comparison chart
# Opens: outputs/comparison/comparison_chart.png

# Check detailed metrics
cat outputs/comparison/comparison_results.json
```

## Individual Model Testing

### Test YOLOv12
```bash
python inference.py --checkpoint outputs/comparison/yolov12_best.pth \
                    --mode test --data_dir data/faces
```

### Test SCRFD
```bash
python inference.py --checkpoint outputs/comparison/scrfd_best.pth \
                    --mode test --data_dir data/faces
```

### Test Combined
```bash
python inference.py --checkpoint outputs/comparison/combined_best.pth \
                    --mode test --data_dir data/faces
```

## Understanding the Difference

### YOLOv12 Model
- **Backbone**: CSP blocks only
- **Focus**: General robust feature extraction
- **Speed**: Fastest
- **Best for**: Speed-critical applications

### SCRFD Model
- **Backbone**: Attention-enhanced blocks
- **Focus**: Face-specific features
- **Accuracy**: High for faces
- **Best for**: Security-critical applications (low FAR)

### Combined Model
- **Backbone**: CSP + Attention integration
- **Focus**: Both general and face-specific
- **Balance**: Best overall performance
- **Best for**: Production systems requiring both speed and accuracy

## Customization

### Adjust Comparison Parameters

Edit `compare_models.py` or use command-line args:

```bash
python compare_models.py \
    --data_dir data/faces \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0005
```

### Compare on Different Metrics

Modify the comparison script to prioritize different metrics:
- Security-focused: Minimize FAR
- User-friendly: Minimize FRR
- Balanced: Minimize EER
- Overall: Maximize Accuracy

## Troubleshooting

### Issue: "Out of memory"
- Reduce batch_size: `--batch_size 4`
- Train models separately instead of all at once

### Issue: "Comparison takes too long"
- Reduce epochs: `--epochs 20`
- Use smaller dataset for quick test

### Issue: "All models have similar performance"
- Need more training data
- Increase epochs
- Ensure data quality

## Complete Command Reference

```bash
# Quick test (20 epochs)
python compare_models.py --data_dir data/faces --epochs 20 --batch_size 8

# Standard comparison (50 epochs)
python compare_models.py --data_dir data/faces --epochs 50 --batch_size 16

# Full comparison (100 epochs)
python compare_models.py --data_dir data/faces --epochs 100 --batch_size 16

# Low memory mode
python compare_models.py --data_dir data/faces --epochs 50 --batch_size 4
```

## File Structure

```
face_recognition_combined/
├── models/
│   ├── yolov12_model.py           # Standalone YOLOv12
│   ├── scrfd_model.py             # Standalone SCRFD
│   └── yolov12_scrfd_combined.py  # Combined model
├── compare_models.py              # Comparison script
└── outputs/
    └── comparison/
        ├── yolov12_best.pth
        ├── scrfd_best.pth
        ├── combined_best.pth
        ├── comparison_results.json
        └── comparison_chart.png
```

---

**Ready to compare? Run:**
```bash
python compare_models.py --data_dir data/faces --epochs 50
```

**This will train all 3 models and show you which one performs best! 🏆**
