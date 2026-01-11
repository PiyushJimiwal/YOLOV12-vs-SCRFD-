# YOLOv12 vs SCRFD Face Recognition Comparison

A simple and fast face recognition system that compares YOLOv12 and SCRFD models for face detection and recognition. No complex training required - just add your photos and run!

## 🌟 Features

- **Dual Model Comparison**: Compare YOLOv12 vs SCRFD side-by-side
- **Fast Training**: Trains in seconds, not hours
- **Real-time Webcam**: See both models detecting faces simultaneously
- **Accuracy Metrics**: Get per-person and overall accuracy for each model
- **Simple Setup**: Just add photos and run one command
- **No GPU Required**: Works on CPU

## 📁 Project Structure

```
face_recognition_combined/
├── quick_comparison.py            # Main comparison script
├── models/
│   ├── yolov12_model.py          # YOLOv12 model
│   └── scrfd_model.py            # SCRFD model
├── dataset/
│   └── face_dataset.py           # Dataset loader
├── utils/
│   └── metrics.py                # Metrics calculation
├── data/faces/
│   ├── train/                    # Training images
│   └── val/                      # Validation images
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate      # Linux/Mac

# Install required packages
pip install -r requirements.txt
```

### 2. Add Your Photos

Organize photos in this structure:

```
data/faces/
├── train/
│   ├── person_1/              # Your photos
│   │   ├── photo1.jpg
│   │   ├── photo2.jpg
│   │   └── ... (10+ images recommended)
│   ├── person_2/              # Friend's photos
│   │   ├── photo1.jpg
│   │   └── ... (10+ images)
│   └── ...
└── val/
    ├── person_1/              # 2-5 test images
    │   └── test1.jpg
    ├── person_2/
    │   └── test1.jpg
    └── ...
```

**Requirements:**
- At least **2 different people** (minimum 10 photos each)
- Clear face photos (JPG, PNG)
- Different angles and lighting for better results

### 3. Run Comparison

**Test on Validation Set:**
```bash
python inference.py --checkpoint outputs/checkpoint_best.pth --mode test --data_dir data/faces
```

**Single Image Inference:**
```bash
python inference.py --checkpoint outputs/checkpoint_best.pth --mode image --image test.jpg --output result.jpg
```

**Run the comparison:**
```bash
python quick_comparison.py
```

This will:
1. ✅ Train YOLOv12 model (takes ~10-15 seconds)
2. ✅ Train SCRFD model (takes ~10-15 seconds)
3. ✅ Test both models on validation data
4. ✅ Show accuracy metrics for each model
5. ✅ Ask if you want to run webcam demo

### 4. View Results

After running, you'll see:

```
======================================================================
FINAL COMPARISON
======================================================================
YOLOv12 Accuracy:  75.00%
SCRFD Accuracy:    83.33%

🏆 Winner: SCRFD

✅ Results saved to comparison_results.json
```

### 5. Webcam Demo

When prompted, type `y` to see real-time face recognition:
- **Blue boxes**: YOLOv12 predictions
- **Green boxes**: SCRFD predictions
- Press **'q'** to quit

## 📊 Output Files

- `comparison_results.json`: Detailed metrics for both models
  - Overall accuracy
  - Per-person accuracy
  - Number of correct/total predictions

## 🎯 How It Works

### YOLOv12 Model
- Uses gradient-based feature extraction
- Focuses on edge detection and patterns
- Fast and efficient

### SCRFD Model
- Uses attention mechanism for face regions
- Focuses on key facial features (eyes, nose, mouth)
- More specialized for faces

## 💡 Tips for Better Results

1. **More Training Photos**: Add 15-20 photos per person
2. **Variety**: Include different angles, lighting, expressions
3. **Quality**: Use clear, well-lit photos
4. **Balance**: Have similar number of photos for each person

## ❓ Troubleshooting

### Low Accuracy
- Add more training photos (15+ per person)
- Ensure photos are clear and faces are visible
- Add more variety in poses and lighting

### Models Both Wrong
- Need more than 2-3 people for better comparison
- Each person needs at least 10 training photos

### Webcam Not Working
- Check if camera is connected
- Grant camera permissions
- Try closing other apps using the camera

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to open issues or submit pull requests!
    student_id='student_new',
    image_paths=['path/to/img1.jpg', 'path/to/img2.jpg']
)

# Save database
inference.save_database('student_database.json')
```

## 📊 Output Files

After training, you'll find:

```
outputs/
├── checkpoint_best.pth           # Best model checkpoint
├── checkpoint_latest.pth         # Latest checkpoint
├── checkpoint_epoch_X.pth        # Periodic checkpoints
├── logs/                         # Tensorboard logs
└── inference/
    ├── confusion_matrix_val.png  # Confusion matrix
    ├── roc_curve_val.png        # ROC curve (FAR vs FRR)
    └── student_database.json     # Student embeddings database
```

## 🎯 Use Cases

1. **Student Attendance System**: Automated classroom attendance
2. **Access Control**: Secure building/room access
3. **Identity Verification**: Person identification in various scenarios
4. **Surveillance**: Real-time face recognition in video streams

## 🛠️ Customization

### Modify Model Architecture
Edit `models/yolov12_scrfd_combined.py` to adjust:
- Number of CSP blocks
- Feature channels
- Embedding dimensions
- Detection scales

### Adjust Data Augmentation
Edit `dataset/face_dataset.py` to customize:
- Image transformations
- Augmentation probabilities
- Input image size

### Custom Metrics
Extend `utils/metrics.py` to add:
- Additional evaluation metrics
- Custom visualization
- Performance analysis tools

## 📝 Performance Tips

1. **Data Quality**: Use high-quality, diverse images for each student
2. **Balanced Dataset**: Aim for similar number of images per student
3. **Augmentation**: Enable augmentation for smaller datasets
4. **Threshold Tuning**: Adjust recognition threshold based on your FAR/FRR requirements
5. **GPU Usage**: Use CUDA for faster training and inference

## 🐛 Troubleshooting

### Low Accuracy
- Increase training epochs
- Add more training data per student
- Enable data augmentation
- Adjust learning rate

### High FAR
- Increase recognition threshold
- Add negative samples (non-student faces)
- Train longer

### High FRR
- Decrease recognition threshold
- Add more diverse images per student
- Check image quality

## 📚 References

- YOLOv12: You Only Look Once
- SCRFD: Sample and Computation Redistribution for Efficient Face Detection
- Face Recognition with Deep Learning

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Contact

For questions and support, please create an issue in the repository.

---

**Happy Face Recognition! 🎭**
