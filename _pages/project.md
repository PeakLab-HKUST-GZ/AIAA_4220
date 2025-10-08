---
layout: page
permalink: /EAI-fall-2025/project/
title: Course Project
description: Guidelines and suggestions for course projects
---

# AIAA 4220 - Introduction to Embodied AI Project 2: Egocentric Object Detection Challenge

## Overview

**AIAA 4220 - Introduction to Embodied AI Project 2: Egocentric Object Detection Challenge**

Welcome to the second project of AIAA 4220 Introduction to Embodied AI course! This competition challenges you to develop robust object detection algorithms for first-person perspective images.

🏆 **Kaggle Competition**: https://www.kaggle.com/t/beb9f7424edf46799536b93620e26f2f

📖 **How to Join Kaggle Competition**: https://www.kaggle.com/discussions/general/400680

**Competition Highlights:**

- **Dataset**: 13,000 egocentric images from EgoObjects dataset
- **Task**: Detect 10 common object categories in first-person view
- **Framework Freedom**: Use any detection method (YOLO, Faster R-CNN, DETR, etc.)
- **Team Size**: 1-3 students per team
- **Baseline Performance**: mAP@0.5:0.95 of 0.759-0.788 provided (MMDetection)
- **Prize**: Top 3 teams present methods on Dec 3 (4%/2%/1% extra credit)

This is a **learning-focused competition** where methodology explanation and code quality matter as much as leaderboard ranking. Perfect for exploring modern object detection techniques while working with challenging real-world data.

---

## Description

### The Challenge

Egocentric (first-person) vision presents unique challenges compared to traditional third-person object detection. Objects appear at unusual angles, suffer from motion blur, and exhibit extreme scale variations. Your task is to build a detector that can reliably identify everyday objects in this challenging domain.

### Dataset Details

- **Source**: Curated from [EgoObjects](https://arxiv.org/abs/2309.08816) dataset
- **Total Images**: 13,000 first-person perspective images
- **Split**: 10,000 train / 1,000 validation / 2,000 test
- **Annotation Format**: COCO-style JSON with 2D bounding boxes

**Object Categories (10 classes):**

1. Box
2. Book
3. Bottle
4. Chair
5. Mug
6. Door
7. Shelf
8. Plate
9. Sofa

### Technical Requirements

**Complete Submission Package:**

1. **Source Code**: All training and inference scripts
2. **Technical Report**: Method explanation, architecture choices, hyperparameters
3. **Resource Documentation**: GPU usage, training time, model parameters
4. **Predictions**: Test set results in specified JSON format

**Prediction Format:**

```json
[
    {
        "image_id": 12345,
        "category_id": 3,
        "bbox": [x_min, y_min, width, height],
        "score": 0.85
    }
]
```

### Baseline Method Provided

We provide a complete MMDetection pipeline to get you started:

**MMDetection Framework**

- Professional-grade detection toolkit
- Includes Faster R-CNN and Deformable DETR implementations
- Expected mAP@0.5:0.95: 0.759-0.788
- Complete training/validation/testing pipeline with visualization tools

### Alternative Detection Methods (Explore on Your Own!)

For additional challenge and learning, consider exploring these state-of-the-art detection algorithms:

- **YOLOv10-N/S** (NMS-free, high throughput), **RT-DETRv2-Lite** (lightweight Transformer), **RTMDet-tiny/nano** (OpenMMLab family), **NanoDet-Plus** (mobile-friendly), **PP-PicoDet/PP-YOLOE** (mobile-optimized), **LW-DETR** (lightweight DETR alternative)

### Learning Objectives

- Understand modern object detection architectures
- Experience training on challenging real-world data
- Learn to balance accuracy vs. computational efficiency
- Practice professional ML workflow (train/val/test pipeline)

This competition emphasizes **engineering excellence** over pure performance optimization. Clean, well-documented code that demonstrates understanding of the underlying principles will be highly valued.

---

## Getting Started Guide

### Environment Setup

Before starting, install **Conda** and **PyTorch**. If you haven't already, download and install Miniconda:

👉 [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/install)

Then create and activate a clean environment:

```bash
# Create a conda environment
conda create -n ego2d python=3.9
conda activate ego2d
```

Install PyTorch (ensure it matches your CUDA version):

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### MMDetection Setup

Install MMDetection and dependencies:

```bash
pip install openmim
mim install mmengine==0.10.7 mmcv==2.1.0 mmdet==3.3.0
```

We provide two baseline implementations:

- **Faster R-CNN** (anchor-based)
- **Deformable DETR** (end-to-end)

### Training

Run training using the distributed training script (e.g., using 4 GPUs):

```bash
# Change to mm directory
cd mm

# Train Deformable DETR
bash tools/dist_train.sh config/deformable-detr_r50.py 4

# Or train Faster R-CNN
bash tools/dist_train.sh config/faster-rcnn_r50_fpn_giou_1x.py 4
```

### Testing and Submission

**Step 1: Generate Predictions**

Run inference on the test set to generate predictions:

```bash
# For Faster R-CNN
bash tools/dist_test.sh config/faster-rcnn_r50_fpn_giou_20e.py work_dirs/faster-rcnn_r50_fpn_giou_20e/epoch_20.pth 4

# For Deformable DETR  
bash tools/dist_test.sh config/deformable-detr_r50.py work_dirs/deformable-detr_r50/epoch_50.pth 4
```

This will produce a prediction file (e.g., `test.bbox.json`) in the corresponding `work_dirs` folder.

**Step 2: Convert to Kaggle Format**

Use the provided conversion script to convert MMDetection output to Kaggle submission format:

```bash
# Return to project root
cd ..

# Convert predictions to submission.csv
python convert_to_submission.py --pred mm/work_dirs/deformable-detr_r50/test.bbox.json --output submission.csv
```

Expected output:

```
Loaded 200000 predictions from mm/work_dirs/deformable-detr_r50/test.bbox.json
Found 2000 unique images
✓ Saved submission to submission.csv
```

**Step 3: Upload to Kaggle**

1. Go to the Kaggle competition page
2. Click "Submit Predictions"
3. Upload `submission.csv`
4. **Important**: Add your **student ID** in the submission description
5. Click "Make Submission"

### Format Verification

Before generating test results for final submission, **verify the output format** using the validation set:

**Steps:**

1. Modify `test_dataloader` and `test_evaluator` in `mm/config/data.py` to use the validation set
  
2. Run inference on the validation set to generate predictions:
  
  ```bash
  bash tools/dist_test.sh config/faster-rcnn_r50_fpn_giou_20e.py work_dirs/faster-rcnn_r50_fpn_giou_20e/epoch_20.pth 4
  ```
  
3. Evaluate the predicted results using the provided evaluation script:
  
  ```bash
  python eval.py --gt data/val.json --pred mm/work_dirs/faster-rcnn_r50_fpn_giou_20e/test.bbox.json
  ```
  

### Visualization

To inspect the dataset:

```bash
cd tools
python browse_dataset.py ../mm/config/deformable-detr_r50.py --output-dir vis
```

To visualize prediction results:

```bash
bash tools/dist_test.sh config/faster-rcnn_r50_fpn_giou_20e.py work_dirs/faster-rcnn_r50_fpn_giou_20e/epoch_20.pth 4 --show --show-dir vis
```

---

## Evaluation

### Primary Metric

**Mean Average Precision (mAP) at IoU 0.50:0.95**

This is the standard COCO evaluation metric that averages precision across IoU thresholds from 0.5 to 0.95 (step 0.05). It provides a comprehensive measure of both localization accuracy and detection confidence.

### Detailed Metrics Reported

- **mAP@0.5:0.95**: Primary ranking metric (IoU thresholds 0.5-0.95)
- **mAP@0.5**: Detection accuracy at relaxed IoU threshold
- **mAP@0.75**: Detection accuracy at strict IoU threshold
- **mAP by object size**: Small/Medium/Large object performance
- **Average Recall (AR)**: Maximum recall at 1, 10, 100 detections

### Submission Requirements

**File Format**: CSV file named `submission.csv` with the following structure:

| Column | Description |
| --- | --- |
| `id` | Image ID (2000 rows, one per test image) |
| `predictions` | JSON array of detections for each image |

Each detection in the `predictions` column must include:

```json
{
  "image_id": <int>,
  "category_id": <int>,     // 0-9 for the 10 object classes
  "bbox": [x, y, w, h],     // COCO format: [x_min, y_min, width, height]
  "score": <float>          // Confidence score between 0 and 1
}
```

**Important**: Use the provided `convert_to_submission.py` script to ensure correct format. Do not manually edit the CSV file.

**Kaggle Submission**: When submitting to Kaggle, **include your student ID** in the submission description to ensure proper grading.

### Academic Integrity

- **Collaboration**: Teams of 1-3 students allowed
- **External Data**: Only provided dataset permitted
- **Pre-trained Models**: Allowed and encouraged (ImageNet, COCO weights)
- **Framework Freedom**: Any detection framework/architecture permitted

### Timeline

- **Competition Start**: October 11, 2024
- **Submission Deadline**: December 3, 2024, 11:59 PM
- **Results Presentation**: Top 3 teams present on December 3

These serve as reference implementations to help you get started. Your final ranking will be determined by your performance relative to other teams on the Kaggle leaderboard.

---

## Submission Checklist

### Required Deliverables

✅ **Source Code**: Complete training and inference scripts with clear documentation

✅ **Technical Report**: Detailed explanation of your method including:

- Detection framework/model used
- Any modifications or improvements made
- Training configurations and hyperparameters
- Resources used (GPU type, number of GPUs, training time)
- Model size (number of parameters)

✅ **Prediction Results**: Test set predictions converted to `submission.csv` using the provided conversion script

✅ **Kaggle Submission**: Submit `submission.csv` to Kaggle with **student ID** included in description

### Final Submission Format

Your submission should include:

- **Kaggle submission**: Upload `submission.csv` with student ID in description
- All source code files
- Technical documentation (PDF or Markdown)
- Instructions to reproduce your results

### Quick Reference: Complete Workflow

```bash
# 1. Train your model
cd mm
bash tools/dist_train.sh config/your_config.py 4

# 2. Generate predictions on test set
bash tools/dist_test.sh config/your_config.py work_dirs/your_model/epoch_X.pth 4

# 3. Convert to Kaggle format
cd ..
python convert_to_submission.py --pred mm/work_dirs/your_model/test.bbox.json --output submission.csv

# 4. Upload submission.csv to Kaggle (include student ID in description)
```

Good luck, and may your detection boxes be precise! 🎯

***

* (The list will be replaced with the table of contents.)
{:toc}

***