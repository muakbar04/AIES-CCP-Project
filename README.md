# Helmet Detection System with YOLOv11

A professional-grade, end-to-end pipeline for helmet detection built on the state-of-the-art YOLOv11 architecture. This repository contains a Google Colab notebook that guides you through every phase of development—starting from environment validation, through dataset acquisition and model training, to advanced visualization, deployment on local video streams, and real-time inference via webcam.

---

## 1. Verify GPU Environment

**Objective:** Confirm the presence of a CUDA-enabled NVIDIA GPU and sufficient VRAM.
**Why:** Ensures high-throughput training and inference performance, prevents runtime bottlenecks.
**Outcome:** Hardware report with GPU model, driver version, and memory utilization.

---

## 2. Install Ultralytics YOLO Package

**Objective:** Install the official Ultralytics library for YOLOv8/v11.
**Why:** Provides a unified CLI and Python API for model operations—training, evaluation, and export.
**Outcome:** Access to `yolo` commands and Python classes for seamless workflow integration.

---

## 3. Sanity Check & Dependency Validation

**Objective:** Run `ultralytics.checks()` to validate Python, PyTorch, and CUDA configurations.
**Why:** Verifies compatibility of critical deep learning frameworks and prevents misconfiguration errors.
**Outcome:** Detailed report on versions, GPU availability, and missing/incorrect dependencies.

---

## 4. Import Core APIs & Visualization Tools

**Objective:** Load the `YOLO` class and Jupyter display utilities.
**Why:** Establishes the core interfaces for model manipulation and inline result rendering.
**Outcome:** Ready-to-use Python objects for training workflows and image/video display.

---

## 5. Acquire Dataset via Roboflow SDK

**Objective:** Programmatically download the helmet detection dataset in YOLOv11 format.
**Why:** Automates dataset retrieval, ensures correct directory structure, and standardizes splits.
**Outcome:** Local directory containing `data.yaml`, and `train/`, `valid/`, `test/` partitions with images and labels.

---

## 6. Train YOLOv11‑Nano Model

**Objective:** Execute a high-performance training session using the YOLOv11‑Nano variant.
**Why:** Balances model size and speed for rapid iteration, suitable for limited-resource environments.
**Key Parameters:**

* `data` path to dataset config
* Pretrained `yolo11n.pt` weights
* `epochs` count for convergence
* `imgsz` resolution
  **Outcome:** Trained model checkpoints saved to `runs/detect` directory.

---

## 7. Analyze Confusion Matrix

**Objective:** Visualize class-level performance via a confusion matrix.
**Why:** Identifies high-error classes for targeted data augmentation or rebalancing.
**Outcome:** PNG plot showing true positives, false positives, and false negatives per class.

---

## 8. Evaluate Label Distribution

**Objective:** Plot the frequency of bounding-box labels across all classes.
**Why:** Detects dataset imbalance, informs oversampling or augmentation strategies.
**Outcome:** Bar chart illustrating the number of annotated instances per class.

---

## 9. Track Loss & mAP Metrics

**Objective:** Display training/validation loss curves and mean Average Precision (mAP).
**Why:** Monitors convergence behavior, detects overfitting, and measures detection quality.
**Outcome:** Line plots for classification, localization, and DFL losses, alongside mAP\@0.5 and mAP\@0.5:0.95.

---

## 10. Inspect Augmented Training Samples

**Objective:** Preview the first batch of augmented images and bounding boxes.
**Why:** Validates augmentation pipelines (mosaic, flips, color transforms) and label alignment.
**Outcome:** JPEG image showing preprocessed and augmented training frames.

---

## 11. Perform Batch Inference on Test Set

**Objective:** Run detection across the entire test split and save annotated outputs.
**Why:** Evaluates model generalization on unseen data, generates qualitative results for review.
**Outcome:** Annotated images stored under `runs/detect/predict` with confidence overlays.

---

## 12. Single-Image Inference Example

**Objective:** Demonstrate model versatility by detecting helmets in a custom WebP image.
**Why:** Shows real-world applicability, supports multiple image formats.
**Outcome:** Inline display of detection results, annotated with bounding boxes and confidence scores.

---

## 13. Video Stream Processing

**Objective:** Apply the trained model to an input video file frame by frame.
**Why:** Enables automated post-processing for recorded footage, supports archival analysis.
**Outcome:** Output video saved with per-frame annotation in AVI format.

---

## 14. Google Drive Integration

**Objective:** Mount Google Drive to access and persist datasets, model weights, and outputs.
**Why:** Ensures data persistence across sessions, simplifies file management in Colab.
**Outcome:** Drive directory available under `/content/drive/MyDrive/`.

---

## 15. Real‑Time Webcam Deployment

**Objective:** Stream live video from a webcam to the YOLO model for on‑the‑fly detection.
**Why:** Demonstrates end-to-end deployment, supports interactive safety monitoring applications.
**Outcome:** Live display window with bounding boxes, running at near real-time speeds.

---
