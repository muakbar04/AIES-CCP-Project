# Helmet Detection System with YOLOv11

A professional-grade, end-to-end pipeline for helmet detection powered by the cutting-edge YOLOv11 architecture. This repository provides a fully automated and modular Google Colab notebook that walks through every critical phase of model development—ranging from environment setup and dataset acquisition to model training, evaluation, and deployment. Designed for both academic exploration and industrial applications, this system offers robust performance in real-time video stream analysis and on-device inference.

---

## 📚 Table of Contents

1. [Verify GPU Environment](#1-verify-gpu-environment)
2. [Install Ultralytics YOLO Package](#2-install-ultralytics-yolo-package)
3. [Sanity Check & Dependency Validation](#3-sanity-check--dependency-validation)
4. [Import Core APIs & Visualization Tools](#4-import-core-apis--visualization-tools)
5. [Acquire Dataset via Roboflow SDK](#5-acquire-dataset-via-roboflow-sdk)
6. [Train YOLOv11‑Nano Model](#6-train-yolov11nano-model)
7. [Analyze Confusion Matrix](#7-analyze-confusion-matrix)
8. [Evaluate Label Distribution](#8-evaluate-label-distribution)
9. [Track Loss & mAP Metrics](#9-track-loss--map-metrics)
10. [Inspect Augmented Training Samples](#10-inspect-augmented-training-samples)
11. [Perform Batch Inference on Test Set](#11-perform-batch-inference-on-test-set)
12. [Single-Image Inference Example](#12-single-image-inference-example)
13. [Video Stream Processing](#13-video-stream-processing)
14. [Google Drive Integration](#14-google-drive-integration)
15. [Real‑Time Webcam Deployment](#15-real-time-webcam-deployment)

---

## 1. Verify GPU Environment

**Objective:** Confirm the presence of a CUDA-enabled NVIDIA GPU and check available VRAM.
**Rationale:** Ensures optimal performance during training and inference; prevents potential runtime limitations.
**Outcome:** Hardware report detailing GPU name, driver version, CUDA compatibility, and memory usage.

---

## 2. Install Ultralytics YOLO Package

**Objective:** Install the official `ultralytics` package which includes support for YOLOv8/v11 models.
**Rationale:** Offers seamless access to command-line tools and Python APIs for model training, validation, and deployment.
**Outcome:** Enables use of the `yolo` CLI and Python class-based workflows.

---

## 3. Sanity Check & Dependency Validation

**Objective:** Use `ultralytics.checks()` to validate the setup.
**Rationale:** Verifies that Python, PyTorch, and CUDA environments are properly configured and compatible.
**Outcome:** Diagnostic report covering installed packages, hardware compatibility, and missing dependencies.

---

## 4. Import Core APIs & Visualization Tools

**Objective:** Import YOLO interface, plotting libraries, and display utilities.
**Rationale:** Prepares the notebook environment for training workflows, metric visualization, and inference outputs.
**Outcome:** All required modules loaded for immediate use in subsequent steps.

---

## 5. Acquire Dataset via Roboflow SDK

**Objective:** Fetch the helmet detection dataset in YOLO-compatible format using Roboflow.
**Rationale:** Streamlines data collection, ensures consistent structure, and simplifies reproducibility.
**Outcome:** A structured local dataset with `train`, `valid`, and `test` folders and a `data.yaml` file.

---

## 6. Train YOLOv11‑Nano Model

**Objective:** Launch training using the YOLOv11-Nano architecture.
**Rationale:** Combines lightweight model design with high detection accuracy for fast prototyping and deployment.
**Key Parameters:**

* Dataset configuration path (`data`)
* Initial weights (`yolo11n.pt`)
* Number of training epochs
* Input image size (`imgsz`)
  **Outcome:** Saved checkpoints and training logs under the `runs/detect` directory.

---

## 7. Analyze Confusion Matrix

**Objective:** Generate a confusion matrix for visualizing model performance by class.
**Rationale:** Highlights misclassified instances and supports error analysis.
**Outcome:** A matrix plot showing true positives, false positives, and false negatives.

---

## 8. Evaluate Label Distribution

**Objective:** Visualize class distribution across the dataset.
**Rationale:** Reveals data imbalance that may require augmentation or resampling.
**Outcome:** Bar graph showing the count of annotations per label.

---

## 9. Track Loss & mAP Metrics

**Objective:** Plot training loss curves and mAP (mean Average Precision) across epochs.
**Rationale:** Tracks learning progress, detects underfitting/overfitting, and evaluates model robustness.
**Outcome:** Time-series plots for classification loss, box regression loss, and mAP\@0.5 and mAP\@0.5:0.95.

---

## 10. Inspect Augmented Training Samples

**Objective:** Display augmented images with labeled bounding boxes.
**Rationale:** Validates augmentation pipelines (e.g., mosaic, flipping, scaling) and correct label mapping.
**Outcome:** Grid of augmented training images displayed inline.

---

## 11. Perform Batch Inference on Test Set

**Objective:** Apply the model to all test images and store predictions.
**Rationale:** Demonstrates the model’s performance on unseen data.
**Outcome:** Annotated images saved to `runs/detect/predict/` with confidence thresholds applied.

---

## 12. Single-Image Inference Example

**Objective:** Test the model on a user-specified input image.
**Rationale:** Provides a flexible demo for practical testing or visual presentation.
**Outcome:** Display of predicted bounding boxes and class labels on the given image.

---

## 13. Video Stream Processing

**Objective:** Run the model on each frame of an input video file.
**Rationale:** Enables offline analysis of surveillance or traffic footage.
**Outcome:** Output video with detection overlays saved in standard video format.

---

## 14. Google Drive Integration

**Objective:** Mount Google Drive for persistent storage access.
**Rationale:** Supports saving and loading models, datasets, and outputs between sessions.
**Outcome:** Drive accessible under `/content/drive/MyDrive/` path.

---

## 15. Real‑Time Webcam Deployment

**Objective:** Deploy the trained model on live webcam input.
**Rationale:** Demonstrates real-time use-case capability for safety and surveillance systems.
**Outcome:** Interactive detection results displayed in a live video window.

---
