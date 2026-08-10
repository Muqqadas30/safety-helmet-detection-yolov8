# Safety Helmet Detection System Using YOLOv8

An end-to-end Computer Vision object detection project built as part of the AIRI Team PITB AI Internship (Task 1). The system detects **Person**, **Helmet**, and **Without Helmet** in images to support basic workplace/road safety monitoring.

## Project Overview
Not wearing a safety helmet on construction sites, factories, or while riding a motorcycle is a major safety risk. Manually monitoring this is time-consuming and error-prone. This project trains a YOLOv8 object detection model to automatically flag people who are not wearing a helmet, drawing bounding boxes with confidence scores on new images.

## Dataset
- Source: Roboflow Universe (helmet-detection-kkt4z) + manual review and cleanup
- Total images used: 211
- Classes: person, helmet, without_helmet
- 158 images manually annotated for the "person" class using Roboflow
- Split: 70% train / 21% validation / 9% test

## Tools & Technologies
Python, Google Colab, Google Drive, YOLOv8 (Ultralytics), Roboflow, OpenCV, Matplotlib

## Model Training
- Model: YOLOv8n
- Epochs: 30
- Image size: 640x640
- Platform: Google Colab (Tesla T4 GPU)

## Evaluation Results
- Precision: 0.935
- Recall: 0.940
- mAP@0.5: 0.963
- mAP@0.5:0.95: 0.766

 ## Demo Link
 https://app.roboflow.com/muqqadas-iftikhar/safety-helmet-detection-e4zru/models/muqqadas-iftikhar/safety-helmet-yolov8
