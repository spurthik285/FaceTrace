# Face Trace: Snap, Track, Attend

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRLQ4I9Bze_atlGwnH9yXWpUg-bL1IvxClzog&s" alt="Project Image">
</p>

## Overview
This project implements an automated student attendance system using face recognition technology. The system captures real-time images through a camera, detects faces using OpenCV and Haar-cascade, processes them to enhance features using Local Binary Pattern Histogram (LBPH) and Principal Component Analysis (PCA), and finally registers attendance in an Excel file.

## Features
- **Face Detection:** Utilizes Haar-cascade classifier to detect frontal faces in real-time video frames.
- **Pre-processing:** Includes image scaling, median filtering, and conversion to grayscale to enhance image quality and reduce noise.
- **Feature Extraction:** Implements LBPH for local feature extraction and PCA for dimensionality reduction and feature selection.
- **Attendance Recording:** Marks attendance of recognized students and saves records in Excel format.
- **User Interface:** Built using Tkinter for a user-friendly interaction to capture images and display recognition results.

## Technologies Used
- **OpenCV:** For image processing tasks such as face detection, image enhancement, and feature extraction.
- **Tkinter:** Used to create the GUI for capturing images and displaying attendance results.
- **Python Libraries:** Pandas for handling Excel files, NumPy for numerical computations, and Matplotlib for result visualization.

## Working
1. **Face Detection:** The system detects faces in real-time video frames using Haar-cascade.
2. **Image Capture:** Captures images of students and preprocesses them to standardize quality.
3. **Feature Extraction:** Applies LBPH and PCA to extract features and reduce dimensionality.
4. **Recognition:** Compares test images with trained images using extracted features to recognize students.
5. **Attendance Marking:** Records attendance of recognized students in an Excel file and manages unknown faces separately.
