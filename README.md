# National-Exam-Grading-System
Automatic THPTQG exam grading system using OpenCV and CNN. The project detects and processes scanned answer sheets, extracts student ID and exam code, recognizes multiple-choice bubbles (A–D), and compares results with answer keys in Excel to calculate final scores automatically.

# Overview
This project automatically processes scanned multiple-choice answer sheets, extracts student information (SBD - Student ID, MDT - Exam Code), detects filled bubbles, and calculates final exam scores by comparing predictions with the official answer key stored in Excel.
The system reduces manual grading effort, improves accuracy, and speeds up large-scale exam evaluation.

# Features
- Automatic detection of answer sheet regions from scanned images  
- Recognition of Student ID (SBD) and Exam Code (MDT) using CNN  
- Segmentation of multiple-choice answer bubbles (A, B, C, D)  
- CNN-based classification of filled vs unfilled bubbles  
- Automatic comparison with ground-truth answer key (Excel file)  
- Score calculation and export results to Excel  

# Technologies Used
- Python  
- OpenCV (image processing)  
- NumPy, Pandas  
- TensorFlow / Keras (CNN model)  
- Matplotlib  
- Imutils  

# Project Workflow
1. Load scanned exam image  
2. Detect contours and extract regions (SBD, MDT, answers)  
3. Split regions into grid cells  
4. Preprocess images (grayscale, resize, threshold)  
5. CNN model predicts filled bubbles  
6. Map predictions to answers (A, B, C, D)  
7. Compare with answer key from Excel  
8. Compute and export final score  

# Input Requirements
- Scanned answer sheet image (e.g. test4.jpg)
- Excel file containing answer key (Dapandethi.xlsx)
- Pretrained CNN model weights (weight.h5)

# Output
- Student ID (SBD)
- Exam Code (MDT)
- Predicted answers per question
- Final score (0–10 scale)
- Result file exported as result.xlsx

# Goal
To automate the grading process of multiple-choice exam papers and support large-scale educational assessment with higher speed and accuracy.

# Notes
- Model confidence threshold: > 0.9 for valid selection
- Each question has 4 choices (A–D)
- Answer key is stored in Excel by exam code sheet

# Author
Nguyen Van Dung, 
Data Science Student, 
University of Da Lat Vietnam
