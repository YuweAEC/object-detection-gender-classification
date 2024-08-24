# Object Detection and Gender Classification

This project combines YOLOv3 and a TensorFlow-based gender classification model to perform real-time object detection and gender classification using a webcam. It also includes an optional Flask web interface for running the system.

## Project Structure

object-detection-gender-classification/
│
├── yolov3/
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   ├── coco.names
│
├── model/
│   ├── gender_model.h5       # Pre-trained TensorFlow model for gender classification
│   ├── train_gender_model.py # Optional: Training script for gender classification
│
├── app/
│   ├── __init__.py           # Flask app initialization (if using a web interface)
│   ├── routes.py             # Flask routes for the web interface (optional)
│   ├── camera.py             # Code to handle camera input and run detection
│
├── static/                   # Static files (CSS, JS) for the web interface (if using Flask)
│   ├── css/
│   ├── js/
│
├── templates/                # HTML templates for the Flask web interface
│   ├── index.html
│
├── README.md                 # Project documentation
├── requirements.txt          # List of Python dependencies
├── main.py                   # Main script to run the object detection and classification
├── run.sh                    # Shell script to run the application (optional)
└── Dockerfile                # Dockerfile for containerizing the project (optional)
