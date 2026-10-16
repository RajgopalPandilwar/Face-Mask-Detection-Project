# -*- coding: utf-8 -*-
"""
Face Mask Detection System
==========================
A deep learning-based system to detect faces and classify mask usage in real-time.

Author: Rajgopal Pandilwar

This script uses:
- Haar Cascade Classifier for face detection
- A CNN-based approach for mask classification (when model is available)
"""

import os
import cv2
import numpy as np


def detect_faces(image_path=None, use_webcam=True):
    """
    Detect faces in an image or from webcam feed.
    
    Args:
        image_path (str): Path to image file (optional)
        use_webcam (bool): Whether to use webcam feed
        
    Returns:
        None: Displays detected faces in a window
    """
    # Load the Haar Cascade face detection model
    # This XML file contains the trained model for frontal face detection
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_classifier = cv2.CascadeClassifier(face_cascade_path)
    
    if use_webcam:
        # Initialize webcam capture (0 = default camera)
        cap = cv2.VideoCapture(0)
        
        print("Face Detection Started - Press 'q' to quit")
        print("=" * 40)
        
        while True:
            # Read frame from webcam
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame from webcam")
                break
            
            # Convert to grayscale for face detection
            # Haar Cascade works on grayscale images
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with the cascade classifier
            # scaleFactor: how much the image size is reduced at each scale
            # minNeighbors: how many neighbors each candidate rectangle should have
            faces = face_classifier.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5
            )
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Add label
                cv2.putText(frame, 'Face', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display face count
            face_count = len(faces)
            cv2.putText(frame, f'Faces: {face_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show the output
            cv2.imshow("Face Detection", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
    elif image_path:
        # Load and process a single image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 5)
        
        print(f"Found {len(faces)} face(s) in {image_path}")
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Face Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def classify_mask(face_img, model_path=None):
    """
    Classify whether a detected face is wearing a mask.
    
    Args:
        face_img (numpy array): Cropped face image
        model_path (str): Path to trained mask classification model
        
    Returns:
        str: 'Mask' or 'No Mask'
    """
    # This function requires a trained CNN model for mask classification
    # The original project uses a MobileNetV2-based classifier
    
    if model_path and os.path.exists(model_path):
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        
        # Load the pre-trained mask detection model
        model = load_model(model_path)
        
        # Preprocess the face image
        face_img = cv2.resize(face_img, (224, 224))
        face_img = image.img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0  # Normalize
        
        # Predict
        result = model.predict(face_img)
        return 'Mask' if result[0][0] > 0.5 else 'No Mask'
    else:
        return "Model not available - using face detection only"


if __name__ == "__main__":
    print("Face Mask Detection System")
    print("Author: Rajgopal Pandilwar")
    
    # Run face detection from webcam
    detect_faces(use_webcam=True)
    
    # To classify masks, provide a trained model:
    # result = classify_mask(face_image, model_path='mask_model.h5')
