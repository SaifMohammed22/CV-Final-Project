import cv2
import os
import numpy as np


def detect_and_crop_faces(image_path, target_size=(100, 100)):
    # Load OpenCV pre-trained Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # If no face is detected, resize and return the whole grayscale image
        return cv2.resize(gray, target_size)

    # Take the first detected face (x, y, width, height)
    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]

    # Resize to a fixed size so all matrices match in PCA
    resized_face = cv2.resize(face_roi, target_size)

    return resized_face


def load_dataset(dataset_path):
    X = []
    y = []
    class_names = []
    label_id = 0

    # Loop through folders in the data directory
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)

        # Skip if not a directory
        if not os.path.isdir(person_dir):
            continue

        class_names.append(person_name)

        # Read all images for this person
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            face = detect_and_crop_faces(image_path)

            if face is not None:
                # Flatten the 2D image array into a 1D vector for PCA
                X.append(face.flatten())
                y.append(label_id)

        label_id += 1

    return np.array(X), np.array(y), class_names