import cv2
import os
import numpy as np

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# JAFFE Emotion Labels (from README)
EMOTION_MAPPING = {
    'NE': 'Neutral',
    'HA': 'Happy',
    'SA': 'Sad',
    'SU': 'Surprised',
    'AN': 'Angry',
    'DI': 'Disgusted',
    'FE': 'Fear'
}


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
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # If no face is detected, resize and return the whole grayscale image
        return cv2.resize(gray, target_size)

    # Take the first detected face (x, y, width, height)
    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]

    # Resize to a fixed size so all matrices match in PCA
    resized_face = cv2.resize(face_roi, target_size)

    return resized_face


def _get_emotion_label(dataset_path, image_path):
    """Extract emotion label from JAFFE filename.

    Format: POSER.EMOTION_NUMBER.ID.tiff
    Example: KA.AN1.39.tiff -> emotion is 'AN'

    Emotions: NE (Neutral), HA (Happy), SA (Sad), SU (Surprised), 
              AN (Angry), DI (Disgusted), FE (Fear)
    """
    parent_dir = os.path.relpath(os.path.dirname(image_path), dataset_path)

    if parent_dir == '.':
        # Flat directory: extract from filename
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        parts = file_name.split('.')
        if len(parts) >= 2:
            # Second part is EMOTION_NUMBER (e.g., 'AN1' -> take 'AN')
            return parts[1][:2]

    # Nested directory: use folder name as label (fallback)
    return parent_dir.split(os.sep)[0]


def load_dataset(dataset_path):
    """Load JAFFE dataset and extract emotions as class labels.

    Returns:
        X: Feature matrix (flattened face images)
        y: Emotion labels (0-6 for 7 emotions)
        class_names: List of emotion names [NE, HA, SA, SU, AN, DI, FE]
        image_paths: List of image file paths
    """
    X = []
    y = []
    class_names = []
    class_to_label = {}
    image_paths = []

    # Loop through image files in the dataset directory tree
    for root, _, files in os.walk(dataset_path):
        for image_name in sorted(files):
            if os.path.splitext(image_name)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            image_path = os.path.join(root, image_name)
            emotion = _get_emotion_label(dataset_path, image_path)

            if emotion not in class_to_label:
                class_to_label[emotion] = len(class_names)
                class_names.append(emotion)

            face = detect_and_crop_faces(image_path)

            if face is not None:
                # Flatten the 2D image array into a 1D vector for PCA
                X.append(face.flatten())
                y.append(class_to_label[emotion])
                image_paths.append(image_path)

    return np.array(X), np.array(y), class_names, image_paths
