import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import argparse
from pathlib import Path
import os
import ntpath

# Argument parser
parser = argparse.ArgumentParser(description='Easy Facial Recognition App')
parser.add_argument('-i', '--input', type=str, required=True, help='Directory of input known faces')

print('[INFO] Starting System...')
print('[INFO] Importing pretrained model...')
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print('[INFO] Pretrained model imported.')

def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces

def encode_face(image):
    if len(image.shape) != 3:
        print(f"[ERROR] The input image does not have three dimensions. Current shape: {image.shape}")
        return [], [], []

    # Ensure the image is in RGB format
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        print(f"[ERROR] Unsupported number of channels: {image.shape[2]}.")
        return [], [], []

    # Ensure the image is in the correct format
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"[ERROR] Failed to convert image to grayscale. Error: {e}")
        return [], [], []

    try:
        face_locations = face_detector(gray_image, 1)
    except Exception as e:
        print(f"[ERROR] Failed to detect faces. Error: {e}")
        return [], [], []

    if len(face_locations) == 0:
        print("[WARNING] No faces detected in the image.")
        return [], [], []

    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)
    
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list

def easy_face_reco(frame, known_face_encodings, known_face_names):
    print(np.shape(frame))
    rgb_small_frame = frame[:, :, ::-1]
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            continue
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = [vector <= tolerance for vector in vectors]
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Unknown"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)

if __name__ == '__main__':
    args = parser.parse_args()

    print('[INFO] Importing faces...')
    face_to_encode_path = Path(args.input)
    files = list(face_to_encode_path.rglob('*.jpg')) + list(face_to_encode_path.rglob('*.png'))
    
    if len(files) == 0:
        raise ValueError(f'No faces detected in the directory: {face_to_encode_path}')
    
    known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]
    known_face_encodings = []

    for file_ in files:
        try:
            image = PIL.Image.open(file_)
            image = np.array(image)

            # Debugging: Print image shape and type
            print(f"[INFO] Processing {file_} with image shape: {image.shape} and dtype: {image.dtype}")

            # Check if the image is empty or has an unexpected shape
            if image.size == 0:
                print(f"[ERROR] The image {file_} is empty or cannot be read.")
                continue

            encodings, _, _ = encode_face(image)
            if not encodings:
                print(f"[WARNING] No faces detected in {file_}. Skipping this file.")
                continue
            known_face_encodings.append(encodings[0])
        except Exception as e:
            print(f"[ERROR] Failed to process {file_}. Error: {e}")

    print('[INFO] Faces well imported')
    print('[INFO] Starting Webcam...')
    video_capture = cv2.VideoCapture(0)
    print('[INFO] Webcam well started')
    print('[INFO] Detecting...')

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to capture image from webcam.")
            break

        easy_face_reco(frame, known_face_encodings, known_face_names)
        cv2.imshow('Easy Facial Recognition App', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('[INFO] Stopping System')
    video_capture.release()
    cv2.destroyAllWindows()
