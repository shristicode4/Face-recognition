import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

def preprocess_image(img):
    img = cv2.resize(img, (64, 64))  
    img = img.astype('float32') / 255.0  # Normalize the image
    return img


def get_embedding(model, img):
    img = np.expand_dims(img, axis=0)
    return model.predict(img)

def verify_images(model, img1, img2):
    embedding1 = get_embedding(model, img1)
    embedding2 = get_embedding(model, img2)
    distance = np.linalg.norm(embedding1 - embedding2)

    threshold = 0.5

    print(f"Distance between embeddings: {distance}")

    if distance < threshold:
        return True
    else:
        return False

# Capture an image from the webcam
def capture_image_from_camera(window_name="Capture Image"):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow(window_name, frame)

        # Press 'c' to capture the image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            captured_image = frame
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_image

if __name__ == "__main__":
    # Load the trained model
    model_path = 'C:/Users/ss/OneDrive/Desktop/face_ver/cnn_face_verification.h5'
    model = load_model(model_path)

    # Capture the first image from the camera
    print("Capture the first image by pressing 'c'")
    img1 = capture_image_from_camera()
    img1_processed = preprocess_image(img1)

    # Capture the second image from the camera
    print("Capture the second image by pressing 'c'")
    img2 = capture_image_from_camera()
    img2_processed = preprocess_image(img2)

    # Verify if the two images are of the same person
    is_same_person = verify_images(model, img1_processed, img2_processed)

    if is_same_person:
        print("The images are of the same person.")
    else:
        print("The images are of different people.")
