import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
ids = []

dataset_path = "dataset"

for img_name in os.listdir(dataset_path):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    try:
        label = int(img_name.split('.')[1])  # Extract 102 from 'User.102.Name.27.jpg'
    except (IndexError, ValueError):
        print(f"Skipping invalid file name: {img_name}")
        continue

    img_path = os.path.join(dataset_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping unreadable image: {img_name}")
        continue

    detected = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in detected:
        face_roi = img[y:y+h, x:x+w]
        faces.append(face_roi)
        ids.append(label)

# Train and save
if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    recognizer.save("trainer.yml")
    print("Training completed and saved")
    print(f"Trained on {len(set(ids))} unique IDs, {len(faces)} face samples.")

else:
    print("No faces detected! Please check your dataset images.")

