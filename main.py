import cv2
import os
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox

#--#
root = tk.Tk()
root.withdraw()

user_ID = simpledialog.askstring("Input","Enter your ID:")
user_Name = simpledialog.askstring("Input","Enter your Name:")

if not user_ID or not user_Name:
    messagebox.showerror("Error!","User_id and User_name are required")

#----#
# Face_detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_frontalface_default.xml")

#dataset

dataset_dir = "Dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)


# Image Capture

cap_img = cv2.VideoCapture(0)

count = 0
user_data = []

while True:
    ret,frame = cap_img.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        count +=1
        face_img = gray[y:y+h,x:x+w]
        file_name = f"{dataset_dir}/User.{user_ID}.{user_Name}.{count}.jpg"
        cv2.imwrite(file_name,face_img)

        user_data.append({
            "ID": user_ID,
            "User-Name":user_Name,
            "Face-Image":face_img,
            "File-Name":file_name
        })
        cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 55, 255), 2)

        

    cv2.imshow("Cpture faces",frame)

    key = cv2.waitKey(1000) & 0xFF  
    if key == ord('q'):
        break




cap_img.release()
cv2.destroyAllWindows()