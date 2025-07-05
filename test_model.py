import cv2
import os
import pandas as pd
from datetime import datetime

# Face_detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("trainer.yml")

# Image Capture
cap_img = cv2.VideoCapture(0)

Name_Dict = {
    101: "Saurabh Maurya",
    102:"Harsh Maurya",
    103:"Shobhit Maurya",
    104:"Nidhi Maurya",
    105:"Rahul Meel"
}

attandence_file = "attendence.xlsx"


if os.path.exists(attandence_file):
    df_attendance = pd.read_excel(attandence_file)
else:
    df_attendance = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])

marked_ids = set()

while True:
    ret, frame = cap_img.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        serial, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        if confidence < 100:
            name = Name_Dict.get(serial, "Unknown")
            cv2.putText(frame, name, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 55), 2)

            if serial not in marked_ids:
                marked_ids.add(serial)
                new_row = {
                    "ID": serial,
                    "Name": name,
                    "Date": date_str,
                    "Time": time_str
                }
                df_attendance = pd.concat([df_attendance, pd.DataFrame([new_row])], ignore_index=True)

        else:
            name = "Unknown"
            cv2.putText(frame, name, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 55), 2)

            new_row = {
                "ID": "Unknown",
                "Name": "Unknown",
                "Date": date_str,
                "Time": time_str
            }
            df_attendance = pd.concat([df_attendance, pd.DataFrame([new_row])], ignore_index=True)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 55, 255), 2)

    cv2.imshow("Capture faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Save to Excel
df_attendance.to_excel(attandence_file, index=False)

cap_img.release()
cv2.destroyAllWindows()

print("Attendance saved ")
