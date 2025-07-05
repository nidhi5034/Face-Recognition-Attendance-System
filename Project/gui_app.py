import tkinter as tk
import subprocess

root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("400x300")

def register_face():
    subprocess.run(["python", "main.py"])

def train_model():
    subprocess.run(["python", "training.py"])

def take_attendance():
    subprocess.run(["python", "test_model.py"])

tk.Label(root, text="Face Recognition Attendance System", font=("Helvetica", 16)).pack(pady=20)

tk.Button(root, text="Register New Face", command=register_face, width=25).pack(pady=10)
tk.Button(root, text="Train Model", command=train_model, width=25).pack(pady=10)
tk.Button(root, text="Start Attendance", command=take_attendance, width=25).pack(pady=10)
tk.Button(root, text="Quit", command=root.quit, width=25).pack(pady=10)

root.mainloop()
