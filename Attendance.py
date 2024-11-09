from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import time
import sqlite3
from win32com.client import Dispatch
from datetime import datetime, date

# Function to initialize the SQLite database and create the required tables
def create_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    # Create table for students (if it doesn't exist)
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id TEXT PRIMARY KEY, name TEXT)''')
    
    # Create table for attendance
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id TEXT, name TEXT, date TEXT, time TEXT, status TEXT)''')
    
    conn.commit()
    conn.close()

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Function to mark attendance in the database
def mark_attendance(user_id, name, timestamp, status="Present"):
    today_date = date.today().strftime("%d-%m-%Y")
    
    # Connect to SQLite database
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    # Check if attendance has already been marked today
    c.execute("SELECT * FROM attendance WHERE id = ? AND date = ?", (user_id, today_date))
    result = c.fetchone()
    
    if result:
        print(f"Attendance for {name} (ID: {user_id}) already marked today.")
        if status == "Present":
            speak(f"Attendance for {name} already marked today.")
    else:
        # Insert attendance if not already present
        c.execute("INSERT INTO attendance (id, name, date, time, status) VALUES (?, ?, ?, ?, ?)", 
                  (user_id, name, today_date, timestamp, status))
        conn.commit()
        print(f"Attendance marked for {name} (ID: {user_id}) at {timestamp}. Status: {status}.")
        
        # Only call speak for 'Present' students
        if status == "Present":
            speak(f"Attendance marked for {name}")
    
    conn.close()


# Function to mark absent students
def mark_absent_students():
    today_date = date.today().strftime("%d-%m-%Y")
    
    # Connect to SQLite database
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    # Fetch all students
    c.execute("SELECT * FROM students")
    all_students = c.fetchall()
    
    # Check which students have not been marked as present today
    for student in all_students:
        student_id, name = student
        c.execute("SELECT * FROM attendance WHERE id = ? AND date = ?", (student_id, today_date))
        result = c.fetchone()
        
        if not result:
            # Mark the student as absent
            ts = time.time()
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            mark_attendance(student_id, name, timestamp, status="Absent")
    
    conn.close()

# Initialize the database
create_db()

def add_student(user_id, name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO students (id, name) VALUES (?, ?)", (user_id, name))
    conn.commit()
    conn.close()

# Load face data, names, and IDs
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/ids.pkl', 'rb') as i:
    IDS = pickle.load(i)

# Adding all students from pickle files to the database
for i in range(len(IDS)):
    user_id = IDS[i]
    name = LABELS[i]
    add_student(user_id, name)

print("All students have been added to the database.")

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("bg.png")

attendance_to_save = None  # To hold the attendance details until 'o' is pressed

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (75, 75)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        
        name = output[0]
        user_id = IDS[LABELS.index(name)]
        
        ts = time.time()
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        
        # Prepare attendance details to save later
        attendance_to_save = (user_id, name, timestamp)
        
        # Draw rectangles and text on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(frame, "Name:" + str(name), (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Roll no:" + str(user_id), (x, y + h + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)

    k = cv2.waitKey(1)
    
    if k == ord('o'):
        if attendance_to_save:
            user_id, name, timestamp = attendance_to_save
            mark_attendance(user_id, name, timestamp, status="Present")  # Save attendance as Present when 'o' is pressed
            speak("Attendance saved.")
            attendance_to_save = None  # Reset after saving
    
    if k == ord('q'):
        # Before exiting, mark absent students
        mark_absent_students()
        break

video.release()
cv2.destroyAllWindows()

