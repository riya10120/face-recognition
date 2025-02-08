import cv2
import face_recognition
import numpy as np
import database
import pyttsx3
from deepface import DeepFace

# Initialize text-to-speech engine
engine = pyttsx3.init()

def fetch_known_faces():
    conn = database.connect_db()
    cursor = conn.cursor()

    # Fetch stored face encodings and their corresponding names
    cursor.execute("SELECT name, face_encoding FROM users")
    
    known_faces = []
    known_names = []

    for name, encoding_str in cursor.fetchall():
        # Convert the stored encoding from string back to a numpy array
        encoding = np.array(list(map(float, encoding_str.split(","))))
        known_faces.append(encoding)
        known_names.append(name)

    cursor.close()
    conn.close()
    return known_faces, known_names

def recognize_face():
    known_faces, known_names = fetch_known_faces()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Use "cnn" for better accuracy (slower)
        
        # Encode the faces found in the current frame
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare the face with known faces using a lower tolerance for better matching
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
            name = "Unknown"  # Default name for unrecognized faces

            # Calculate face distances for better matching
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]  # Get the name of the matched face

            # Draw rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Detect emotion using DeepFace
            try:
                face_image = frame[top:bottom, left:right]
                emotion_result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
                emotion = emotion_result[0]['dominant_emotion']
                emotion_text = f"{name} is {emotion}"
                cv2.putText(frame, emotion_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Speak the name and emotion
                engine.say(f"{name} is {emotion}")
                engine.runAndWait()
            except Exception as e:
                print(f"Error in emotion detection: {e}")

        # Show the frame with the face(s) recognized
        cv2.imshow("Face Recognition", frame)

        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

recognize_face()