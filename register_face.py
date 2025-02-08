import cv2
import face_recognition
import numpy as np
import database

def register_face():
    conn = database.connect_db()
    cursor = conn.cursor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'S' to save your face.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Display the frame
        cv2.imshow("Register Face - Press 'S' to Save", frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Trigger on pressing 'S'
            print("Attempting to detect face...")

            # Detect faces in the captured frame
            face_locations = face_recognition.face_locations(frame, model="hog")  # Use "cnn" for better accuracy (slower)

            if face_locations:
                print("Face detected! Processing...")

                # Only proceed if a face is detected
                face_encoding = face_recognition.face_encodings(frame, face_locations)[0]

                print("Please enter your name.")
                name = input("Enter Name: ")

                # Save the face encoding and name into MySQL
                encoding_str = ",".join(map(str, face_encoding))
                cursor.execute("INSERT INTO users (name, face_encoding) VALUES (%s, %s)", (name, encoding_str))
                conn.commit()
                print(f"✅ Face registered for {name}")
                break  # Exit after registering the face
            else:
                print("⚠️ No face detected. Please ensure your face is visible and well-lit.")
                print("Press 'S' to try again or 'Q' to quit.")

        elif key == ord('q'):  # Exit on pressing 'Q'
            print("Exiting without registering a face.")
            break

    cap.release()
    cv2.destroyAllWindows()
    cursor.close()
    conn.close()

register_face()