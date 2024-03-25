import cv2
import numpy as np
import os

# Function to load known faces and their corresponding labels
def load_known_faces(data_path):
    known_faces = {}
    for filename in os.listdir(data_path):
        if filename.endswith('.npy'):
            label = int(filename[:-4])
            known_faces[label] = np.load(os.path.join(data_path, filename))
    return known_faces

# Load known faces and their labels
known_faces = load_known_faces('Known_Faces')

# Load the pre-trained LBPH Face Recognizer model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trained_model.yml")

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

        return img, roi, (x, y, w, h)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face, (x, y, w, h) = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))

        if confidence > 82:
            label = result[0]
            # If the recognized face is known, display the name
            if label in known_faces:
                name = known_faces[label]
                cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                # If the recognized face is unknown, prompt for login and store data
                cv2.putText(image, "Unknown Face", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "Press 'y' to log in", (x, y + h + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
                key = cv2.waitKey(1)
                if key == ord('y'):
                    new_name = input("Enter your name: ")
                    # Save the new face and its label
                    new_label = max(known_faces.keys()) + 1 if known_faces else 0
                    np.save(os.path.join('Known_Faces', f'{new_label}.npy'), new_name)
                    known_faces[new_label] = new_name
                    print("New face added and logged in!")
        else:
            cv2.putText(image, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face Cropper', image)

    except Exception as e:
        print(f"Error: {str(e)}")
        cv2.putText(image, "Face not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
