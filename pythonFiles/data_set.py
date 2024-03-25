import cv2
import os

# Create a directory to store images if it doesn't exist
data_path = 'Images'
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Function to capture images from the camera
def capture_images():
    name = input("Enter your name: ")
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Create a face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Counter for image filenames
    img_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Increment the image counter
            img_count += 1
            # Save the detected face to the 'Images' folder
            cv2.imwrite(os.path.join(data_path, f"{name}_{img_count}.jpg"), gray[y:y+h, x:x+w])

        # Display the frame
        cv2.imshow('Capture Faces', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Capture images
capture_images()
