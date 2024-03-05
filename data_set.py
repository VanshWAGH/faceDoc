import cv2
import os

# Load the pre-trained Haar Cascade classifier for face detection
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Function to extract faces from an image
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is None or len(faces) == 0:
        return None

    # Assuming you want to extract only the first face found
    (x, y, w, h) = faces[0]
    cropped_face = img[y:y + h, x:x + w]

    return cropped_face


# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Create a directory to save the captured images
output_directory = 'Images'
os.makedirs(output_directory, exist_ok=True)

# Initialize a counter for the number of captured images
count = 0

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if a face is found in the current frame
    if face_extractor(frame) is not None:
        count += 1

        # Resize and convert the face to grayscale
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save the captured face image
        file_name_path = os.path.join(output_directory, f'Image_{count}.jpg')
        cv2.imwrite(file_name_path, face)

        # Display the count on the captured face
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Captured Face', face)

    else:
        print("Face not Found")

    # Break the loop if 'Enter' key is pressed or the specified number of images is captured
    if cv2.waitKey(1) == 13 or count == 5:
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()

print(f"{count} samples received and saved in {output_directory}")
