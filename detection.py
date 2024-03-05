import face_recognition
from os import listdir
import cv2
import numpy as np
from os.path import isfile, join

data_path = 'Images'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

# Set a common size for all images
common_size = (100, 100)

for i, files in enumerate(onlyfiles):
    images_path = join(data_path, onlyfiles[i])
    images = cv2.imread(images_path, cv2.IMREAD_GRAYSCALE)

    if images is not None:
        # Resize the image to a common size
        resized_image = cv2.resize(images, common_size)
        Training_Data.append(np.asarray(resized_image, dtype=np.uint8))
        Labels.append(i)
    else:
        print(f"Unable to read image: {images_path}")

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Dataset model training complete")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    print(f"Number of faces detected: {len(faces)}")
    print(f"Faces coordinates: {faces}")

    if len(faces) == 0:
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

        return img, roi


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))

        if confidence > 82:
            cv2.putText(image, "Vansh", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Face Cropper', image)
        else:
            cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Face Cropper', image)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
