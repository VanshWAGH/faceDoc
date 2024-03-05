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
