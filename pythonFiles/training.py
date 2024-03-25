import cv2
import numpy as np
from os import listdir
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
        # Extract label from the file name (assuming the file name is in the format "Name_<index>.jpg")
        label = int(onlyfiles[i].split('_')[1].split('.')[0])  # Extracting the index from the file name
        Labels.append(label)
    else:
        print(f"Unable to read image: {images_path}")

# Convert Labels to numpy array
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize LBPH Face Recognizer
model = cv2.face.LBPHFaceRecognizer_create()

# Train the model
model.train(np.asarray(Training_Data), Labels)

# Save the trained model
model.save("trained_model.yml")

print("Dataset model training complete")
