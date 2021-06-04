from deepface.commons import functions
import os
from matplotlib import pyplot as plt

backends = ['retinaface']

INPUT_PATH = "G:/datasets/Alice/Data/Chinese/original/angry/"
OUTPUT_PATH = "G:/datasets/Alice/Data/Chinese/cropped_faces/angry_cropped_faces/"
EMOTION = "angry"
ETH = "chn"
CHANNELS = 3
INPUT_SHAPE = (100, 100)

# Read all files in the folder
files = os.listdir(INPUT_PATH)

for i, f in enumerate(files):

    img_path = INPUT_PATH + f

    # Detect faces in image
    try:
        img = functions.preprocess_face(img_path, target_size=INPUT_SHAPE, detector_backend=backends[0])

        img = img.reshape(INPUT_SHAPE[0], INPUT_SHAPE[0], CHANNELS)

        dest = OUTPUT_PATH + ETH + "_" + EMOTION + "_" + str(i) + ".jpg"
        # dest = "../test/" + ETH + "_" + EMOTION + "_" + str(i) + ".jpg"
        # Save the image file
        plt.imsave(dest, img[:, :, ::-1], format="jpg")
        c = i + 1
        print("Processed: " + str(c) + " images")
    except:
        print("No face detected! -> "+files[i])

print("done")
