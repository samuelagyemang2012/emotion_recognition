from deepface.commons import functions
import os
from matplotlib import pyplot as plt

backends = ['retinaface']

INPUT_PATH = "G:/datasets/Alice/Data/African/original/happy_edited/"
OUTPUT_PATH = "G:/datasets/Alice/Data/African/cropped_faces/happy_cropped_faces/"
EMOTION = "happy"
ETH = "afr"
CHANNELS = 3

INPUT_SHAPE = (100, 100)

files = os.listdir(INPUT_PATH)

for i, f in enumerate(files):

    img_path = INPUT_PATH + f
    img = functions.preprocess_face(img_path, target_size=INPUT_SHAPE, detector_backend=backends[0])
    if img.size != 0:
        img = img.reshape(INPUT_SHAPE[0], INPUT_SHAPE[0], CHANNELS)
        dest = OUTPUT_PATH+ETH + "_" + EMOTION + "_" + str(i)+".jpg"
        # dest = "../test/" + ETH + "_" + EMOTION + "_" + str(i) + ".jpg"
        plt.imsave(dest, img[:, :, ::-1], format="jpg")
        c = i+1
        print("Processed: " + str(c) + " images")
    else:
        print("No face detected!")

print("done")
