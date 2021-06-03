import cv2
import numpy as np
import face_recognition
from imutils import paths

# INPUT_FOLDER = "G:\datasets\Alice\Data\Chinese\original/sad"
# OUTPUT_FOLDER = "G:\datasets\Alice\Data\Chinese\cropped_faces/sad_cropped_faces"

INPUT_FOLDER = "C:/Users/Administrator/Documents/WeChat Files/wxid_avb9aus2fd1622/FileStorage/File/2021-06/africans/africans"
OUTPUT_FOLDER = "test"

# Read all image files
files = list(paths.list_images(INPUT_FOLDER))
count = 0

for p in files:
    img = cv2.imread(p)
    img = cv2.resize(img, dsize=(400, 400), interpolation=cv2.INTER_LINEAR)
    faceLoc = face_recognition.face_locations(img)

    for top, right, bottom, left in faceLoc:
        # Draw a box around the face
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        count += 1
        crop_img = img[top:bottom, left:right]
        crop_img = cv2.resize(crop_img, dsize=(100, 100))
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        dest = OUTPUT_FOLDER + "/sad_cn_" + str(count) + ".jpg"
        cv2.imwrite(dest, crop_img)

        print("Processed: " + str(count) + " faces")

print("done")
