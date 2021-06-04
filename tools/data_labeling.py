from imutils import paths
import os
import csv

# MAIN_FOLDER = "G:\datasets\Alice\Data\Chinese\cropped_faces"
MAIN_FOLDER = "G:\datasets\Alice\Data\African\cropped_faces"
ETHNICITY = "AFRICAN"
OUTPUT_FOLDER = "../data"

folders = os.listdir(MAIN_FOLDER)
labels_num = [0, 1, 2, 3]
labels_str = ["angry", "happy", "neutral", "sad"]
csv_data = []

"""
0->angry
1->happy
2->neutral
3->sad
"""
for f in folders:

    if f == "angry_cropped_faces":
        files = (list(paths.list_images(MAIN_FOLDER + '\\' + f)))
        for ff in files:
            dd = ff.split("\\")
            name = dd[len(dd) - 1]
            csv_data.append([name, 0, labels_str[0], ETHNICITY])

    if f == "happy_cropped_faces":
        files = (list(paths.list_images(MAIN_FOLDER + '\\' + f)))
        for ff in files:
            dd = ff.split("\\")
            name = dd[len(dd) - 1]
            csv_data.append([name, 1, labels_str[1], ETHNICITY])

    if f == "neutral_cropped_faces":
        files = (list(paths.list_images(MAIN_FOLDER + '\\' + f)))
        for ff in files:
            dd = ff.split("\\")
            name = dd[len(dd) - 1]
            csv_data.append([name, 2, labels_str[2], ETHNICITY])

    if f == "sad_cropped_faces":
        files = (list(paths.list_images(MAIN_FOLDER + '\\' + f)))
        for ff in files:
            dd = ff.split("\\")
            name = dd[len(dd) - 1]
            csv_data.append([name, 3, labels_str[3], ETHNICITY])

with open(OUTPUT_FOLDER + "/all_data_african.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
