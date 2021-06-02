import pandas as pd
import random
import csv

DATA_PATH = "../data/all_data_african.csv"
OUTPUT_PATH = "../data/african"
SPLIT_SIZE = 0.8
TRAIN = OUTPUT_PATH + "/" + "train_african_" + str(SPLIT_SIZE) + ".csv"
TEST = OUTPUT_PATH + "/" + "test_african_" + str(SPLIT_SIZE) + ".csv"
TRAIN_CSV = []
TEST_CSV = []

df = pd.read_csv(DATA_PATH, header=None)

# Read images and labels into list
images = df[0]
labels = df[1]

# Shuffle
temp = list(zip(images, labels))
random.shuffle(temp)
n_images, n_labels = zip(*temp)

# Split
train_size = int(len(n_images) * SPLIT_SIZE)

train_images = n_images[0:train_size]
train_labels = n_labels[0:train_size]

test_images = n_images[train_size:]
test_labels = n_labels[train_size:]

# Save to file
for i, img in enumerate(train_images):
    TRAIN_CSV.append([img, train_labels[i]])

for i, img in enumerate(test_images):
    TEST_CSV.append([img, test_labels[i]])

with open(TRAIN, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(TRAIN_CSV)

with open(TEST, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(TEST_CSV)

print("train images: "+str(len(TRAIN_CSV))+", test images: "+str(len(TEST_CSV))+", total: "+str((len(TRAIN_CSV)+len(TEST_CSV))))
print("Done")
