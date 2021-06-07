import pandas as pd
import cv2
import scipy
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from evaluation.evaluation import *
from models.model import *

ETH = 'afr'
EPOCHS = 100
INPUT_SHAPE = (100, 100, 1)
BATCH_SIZE = 8
NUM_CLASSES = 4
VAL_SPLIT = 0.1
LEARNING_RATE = 0.0001
IMG_BASE_PATH = "C:/Users/Administrator/Desktop/Sam/facial_recognition/african/all_data/"
TRAIN_DATA_PATH = "../data/african/train_african_0.8.csv"
TEST_DATA_PATH = "../data/african/test_african_0.8.csv"
BEST_MODEL_PATH = "../trained_models/"
TRAIN_DATA = []
TRAIN_LABELS = []
TEST_DATA = []
TEST_LABELS = []

# Load the data and labels
train_df = pd.read_csv(TRAIN_DATA_PATH, header=None)
test_df = pd.read_csv(TEST_DATA_PATH, header=None)

train_imgs = train_df[0]
train_labels = train_df[1]

test_imgs = test_df[0]
test_labels = test_df[1]

for ti in train_imgs:
    img_path = IMG_BASE_PATH + ti
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    TRAIN_DATA.append(img)

for tt in test_imgs:
    img_path = IMG_BASE_PATH + tt
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    TEST_DATA.append(img)

# Normalize data
TRAIN_DATA = np.array(TRAIN_DATA)
TEST_DATA = np.array(TEST_DATA)

TRAIN_DATA = TRAIN_DATA.astype('float32')
TEST_DATA = TEST_DATA.astype('float32')

# one-hot encode labels
TRAIN_LABELS = to_categorical(train_labels)
TEST_LABELS = to_categorical(test_labels)

# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    validation_split=VAL_SPLIT,
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

TRAIN_DATA = np.reshape(TRAIN_DATA, (TRAIN_DATA.shape[0], TRAIN_DATA.shape[1], TRAIN_DATA.shape[2], 1))
TEST_DATA = np.reshape(TEST_DATA, (TEST_DATA.shape[0], TEST_DATA.shape[1], TEST_DATA.shape[2], 1))

train_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE)
val_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE)
test_gen = test_datagen.flow(TEST_DATA)

# setup callbacks
name = "eth_emotion_model.h5"
callbacks = create_callbacks(BEST_MODEL_PATH + name, "loss", "min", 3)

# build model
resnet_base = resnet_50(INPUT_SHAPE)
fully_connected = fully_connected(NUM_CLASSES)
model = build_model(resnet_base, fully_connected)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# evaluate model
