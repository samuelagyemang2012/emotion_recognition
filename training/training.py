import pandas as pd
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from evaluation.evaluation import *
from _models.model import *

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

print("Loading training data")
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
print("Normalizing data")
TRAIN_DATA = np.array(TRAIN_DATA)
TEST_DATA = np.array(TEST_DATA)

TRAIN_DATA = TRAIN_DATA.astype('float32')
TEST_DATA = TEST_DATA.astype('float32')

# one-hot encode labels
print("One-hot encoding labels")
TRAIN_LABELS = to_categorical(train_labels)
TEST_LABELS = to_categorical(test_labels)

# data augmentation
print("Augmenting data")
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

train_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='training')
val_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='validation')
test_gen = test_datagen.flow(TEST_DATA)

# setup callbacks
print("Setting up callbacks")
file_ext = ".h5"
name = ETH + "_emotion_model"
callbacks = create_callbacks(BEST_MODEL_PATH + name + file_ext, "loss", "min", 3)

# build model
print("Building model")
input_tensor = Input(shape=(100, 100, 1))
resnet_base = resnet_50(input_tensor, INPUT_SHAPE, None)
fully_connected = fully_connected(NUM_CLASSES)
model = build_model(resnet_base, fully_connected)
opt = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

# train model
print("Training model")
history = model.fit(train_gen,
                    validation_data=val_gen,
                    callbacks=callbacks,
                    epochs=1)

# evaluate model
print("Evaluating model")
acc = model.evaluate(TEST_DATA, TEST_LABELS, batch_size=BATCH_SIZE)
preds = model.predict(TEST_DATA, verbose=0)
preds = np.argmax(preds, axis=1)
model_loss_path = "../graphs/" + name + "_loss.png"
model_acc_path = "../graphs/" + name + "_acc.png"
acc_loss_graphs_to_file(name, history, ['train', 'val'], 'upper left', model_loss_path, model_acc_path)
model_metrics_path = "../results/" + name + "_metrics.txt"
metrics_to_file(name, model_metrics_path, TEST_LABELS, preds, ['angry', 'happy', 'neutral', 'sad'], acc)

print("Done!")
