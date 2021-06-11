from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, GRU, Bidirectional, Dropout, Input, \
    Reshape, Conv2D, Conv1D, GlobalMaxPooling1D, MaxPooling2D, MaxPooling1D, Concatenate, TimeDistributed, ReLU
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, \
    accuracy_score
from tensorflow.keras import regularizers
# from keras._models import Model

from tensorflow.keras.applications import ResNet50, ResNet50V2


# Resnet 50
def resnet_50(input_tensor, input_shape, weights):
    base = ResNet50(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    base.trainable = False
    return base


# Resnet 50 v2
# input_tensor = Input(shape=(100, 100, 1))

def resnet_50_v2(input_tensor, input_shape, weights):
    base = ResNet50V2(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    base.trainable = False
    return base


# Feed forward
def fully_connected(num_classes):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_model(base, forward):
    model = Sequential()
    model.add(base)
    model.add(forward)
    return model
