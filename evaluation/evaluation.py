import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def plot_confusion_matrix(y_test, preds, labels, title, path):
    cm = confusion_matrix(y_test.argmax(axis=1), preds)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df, annot=True, cmap="Blues")
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)
    plt.clf()


def show_classification_report(y_test, preds, names):
    res = classification_report(y_test.argmax(axis=1), preds, target_names=names)
    return "*Classification Report*" + "\n" + res + "\n"


def get_metrics(acc, y_test, preds):
    data = "Accuracy: " + str(acc[1]) + "\n"
    recall = recall_score(y_test.argmax(axis=1), preds, average='micro')
    data += 'Recall: ' + str(recall) + "\n"
    precision = precision_score(y_test.argmax(axis=1), preds, average='micro')
    data += 'Precision: ' + str(precision) + "\n"
    f1 = f1_score(y_test.argmax(axis=1), preds, average='micro')
    data += 'F1_score: ' + str(f1) + "\n"
    return data


def metrics_to_file(file_title, path, y_test, preds, labels, acc):
    data = file_title + "\n"
    # data += show_confusion_matrix(y_test, preds, labels)
    data += show_classification_report(y_test, preds, labels)
    data += get_metrics(acc, y_test, preds)
    write_to_file(data, path)


def acc_loss_graphs_to_file(model_name, history, legend, legend_loc, loss_path, acc_path):
    loss_title = model_name + " Loss Graph"
    acc_title = model_name + " Accuracy Graph"
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    draw_training_graphs(loss_title, train_loss, val_loss, "epochs", "loss", legend, legend_loc, loss_path)
    draw_training_graphs(acc_title, train_acc, val_acc, "epochs", "accuracy", legend, legend_loc, acc_path)


def draw_training_graphs(title, train_hist, val_hist, x_label, y_label, legend, loc, path):
    plt.plot(train_hist)
    plt.plot(val_hist)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend, loc=loc)
    plt.savefig(path)
    plt.clf()


def create_callbacks(): #(best_model_path, monitor, mode, patience):
    es = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00005,
        patience=11,
        verbose=1,
        restore_best_weights=True,
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1,
    )

    # mc = ModelCheckpoint(best_model_path, monitor=monitor, mode=mode, verbose=1, save_best_only=True)
    callbacks = [es, lr_scheduler]  # , mc]
    return callbacks


def write_to_file(data, path):
    f = open(path, "w")
    f.write(data)
    f.close()
