import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, \
    accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def show_confusion_matrix(y_test, preds):
    res = confusion_matrix(y_test.argmax(axis=1), preds)
    return "*Confusion Matrix*" + "\n" + np.array_str(res) + "\n"


def show_classification_report(y_test, preds, names):
    res = classification_report(y_test.argmax(axis=1), preds, target_names=names)
    return "*Classification Report*" + "\n" + res + "\n"


def get_metrics(acc, y_test, preds):
    data = "Accuracy: " + str(acc[1]) + "\n"
    tn, fp, fn, tp = confusion_matrix(y_test.argmax(axis=1), preds).ravel()
    data += 'False positive rate: ' + str(fp / (fp + tn)) + "\n"
    data += 'False negative rate: ' + str(fn / (fn + tp)) + "\n"
    recall = tp / (tp + fn)
    data += 'Recall: ' + str(recall) + "\n"
    precision = tp / (tp + fp)
    data += 'Precision: ' + str(precision) + "\n"
    f1_score = ((2 * precision * recall) / (precision + recall))
    data += 'F1 score: ' + str(f1_score) + "\n"
    return data


def get_fpr(y_test, preds):
    tn, fp, fn, tp = confusion_matrix(y_test.argmax(axis=1), preds).ravel()
    fpr = (fp / (fp + tn))
    return fpr


def get_fnr(y_test, preds):
    tn, fp, fn, tp = confusion_matrix(y_test.argmax(axis=1), preds).ravel()
    fnr = (fn / (fn + tp))
    return fnr


def metrics_to_file(file_title, path, y_test, preds, names, acc):
    data = file_title + "\n"
    data += show_confusion_matrix(y_test, preds)
    data += show_classification_report(y_test, preds, names)
    data += get_metrics(acc, y_test, preds)
    write_to_file(data, path)


def acc_loss_graphs_to_file(model_name, history, legend, legend_loc, loss_path, acc_path):
    loss_title = model_name + " Loss Graph"
    acc_title = model_name + " Accuracy Graph"
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
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


def create_callbacks(best_model_path, monitor, mode, patience):
    es = EarlyStopping(monitor=monitor, mode=mode, verbose=1, patience=patience)
    mc = ModelCheckpoint(best_model_path, monitor=monitor, mode=mode, verbose=1, save_best_only=True)
    callbacks = [es, mc]
    return callbacks


def write_to_file(data, path):
    f = open(path, "w")
    f.write(data)
    f.close()