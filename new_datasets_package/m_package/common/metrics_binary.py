import numpy as np
from sklearn.metrics import *

def AUC(y_test_hot, y_pred_proba):
    return roc_auc_score(y_test_hot, y_pred_proba)

def ACC(y_test_labels, y_pred_labels):
    return accuracy_score(y_test_labels, y_pred_labels)

def PR(y_test_labels, y_pred_labels):
    return precision_score(y_test_labels, y_pred_labels)

def RECC(y_test_labels, y_pred_labels):
    return recall_score(y_test_labels, y_pred_labels)

def F1(y_test_labels, y_pred_labels):
    return f1_score(y_test_labels, y_pred_labels)


def metrics_per_fold_binary(model, test_dataset, metrics_dict):
    fl = False
    y_pred, y_test = None, None

    for x_batch_test, y_batch_test in test_dataset:
        label_proba = model(x_batch_test, training=False)

        if fl:
            y_pred, y_test = np.concatenate((y_pred, label_proba.numpy()), axis=0), np.concatenate((y_test, y_batch_test.numpy()), axis=0)
        else:
            y_pred, y_test = label_proba.numpy(), y_batch_test.numpy()
            fl = True

    #labels creating
    y_test_lab = np.argmax(y_test, axis=1) 
    y_pred_lab = np.argmax(y_pred, axis=1)

    funcs = [AUC, ACC, PR, RECC, F1] #AUC -> proba; else -> labels
    
    for (key, func) in zip(metrics_dict.keys(), funcs):
        if key == "auc_roc":
            metrics_dict[key].append(round(func(y_test, y_pred), 4)) #proba
        else:
            metrics_dict[key].append(round(func(y_test_lab, y_pred_lab), 4)) #labels
    return metrics_dict


def resulting_binary(metrics_dict):
    d = {}
    for k in metrics_dict.keys():
        mean_ = round(np.nanmean(metrics_dict[k]), 4)
        std_ = round(np.std(metrics_dict[k]), 4)
        d[k] = str(mean_) + " pm " + str(std_)
    return d
