import numpy as np
from sklearn.metrics import *

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def AUC(y_true, y_pred):
    try:
        roc_auc_score(y_true, y_pred, average = 'macro', multi_class='ovr')
    except ValueError:
        pass
    else:
        return roc_auc_score(y_true, y_pred, multi_class='ovr', average = 'weighted')

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted')

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted')

def F1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def metrics_calculation(y_true, y_pred, y_true_proba, y_pred_proba):
    acc, auc, pre, rec, f1 = accuracy(y_true_proba, y_pred_proba), AUC(y_true, y_pred), \
        precision(y_true, y_pred), recall(y_true, y_pred), F1(y_true, y_pred)
    return acc, auc, pre, rec, f1


def store_results(y_true, y_pred, y_true_proba, y_pred_proba, metrics, d):
    func = [AUC, accuracy, precision, recall, F1]
    for i in range(len(metrics)):
        if i == 0:
            d[metrics[i]].append(func[i](y_true_proba, y_pred_proba)) 
        else:
            if len(np.unique(np.array(y_true))) > 1:
                d[metrics[i]].append(func[i](y_true, y_pred)) 


def printing_results(y_true, y_pred, y_true_proba, y_pred_proba):
    acc, auc, pre, rec, f1 = metrics_calculation(y_true, y_pred, y_true_proba, y_pred_proba)
    print("Accuracy \tAUC \tPrecision \tRecall")
    print("%.3f" % acc, "%.3f" % auc, "%.3f" % pre, "%.3f" % rec, sep="\t   ")


def counting(results):
    d = {}
    for k in results.keys():
        results[k] = [i for i in results[k] if i != None]
        mean_ = round(np.nanmean(results[k]), 4)
        std_ = round(np.std(results[k]), 4)
        d[k] = str(mean_) + " pm " + str(std_)
        print(f"{k}:")
        print(f"{mean_} \pm {std_}")
    return d

def metrics_per_fold_tr(model, test, final_dict):
    metrics_arr = ['auc_roc', 'accuracy', 'precision', 'recall', 'f1']

    metrics_dict = {
        "auc_roc" : [],
        "accuracy" : [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for x_batch_test, y_batch_test in test:
        label_proba = model(x_batch_test, training=False)
        labels_ = np.argmax(label_proba, axis=1)
        y_test_labels = np.argmax(y_batch_test, axis=1)
        store_results(y_test_labels, labels_, y_batch_test, label_proba, metrics_arr, metrics_dict)

    for k in metrics_dict.keys():
        metrics_dict[k] = [i for i in metrics_dict[k] if i != None]
        mean_ = round(np.nanmean(metrics_dict[k]), 4)
        final_dict[k].append(mean_)

    return final_dict
