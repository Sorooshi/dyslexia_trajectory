from keras import metrics

def accuracy(y_true, y_pred):
    acc = metrics.CategoricalAccuracy()
    acc.update_state(y_true=y_true, y_pred=y_pred)
    return acc.result().numpy()

def AUC(y_true, y_pred):
    auc = metrics.AUC()
    auc.update_state(y_true=y_true, y_pred=y_pred)
    return auc.result().numpy()

def F1score(y_true, y_pred):
    f1 = metrics.F1Score()
    f1.update_state(y_true=y_true, y_pred=y_pred)
    return f1.result().numpy()

def precision(y_true, y_pred):
    precision = metrics.Precision()
    precision.update_state(y_true=y_true, y_pred=y_pred)
    return precision.result().numpy()

def recall(y_true, y_pred):
    recall = metrics.Recall()
    recall.update_state(y_true=y_true, y_pred=y_pred)
    return recall.result().numpy()

def metrics_calculation(y_true, y_pred):
    acc, auc, f1, pre, rec = accuracy(y_true, y_pred), AUC(y_true, y_pred), F1score(y_true, y_pred), \
        precision(y_true, y_pred), recall(y_true, y_pred)
    return acc, auc, f1, pre, rec

def store_results(y_true, y_pred, metrics, d):
    func = [accuracy, AUC, F1score, precision, recall]
    for i in range(len(metrics)):
        d[metrics[i]].append(func[i](y_true, y_pred))
    return d

def printing_results(y_true, y_pred):
    acc, auc, f1, pre, rec = metrics_calculation(y_true, y_pred)
    print("Accuracy \tAUC \tF1score \tPrecision \tRecall")
    print("%.3f" % acc, "%.3f" % auc, "%.3f" % f1, "%.3f" % pre, "%.3f" % rec, sep="\t")
