import argparse
import os
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
import keras


from sklearn.model_selection import KFold, train_test_split
from skopt import BayesSearchCV
from skopt.space import  Categorical
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from m_package.data.fused_creation import window_dataset_creation
from m_package.models.deep import lstm_1d_deep_logits
from m_package.common.metrics_binary import resulting_binary, linear_per_fold

# Global variables
epoch_num = 150 
n_steps = 10
batch_size = 16
dataset_name_ = "fused_data.csv"
hp_name = "1D_windowed_deep_lstm.txt"

def args_parser(arguments):
    
    _run = arguments.run
    _model_name = arguments.model_name.lower()

    return  _run, _model_name


def return_optimizer(best_hps):
    optimizer_name = best_hps.values["optimizer"]
    learning_rate = best_hps.values["lr"]

    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    return optimizer



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run", type=int, default=2,
        help="Run the model or load the saved"
             " 1 is for tuning"
             " 2 is for training the model"
    )

    parser.add_argument(
        "--model_name", type=str, default="conv_grad",
        help="Model's name"
            "mlp for MLPClassifier"
            
    )

    args = parser.parse_args()

    run, model_name = args_parser(arguments=args)

 
    print(
        "configuration: \n",
        "  Model:", model_name, "\n",
        "  run:", run, "\n",
    )

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:    
        try:
            tf.config.set_logical_device_configuration(
                device=gpus[0],
                logical_devices=[
                    tf.config.LogicalDeviceConfiguration(memory_limit=32000)
                ],
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("CPU")

    path = Path("Datasets")
    path_tuner = Path("Hyper_params")

    fix_data, fix_y_data, age = window_dataset_creation(n_steps, path, dataset_name_) #sentences are not separated

    indices = np.arange(fix_data.shape[0])
    train_idx, cv_idx = train_test_split(indices, test_size=0.4)
    train_lstm_idx, tune_mlp_idx = train_test_split(train_idx, test_size=0.5)

    #training LSTM with best hp
    X_data, y_data = fix_data[train_lstm_idx], fix_y_data[train_lstm_idx]
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)



    with open(os.path.join(path_tuner, hp_name),'rb') as f:
        best_hps = pickle.load(f)
    
    model_lstm = lstm_1d_deep_logits(best_hps)
    model_lstm.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
    model_lstm.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

    #tune mlp
    X_data_mlp, y_train_mlp, age_mlp = fix_data[tune_mlp_idx], fix_y_data[tune_mlp_idx], age[tune_mlp_idx]
    scaler_s, scaler_m = StandardScaler(),  MinMaxScaler()

    X_data_mlp_pred = scaler_s.fit_transform(model_lstm.predict(X_data_mlp))
    age_mlp_sc = scaler_m.fit_transform(age_mlp)

    X_train_mlp = np.hstack((X_data_mlp_pred, age_mlp_sc))

    param_space = {
            "activation": Categorical(["logistic", "tanh", "relu"]),
            "solver": Categorical(["lbfgs", "sgd", "adam"]),
            "learning_rate": Categorical(["constant", "invscaling", "adaptive"])
            }
    model_mlp = MLPClassifier()

    print("Tuning has begun")
    bayes_search = BayesSearchCV(model_mlp, param_space, n_iter=100, cv=5, n_jobs=5)

    np.int = int
    bayes_search.fit(X_train_mlp , np.argmax(y_train_mlp, axis=1))

    best_estimator = bayes_search.best_estimator_
    best_params = bayes_search.best_params_
    print(f"Best parameters found:")
    print(best_params)

    #5cv

    metrics_results = {
                "auc_roc" : [],
                "accuracy" : [],
                "precision": [],
                "recall": [],
                "f1": []
            }
    


    X_cv_mlp, y_val_mlp, age_cv_mlp = fix_data[cv_idx], fix_y_data[cv_idx], age[cv_idx]

    scaler_s, scaler_m = StandardScaler(),  MinMaxScaler()
    X_data_mlp_pred = scaler_s.fit_transform(model_lstm.predict(X_cv_mlp))
    age_mlp_sc_cv = scaler_m.fit_transform(age_cv_mlp)

    X_val_mlp = np.hstack((X_data_mlp_pred, age_mlp_sc_cv))

    kf = KFold(n_splits=5)
    for train_index , test_index in kf.split(X_val_mlp):
        X_train , X_test = X_val_mlp[train_index], X_val_mlp[test_index]
        y_train , y_test = np.argmax(y_val_mlp[train_index], axis=1),np.argmax(y_val_mlp[test_index], axis=1)

        model_best = model_mlp.set_params(**best_params)
        model_best.fit(X_train,y_train)
        pred_values = model_best.predict(X_test)
        pred_proba = model_best.predict_proba(X_test)[:, 1]
        metrics_results = linear_per_fold(y_test, pred_proba, pred_values, metrics_results)

    final_results = resulting_binary(metrics_results)
    print(final_results)

