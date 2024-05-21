import argparse
import os
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
import keras
from keras_tuner.tuners import BayesianOptimization
import keras_tuner as kt


from sklearn.model_selection import KFold, train_test_split
from skopt import BayesSearchCV
from skopt.space import  *
from sklearn.neural_network import MLPClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, roc_auc_score

from m_package.data.fused_creation import window_dataset_creation, window_dataset_creation_fused
from m_package.models.deep import lstm_1d_deep, lstm_1d_deep_fused
from m_package.common.metrics_binary import metrics_per_fold_binary, resulting_binary, linear_per_fold
from m_package.common.utils import plot_history, make_prediction, conf_matrix, plot_history_metric_final, plot_history_loss_final



# Global variables
epoch_num = 150 
n_steps = 10
batch_size = 16
dataset_name_ = "fused_data.csv"
hp_name = "1D_windowed_deep_lstm.txt"
num_tune_epochs = 50
num_trials = 20
num_points = 5

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


def split_data(X, y):
    # Multi-class labels: 0 -> norm; 1 -> risk; 2-> dyslexia
    # Binary labels: 0 -> norm; 1 -> dyslexia
    X_train, X_valt, y_train, y_valt = train_test_split(X, y, test_size=0.35, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_valt, y_valt, test_size=0.5, stratify=y_valt)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    return train_dataset, val_dataset, test_dataset


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
            "tabnet for TabNet"
            "lstm for LSTM model"
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

    if model_name == "lstm":
        fix_data, fix_y_data = window_dataset_creation_fused(n_steps, path, dataset_name_) 
        proj_name = "fused_lstm"
        if run == 1:
            train_dataset, val_dataset, test_dataset = split_data(fix_data, fix_y_data)
            tuner = BayesianOptimization(
                lstm_1d_deep_fused,
                objective=kt.Objective('val_auc', direction='max'),
                max_trials=num_trials,
                num_initial_points=num_points,
                overwrite=True,
                directory='tuning_dir',
                project_name=proj_name)
            
            tuner.search(train_dataset, epochs=num_tune_epochs, validation_data=val_dataset)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model = lstm_1d_deep_fused(best_hps)

            with open(os.path.join(path_tuner, proj_name + '.txt'),'wb') as f:
                pickle.dump(best_hps, f)

            #tune num of epochs
            train_dataset, val_dataset, test_dataset = split_data(fix_data, fix_y_data)
            best_model.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
            history = best_model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)
            
            path = "Figures"
            plot_history(history.history['loss'], history.history['val_auc'], history.history['auc'], path, proj_name, history.history['val_loss'])
        elif run == 2:
            metrics_results = {
                "auc_roc" : [],
                "accuracy" : [],
                "precision": [],
                "recall": [],
                "f1": []
                }

            with open(os.path.join(path_tuner, proj_name + '.txt'),'rb') as f:
                best_hps = pickle.load(f)

            for key in best_hps.values:
                print(key, best_hps[key])
            
            train_loss, valid_loss = [], []
            train_auc, valid_auc = [], []
            y_true_arr, y_pred_arr = [], []
            for i in range(5):
                train_dataset, val_dataset, test_dataset = split_data(fix_data, fix_y_data)
                model = lstm_1d_deep_fused(best_hps)
                model.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
                history = model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)
                train_loss.append(history.history['loss'])
                valid_loss.append(history.history['val_loss'])
                history_keys = list(history.history.keys())
                auc_str = history_keys[1]
                auc_val_str = history_keys[-1]
                train_auc.append(history.history[auc_str])
                valid_auc.append(history.history[auc_val_str])
                #make_prediction
                y_pred, y_test = make_prediction(model, test_dataset)
                y_true_arr.append(y_test)
                y_pred_arr.append(y_pred)

                #calc metrics 
                metrics_results = metrics_per_fold_binary(model, test_dataset, metrics_results)

            print(history.history.keys())
            plot_history_loss_final(train_loss, valid_loss, "Figures", proj_name)
            plot_history_metric_final(train_auc, valid_auc, "Figures", proj_name)
            final_results = resulting_binary(metrics_results)
            print(f"RESULTS: for {proj_name}\n")
            print(final_results)

            conf_matrix(y_pred_arr, y_true_arr, f"std_{proj_name}")
    else:
        if run == 1:
            #run 1 tune mlp and train lstm
            fix_data, fix_y_data, age = window_dataset_creation(n_steps, path, dataset_name_) 
            indices = np.arange(fix_data.shape[0])
            train_idx, val_idx = train_test_split(indices, test_size=0.35)

            X_train, y_train, age_t = fix_data[train_idx], fix_y_data[train_idx], age[train_idx]
            X_val, y_val = fix_data[val_idx], fix_y_data[val_idx]
            scaler_m = MinMaxScaler() 
            age_t = scaler_m.fit_transform(age_t)

            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(batch_size)

            with open(os.path.join(path_tuner, hp_name),'rb') as f:
                best_hps = pickle.load(f)
    
            model_lstm = lstm_1d_deep(best_hps)
            model_lstm.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
            model_lstm.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

            X_data = np.hstack((model_lstm.predict(X_train), age_t))

            if model_name == "mlp":
                param_space = {
                        "activation": Categorical(["logistic", "tanh", "relu"]),
                        "solver": Categorical(["lbfgs", "sgd", "adam"]),
                        "learning_rate": Categorical(["constant", "invscaling", "adaptive"])
                        }
                model_clf = MLPClassifier()
        
            elif model_name == "tabnet":
                DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
                print("Using {}".format(DEVICE))

                param_space = {
                        'n_d': Integer(8, 64),
                        'n_a': Integer(8, 64),
                        'n_steps': Integer(3, 10),
                        'gamma': Real(1.0, 2.0),
                        'lambda_sparse': Real(1e-6, 1e-3, prior='log-uniform'),
                        'mask_type': Categorical(['sparsemax', 'entmax']),
                        'n_shared': Integer(1, 5),
                        'momentum': Real(0.01, 0.4)
                        #'device_name': [DEVICE]
                }
                model_clf = TabNetClassifier(device_name=DEVICE)


            print("Tuning has begun")
            bayes_search = BayesSearchCV(model_clf, param_space, n_iter=100, cv=5, n_jobs=5, scoring=make_scorer(roc_auc_score))

            np.int = int
            bayes_search.fit(X_data , np.argmax(y_train, axis=1))

            best_estimator = bayes_search.best_estimator_
            best_params = bayes_search.best_params_

            print(f"Best parameters found:")
            print(best_params)

            # run 2 => 5 cv

            metrics_results = {
                "auc_roc" : [],
                "accuracy" : [],
                "precision": [],
                "recall": [],
                "f1": []
                }
        
            y_true_arr, y_pred_arr = [], []
            for i in range(1):
                train_idx, val_idx = train_test_split(indices, test_size=0.35)
                val_idx, test_idx = train_test_split(indices, test_size=0.5)

                X_train, y_train, age_t = fix_data[train_idx], fix_y_data[train_idx], age[train_idx]
                age_t = scaler_m.transform(age_t)
                X_val, y_val = fix_data[val_idx], fix_y_data[val_idx]
                X_test, y_test, age_test = fix_data[test_idx], np.argmax(fix_y_data[test_idx], axis=1), age[test_idx]
                age_test = scaler_m.transform(age_test)
                #retrain lstm and mlp
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

                val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                val_dataset = val_dataset.batch(batch_size)
    
                model_lstm = lstm_1d_deep(best_hps)
                model_lstm.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
                model_lstm.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

                X_train_data = np.hstack((model_lstm.predict(X_train), age_t))
                X_test_data = np.hstack((model_lstm.predict(X_test), age_test))

                model_best = model_clf.set_params(**best_params)
                model_best.fit(X_train_data, np.argmax(y_train, axis=1))
                pred_values = model_best.predict(X_test_data)
                pred_proba = model_best.predict_proba(X_test_data)[:, 1]
                y_true_arr.append(y_test)
                y_pred_arr.append(pred_values)
                metrics_results = linear_per_fold(y_test, pred_proba, pred_values, metrics_results)
        
            final_results = resulting_binary(metrics_results)
            print(final_results)
            conf_matrix(y_pred_arr, y_true_arr, f"std_{model_name}_linear_run={run}")

        elif run == 2 and model_name == "mlp":
            fix_data, fix_y_data, age = window_dataset_creation(n_steps, path, dataset_name_) 
            indices = np.arange(fix_data.shape[0])
            scaler_m = MinMaxScaler() 

            with open(os.path.join(path_tuner, hp_name),'rb') as f:
                best_hps = pickle.load(f)

            best_params = OrderedDict([('activation', 'relu'), ('learning_rate', 'constant'), ('solver', 'lbfgs')])

            print(f"Best parameters found:")
            print(best_params)

            model_clf = MLPClassifier()

            metrics_results = {
                "auc_roc" : [],
                "accuracy" : [],
                "precision": [],
                "recall": [],
                "f1": []
                }
            
            y_true_arr, y_pred_arr = [], []


            for i in range(5):
                train_idx, val_idx = train_test_split(indices, test_size=0.35)
                val_idx, test_idx = train_test_split(indices, test_size=0.5)

                X_train, y_train, age_t = fix_data[train_idx], fix_y_data[train_idx], age[train_idx]
                age_t = scaler_m.fit_transform(age_t)
                X_val, y_val = fix_data[val_idx], fix_y_data[val_idx]
                X_test, y_test, age_test = fix_data[test_idx], np.argmax(fix_y_data[test_idx], axis=1), age[test_idx]
                age_test = scaler_m.transform(age_test)
                #retrain lstm and mlp
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

                val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                val_dataset = val_dataset.batch(batch_size)
    
                model_lstm = lstm_1d_deep(best_hps)
                model_lstm.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
                model_lstm.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

                X_train_data = np.hstack((model_lstm.predict(X_train), age_t))
                X_test_data = np.hstack((model_lstm.predict(X_test), age_test))

                model_best = model_clf.set_params(**best_params)
                model_best.fit(X_train_data, np.argmax(y_train, axis=1))
                pred_values = model_best.predict(X_test_data)
                pred_proba = model_best.predict_proba(X_test_data)[:, 1]
                y_true_arr.append(y_test)
                y_pred_arr.append(pred_values)
                metrics_results = linear_per_fold(y_test, pred_proba, pred_values, metrics_results)
        
            final_results = resulting_binary(metrics_results)
            print(final_results)
            conf_matrix(y_pred_arr, y_true_arr, f"std_{model_name}_linear")
