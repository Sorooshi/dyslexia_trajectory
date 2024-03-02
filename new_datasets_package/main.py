import argparse
import pickle
import os
import keras
from pathlib import Path

import tensorflow as tf
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from kerastuner.tuners import BayesianOptimization
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import GradientBoostingClassifier
import keras_tuner as kt

from m_package.data.creartion import DyslexiaVizualization, img_dataset_creation, window_dataset_creation
from m_package.models.basic import conv_2d_basic, lstm_2d_basic, convlstm_3d_basic_huddled, convlstm_3d_basic, conv3d_basic, conv3d_basic_huddled
from m_package.models.deep import conv_2d_deep, lstm_2d_deep, conv3d_deep, conv3d_deep_huddled, convlstm_3d_deep, convlstm_3d_deep_huddled
from m_package.common.utils import plot_history, conf_matrix
from m_package.common.metrics_binary import metrics_per_fold_binary, resulting_binary, linear_per_fold


def args_parser(arguments):

    _run = arguments.run
    _epoch_num = arguments.epoch_num
    _data_name = arguments.data_name.lower()
    _model_name = arguments.model_name.lower()
    _num_classes = arguments.num_classes
    _type_name = arguments.type_name.lower()

    return  _run, _epoch_num, _data_name, _model_name, _num_classes, _type_name


#data splitting
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
             " 0 is for creating the datasets"
             " 1 is for tuning"
             " 2 is for training the model"
    )

    parser.add_argument(
        "--epoch_num", type=int, default=5,
        help="Run the model the number of epochs"
    )

    parser.add_argument(
        "--data_name", type=str, default="_huddled",
        help="Dataset's name"
             "_by_size is for By size (without trajectories- 20 frames)"
             "_traj is for By size (with trajectories - 20 frames)"
             "_huddled is for By size (huddled - 20 frames)"
             "_hog_by_size is for hog By size (without trajectories- 20 frames)"
             "_hog_traj is for hog By size (with trajectories - 20 frames)"
             "_hog_huddled is for hog By size (huddled - 20 frames)"
             "_img_fixation is for images"
             "_windowed is for window dataset"
    )

    parser.add_argument(
        "--model_name", type=str, default="conv_grad",
        help="Model's name"
            "basic is for basic neural network"
            "deep is for deeper neural network"
            "gbc for GradientBoostingClassifier"
    )

    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="Number of classes"
            " 2 is for binary classification"
            " 3 is for multi-class"
    )

    parser.add_argument(
        "--type_name", type=str, default="conv",
        help="type_name"
            "conv is for convolutional type"
            "lstm is for lstm type"
            "convlstm is for convlstm type"
    )

    args = parser.parse_args()

    run, epoch_num, data_name, model_name, num_classes, type_name = args_parser(arguments=args)

    # constants 
    batch_size = 16
    num_tune_epochs = 50
    num_trials = 20
    num_points = 5
    path_tuner = "Hyper_params"
    n_steps = 10

    #dataset representation type
    if data_name == "_by_size" or data_name == "_traj" or data_name == "_huddled":
        data_rep = "3D"
    elif  data_name == "_img_fixation":
        data_rep = "2D"
    else:
        data_rep = "1D"

    #fix name of the dataset
    if num_classes == 3:
        dataset_name_ = "Fixation_cutted_frames.csv"
    elif num_classes == 2:
        dataset_name_ = "Fixation_cutted_binary.csv"

    print(
        "configuration: \n",
        "  Model:", model_name, "\n",
        "  data_name:", data_name, "\n",
        "  run:", run, "\n",
        "  epochs:", epoch_num,"\n",
        "  num_classes", num_classes, "\n",
        "  type", type_name, "\n",
        "  representation type", data_rep, "\n",
        "  Dataset name", dataset_name_, "\n"
    )

    path = Path("Datasets")


    #dataset loading
    if (data_name == "_by_size" or data_name == "_hog_by_size") and run > 0:
        with open(os.path.join(path, f'X_by_size_{num_classes}.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_by_size_{num_classes}.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 16, 64]
        print("by_size dataset has been loaded")
    elif (data_name == "_traj" or data_name == "_hog_traj") and run > 0:
        with open(os.path.join(path, f'X_traj_{num_classes}.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_traj_{num_classes}.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 16, 64]
        print("_traj dataset has been loaded")
    elif (data_name == "_huddled" or data_name == "_hog_huddled") and run > 0:
        with open(os.path.join(path, f'X_huddled_{num_classes}.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_huddled_{num_classes}.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 32, 64]
        print("_huddled dataset has been loaded")
    elif data_name == "_img_fixation"  and run > 0:
        with open(os.path.join(path, f'X_img_{num_classes}.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_img_{num_classes}.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [60, 180]
        print("Img dataset has been loaded")
    elif data_name == "_windowed" and run > 0:
        X_data, y_data = window_dataset_creation(n_steps, path, dataset_name_)

    #dataset creation
    if run == 0: 
        print("Start of the dataset creation")
        X_img, y_img = img_dataset_creation(path="Datasets", dataset_name=dataset_name_)

        with open(os.path.join(path, f'X_img_{num_classes}.txt'),'wb') as f:
            pickle.dump(X_img, f)

        with open(os.path.join(path, f'y_img_{num_classes}.txt'),'wb') as f:
            pickle.dump(y_img, f)
        print("Img dataset has been created\n")

        #huddled
        dataset_creator_huddled = DyslexiaVizualization([32, 64], dataset_name=dataset_name_, path="Datasets", file_format="csv")
        X_h, y_h = dataset_creator_huddled.get_datas("huddle")

        with open(os.path.join(path, f'X_huddled_{num_classes}.txt'),'wb') as f:
            pickle.dump(X_h, f)

        with open(os.path.join(path, f'y_huddled_{num_classes}.txt'),'wb') as f:
            pickle.dump(y_h, f)
        print("Huddled dataset has been created\n")

        #traj
        dataset_creator_traj = DyslexiaVizualization([16, 64], dataset_name=dataset_name_, path="Datasets", file_format="csv")
        X_t, y_t = dataset_creator_traj.get_datas("traj")

        with open(os.path.join(path, f'X_traj_{num_classes}.txt'),'wb') as f:
            pickle.dump(X_t, f)

        with open(os.path.join(path, f'y_traj_{num_classes}.txt'),'wb') as f:
            pickle.dump(y_t, f)
        print("Trajectory dataset has been created\n")
        
        #size
        dataset_creator_size = DyslexiaVizualization([16, 64], dataset_name=dataset_name_, path="Datasets", file_format="csv")
        X_s, y_s = dataset_creator_size.get_datas("by_size")

        with open(os.path.join(path, f'X_by_size_{num_classes}.txt'),'wb') as f:
            pickle.dump(X_s, f)

        with open(os.path.join(path, f'y_by_size_{num_classes}.txt'),'wb') as f:
            pickle.dump(y_s, f)
        print("By size dataset has been created\n")

    if run == 1 or run == 2: #not for dataset creation and drawing matrices
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


    if model_name == "basic" or model_name == "deep":
        if data_rep == "3D":
            if model_name == "basic":
                if type_name == "conv":
                    if data_name == "_huddled":
                        model_build_func = conv3d_basic_huddled
                    else:
                        model_build_func = conv3d_basic
                elif type_name == "convlstm":
                    if data_name == "_huddled":
                        model_build_func = convlstm_3d_basic_huddled
                    else:
                        model_build_func = convlstm_3d_basic
            elif model_name == "deep":
                if type_name == "conv":
                    if data_name == "_huddled":
                        model_build_func = conv3d_deep_huddled
                    else:
                        model_build_func = conv3d_deep
                elif type_name == "convlstm":
                    if data_name == "_huddled":
                        model_build_func = convlstm_3d_deep_huddled
                    else:
                        model_build_func = convlstm_3d_deep
        elif data_rep == "2D":
            if model_name == "basic":
                model_build_func = conv_2d_basic
            elif model_name == "deep":
                model_build_func = conv_2d_deep
        elif data_rep == "1D":
            if model_name == "basic":
                model_build_func = lstm_2d_basic
            elif model_name == "deep":
                model_build_func = lstm_2d_deep
        proj_name = f'{data_rep}{data_name}_{model_name}_{type_name}'
        model_name_save = f"{data_rep}_{epoch_num}{data_name}_{model_name}_{num_classes}_{type_name}"
        if run == 1:
            train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)
            tuner = BayesianOptimization(
                model_build_func,
                objective=kt.Objective('val_auc', direction='max'),
                max_trials=num_trials,
                num_initial_points=num_points,
                overwrite=True,
                directory='tuning_dir',
                project_name=proj_name)
            
            tuner.search(train_dataset, epochs=num_tune_epochs, validation_data=val_dataset)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model = model_build_func(best_hps)

            with open(os.path.join(path_tuner, proj_name + '.txt'),'wb') as f:
                pickle.dump(best_hps, f)

            #tune num of epochs
            train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)
            best_model.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
            history = best_model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)
            
            path = "Figures"
            plot_history(history.history['loss'], history.history['val_auc'], history.history['auc'], path, model_name_save, history.history['val_loss'])
        if run == 2:
            # before training params
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

            for _ in range(5):
                #creating the datasets
                train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)
                #build and train model on huge number of epochs
                model = model_build_func(best_hps)
                model.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
                model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

                #calc metrics 
                metrics_results = metrics_per_fold_binary(model, test_dataset, metrics_results)

            final_results = resulting_binary(metrics_results)
            print(f"RESULTS: for {proj_name}\n")
            print(final_results)

            conf_matrix(model, test_dataset, f"model_{model_name_save}")

    elif model_name == "gbc":
        X_reshaped = X_data.reshape(X_data.shape[0], -1)
        y_data = np.argmax(y_data, axis=1)
        X_train_t, X_val, y_train_t, y_val = train_test_split(X_reshaped, y_data, test_size=0.65, stratify=y_data) 

        print("Tuning has begun")
        param_space = {
            'n_estimators': Integer(100, 1000),
            'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
            'max_depth': Integer(3, 10),
        }

        gbc = GradientBoostingClassifier()
        bayes_search = BayesSearchCV(gbc, param_space, n_iter=2, cv=5, n_jobs=1)

        np.int = int
        bayes_search.fit(X_train_t, y_train_t)

        best_estimator = bayes_search.best_estimator_
        best_params = bayes_search.best_params_
        print("Best parameters found:")
        print(best_params)

        metrics_results = {
            "auc_roc" : [],
            "accuracy" : [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        kf = KFold(n_splits=5)
        for train_index , test_index in kf.split(X_val):
            X_train , X_test = X_val[train_index], X_val[test_index]
            y_train , y_test = y_val[train_index] , y_val[test_index]

            model_best = GradientBoostingClassifier().set_params(**best_params)
            model_best.fit(X_train,y_train)
            pred_values = model_best.predict(X_test)
            pred_proba = model_best.predict_proba(X_test)[:, 1]
            metrics_results = linear_per_fold(y_test, pred_proba, pred_values, metrics_results)

        final_results = resulting_binary(metrics_results)
        print(final_results)
