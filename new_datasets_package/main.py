import argparse
import os
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications import ResNet50, VGG16
from keras_tuner.tuners import BayesianOptimization
import keras_tuner as kt

from sklearn.model_selection import KFold, train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

from m_package.data.creartion import DyslexiaVizualization, img_dataset_creation, window_dataset_creation, img_dataset_creation_colored
from m_package.models.basic import conv_2d_basic, lstm_1d_basic, convlstm_3d_basic_huddled, convlstm_3d_basic, conv3d_basic, conv3d_basic_huddled, convlstm_1d_basic, conv_1d_basic
from m_package.models.deep import conv_2d_deep, lstm_1d_deep, conv3d_deep, conv3d_deep_huddled, convlstm_3d_deep, convlstm_3d_deep_huddled, convlstm_1d_deep, conv_1d_deep
from m_package.models.sitted import sitted_deep_ConvLSTM1D, sitted_basic_ConvLSTM1D, sitted_basic_ConvLSTM3D, sitted_deep_ConvLSTM3D
from m_package.models.GAN import GANModel, build_discriminator, build_generator_ver1, build_generator_ver2, ImageGenerationCallback, desicion_model
from m_package.common.utils import plot_history, conf_matrix, GAN_plot, save_model
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


def hog_dataset(X, y, pixels_cell, cell_block):
    y = np.argmax(y, axis=1)
    X_hog = []
    X_features_hog = []
    for video in X:
        video_arr = []
        features_arr = []
        for frame in video:
            fd, hog_frame = hog(frame, orientations=8, pixels_per_cell=(pixels_cell, pixels_cell),
                                cells_per_block=(cell_block, cell_block), visualize=True)
            video_arr.append(hog_frame)
            features_arr.append(fd)
        X_hog.append(video_arr)
        X_features_hog.append(features_arr)
    return np.array(X_hog), np.array(X_features_hog), y


def GAN_data():
    X_dys, y_dys = img_dataset_creation(path="Datasets", dataset_name="Fixation_cutted_binary_dys.csv")
    print(X_dys.shape)
    X_dys = X_dys/255
    y_dys = np.argmax(y_dys, axis=1)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_dys))
    train_dataset_dys = train_dataset.shuffle(buffer_size=len(X_dys)).batch(batch_size)

    return train_dataset_dys


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
             "_img_fixation_c is for colored images"
             "_windowed is for window dataset"
    )

    parser.add_argument(
        "--model_name", type=str, default="conv_grad",
        help="Model's name"
            "basic is for basic neural network"
            "deep is for deeper neural network"
            "gbc for GradientBoostingClassifier"
            "rf for RandomForestClassifier"
            "mlp for MLPClassifier"
            "resnet is for pretrained ResNet50"
            "vgg is for pretrained VGG16"
            "inception is for pretrained InceptionV3"
            "sitted_basic for baby sitted basic models"
            "sitted_deep for baby sitted deep models" 
            "gan for generative gan (both versions)"
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
            "pretrained is for pretrained models"
            "ver1/ver2 for GAN versions of generator"
            "ind for Independent test"
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

    # from scratch models
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
            if type_name == "lstm" or type_name == "ind":
                if model_name == "basic":
                    model_build_func = lstm_1d_basic
                elif model_name == "deep":
                    model_build_func = lstm_1d_deep
            elif type_name == "convlstm":
                if model_name == "basic":
                    model_build_func = convlstm_1d_basic
                    print(X_data.shape)
                elif model_name == "deep":
                    model_build_func = convlstm_1d_deep
            elif type_name == "conv":
                if model_name == "basic":
                    model_build_func = conv_1d_basic
                elif model_name == "deep":
                    model_build_func = conv_1d_deep
        if type_name == "ind":
            proj_name = f'{data_rep}{data_name}_{model_name}_lstm'
            #proj_name = f'{data_rep}{data_name}_{model_name}_conv'
        else:
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
                if type_name == "ind":
                        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.35, stratify=y_data)
                        print(X_train.shape)

                        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

                        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
                        # test fom ind dataset
                        X_test, y_test = window_dataset_creation(n_steps, path, "Independent_test.csv")
                        print(X_test.shape)
                        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

                else:
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

    elif model_name == "gbc" or model_name == "rf" or model_name == "mlp":
        if model_name == "gbc":
            param_space = {
                'n_estimators': Integer(100, 1000),
                'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
                'max_depth': Integer(3, 10),
            }

            model = GradientBoostingClassifier()
        elif model_name == "rf":
            param_space = {
                'n_estimators': Integer(2000, 50000),
                'max_depth': Integer(2, 20),
                'min_samples_split': Integer(2, 50),
                'min_samples_leaf': Integer(1, 50)
            }

            model = RandomForestClassifier()
        elif model_name == "mlp":
            param_space = {
                "activation": Categorical(["logistic", "tanh", "relu"]),
                "solver": Categorical(["lbfgs", "sgd", "adam"]),
                "learning_rate": Categorical(["constant", "invscaling", "adaptive"])
            }
            model = MLPClassifier()

        if data_name[:4] != '_hog':

            X_reshaped = X_data.reshape(X_data.shape[0], -1)
            y_data = np.argmax(y_data, axis=1) 
            if model_name == "mlp":
                scaler = StandardScaler()
                X_reshaped = scaler.fit_transform(X_reshaped)
            X_train_t, X_val, y_train_t, y_val = train_test_split(X_reshaped, y_data, test_size=0.65, stratify=y_data)
            print("Tuning has begun")
            bayes_search = BayesSearchCV(model, param_space, n_iter=100, cv=5, n_jobs=5)

            np.int = int
            bayes_search.fit(X_train_t, y_train_t)

            best_estimator = bayes_search.best_estimator_
            best_params = bayes_search.best_params_
            print(f"Best parameters found for {model_name}:")
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

                model_best = model.set_params(**best_params)
                model_best.fit(X_train,y_train)
                pred_values = model_best.predict(X_test)
                pred_proba = model_best.predict_proba(X_test)[:, 1]
                metrics_results = linear_per_fold(y_test, pred_proba, pred_values, metrics_results)

            final_results = resulting_binary(metrics_results)
            print(final_results)
        else:
            for pixels_cell in range(1,4):
                for cell_block in range(1,3):
                    print(f"Pixels per cell: {pixels_cell} \t Cells per block: {cell_block}")
                    X_h, X_h_f, y = hog_dataset(X_data, y_data, pixels_cell, cell_block)
                    X_f_h_concated = np.reshape(X_h_f, (X_h_f.shape[0], X_h_f.shape[1]*X_h_f.shape[2]))
                    scaler = StandardScaler()
                    X_f_h_concated = scaler.fit_transform(X_f_h_concated)
                    X_train_t, X_val, y_train_t, y_val = train_test_split(X_f_h_concated, y, test_size=0.5, stratify=y)

                    bayes_search = BayesSearchCV(model, param_space, n_iter=100, cv=5, n_jobs=5)

                    np.int = int
                    bayes_search.fit(X_train_t, y_train_t)

                    best_estimator = bayes_search.best_estimator_
                    best_params = bayes_search.best_params_
                    print(f"Best parameters found for {model_name}:")
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

                        model_best = model.set_params(**best_params)
                        model_best.fit(X_train,y_train)
                        pred_values = model_best.predict(X_test)
                        pred_proba = model_best.predict_proba(X_test)[:, 1]
                        metrics_results = linear_per_fold(y_test, pred_proba, pred_values, metrics_results)

                    final_results = resulting_binary(metrics_results)
                    print(f"Results for {model_name} \n Best params: {best_params} \n Pixels per cell: {pixels_cell} \t Cells per block: {cell_block}")
                    print(f"Dict with Results: {final_results}")
                    print("Start another tuning \n")

    #pretrained models
    if type_name == "pretrained":
        print("in pr")
        X_data = np.repeat(X_data[..., np.newaxis], 3, -1)
        if model_name == "resnet":
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(60, 180, 3))
        elif model_name == "vgg":
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(60, 180, 3))
        
        if run == 1: #tune number of epochs
            train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(256, activation='relu')(x)
            predictions = Dense(2, activation='sigmoid')(x)  

            model = Model(inputs=base_model.input, outputs=predictions)

            for layer in base_model.layers:
                layer.trainable = False

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())

            history = model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)
            path = "Figures"
            model_name_save = f"{type_name}_{model_name}{data_name}_{epoch_num}"
            plot_history(history.history['loss'], history.history['val_auc'], history.history['auc'], path, model_name_save, history.history['val_loss'])

        elif run == 2:
            metrics_results = {
                "auc_roc" : [],
                "accuracy" : [],
                "precision": [],
                "recall": [],
                "f1": []
            }

            for _ in range(5):
                train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)

                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dense(256, activation='relu')(x)
                predictions = Dense(2, activation='sigmoid')(x)  

                model = Model(inputs=base_model.input, outputs=predictions)

                for layer in base_model.layers:
                    layer.trainable = False

                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
                model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

                #calc metrics 
                metrics_results = metrics_per_fold_binary(model, test_dataset, metrics_results)

            final_results = resulting_binary(metrics_results)
            print(f"RESULTS: for {model_name}\n")
            print(final_results)
            model_name_save = f"{data_rep}_{epoch_num}{data_name}_{model_name}_{num_classes}_{type_name}"
            conf_matrix(model, test_dataset, f"model_{model_name_save}")

    if model_name == "sitted_basic" or model_name == "sitted_deep":
        if data_rep == "1D":
            if type_name == "convlstm":
                if model_name == "sitted_basic":
                    model = sitted_basic_ConvLSTM1D()
                elif model_name == "sitted_deep":
                    model = sitted_deep_ConvLSTM1D()
        elif data_rep == "3D":
            if type_name == "convlstm":
                if model_name == "sitted_basic":
                    model = sitted_basic_ConvLSTM3D(size)
                elif model_name == "sitted_deep":
                    model = sitted_deep_ConvLSTM3D(size)
        
        if run == 1:
            train_dataset, val_dataset, test_dataset = split_data(X_data, y_data) 
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
            history = model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

            path = "Figures"

            plot_history(history.history['loss'], history.history['val_auc'], history.history['auc'], path, f"{data_rep}_{epoch_num}{data_name}_{model_name}_{type_name}", history.history['val_loss'])
        elif run == 2:
            metrics_results = {
                "auc_roc" : [],
                "accuracy" : [],
                "precision": [],
                "recall": [],
                "f1": []
            }

            for _ in range(5):
                train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
                model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

                metrics_results = metrics_per_fold_binary(model, test_dataset, metrics_results)
                print(metrics_results)

            final_results = resulting_binary(metrics_results)
            print(final_results)
            
            #saving conf_matrix
            conf_matrix(model, test_dataset, f"{data_rep}_{epoch_num}{data_name}_{model_name}_{type_name}")

    if model_name == "gan":
        if data_name == "_img_fixation_c":
            print("Start of the dataset creation")
            #dataset_name_= "Fixation_testing_2.csv"
            X_data, y_data = img_dataset_creation_colored(path="Datasets", dataset_name=dataset_name_)
            image_shape = X_data[0].shape
            color = 3
        else:
            image_shape = tuple((60,180,1))
            color = 1
        save_freq = 50
        #dyslexia
        noise_shape = 128
        dense_image_shape = np.prod(X_data[0].shape)
        train_dataset = GAN_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((X_data))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_data)).batch(batch_size)

        if run == 1:
            moments = np.linspace(0.01, 0.99, num=10)
            lr_g = 1e-5
            lr_d_int =  np.linspace(1e-10, 1e-5, num=10)
            for moment in moments:
                for lr_d in lr_d_int:
                    if type_name == "ver1":
                        generator = build_generator_ver1()
                    elif type_name == "ver2":
                        generator = build_generator_ver2()

                    model_name_save_n = f"fixed_lr_g_{type_name}_gan_model_{epoch_num}_lr_d={lr_d}_momentum={moment}"
                    print(model_name_save_n)
                    discriminator = build_discriminator(size, moment)
                    gan = GANModel(generator, discriminator)

                    generator_optimizer = keras.optimizers.SGD(lr_g)
                    discriminator_optimizer = keras.optimizers.SGD(lr_d)
                    g_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
                    d_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)

                    gan.compile(generator_optimizer, discriminator_optimizer, g_loss, d_loss)
                    hist = gan.fit(train_dataset, epochs=epoch_num) #callbacks=[ImageGenerationCallback(generator, "generated_images", save_freq, type_name)])

                    path = "Figures"
                    GAN_plot(hist, path, model_name_save_n)
                    #save_model(generator, model_name_save_n)
        elif run == 2:
            lr_g = 1e-5
            lr_d = 1e-5
            moment = 0.01


            if type_name == "ver1":
                generator = build_generator_ver1(color)
            elif type_name == "ver2":
                generator = build_generator_ver2(color)

            model_name_save_n = f"colored_images_{type_name}_gan_model_{epoch_num}_lr_d={lr_d}_momentum={moment}"
            print(model_name_save_n)
            discriminator = build_discriminator(image_shape, moment)
            gan = GANModel(generator, discriminator)

            generator_optimizer = keras.optimizers.SGD(lr_g)
            discriminator_optimizer = keras.optimizers.SGD(lr_d)
            g_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
            d_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)

            gan.compile(generator_optimizer, discriminator_optimizer, g_loss, d_loss)
            name_pic = f"GAN_colored_test"
            hist = gan.fit(train_dataset, epochs=epoch_num, callbacks=[ImageGenerationCallback(generator, "generated_images", save_freq, name_pic, color)])

            path = "Figures"
            GAN_plot(hist, path, model_name_save_n)


