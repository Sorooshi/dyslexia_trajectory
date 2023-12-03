import argparse
import pickle
import os
import tensorflow as tf
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

from m_package.models.CE_GAN import build_generator, build_discriminator, GAN, class_expert_model
from m_package.models.Conv_LSTM_grad import build_basic, build_deep, ConvLSTM
from m_package.models.ResNet import Resnet
from m_package.data.creartion import DyslexiaVizualization
from m_package.common.metrics_multi import metrics_per_fold, resulting
from m_package.common.metrics_binary import metrics_per_fold_binary, resulting_binary
from m_package.common.utils import plot_history, plot_loss, saving_results, save_model, conf_matrix

def args_parser(arguments):

    _run = arguments.run
    _epoch_num = arguments.epoch_num
    _data_name = arguments.data_name.lower()
    _model_name = arguments.model_name.lower()
    _num_classes = arguments.num_classes

    return  _run, _epoch_num, _data_name, _model_name, _num_classes

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run", type=int, default=2,
        help="Run the model or load the saved"
             " 0 is for creating the datasets"
             " 1 is for tuning the number of epochs"
             " 2 is for training the model"
             " 3 is for drawing metrics"
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
    )

    parser.add_argument(
         "--model_name", type=str, default="conv_grad",
        help="Model's name"
            "conv_grad is for convolutional neural network"
            "conv_grad_deep is for deeper convolutional neural network"
            "resnet is for ResNet model"
            "ce_gan is for generative adversarial model"
    )

    parser.add_argument(
         "--num_classes", type=int, default=2,
        help="Number of classes"
            " 2 is for binary classification"
            " 3 is for multi-class"
    )

    args = parser.parse_args()

    run, epoch_num, data_name, model_name, num_classes = args_parser(arguments=args)
    batch_size = 16

    print(
        "configuration: \n",
        "  Model:", model_name, "\n",
        "  data_name:", data_name, "\n",
        "  run:", run, "\n",
        "  epochs:", epoch_num,"\n",
        "  num_classes", num_classes, "\n"
    )

    path = Path("Datasets")

    if data_name == "_by_size" and run > 0:
        with open(os.path.join(path, f'X_by_size_{num_classes}.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_by_size_{num_classes}.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 16, 64]
        print("by_size dataset has been loaded")
    elif data_name == "_traj" and run > 0:
        with open(os.path.join(path, f'X_traj_{num_classes}.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_traj_{num_classes}.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 16, 64]
        print("_traj dataset has been loaded")
    elif data_name == "_huddled" and run > 0:
        with open(os.path.join(path, f'X_huddled_{num_classes}.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_huddled_{num_classes}.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 32, 64]
        print("_huddled dataset has been loaded")

    if run == 0: #dataset creation
        if num_classes == 3:
            dataset_name_ = "Fixation_cutted_frames.csv"
        elif num_classes == 2:
            dataset_name_ = "Fixation_cutted_binary.csv"

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
    

    def GAN_data(data_name):
        if data_name == "_by_size":
            type_creation, size_cr = "by_size", [16, 64]
        elif data_name == "_traj":
            type_creation, size_cr = "traj", [16, 64]
        else:
            type_creation, size_cr = "huddle", [32, 64]

        #norm class
        dataset_creator_traj = DyslexiaVizualization(size_cr, dataset_name="Fixation_cutted_binary_norm.csv", path="Datasets", file_format="csv")
        X_norm, y_norm = dataset_creator_traj.get_datas(type_creation)
        y_norm = np.argmax(y_norm, axis=1)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_norm, y_norm))
        train_dataset_norm = train_dataset.shuffle(buffer_size=len(X_norm)).batch(batch_size)

        #dys class
        dataset_creator_traj = DyslexiaVizualization(size_cr, dataset_name="Fixation_cutted_binary_dys.csv", path="Datasets", file_format="csv")
        X_dys, y_dys = dataset_creator_traj.get_datas(type_creation)
        y_dys = np.argmax(y_dys, axis=1)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_dys, y_dys))
        train_dataset_dys = train_dataset.shuffle(buffer_size=len(X_dys)).batch(batch_size)

        print(X_norm.shape, X_dys.shape)

        return train_dataset_norm, train_dataset_dys
    

    def build(n_classes, model_n):
        if n_classes == 3:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            train_metric = tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=3)
            val_metric = tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=3)
            activation_dense = "softmax"
        elif n_classes == 2:
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            train_metric = tf.keras.metrics.AUC(name='auc', multi_label=False)
            val_metric = tf.keras.metrics.AUC(name='auc', multi_label=False)
            activation_dense = "sigmoid"

        if model_n == "conv_grad_deep":
            model = build_deep('relu', size, num_classes, activation_dense) 
        elif model_n == "conv_grad":
            model = build_basic('relu', size, num_classes, activation_dense)
        
        return loss_fn, train_metric, val_metric, model
    
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


    if (model_name == "conv_grad" or model_name == "conv_grad_deep") and run > 0:
        if run == 1: # tune the number of epoch (done)
            #creating the datasets
            train_dataset, val_dataset, test_dataset = split_data(X_data, y_data) 

            #build and train model on huge number of epochs
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
            loss_fn, train_metric, val_metric, model = build(num_classes, model_name)

            conv_model = ConvLSTM(model, optimizer, loss_fn, train_metric, val_metric)
            conv_model.fit(epoch_num, train_dataset, val_dataset)

            path = "Figures"

            loss = conv_model.loss_per_training
            valid_auc_ = conv_model.valid_auc
            train_auc_ = conv_model.training_auc

            model_name_save = f"{epoch_num}{data_name}_{model_name}_{num_classes}"

            plot_history(loss, valid_auc_, train_auc_, path, model_name_save)
            plot_loss(loss, path, model_name_save)

            
        elif run == 2: #running the model for k times (k == 5) (done)
            # before training params
            metrics_results = {
                "auc_roc" : [],
                "accuracy" : [],
                "precision": [],
                "recall": [],
                "f1": []
            }

            for _ in range(5):
                #creating the datasets
                train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)
                #build and train model on huge number of epochs
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
                loss_fn, train_metric, val_metric, model = build(num_classes, model_name)

                conv_model = ConvLSTM(model, optimizer, loss_fn, train_metric, val_metric)
                conv_model.fit(epoch_num, train_dataset, val_dataset)

                model_trained = conv_model.ret() 

                #calc metrics 
                if num_classes == 3:
                    metrics_results = metrics_per_fold(model_trained, test_dataset, metrics_results)
                elif num_classes == 2:
                    metrics_results = metrics_per_fold_binary(model_trained, test_dataset, metrics_results)
                

            #calc and save results per all folds
            if num_classes == 3:
                final_results = resulting(metrics_results)
            elif num_classes == 2:
                final_results = resulting_binary(metrics_results)
            print(final_results)

            model_name_save = f"{epoch_num}{data_name}_{model_name}_{num_classes}"

            saving_results(final_results, model_name_save)
            save_model(model_trained, f"model_{model_name_save}")
            
            #saving conf_matrix
            conf_matrix(model_trained, test_dataset, f"model_{model_name_save}")
    
    
    elif model_name == "resnet" and num_classes == 2:
        if run == 1: #done
            train_dataset, val_dataset, test_dataset = split_data(X_data, y_data) 
            model = Resnet(tuple(size + [1]))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
            history = model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

            path = "Figures"
            model_name_save = f"{epoch_num}{data_name}_{model_name}_{num_classes}"

            plot_history(history.history['loss'], history.history['val_auc'], history.history['auc'], path, model_name_save, history.history['val_loss'])
            plot_loss(history.history['loss'], path, model_name_save)
        elif run == 2:
            # before training params
            metrics_results = {
                "auc_roc" : [],
                "accuracy" : [],
                "precision": [],
                "recall": [],
                "f1": []
            }

            for _ in range(5):
                #creating the datasets
                train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)
                #build and train model on huge number of epochs
                model = Resnet(tuple(size + [1]))
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
                model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

                #calc metrics 
                metrics_results = metrics_per_fold_binary(model, test_dataset, metrics_results)
                print(metrics_results)

            final_results = resulting_binary(metrics_results)
            print(final_results)

            model_name_save = f"{epoch_num}{data_name}_{model_name}_{num_classes}"

            saving_results(final_results, model_name_save)
            save_model(model, f"model_{model_name_save}")
            
            #saving conf_matrix
            conf_matrix(model, test_dataset, f"model_{model_name_save}")

    elif model_name == "ce_gan" and num_classes == 2:
        if run == 1: #train gan and plot amount of epoch for ce model 
            #data for ce_model
            train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)

            #data for gan model
            norm_dataset, dys_dataset = GAN_data(data_name)

            image_shape = size + [1]
            gan_epoch = 200
            #build and train generator for norm class
            generator_norm = build_generator(image_shape=tuple(image_shape), dense_image_shape=np.prod(image_shape))
            discriminator_norm = build_discriminator(size)
            gan_norm = GAN(generator_norm, discriminator_norm, batch_size=batch_size)
            gan_norm.compile(g_opt=tf.keras.optimizers.Adam(1e-4),
                             d_opt=tf.keras.optimizers.Adam(1e-4),
                             g_loss=tf.keras.losses.BinaryCrossentropy(),
                             d_loss=tf.keras.losses.BinaryCrossentropy())

            path = "Figures"
            model_name_save_n = f"{gan_epoch}{data_name}_GAN_norm_{num_classes}"
            gan_norm.fit(norm_dataset, epochs=gan_epoch)
            gan_norm.plot_losses(path, model_name_save_n)

            #build and train generator for dys class
            generator_dys = build_generator(image_shape=tuple(image_shape), dense_image_shape=np.prod(image_shape))
            discriminator_dys = build_discriminator(size)
            gan_dys = GAN(generator_dys, discriminator_dys, batch_size=batch_size)
            gan_dys.compile(g_opt=tf.keras.optimizers.Adam(1e-4),
                             d_opt=tf.keras.optimizers.Adam(1e-4),
                             g_loss=tf.keras.losses.BinaryCrossentropy(),
                             d_loss=tf.keras.losses.BinaryCrossentropy())

            model_name_save_d = f"{gan_epoch}{data_name}_GAN_dys_{num_classes}"
            gan_norm.fit(dys_dataset, epochs=gan_epoch)
            gan_norm.plot_losses(path, model_name_save_d)

            #build and train ce_model
            weights_ce = [discriminator_norm.layers[6].get_weights(), discriminator_dys.layers[6].get_weights()]
            ce_model = class_expert_model(weights_ce, tuple(image_shape))

            ce_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=tf.keras.metrics.AUC())
            history = ce_model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

            path = "Figures"
            model_name_save = f"{epoch_num}{data_name}_{model_name}_{num_classes}"

            plot_history(history.history['loss'], history.history['val_auc'], history.history['auc'], path, model_name_save, history.history['val_loss'])
            plot_loss(history.history['loss'], path, model_name_save)

        elif run == 2:
            pass

