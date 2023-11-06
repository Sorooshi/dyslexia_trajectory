import argparse
import pickle
import os
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split

from m_package.models.Conv_LSTM_grad import build, ConvLSTM
from m_package.data.creartion import DyslexiaVizualization
from m_package.common.metrics import metrics_per_fold, resulting
from m_package.common.metrics_tr import metrics_per_fold_tr, counting
from m_package.common.utils import plot_history, plot_loss, saving_results, save_model

def args_parser(arguments):

    _run = arguments.run
    _epoch_num = arguments.epoch_num
    _data_name = arguments.data_name.lower()
    _model_name = arguments.model_name.lower()

    return  _run, _epoch_num, _data_name, _model_name

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run", type=int, default=2,
        help="Run the model or load the saved"
             " 0 is for creating the datasets"
             " 1 is for tuning the number of epochs"
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
    )

    parser.add_argument(
         "--model_name", type=str, default="conv_grad",
        help="Model's name"
            "conv_grad is for convolutional neural network"
    )

    args = parser.parse_args()

    run, epoch_num, data_name, model_name  = args_parser(arguments=args)

    print(
        "configuration: \n",
        "  Model:", model_name, "\n",
        "  data_name:", data_name, "\n",
        "  run:", run, "\n",
        "  epochs:", epoch_num,"\n"
    )

    path = Path("Datasets")

    if data_name == "_by_size" and run > 0:
        with open(os.path.join(path, 'X_by_size.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, 'y_by_size.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 16, 64]
        print("by_size dataset has been loaded")
    elif data_name == "_traj" and run > 0:
        with open(os.path.join(path, 'X_traj.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, 'y_traj.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 16, 64]
        print("_traj dataset has been loaded")
    elif data_name == "_huddled" and run > 0:
        with open(os.path.join(path, 'X_huddled.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, 'y_huddled.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 32, 64]
        print("_huddled dataset has been loaded")

    if run == 0:
        #huddled
        dataset_creator_huddled = DyslexiaVizualization([32, 64], dataset_name="Fixation_cutted_frames.csv", path="Datasets", file_format="csv")
        X_h, y_h = dataset_creator_huddled.get_datas("huddle")

        with open(os.path.join(path, 'X_huddled.txt'),'wb') as f:
            pickle.dump(X_h, f)

        with open(os.path.join(path, 'y_huddled.txt'),'wb') as f:
            pickle.dump(y_h, f)
        print("Huddled dataset has been created\n")

        #traj
        dataset_creator_traj = DyslexiaVizualization([16, 64], dataset_name="Fixation_cutted_frames.csv", path="Datasets", file_format="csv")
        X_t, y_t = dataset_creator_traj.get_datas("traj")

        with open(os.path.join(path, 'X_traj.txt'),'wb') as f:
            pickle.dump(X_t, f)

        with open(os.path.join(path, 'y_traj.txt'),'wb') as f:
            pickle.dump(y_t, f)
        print("Trajectory dataset has been created\n")
        
        #size
        dataset_creator_size = DyslexiaVizualization([16, 64], dataset_name="Fixation_cutted_frames.csv", path="Datasets", file_format="csv")
        X_s, y_s = dataset_creator_size.get_datas("by_size")

        with open(os.path.join(path, 'X_by_size.txt'),'wb') as f:
            pickle.dump(X_s, f)

        with open(os.path.join(path, 'y_by_size.txt'),'wb') as f:
            pickle.dump(y_s, f)
        print("By size dataset has been created\n")

    def split_data(X, y):
        batch_size = 16
        #labels: 0 -> norm; 1 -> risk; 2-> dyslexia
        X_train, X_valt, y_train, y_valt = train_test_split(X, y, test_size=0.35, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_valt, y_valt, test_size=0.5, stratify=y_valt)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

        return train_dataset, val_dataset, test_dataset


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


    if model_name == "conv_grad" and run > 0:
        if run == 1: # tune the number of epoch (done)
            #creating the datasets
            train_dataset, val_dataset, test_dataset = split_data(X_data, y_data) 

            #build and train model on huge number of epochs
            model = build('relu', size)

            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

            train_metric = tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=3)
            val_metric = tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=3)

            conv_model = ConvLSTM(model, optimizer, loss_fn, train_metric, val_metric)
            conv_model.fit(epoch_num, train_dataset, val_dataset)

            path = "Figures"
            name = str(epoch_num) + data_name

            loss = conv_model.loss_per_training
            valid_auc_ = conv_model.valid_auc
            train_auc_ = conv_model.training_auc

            plot_history(loss, valid_auc_, train_auc_, path, name)
            plot_loss(loss, path, name)

            
        elif run == 2: #running the model for k times (k == 5) (done)
            # before training params

            metrics_results = {
                "auc_roc" : [],
                "accuracy" : [],
                "precision": [],
                "recall": [],
                "f1": []
            }

            metrics_results_tr = {
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
                model = build('relu', size)

                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

                train_metric = tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=3)
                val_metric = tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=3)

                conv_model = ConvLSTM(model, optimizer, loss_fn, train_metric, val_metric)
                conv_model.fit(epoch_num, train_dataset, val_dataset)

                model_trained = conv_model.ret() 

                #calc metrics 
                metrics_results = metrics_per_fold(model_trained, test_dataset, metrics_results) # on full test set
                metrics_results_tr = metrics_per_fold_tr(model_trained, test_dataset, metrics_results_tr) #calc by batch

            #calc and save results per all folds
            final_results = resulting(metrics_results)
            final_results_tr = counting(metrics_results_tr)

            saving_results(final_results, f"{epoch_num}{data_name}_full")
            saving_results(final_results_tr, f"{epoch_num}{data_name}_per_barches")

            save_model(model_trained, f"model_{epoch_num}{data_name}")