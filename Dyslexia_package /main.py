import argparse
import numpy as np
import torch
import keras
import json

from d_package.models.ConvLSTM import ConvLstmestimator, tune_model
from d_package.data.dyslexia_data import DyslexiaVizualization
from d_package.common.metrics import store_results, counting, printing_results
from d_package.common.utils import dict_with_results

from sklearn.model_selection import train_test_split

SEED_CONSTANT = 27

def args_parser(arguments):

    _run = arguments.run
    _data_name = arguments.data_name.lower()
    _model_name = arguments.model_name.lower()
    # _target_is_org = arguments.target_is_org
    _to_exclude_at_risk = arguments.to_exclude_at_risk

    return  _run, _data_name, _model_name, _to_exclude_at_risk


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run", type=int, default=1,
        help="Run the model or load the saved"
             " 1 is for tunning and training model"
             " 2 is for training model with the best parameters"
             " 3 is for printing saved results and model parameters"
    )

    parser.add_argument(
        "--data_name", type=str, default="by_size",
        help="Dataset's name"
             "The following (lowercase) strings are supported"
             " 1) Fixation is measured by size of the dot = by_size"
             " 2) Real time fixation = by_time"
             " 3) Demo fixation is measured by size = by_size_demo"
             " 4) Demo real time fixation = by_time_demo"
    )

    parser.add_argument(
        "--model_name", type=str, default="convlstm",
        help="Model's name"
            "The following (lowercase) strings are supported"
            " 1) Convolutional LSTM model = convlstm"
            " 2) Generative Adversarial Networks = gan"
    )

    parser.add_argument(
        "--to_exclude_at_risk", type=int, default=0,
        help="Whether to exclude at-risk class from experiments or not."
             " Only setting it to one (to_exclude_at_risk=1) will exclude this class. "
    )

    args = parser.parse_args()

    run, data_name, model_name, to_exclude_at_risk = args_parser(arguments=args)
    
    print(
        "configuration: \n",
        "  Model:", model_name, "\n",
        "  data_name:", data_name, "\n",
        "  run:", run, "\n",
        "  to_exclude_at-risk:", to_exclude_at_risk, "\n",
    )


    if data_name == "by_size":
        dataset_name, type  = "Fixation_report.xlsx", "by_size"
    elif data_name == "by_size_demo":
        dataset_name, type  = "Fixation_report_demo.xlsx", "by_size"
    elif data_name == "by_time":
        dataset_name, type  = "Fixation_report.xlsx", "by_time"
    elif data_name == "by_time_demo":
        dataset_name, type  = "Fixation_report_demo.xlsx", "by_time"


    if to_exclude_at_risk == 1:
        sheet_names = [0,2]
    else:
        sheet_names = [0,1,2]


    dd = DyslexiaVizualization(dataset_name=dataset_name, sheet_name=sheet_names)
    X, y = dd.get_datas(type=type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, random_state=SEED_CONSTANT, shuffle=True)
    print("Train and test shapes: \n",
          "  Train shape:", len(y_train), "\n",
          "  Test shape is:", len(y_test), "\n")

    if model_name == "convlstm":
        if run == 1:
            hyperparams, config = tune_model(X_train=X_train, y_train=y_train)
            print("Tunning has finished, hyperparams of the best model are: \n")
            for k in hyperparams.keys():
                print("\t", k , hyperparams[k])
            print("\n")
            
            with open("Models/best_model_config.json", "w") as outfile:
                json.dump(config, outfile)

            metrics = ['acc', 'auc', 'f1', 'pre', 'rec']
            results = dict_with_results(metrics)
            for i in range(10):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                stratify=y, shuffle=True)
                model = ConvLstmestimator(config, hyperparams['epochs'], hyperparams['optimizer'], hyperparams['batch_size'])
                model.train_model(X_train, y_train, 0.2)
                print(f"Results of the best model for iter {i}\n")
                model.eval_model(X_test, y_test)
                results = store_results(y_test, model.prediction(X_test), metrics, results.copy())
            with open("Results/best_model_results.json", "w") as outfile:
                json.dump(results, outfile)
            counting(results)
        elif run == 2:
            pass
        elif run == 3:
            model = keras.models.load_model('Models/bestmodel.keras')
            y_pred = model.predict(X_test)
            printing_results(y_test, y_pred)
    elif model_name == "gan":
        if run == 1:
            pass
        elif run == 2:
            pass
        elif run == 3:
            pass