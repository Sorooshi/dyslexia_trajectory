import numpy as np
import pandas as pd
import os
import tensorflow as tf



def splitting_window(features, targets, ages, n_steps):
    x_seq, y_seq, age_seq = [], [], []
    for i in range(len(features)):
        end_idx = i + n_steps
        if end_idx >= len(features):
            break
        x_seq.append(features[i:end_idx, :])
        y_seq.append(targets[end_idx])
        age_seq.append(ages[end_idx])

    return np.asarray(x_seq), np.asarray(y_seq), np.asarray(age_seq)

def window_dataset_creation(n_steps, path, dataset_name):
    fixation = os.path.join(path, dataset_name)
    data = pd.read_csv(fixation)

    df_id = data[data['SubjectID'] == data["SubjectID"].unique()[0]]
    age_train = df_id.loc[:, ["Age"]]
    y_train = df_id.loc[:, ["Group"]]
    x_train = df_id.loc[:, ["FIX_X", "FIX_Y", "FIX_DURATION"]]
    x_train_windowed_final, y_train_windowed_final, age_final = splitting_window(x_train.values, y_train.values, age_train.values, n_steps)


    for ids in data["SubjectID"].unique()[1:]:
        df_id = data[data['SubjectID'] == ids]
        age_train = df_id.loc[:, ["Age"]]
        y_train = df_id.loc[:, ["Group"]]
        x_train = df_id.loc[:, ["FIX_X", "FIX_Y", "FIX_DURATION"]]
        x_train_windowed, y_train_windowed, age_windowed = splitting_window(x_train.values, y_train.values, age_train.values, n_steps)
        x_train_windowed_final = np.concatenate((x_train_windowed_final, x_train_windowed))
        y_train_windowed_final = np.concatenate((y_train_windowed_final, y_train_windowed))
        age_final = np.concatenate((age_final, age_windowed))

    y_train_windowed_final = y_train_windowed_final.reshape(-1) - 1
    y_train_windowed_final  = tf.keras.utils.to_categorical(y_train_windowed_final )
    y_train_windowed_final  = np.array(y_train_windowed_final)

    return x_train_windowed_final, y_train_windowed_final, age_final