import cv2 as cv
import numpy as np
import pandas as pd
from IPython.display import display
import os
from pathlib import Path
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import io


def draw_scatter(X, Y, dur):
    fig, ax = plt.subplots(figsize=(18, 6), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_axis_off()
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    ax.invert_yaxis()
    plt.scatter(X, Y, c=dur, s=dur, cmap='gray', vmin=0)
    return fig

def fig_to_np(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv.imdecode(img_arr, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (180, 60), interpolation=cv.INTER_LINEAR)
    img[0, :] = img[img.shape[0] - 1, :] = img[:, 0] = img[:, img.shape[1] - 1] = 0
    plt.close(fig)
    return img

def img_dataset_creation(path, dataset_name):
    fixation = os.path.join(path, dataset_name)
    fix_data = pd.read_csv(fixation)

    x_train_img, y_train_img = [], []

    for ids in fix_data["SubjectID"].unique():
        X = fix_data[fix_data["SubjectID"] == ids]["FIX_X"]
        Y = fix_data[fix_data["SubjectID"] == ids]["FIX_Y"]
        dur = fix_data[fix_data["SubjectID"] == ids]["FIX_DURATION"]

        img = draw_scatter(X, Y, dur)
        np_img = fig_to_np(img)
        x_train_img.append(np_img)
        y_train_img.append(fix_data[fix_data["SubjectID"] == ids]["Group"].values[0] - 1)

    x_train_img = np.array(x_train_img)
    y_train_img = tf.keras.utils.to_categorical(y_train_img)
    y_train_img = np.array(y_train_img)

    return x_train_img, y_train_img


def splitting_window(features, targets, n_steps):
    x_seq, y_seq = [], []
    for i in range(len(features)):
        end_idx = i + n_steps
        if end_idx >= len(features):
            break
        x_seq.append(features[i:end_idx, :])
        y_seq.append(targets[end_idx])

    return np.asarray(x_seq), np.asarray(y_seq)

def window_dataset_creation(n_steps, path, dataset_name):
    fixation = os.path.join(path, dataset_name)
    data = pd.read_csv(fixation)

    df_id = data[data['SubjectID'] == data["SubjectID"].unique()[0]]
    y_train = df_id.loc[:, ["Group"]]
    x_train = df_id.loc[:, ["FIX_X", "FIX_Y", "FIX_DURATION"]]
    x_train_windowed_final, y_train_windowed_final = splitting_window(x_train.values, y_train.values, n_steps)


    for ids in data["SubjectID"].unique()[1:]:
        df_id = data[data['SubjectID'] == ids]
        y_train = df_id.loc[:, ["Group"]]
        x_train = df_id.loc[:, ["FIX_X", "FIX_Y", "FIX_DURATION"]]
        x_train_windowed, y_train_windowed = splitting_window(x_train.values, y_train.values, n_steps)
        x_train_windowed_final = np.concatenate((x_train_windowed_final, x_train_windowed))
        y_train_windowed_final = np.concatenate((y_train_windowed_final, y_train_windowed))


    y_train_windowed_final = y_train_windowed_final.reshape(-1) - 1
    y_train_windowed_final  = tf.keras.utils.to_categorical(y_train_windowed_final )
    y_train_windowed_final  = np.array(y_train_windowed_final)

    return x_train_windowed_final, y_train_windowed_final


class DyslexiaVizualization:
    def __init__(self, shape, sheet_name: list = [0,1,2], dataset_name: str = "Fixation_report.xlsx", path: Path= Path("../Datasets"), file_format: str = "xlsx") -> None:
        self.shape = shape  # original shape
        self.x_coord_norm = self.shape[1]/1500
        self.y_coord_norm = self.shape[0]/600

        self.sheet_names = sheet_name
        self.fixation = os.path.join(path, dataset_name)
        self.file_format = file_format

        self.fix_data = None
        self.seq_length = 335  # longest video length

        self.cols = [(102,102,255), (102,178,255), (102, 255, 178), (178, 255, 102), 
                (255,255,102), (255,102,102), (255, 102, 178), (255,102, 255), (178,102,255)]

        self.labels = []
        self.features = []

    def __Fixation_dataset(self):
        if self.file_format == "xlsx":
            fix_xl = pd.read_excel(self.fixation, sheet_name=self.sheet_names)
            self.fix_data = pd.concat([fix_xl[i] for i in fix_xl.keys()], ignore_index=True, sort=False)
        elif self.file_format == "csv":
            self.fix_data = pd.read_csv(self.fixation)

    def __id_getter(self):
        return self.fix_data['SubjectID'].unique()
        
    def size_change_video_creation(self, norm_size: int = 3, shift: int = 6, description: bool = False, padding: bool = False):
        image = np.zeros([self.shape[0],self.shape[1],3],dtype=np.uint8)
        image.fill(255)

        self.__Fixation_dataset()

        fix_norm = self.fix_data.copy()
        x = fix_norm['FIX_DURATION'].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        fix_norm['FIX_DURATION'] = (x_scaled * norm_size) + 2

        ids = self.__id_getter()
        labels = []

        for id in ids:
            frames_l = []
            df_id = fix_norm[fix_norm['SubjectID'] == id].iloc[-20:]
            num = 0
            for _, row in df_id.iterrows():
                image_copy = image.copy()
                if description:
                    cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X'] * self.x_coord_norm), int(row['FIX_Y']* self.y_coord_norm - shift))
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), self.cols[row['Word_Number'] - 1], -1)
                num += 1
                #frame normalization
                #resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                normalized_frame = normalized_frame / 255
                frames_l.append(normalized_frame)
            if padding:
                image_copy = image.copy()
                while num < self.seq_length:
                    #resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                    normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                    normalized_frame = normalized_frame / 255
                    frames_l.append(normalized_frame)
                    num += 1
            self.features.append(frames_l[:self.seq_length])
            labels.append(df_id['Group'].unique()[0] - 1)
            self.labels = tf.keras.utils.to_categorical(labels)

    def trajectory_creation(self, norm_size: int = 3, shift: int = 6, description: bool = False, padding: bool = False):
        image = np.zeros([self.shape[0], self.shape[1], 3],dtype=np.uint8)
        image.fill(255)

        self.__Fixation_dataset()

        fix_norm = self.fix_data.copy()
        x = fix_norm['FIX_DURATION'].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        fix_norm['FIX_DURATION'] = (x_scaled * norm_size) + 2 

        ids = self.__id_getter()
        labels = []

        for id in ids:
            frames_l = []
            df_id = fix_norm[fix_norm['SubjectID'] == id].iloc[-20:]
            num = 0
            image_copy = image.copy()
            prev_point = False
            for _, row in df_id.iterrows():
                if description:
                    cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X'] * self.x_coord_norm), int(row['FIX_Y']* self.y_coord_norm - shift))
                #print(center_coordinates)
                if prev_point:
                    cv.line(image_copy, prev_point, center_coordinates, self.cols[row['Word_Number'] - 1],1)
                    prev_point =  center_coordinates
                else:
                    prev_point = center_coordinates
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), self.cols[row['Word_Number'] - 1], -1)
                normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                normalized_frame = normalized_frame / 255
                frames_l.append(normalized_frame)
            if padding:
                image_copy = image.copy()
                while num < self.seq_length:
                    #resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                    normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                    normalized_frame = normalized_frame / 255
                    frames_l.append(normalized_frame)
                    num += 1
            self.features.append(frames_l[:self.seq_length])
            labels.append(df_id['Group'].unique()[0] - 1)
            self.labels = tf.keras.utils.to_categorical(labels)
            
    def huddled_creation(self, norm_size: int = 4, shift: int = 25, moving: int = 3, description: bool = False, padding: bool = False):
        image = np.zeros([self.shape[0], self.shape[1], 3],dtype=np.uint8)
        image.fill(255)

        self.__Fixation_dataset()

        fix_norm = self.fix_data.copy()
        x = fix_norm['FIX_DURATION'].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        fix_norm['FIX_DURATION'] = (x_scaled * norm_size) + 2 

        ids = self.__id_getter()
        labels = []

        for id in ids:
            frames_l = []
            df_id = fix_norm[fix_norm['SubjectID'] == id].iloc[-20:]
            num = 0
            image_copy = image.copy()
            for _, row in df_id.iterrows():
                if description:
                    cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X'] * self.x_coord_norm), int(row['FIX_Y']* self.y_coord_norm - (shift - moving*(row['Word_Number'] - 1))))
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), self.cols[row['Word_Number'] - 1], -1)
                normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                normalized_frame = normalized_frame / 255
                frames_l.append(normalized_frame)
            if padding:
                image_copy = image.copy()
                while num < self.seq_length:
                    #resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                    normalized_frame = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
                    normalized_frame = normalized_frame / 255
                    frames_l.append(normalized_frame)
                    num += 1
            self.features.append(frames_l[:self.seq_length])
            labels.append(df_id['Group'].unique()[0] - 1)
            self.labels = tf.keras.utils.to_categorical(labels)

    def get_datas(self, type_: str = "by_size"):
        self.features, self.labels = [], []
        if type_ == "by_size":
            self.size_change_video_creation()
        elif type_ == "traj":
            self.trajectory_creation()
        elif type_ == "huddle":
            self.huddled_creation()

        self.features = np.asarray(self.features)
        self.labels = np.array(self.labels)
        return self.features, self.labels