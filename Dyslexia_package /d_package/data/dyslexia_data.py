import os
import re
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple
from sklearn import preprocessing
from collections import defaultdict
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold, train_test_split


class DyslexiaVizualization:
    def __init__(self, path: Path = Path("../Datasets"), dataset_name: str = "Fixation_report.xlsx", sheet_name: list = [0,1,2]) -> None:
        self.path = path
        self.shape = [600,1500]  # original shape 
        self.norm_shape = [300, 120]  # to resize 
        self.sheet_names = sheet_name

        self.fixation = pd.ExcelFile(os.path.join(self.path, dataset_name))
        self.fix_data = None
        self.seq_length = 1209  # longest video length
        self.video_length = 2

        self.cols = [(102,102,255), (102,178,255), (102, 255, 178), (178, 255, 102), 
                (255,255,102), (255,102,102), (255, 102, 178), (255,102, 255), (178,102,255)]
        
        self.labels = []
        self.features = []


    def __Fixation_dataset(self):
        fix_xl = pd.read_excel(self.fixation, sheet_name=self.sheet_names)
        self.fix_data = pd.concat([fix_xl[i] for i in fix_xl.keys()], ignore_index=True, sort=False)

    def __id_getter(self):
        return self.fix_data['SubjectID'].unique()

    def size_change_video_creation(self, norm_size: int = 15, shift:int = 300, description: bool = False, padding: bool = True):
        image = np.zeros([self.shape[0],self.shape[1],3],dtype=np.uint8)
        image.fill(255)

        self.__Fixation_dataset()

        fix_norm = self.fix_data.copy()
        x = fix_norm['FIX_DURATION'].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        fix_norm['FIX_DURATION'] = (x_scaled * norm_size) + 1 

        ids = self.__id_getter()
        labels = []

        for id in ids:
            frames_l = []
            df_id = fix_norm[fix_norm['SubjectID'] == id]
            num = 0
            for _, row in df_id.iterrows():
                image_copy = image.copy()
                if description:
                    cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X']), int(row['FIX_Y'] - shift))
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), self.cols[row['Word_Number'] - 1], -1)
                num += 1
                #frame normalization
                resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                normalized_frame = resized_frame / 255
                frames_l.append(normalized_frame)
            if padding:
                image_copy = image.copy()
                while num < self.seq_length:
                    resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                    normalized_frame = resized_frame / 255
                    frames_l.append(normalized_frame)
                    num += 1
            self.features.append(frames_l)
            labels.append(df_id['Group'].unique()[0] - 1)
            self.labels = to_categorical(labels)

    def time_change_video_creation(self, framerate: int = 24, shift:int = 300,  description: bool = False, padding: bool = True):
        image = np.zeros([self.shape[0],self.shape[1],3],dtype=np.uint8)
        image.fill(255)

        self.__Fixation_dataset()
        ids = self.__id_getter()
        labels = []

        for id in ids:
            frames_l = []
            df_id = self.fix_data[self.fix_data['SubjectID'] == id]
            num = 0
            for _, row in df_id.iterrows():
                fr_amount = int(np.rint(row['FIX_DURATION']/(1000/framerate)))
                for amount in range(fr_amount):
                    image_copy = image.copy()
                    if description:
                        cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                        cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    center_coordinates = (int(row['FIX_X']), int(row['FIX_Y'] - shift))
                    cv.circle(image_copy, center_coordinates, 10, self.cols[row['Word_Number'] - 1], -1)
                    num += 1
                    #frame normalization
                    resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                    normalized_frame = resized_frame / 255
                    frames_l.append(normalized_frame)
            if padding:
                image_copy = image.copy()
                while num < self.seq_length:
                    resized_frame = cv.resize(image_copy, (self.norm_shape[0], self.norm_shape[1]))
                    normalized_frame = resized_frame / 255
                    frames_l.append(normalized_frame)
                    num += 1
            self.features.append(frames_l)
            labels.append(df_id['Group'].unique()[0] - 1)
            self.labels = to_categorical(labels)

    def get_datas(self, type: str = "by_size"):
        self.features, self.labels = [], []
        if type == "by_size":
            self.size_change_video_creation()
        elif type == "by_time":
            self.time_change_video_creation()

        self.features = np.asarray(self.features)
        self.labels = np.array(self.labels)
        return self.features, self.labels