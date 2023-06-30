import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import preprocessing

global cols, seq_length
cols = [(102,102,255), (102,178,255), (102, 255, 178), (178, 255, 102), 
                (255,255,102), (255,102,102), (255, 102, 178), (255,102, 255), (178,102,255)]
seq_length = 1208


class Visualization:
    def __init__(self, data, ids, shape) -> None:
        self.data = data
        self.id = ids
        self.shape = shape

    def __save(self, filepath: str, image, id):
        global_folder = '/creations/'
        new_folder = str(id) + '/'
        filename = id + filepath
        abpath = os.path.abspath(__file__)
        path = os.path.split(abpath)[0]
        directory = path + global_folder + new_folder

        if not os.path.exists(directory):
            os.makedirs(directory)

        cv.imwrite(directory+filename, image)

    def __ffmpeg_creation(self, id, framerate, name):
        abpath = os.path.abspath(__file__)
        path = os.path.split(abpath)[0] + "/creations"

        video_directory = path + "/videos"

        if not os.path.exists(video_directory):
            os.makedirs(video_directory)

        rez = sorted(os.listdir(path))
        for n, item in enumerate(rez):
            if item != '.DS_Store' and item != 'videos':
                new_dir = path + '/' + item
                os.chdir(new_dir)
                os.system(f"ffmpeg -framerate {framerate} -y -i {id}im.%07d.jpg {id}video{name}.mp4")
                os.replace(new_dir + f"/{id}video{name}.mp4" , video_directory + f"/{id}video{name}.mp4")
                rez_n = sorted(os.listdir(new_dir))
                for n, item_ in enumerate(rez_n):
                    os.remove(item_)
                os.chdir(path)
                os.rmdir(new_dir)
                
                
    def real_time_video_creation(self, shift:int = 300, framerate: int = 24, description: bool = True):
        image = np.zeros([self.shape[0],self.shape[1],3],dtype=np.uint8)
        image.fill(255)

        for id in self.id:
            df_id = self.data[self.data['SubjectID'] == id]
            num = 0
            for iter, row in df_id.iterrows():
                fr_amount = int(np.rint(row['FIX_DURATION']/(1000/framerate)))
                for amount in range(fr_amount):
                    image_copy = image.copy()
                    if description:
                        cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                        cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    center_coordinates = (int(row['FIX_X']), int(row['FIX_Y'] - shift))
                    cv.circle(image_copy, center_coordinates, 10, cols[row['Word_Number'] - 1], -1)
                    self.__save('im.%.7d.jpg'%(num), image_copy, id)
                    num += 1
            self.__ffmpeg_creation(id, framerate, "_by_time")


    def size_change_video_creation(self, norm_size: int = 15, shift:int = 300, framerate: int = 6, description: bool = True, padding: bool = True):
        image = np.zeros([self.shape[0],self.shape[1],3],dtype=np.uint8)
        image.fill(255)

        fix_norm = self.data.copy()
        x = fix_norm['FIX_DURATION'].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        fix_norm['FIX_DURATION'] = (x_scaled * norm_size) + 1 

        for id in self.id:
            df_id = fix_norm[fix_norm['SubjectID'] == id]
            num = 0
            for iter, row in df_id.iterrows():
                image_copy = image.copy()
                if description:
                    cv.putText(image_copy, f"Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                    cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X']), int(row['FIX_Y'] - shift))
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), cols[row['Word_Number'] - 1], -1)
                self.__save('im.%.7d.jpg'%num, image_copy, id)
                num += 1
            if padding:
                image_copy = image.copy()
                while num < seq_length:
                    self.__save('im.%.7d.jpg'%num, image_copy, id)
                    num += 1
            self.__ffmpeg_creation(id, framerate, "_by_size")


    def put_together(self, norm_size: int = 15, framerate: int = 6):
        image = np.zeros([self.shape[0],self.shape[1],3],dtype=np.uint8)
        image.fill(255)

        fix_norm = self.data.copy()
        x = fix_norm['FIX_DURATION'].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        fix_norm['FIX_DURATION'] = (x_scaled * norm_size) + 1 
        df_id1 = fix_norm[fix_norm['SubjectID'] == self.id[0]]
        df_id2 = fix_norm[fix_norm['SubjectID'] == self.id[1]]
        max_iters = max(len(list(df_id1['FIX_X'])), len(list(df_id2['FIX_X'])))

        for i in range(0, max_iters):
            image_copy = image.copy()
            if len(list(df_id1['FIX_X'])) > i: 
                row = df_id1.iloc[i]
                cv.putText(image_copy, f"{self.id[0]} Sentence ID: {row['Sentence_ID']}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X']), int(row['FIX_Y'] - 150))
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), cols[row['Word_Number'] - 1], -1)
            if len(list(df_id2['FIX_X'])) > i: 
                row = df_id2.iloc[i]
                cv.putText(image_copy, f"{self.id[1]} Sentence ID: {row['Sentence_ID']}", (50, 500), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                cv.putText(image_copy, f"Word Number: {row['Word_Number']}", (50, 550), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                center_coordinates = (int(row['FIX_X']), int(row['FIX_Y'] - 250))
                cv.circle(image_copy, center_coordinates, int(row['FIX_DURATION']), cols[row['Word_Number'] - 1], -1)
            self.__save('im.%.7d.jpg'%(i), image_copy, "asd17")
        self.__ffmpeg_creation("asd17", framerate, "_put_together")

