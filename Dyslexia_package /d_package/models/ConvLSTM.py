import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import keras_tuner as kt
from keras_tuner import Objective
from keras_tuner import BayesianOptimization, HyperModel
import tensorflow as tf


from Dyslexia_package.d_package.common.metrics import printing_results

CLASSES_LIST = ['Dyslexia', 'Normal', 'Risk']
length = 150

class ConvLSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(ConvLSTM2D(filters=hp.Int('filters',
                                          min_value=4,
                                          max_value=16,
                                          step=4), 
                            kernel_size=(3,3), 
                            activation=hp.Choice('activation',
                                                values=['relu', 'tanh', 'sigmoid', 'exponential']), 
                            data_format='channels_last',
                            recurrent_dropout=0.2, 
                            return_sequences=True, 
                            input_shape=(length, 120, 300, 3))) 
        model.add(MaxPooling3D(pool_size=(1,2,2), padding='same', data_format='channels_last'))
        model.add(Flatten())
        model.add(Dense(len(CLASSES_LIST), activation='Softmax'))
        model.compile(loss='categorical_crossentropy', 
                            optimizer=hp.Choice('optimizer',
                                               values=['adam', 'SGD', 'rmsprop']), 
                             metrics=[keras.metrics.AUC(name='auc', multi_label=True)])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int("batch_size", 
                              min_value=1,
                              max_value=16,
                              step=2),
            epochs=hp.Int("epochs",
                          min_value=1,
                          max_value=5,
                          step=1
            ),
            **kwargs,
        )
    

def tune_model(X_train, y_train):

    tuner = kt.BayesianOptimization(
    ConvLSTMHyperModel(),
    objective=Objective(name='val_auc', direction='max'),
    max_trials=2,
    overwrite=True,
    directory="my_dir_auc",
    project_name="tune_hypermodel_auc",
    )

    print(tuner.search_space_summary())

    tuner.search(
    X_train,
    y_train,
    validation_split=0.2,
    verbose=1
    )

    best_hp = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hp)

    return best_hp.values, model.get_config()

class ConvLstmestimator:
    def __init__(self, model_config, epochs, optimizer, batch_size) -> None:
        self.model = keras.Sequential.from_config(config=model_config)
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

    def train_model(self, X_train, y_train, val_size):
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=[keras.metrics.AUC(multi_label=True)])
        
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, 
                             shuffle=True, validation_split=val_size)
        
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='valid')
        plt.legend()
        plt.savefig("Figures/history.png")

    def return_pretrained_model(self):
        return self.model
    
    def prediction(self, X_test):
        return self.model.predict(X_test)

    def eval_model(self, X_test, y_test):
        self.model.evaluate(X_test, y_test)
        printing_results(y_test, self.model.predict(X_test))