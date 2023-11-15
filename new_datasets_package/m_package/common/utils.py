from matplotlib import ticker
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle 
from sklearn.metrics import confusion_matrix

def plot_history(history, valid, train, path, name):
    fig, ax = plt.subplots(1,2)
    tick = max(len(history) // 5, 1)

    #model loss
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(tick))
    ax[0].plot(history)
    ax[0].title.set_text('model loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')

    #auc
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(tick))
    ax[1].plot(train, label="train AUC")
    ax[1].plot(valid, label="valid AUC")
    ax[1].title.set_text('metric AUC')
    ax[1].set_ylabel('AUC')
    ax[1].set_xlabel('epoch')
    ax[1].legend()

    fig.tight_layout()
    plt.savefig(f'{path}/{name}_auc_and_loss.png', bbox_inches='tight') 


def plot_loss(history, path, name):
    fig, ax = plt.subplots( )
    tick = max(len(history) // 5, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick))
    ax.plot(history)
    ax.title.set_text('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    plt.savefig(f'{path}/{name}_loss.png', bbox_inches='tight') 



def saving_results(results, name):
    with open(f'Results/{name}.txt','wb') as f:
        pickle.dump(results, f)


def save_model(model, name):
    model.save(f'Models/{name}.keras')


def conf_matrix(model_name, test_dataset):
    fl = False
    y_pred, y_test = None, None
    loaded_model = tf.keras.models.load_model(f"Models/{model_name}.keras") #load model with current name
    for x_batch_test, y_batch_test in test_dataset:
        label_proba = loaded_model(x_batch_test, training=False)
        if fl:
            y_pred, y_test = np.concatenate((y_pred, label_proba.numpy()), axis=0), np.concatenate((y_test, y_batch_test.numpy()), axis=0)
        else:
            y_pred, y_test = label_proba.numpy(), y_batch_test.numpy()
            fl = True

    y_test_lab = np.argmax(y_test, axis=1) 
    y_pred_lab = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test_lab, y_pred_lab)
    with open(f'Results_matrix/matrix_{model_name}.txt', 'wb') as f:
        pickle.dump(cm, f)

