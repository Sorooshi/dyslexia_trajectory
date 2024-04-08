from matplotlib import ticker
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_history(history, valid, train, path, name, val_history = None):
    fig, ax = plt.subplots(1,2)
    tick = max(len(history) // 5, 1)

    #model loss
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(tick))
    ax[0].plot(history, label="train loss")
    if val_history is not None:
        ax[0].plot(val_history, label="valid loss")
    ax[0].title.set_text('model loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()

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


def plot_history_loss_final(loss,val_loss, path, name): #loss, val_auc, auc, val_loss
    fig, ax = plt.subplots(figsize=(12, 6))  
    tick = max(len(loss[0]) // 5, 1)

    mean_loss = np.mean(np.array(loss), axis=0) 
    std_loss = np.std(np.array(loss), axis=0) 

    mean_val_loss = np.mean(np.array(val_loss), axis=0) 
    std_val_loss = np.std(np.array(val_loss), axis=0) 

    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick)) 
    ax.plot(mean_loss, label="train loss", linewidth=2, color='green')
    ax.fill_between(range(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, color='green', alpha=0.2)
 
    ax.plot(mean_val_loss, label="validation loss", linewidth=2, color='olivedrab')
    ax.fill_between(range(len(mean_val_loss)), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color='olivedrab', alpha=0.2)

    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(f'{path}/{name}_LOSS_final.png', bbox_inches='tight')


def plot_history_metric_final(auc, val_auc, path, name): #loss, val_auc, auc, val_loss
    fig, ax = plt.subplots(figsize=(12, 6))  
    tick = max(len(auc[0]) // 5, 1)

    mean_auc = np.mean(np.array(auc), axis=0) 
    std_auc = np.std(np.array(auc), axis=0) 

    mean_val_auc = np.mean(np.array(val_auc), axis=0) 
    std_val_auc = np.std(np.array(val_auc), axis=0) 

    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick)) 
    ax.plot(mean_auc, label="train AUC", linewidth=2, color='green')
    ax.fill_between(range(len(mean_auc)), mean_auc - std_auc, mean_auc + std_auc, color='green', alpha=0.2)
 
    ax.plot(mean_val_auc, label="validation AUC", linewidth=2, color='olivedrab')
    ax.fill_between(range(len(mean_val_auc)), mean_val_auc - std_val_auc, mean_val_auc + std_val_auc, color='olivedrab', alpha=0.2)

    ax.set_ylabel('AUC')
    ax.set_xlabel('Epoch')
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(f'{path}/{name}_AUC_final.png', bbox_inches='tight')



def plot_loss(history, path, name):
    fig, ax = plt.subplots( )
    tick = max(len(history) // 5, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick))
    ax.plot(history)
    ax.title.set_text('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    plt.savefig(f'{path}/{name}_loss.png', bbox_inches='tight') 


def GAN_plot(history, path, name):
    fig, ax = plt.subplots( )
    tick = max(len(history.history["d_loss"]) // 5, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick))
    ax.plot(history.history['d_loss'], label='d_loss')
    ax.plot(history.history['g_loss'], label='g_loss')
    ax.title.set_text('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend()
    plt.savefig(f'{path}/{name}_loss.png', bbox_inches='tight')


def saving_results(results, name):
    with open(f'Results/{name}.txt','wb') as f:
        pickle.dump(results, f)


def save_model(model, name):
    model.save(f'Models/{name}.keras')


def make_prediction(model, test_dataset):
    fl = False
    y_pred, y_test = None, None
    loaded_model = model
    for x_batch_test, y_batch_test in test_dataset:
        label_proba = loaded_model(x_batch_test, training=False)
        if fl:
            y_pred, y_test = np.concatenate((y_pred, label_proba.numpy()), axis=0), np.concatenate((y_test, y_batch_test.numpy()), axis=0)
        else:
            y_pred, y_test = label_proba.numpy(), y_batch_test.numpy()
            fl = True
    return np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)

def conf_matrix(y_pred_lab, y_test_lab, model_name):
    n_experiments = len(y_pred_lab)
    print(n_experiments)
    cms = []
    for i in range(n_experiments):
        cm = confusion_matrix(y_test_lab[i], y_pred_lab[i])
        cms.append(cm)
    cms = np.array(cms)

    mean_cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))  
    cax = ax.matshow(mean_cm, cmap="YlGn")  
    fig.colorbar(cax)
    
    for (i, j), val in np.ndenumerate(mean_cm):
        ax.text(j, i, f'{val:.2f}\n±{std_cm[i,j]:.2f}', ha='center', va='center', color='black')
    
    ax.set_xlabel('Predicted labels', fontsize=14)
    ax.set_ylabel('True labels', fontsize=14)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Dyslexic', 'Dyslexic'], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Non-Dyslexic', 'Dyslexic'], fontsize=12)
    fig.tight_layout()
    plt.savefig(f'Matrices_pictures/matrix_{model_name}.png',  bbox_inches='tight')
