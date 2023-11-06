from matplotlib import ticker
import matplotlib.pyplot as plt
import pickle 

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

