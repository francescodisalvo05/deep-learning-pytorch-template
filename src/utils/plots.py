from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import numpy as np
import random


def plot_auc(path_plots, labels, predictions):

    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    roc_auc = auc(fpr,tpr)

    print(f'AUC = {roc_auc:.2f}')

    lw = 1
    plt.figure()
    plt.plot(fpr, tpr, color='navy',
            lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#A7C7E7', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path_plots,'roc_auc.png'))
    plt.close()


def plot_sample(path_plots, X, y, filename='sample.png', label=None):

    if label:
        # to do: handle erros (e.g. label not in y)
        X = X[y == label]

    # randomize the image order
    random.shuffle(X) 
    
    # create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()

    # plot each image in a subplot
    for i, ax in enumerate(axes):
        ax.imshow(X[i], cmap='gray')
        ax.axis('off')

    # remove gaps between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # display the plot
    plt.savefig(os.path.join(path_plots,filename))
    plt.close()
