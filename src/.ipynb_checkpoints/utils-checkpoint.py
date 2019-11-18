import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import socket
if socket.gethostname() == 'exalearn':
    import matplotlib
    matplotlib.use('agg')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    

def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.close('all')
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes if classes!=None else np.arange(cm.shape[1]), 
           yticklabels=classes if classes!=None else np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return fig, image



def plot_conf_matrix(cm, classes, 
                     normalized=False, cl_names=None):
    '''
    function to plot confusion matrix
    '''
    plt.close('all')
    fig = plt.figure(figsize=(22,16))
    if normalized:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='0.3f', linewidths=.5, 
                    xticklabels=cl_names, yticklabels=cl_names, cmap="GnBu", 
                    annot_kws={'size': 'medium'})
    else:
        sns.heatmap(cm, annot=True, fmt='d', linewidths=.5,
                    xticklabels=cl_names, yticklabels=cl_names, cmap="GnBu", 
                    annot_kws={'size': 'x-large'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return fig, image


def normalize_target(data, scale_to=[0,1], n_feat=2):
    normed = np.zeros_like(data)
    f_min = np.min(data, axis=0)
    f_max = np.max(data, axis=0)
    for f in range(n_feat):
        normed[:,f*2] = (data[:,f*2] - f_min[f*2]) / \
                      (f_max[f*2] - f_min[f*2])
        normed[:,f*2+1] = data[:,f*2+1] / \
                          (f_max[f*2] - f_min[f*2])
    
    return normed


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())