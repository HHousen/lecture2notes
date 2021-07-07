import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import nn


def convert_relu_to_mish(model):
    """Find all of the ``nn.ReLU`` activation functions in ``model`` and replace them with mish."""
    from mish import mish

    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, mish(inplace=True))
        else:
            convert_relu_to_mish(child)


def plot_confusion_matrix(
    y_pred,
    y_true,
    classes,
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
    save_path=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return ax
