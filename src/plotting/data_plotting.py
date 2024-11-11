import matplotlib.pyplot as plt
import numpy as np


def plot_predictors(data_frame, y):
    """
    Plot a series of factors in a dataframe against an output y.

    Args:
        data_frame : Dataframe of sampled factors
        y : Vector of outputs following sampling using the data_frame
    """
    num_rows = int((np.ceil(data_frame.shape[1]) / 2)) + 1 * np.mod(
        data_frame.shape[1], 2
    )
    fig, ax = plt.subplots(num_rows, 2)

    count = 0
    for rows in range(0, num_rows):
        for cols in range(0, 2):

            ax[rows, cols].scatter(data_frame[data_frame.columns[count]], y)
            ax[rows, cols].set_title(data_frame.columns[count])
            count += 1
            if count >= len(data_frame.columns) - 2:
                break

    return ax, fig


def plot_predicted_vs_actual(y, y_hat):
    """
    Plot predicted cost y_hat against actual cost y.

    Args:
        y : Actual y
        y_hat : Predicted y
    """
    ymax = max(max(y), max(y_hat))
    fig, ax = plt.subplots(1, 1)
    ax.scatter(y, y_hat, alpha=0.5)
    ax.plot(np.linspace(1, ymax + 1, 10), np.linspace(1, ymax + 1, 10), color="red")
    ax.set_xlim([0, ymax])
    ax.set_ylim([0, ymax])
    ax.set_xlabel("Actual Cost")
    ax.set_ylabel("Predicted Cost")

    return ax, fig
