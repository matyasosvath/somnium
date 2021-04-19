


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


from sklearn.metrics import roc_curve


def plot_roc_curve(y_train,y_scroes, label=None):
    # Caculate
    # True positive rate (recall) and false positive rate (1- true negative rate)
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)

    # Visualize ROC Curve
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    [...] # Add axis labels and grid


