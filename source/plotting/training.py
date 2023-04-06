import os
import matplotlib.pyplot as plt
import pandas as pd


def remove_prefix(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string
    

def create_metric_names_set(metric_names):
    metric_names = [mn for mn in metric_names if mn != 'step']
    metric_names = [remove_prefix(mn, 'val_') for mn in metric_names]
    metric_names = [remove_prefix(mn, 'train_') for mn in metric_names]
    return set(metric_names)


def plot_training_history(df_metrics, plot_dirpath, filename_format,
                          fileformat='png', show: bool = False):
    if type(fileformat) is str:
        fileformats = [fileformat]
    else:
        fileformats = fileformat

    metric_names = create_metric_names_set(df_metrics.columns.to_list())

    for metric_name in metric_names:
        metric_cols = ['train_' + metric_name, 'val_' + metric_name]
        ax = df_metrics.plot(y=metric_cols)
        ax.set_title(f'{metric_name} over training epochs')
        ax.set_ylabel(metric_name)

        fig = ax.get_figure()
        for extension in fileformats:
            plot_filename = filename_format.format(
                metric_name=metric_name,
                ext=extension)
            plot_filepath = os.path.join(plot_dirpath, plot_filename)
            fig.savefig(plot_filepath)

            if show:
                plt.show()
            plt.close(fig)
