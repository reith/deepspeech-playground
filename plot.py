"""
Plot training/validation curves for multiple models.
"""


from __future__ import division
from __future__ import print_function
import argparse
import matplotlib
import numpy as np
import os
matplotlib.use('Agg')  # This must be called before importing pyplot
import matplotlib.pyplot as plt


COLORS_RGB = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74),
    (152, 78, 163), (255, 127, 0), (31, 75, 90),
    (166, 86, 40), (247, 129, 191), (153, 153, 153),
    (130, 22, 99), (18, 133, 114), (43, 202, 200),
    (141, 219, 221), (45, 10, 159), (7, 78, 47),
    (249, 15, 176), (114, 227, 216), (255, 138, 125)
]

# Scale the RGB values to the [0, 1] range, which is the format
# matplotlib accepts.
colors = [(r / 255, g / 255, b / 255) for r, g, b in COLORS_RGB]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirs', nargs='+', required=True,
                        help='Directories where the model and costs are saved')
    parser.add_argument('-s', '--save_file', type=str, required=True,
                        help='Filename of the output plot')
    return parser.parse_args()


def re_range(ys, step, factor=1):
    """ Compress ys so for each step we'll have mean of that step.
    Params:
        ys: Outputs
        step: Each slice of outputs of this size will be averaged
        factor: Scale inputs by this factor
    """
    n = len(ys)
    rang = [step * (i+1) for i in range(n // step)]
    new_ys = ys[n % step:].reshape((-1, step)).mean(1)
    if n % step:
        rang.append(n)
        new_ys = np.insert(new_ys, [0], ys[:n % step].mean())
    if factor != 1:
        rang = [r*factor for r in rang]
    return new_ys, rang


def have_cost(name, npfile):
    return name in npfile and npfile[name].shape[0] > 0


def graph(dirs, save_file, average_window=100):
    """ Plot the training and validation costs and if exist, word error rate
    over iterations
    Params:
        dirs (list(str)): Directories where the model and costs are saved
        save_file (str): Filename of the output plot
        average_window (int): Window size for smoothening the graphs
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Iters')
    ax.set_ylabel('Loss')
    average_filter = np.ones(average_window) / float(average_window)

    for i, d in enumerate(dirs):
        name = os.path.basename(os.path.abspath(d))
        color = colors[i % len(colors)]
        costs = np.load(os.path.join(d, 'costs.npz'))
        train_costs = costs['train'] if 'train' in costs.files else None
        if have_cost('train', costs):
            train_costs = costs['train']
            iters = train_costs.shape[0]
            if train_costs.ndim == 1:
                train_costs = np.convolve(train_costs, average_filter,
                                          mode='valid')
            ax.plot(train_costs, color=color, label=name + '_train', lw=1.5)
        else:
            assert 'phoneme' in costs.files
        if have_cost('phoneme', costs):
            phoneme_costs = costs['phoneme']
            iters = phoneme_costs.shape[0]
            if phoneme_costs.ndim == 1:
                phoneme_costs = np.convolve(phoneme_costs, average_filter,
                                            mode='valid')
            ax.plot(phoneme_costs, color=color, label=name + '_phoneme',
                    linestyle='--', lw=1.5)
        if have_cost('validation', costs):
            valid_costs = costs['validation']
            valid_ys, valid_xs = re_range(valid_costs, 1,
                                          iters / valid_costs.shape[0])
            ax.plot(valid_xs, valid_ys, '.', color=color,
                    label=name + '_valid')
        if have_cost('wer', costs):
            wers = costs['wer']
            if wers.shape[0] == iters:
                y, x = re_range(wers * 100, average_window)
            else:
                y, x = re_range(wers * 100, 10, iters / wers.shape[0])
            ax.plot(x, y, color=color, label=name + '_wer', marker='*')
        if have_cost('val_wer', costs):
            valid_wers = costs['val_wer']
            y, x = re_range(valid_wers * 100, 1, iters / valid_wers.shape[0])
            ax.plot(x, y, color=color, label=name + '_val_wer', marker='+')
        if have_cost('val_phoneme', costs):
            val_phoneme = costs['val_phoneme']
            y, x = re_range(val_phoneme, 1,
                            iters / val_phoneme.shape[0])
            ax.plot(x, y, color=color, label=name + '_val_phoneme', marker='v')

    ax.grid(True)
    lgd = ax.legend(bbox_to_anchor=(1, 1))
    plt.savefig(save_file, bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    args = parse_args()
    graph(args.dirs, args.save_file)
