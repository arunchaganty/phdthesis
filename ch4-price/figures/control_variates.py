#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variance plots.
"""
import pdb
import json
import sys
import gzip
from collections import namedtuple

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mp

def _output(fname, i, affix=None):
    root, extn = fname.rsplit(".", 1)
    return "{}{}_{}.{}".format(root, affix if affix else "", i, extn)


def do_command(args):
    np.random.seed(0)
    data = np.array([
        [0,   0.2,] ,
        [0.2, 0.3,] ,
        [0.4, 0.4,] ,
        [0.5, 0.3,] ,
        [0.6, 0.3,] ,
        [0.8, 0.2,] ,
        [0.9, 0.4,] ,
        [1.0, 0.6,] ,
        [1.1, 0.7,] ,
        [1.2, 0.8,] ,
        [1.4, 0.7,] ,
        [1.6, 0.6,] ,
        [1.8, 0.5,] ,
        [2.0, 0.4,] ,
        ])

    rho = 0.8
    gs_ = data.T[1] * rho + (1-rho)*0.1*np.random.randn(len(data))
    gs_ -= gs_.mean()
    xs = np.linspace(0, 2, 100)

    #fs = np.polyfit(data.T[0], data.T[1], 4) #UnivariateSpline(xs_, fs_, k=5, s=0)
    fs = UnivariateSpline(data.T[0], data.T[1], k=4, s=0)
    gs = UnivariateSpline(data.T[0] + 0.03 * np.random.randn(len(data)), gs_, k=4, s=0)
    gs, gs_ = gs(xs), gs(data.T[0])
    gs, gs_ = gs - gs_.mean(), gs_ - gs_.mean()

    data = data[1:-2]
    gs_ = gs_[1:-2]

    # just the axes
    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.xlim(0.2, 1.8)
    plt.xticks()
    plt.ylim(-0.4, 1.0)
    plt.xlabel("Input ($z$)")
    plt.plot(xs, 0 * xs, color='k')
    plt.tight_layout()
    plt.savefig(_output(args.output, 0))

    # Now with function
    plt.plot(xs, fs(xs), label="Ground truth $f(z)$")
    lgd = plt.legend(bbox_to_anchor=(0.85, -0.2))
    plt.savefig(_output(args.output, 1), bbox_extra_artists=(lgd,), bbox_inches='tight')

    # Plot bars
    plt.bar(data.T[0] - 0.01, data.T[1], width=0.02)
    lgd = plt.legend(bbox_to_anchor=(0.85, -0.2))
    plt.savefig(_output(args.output, 2), bbox_extra_artists=(lgd,), bbox_inches='tight')

    # with automatic function
    plt.plot(xs, gs, label="Automatic metric $g(z)$")
    lgd = plt.legend(bbox_to_anchor=(0.85, -0.2))
    plt.savefig(_output(args.output, 3), bbox_extra_artists=(lgd,), bbox_inches='tight')

    # with automatic function bars
    plt.bar(data.T[0] + 0.01, data.T[1] - gs_, bottom=gs_, width=0.02)
    plt.savefig(_output(args.output, 4), bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.savefig(args.output, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.clf()

    # new figure
    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.xlabel("Input ($z$)")

    # Just ground truth
    plt.plot([0, 1], [data.T[1].mean(), data.T[1].mean()], '--')
    plt.bar(np.arange(len(data)) * 0.04, data.T[1], width=0.02)
    plt.tight_layout()
    plt.savefig(_output(args.output, 1, 'mean'))

    # with differences
    plt.bar(0.6 + np.arange(len(data)) * 0.04, data.T[1] - gs_, width=0.02)
    plt.tight_layout()
    plt.savefig(_output(args.output, 2, 'mean'))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--output', type=str, default="control_variates.pdf", help="Plot for control variates")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

