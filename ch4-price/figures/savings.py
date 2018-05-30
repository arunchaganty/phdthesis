#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A table of correlations.
"""
import json
import sys
import gzip
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

LABELS = {
        "rouge-l": "ROUGE-L",
        "rouge-1": "ROUGE-1",
        "rouge-2": "ROUGE-2",
        "ter": "TER",
        "sim": "VecSim",
        "meteor": "METEOR",
        "bleu": "BLEU-2",

        "*": "Combined"
        }

def first(x):
    return next(iter(x))

def do_command(args):
    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))
    gamma = np.outer( np.arange(0,1.01,0.01), np.ones(101) )
    rho = np.outer( np.ones(101), np.arange(0,1.01,0.01) )

    ide = np.divide(1-np.square(rho) + gamma,1 + gamma)

    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 

    plt.contourf(gamma, rho,ide, cmap='hot_r', levels=np.arange(0.0,1.1,0.1))
    plt.colorbar(label="Inverse data efficiency")

    plt.xlabel(r'Normalized annotator variance ($\gamma$)')
    plt.ylabel(r'Automatic metric correlation ($\rho$)')

    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--output', type=str, default="savings.pdf", help="An example trajectory for a task")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
