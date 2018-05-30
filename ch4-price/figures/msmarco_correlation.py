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


        "fastqa": "fastqa",
        "fastqa_ext": "fastqa\_ext",
        "snet.single": "snet",
        "snet.ensemble": "snet.ens",
        "*": "All",
        }

def first(x):
    return next(iter(x))

def do_command(args):
    data = json.load(open(args.input))
    data = data[args.data_prompt]
    metrics = ["rouge-l", "rouge-1", "rouge-2", "meteor", "bleu", "sim"]
    systems = ["fastqa", "fastqa_ext", "snet.single", "snet.ensemble", "*"]

    X = np.array([[data[metric][system] for system in systems] for metric in metrics])

    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    fig, ax = plt.subplots()

    plt.imshow(abs(X), cmap="viridis", origin="lower", aspect="auto", vmin=0.1, vmax=0.5)

    ax.set_xticks(np.arange(len(systems)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([LABELS.get(s, s) for s in systems], rotation=45)
    ax.set_yticklabels([LABELS.get(s, s) for s in metrics])
    plt.colorbar(label=r"Pearson $\rho$")
    plt.xlabel("Systems")
    plt.ylabel("Metrics")

    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=str, default="msmarco/msmarco_correlation.json", help="")
    parser.add_argument('-o', '--output', type=str, default="msmarco_correlation.pdf", help="An example trajectory for a task")
    parser.add_argument('-Dp', '--data-prompt', type=str, default="AnyCorrect", help="An example trajectory for a task")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
