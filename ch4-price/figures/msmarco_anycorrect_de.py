#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar plots comparing systems on MSMarco.
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
    "gold": "Upper bound",


    "fastqa": "FastQA",
    "fastqa_ext": "FastQAext",
    "snet.single": "SNet",
    "snet.ensemble": "SNet (ens.)",
    "*": "Combined"
    }

def do_command(args):
    data = json.load(open(args.input))

    prompt = args.data_prompt
    metrics = ["rouge-l", "rouge-1", "rouge-2", "meteor", "bleu", "sim",]# "gold",]
    systems = ["fastqa", "fastqa_ext", "snet.single", "snet.ensemble"]

    plt.rc("font", size=14)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    X = np.array([[data[metric][prompt][system]**2 for system in systems] for metric in metrics])

    plt.rc("font", size=14)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    fig, ax = plt.subplots()

    plt.imshow(abs(X), cmap="viridis", origin="lower", aspect="auto")#, vmin=0.9, vmax=1.15)

    ax.set_xticks(np.arange(len(systems)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([LABELS.get(s, s) for s in systems], rotation=45)
    ax.set_yticklabels([LABELS.get(s, s) for s in metrics])
    plt.colorbar(label="Data efficiency")
    plt.xlabel("Systems")
    plt.ylabel("Metrics")

    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=str, default="msmarco/msmarco_table.json", help="")
    parser.add_argument('-o', '--output', type=str, default="msmarco_anycorrect_de.pdf", help="An example trajectory for a task")
    parser.add_argument('-Dp', '--data-prompt', type=str, default="AnyCorrect", help="An example trajectory for a task")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
