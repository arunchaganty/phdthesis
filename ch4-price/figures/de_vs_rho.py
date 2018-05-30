#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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
        }

GAMMAS = {
        "grammar"      : 1.235,
        "hter"         :   0.357,
        "redundancy"   :   0.608,
        "overall"      : 1.005,
        "AnyCorrect"   :   0.951,
        "AvgCorrect"   :   0.906,
        }

def do_command(args):
    data = json.load(open(args.input_msmarco))
    data_= json.load(open(args.input_lqual))

    corr = json.load(open(args.input_msmarco_correlation))
    corr_= json.load(open(args.input_lqual_correlation))

    colors = cm.Dark2.colors

    xs = []
    for metric, vs in data.items():
        for prompt, vvs in vs.items():
            for system, v in vvs.items():
                if metric != "gold" and system != "reference":
                    xs.append([corr[prompt][metric][system][0], v])
    for metric, vs in data_.items():
        for prompt, vvs in vs.items():
            for system, v in vvs.items():
                if metric != "gold" and system != "reference":
                    xs.append([corr_[prompt][metric][system][0], v])

    xs = abs(np.array(xs))
    print(xs)

    plt.rc("font", size=14)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    plt.xlabel(r"Metric correlation ($\rho$)")
    plt.ylabel(r"Data efficiency")
    plt.scatter(abs(xs.T[0]), xs.T[1])

    xlim = np.array([xs.T[0].min(), xs.T[0].max()])
    coeffs = np.polyfit(xs.T[0], xs.T[1], 1)
    plt.plot(xlim, xlim * coeffs[0] + coeffs[1], linestyle='--', linewidth=2, zorder=-1)

    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-im', '--input-msmarco', type=str, default="msmarco/msmarco_table_gold.json", help="")
    parser.add_argument('-il', '--input-lqual', type=str, default="lqual/lqual_table_gold.json", help="")
    parser.add_argument('-imc', '--input-msmarco-correlation', type=str, default="msmarco/msmarco_correlation.json", help="")
    parser.add_argument('-ilc', '--input-lqual-correlation', type=str, default="lqual/lqual_correlation.json", help="")
    parser.add_argument('-o', '--output', type=str, default="de_gamma.pdf", help="An example trajectory for a task")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
