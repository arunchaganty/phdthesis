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
import scipy.stats as scstats
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

    "*": "Combined"
    }

def do_command(args):
    data = [json.loads(line) for line in open(args.input, "rt")]

    prompt = args.data_prompt
    metric = args.data_metric
    system = args.data_system

    xy = [[int(datum['prompts'][prompt]['gold']), datum['prompts'][prompt][metric]]  for datum in data if datum['system'] == system]
    xy = np.array(xy)

    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    x, y = xy.T[0], xy.T[1]
    print("rho", scstats.pearsonr(xy.T[0], xy.T[1]))


    plt.violinplot([y[x==-1], y[x==0], y[x==1]], [-1, 0, 1])
    plt.scatter(x, y, alpha=0.3, marker='.')

    #coeffs = np.polyfit(x,y, 1)
    #plt.plot([-1,1], [coeffs[1], coeffs[0] + coeffs[1]], ":")
#
#
    plt.xlabel("Human judgment")
    plt.ylabel(LABELS.get(args.data_metric, args.data_metric))
    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=str, default="lqual/lqual_data.json", help="")
    parser.add_argument('-o', '--output', type=str, default="lqual_instance_correlation.pdf", help="An example trajectory for a task")
    parser.add_argument('-Ds', '--data-system', type=str, default="ml+rl", help="An example trajectory for a task")
    parser.add_argument('-Dp', '--data-prompt', type=str, default="overall", help="An example trajectory for a task")
    parser.add_argument('-Dm', '--data-metric', type=str, default="rouge-l", help="An example trajectory for a task")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
