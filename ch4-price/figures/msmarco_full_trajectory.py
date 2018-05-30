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

def do_command(args):
    data = [json.loads(line) for line in gzip.open(args.input, "rt")]
    data = {(obj["system"], obj["metric"], obj["prompt"], obj["estimator"]): obj for obj in data}

    data_gold = [json.loads(line) for line in gzip.open(args.input_gold, "rt")]
    data_gold = {(obj["system"], obj["metric"], obj["prompt"], obj["estimator"]): obj for obj in data_gold}

    data_metric = [json.loads(line) for line in gzip.open(args.input_metric, "rt")]
    data_metric = {(obj["system"], obj["metric"], obj["prompt"], obj["estimator"]): obj for obj in data_metric}

    colors = cm.Dark2.colors

    system = args.data_system
    metric = args.data_metric
    prompt = args.data_prompt

    baseline = np.array(data[system, metric, prompt, "simple"]["summary"])
    baseline_metric = np.array(data_metric[system, metric, prompt, "simple"]["summary"])
    model    = np.array(data[system, metric, prompt, "model_variate"]["summary"])
    model_gold = np.array(data_gold[system, metric, prompt, "model_variate"]["summary"])
    gold     = np.array(data[system, "gold", prompt, "model_variate"]["summary"])

    plt.rc("font", size=14)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    plt.xlabel("Number of samples")
    plt.ylabel(r"Human evaluation score estimate")

    lbls = ["Naive human evaluation",
            "Naive {} evaluation".format(LABELS.get(metric,metric)),
            "Control variates w/ {}".format(LABELS.get(metric,metric)),
#            "Perfect annotators w/ {}".format(LABELS.get(metric,metric)),
#            "Noisy annotators w/ perfect metric",
            ]

    for summary, color, lbl in zip([baseline, baseline_metric, model, model_gold, gold], colors, lbls):
        plt.plot(summary.T[0], color=color, label=lbl, linewidth=0.5)
        #plt.fill_between(xs, summary.T[1], summary.T[2], color=colors[i], alpha=0.3)
        plt.plot(summary.T[1], color=color, linestyle=':', linewidth=0.5)
        plt.plot(summary.T[2], color=color, linestyle=':', linewidth=0.5)

    plt.xlim([0, 500])
    plt.ylim([0.5, 0.7])

    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=str, default="msmarco/msmarco_trajectories.json", help="")
    parser.add_argument('-im', '--input-metric', type=str, default="msmarco/msmarco_trajectories_metric.json", help="")
    parser.add_argument('-ig', '--input-gold', type=str, default="msmarco/msmarco_trajectories_gold.json", help="")
    parser.add_argument('-o', '--output', type=str, default="msmarco_trajectory.pdf", help="An example trajectory for a task")
    parser.add_argument('-Dp', '--data-prompt', type=str, default="AnyCorrect", help="An example trajectory for a task")
    parser.add_argument('-Dm', '--data-metric', type=str, default="rouge-l", help="An example trajectory for a task")
    parser.add_argument('-Ds', '--data-system', type=str, default="fastqa_ext", help="An example trajectory for a task")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
