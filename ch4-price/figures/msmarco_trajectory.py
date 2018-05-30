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
        "rouge-1": "ROUGE-1",
        "sim": "VecSim",
        }

def do_command(args):
    data = [json.loads(line) for line in gzip.open(args.input, "rt")]
    data = {(obj["system"], obj["metric"], obj["prompt"], obj["estimator"]): obj for obj in data}

    data_gold = [json.loads(line) for line in gzip.open(args.input_gold, "rt")]
    data_gold = {(obj["system"], obj["metric"], obj["prompt"], obj["estimator"]): obj for obj in data_gold}

    colors = cm.tab10.colors

    system = args.data_system
    metric = args.data_metric
    prompt = args.data_prompt

    baseline = np.array(data[system, metric, prompt, "simple"]["summary"])
    model    = np.array(data[system, metric, prompt, "model_variate"]["summary"])
    model_foil    = np.array(data[system, "sim", prompt, "model_variate"]["summary"])
    model_gold = np.array(data_gold[system, metric, prompt, "model_variate"]["summary"])
    model_foil_gold = np.array(data_gold[system, "sim", prompt, "model_variate"]["summary"])
    gold     = np.array(data[system, "gold", prompt, "model_variate"]["summary"])

    plt.rc("font", size=16)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    plt.xlabel("Number of samples")
    plt.ylabel(r"80\% confidence interval")
    plt.plot(baseline.T[2] - baseline.T[1], color=colors[0], label="Humans")
    #plt.plot(model_gold.T[2] - model_gold.T[1], color=colors[2], label="Noiseless humans + {}".format(LABELS.get(metric,metric)))
    plt.plot(model_foil.T[2] - model_foil.T[1], color=colors[1], label="Humans + {}".format(LABELS.get("sim",metric)))
    plt.plot(model.T[2] - model.T[1], color=colors[2], label="Humans + {}".format(LABELS.get(metric,metric)))
    #plt.plot(model_foil_gold.T[2] - model_foil_gold.T[1], color=colors[4], label="Noiseless humans + {}".format(LABELS.get("sim",metric)))
    plt.plot(gold.T[2] - gold.T[1], ':', color=colors[4], label="Humans + perfect metric")

    plt.xlim([0, 500])
    plt.ylim([0.03, 0.2])

    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=str, default="msmarco/msmarco_trajectories.json", help="")
    parser.add_argument('-ig', '--input-gold', type=str, default="msmarco/msmarco_trajectories_gold.json", help="")
    parser.add_argument('-o', '--output', type=str, default="msmarco_trajectory.pdf", help="An example trajectory for a task")
    parser.add_argument('-Dp', '--data-prompt', type=str, default="AnyCorrect", help="An example trajectory for a task")
    parser.add_argument('-Dm', '--data-metric', type=str, default="rouge-1", help="An example trajectory for a task")
    parser.add_argument('-Ds', '--data-system', type=str, default="fastqa_ext", help="An example trajectory for a task")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
