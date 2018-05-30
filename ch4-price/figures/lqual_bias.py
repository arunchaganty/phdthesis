
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar plots comparing systems on MSMarco.
"""
import pdb
import json
import sys
import gzip
from collections import namedtuple

import numpy as np
import scipy.stats as scstats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mp

LABELS = {
    "rouge-l": "ROUGE-L",
    "rouge-1": "ROUGE-1",
    "rouge-2": "ROUGE-2",
    "ter": "TER",
    "sim": "VecSim",
    "meteor": "METEOR",
    "bleu": "BLEU-2",
    "gold": "Upper bound",


    "fastqa": "fastqa",
    "fastqa_ext": "fastqa\_ext",
    "snet.single": "snet",
    "snet.ensemble": "snet.ens",
    "*": "Combined"
    }
SYSTEMS = ["seq2seq", "pointer", "ml", "ml+rl"]

def do_command(args):
    data = [json.loads(line) for line in open(args.input)]

    colors = cm.Dark2.colors[:4]
    #ix = SYSTEMS.index(args.data_system)

    ids = [vs[0] for vs in data]
    XY = np.array([vs[1:] for vs in data])
    xy = XY[[ids.index(system) for system in SYSTEMS]]
    xy_lr = XY[[ids.index("{}-{}".format(system, 'lr')) for system in SYSTEMS ]]
    xy_ll = XY[[ids.index("{}-{}".format(system, 'll')) for system in SYSTEMS ]]
    xy_ur = XY[[ids.index("{}-{}".format(system, 'ur')) for system in SYSTEMS ]]
    xy_ul = XY[[ids.index("{}-{}".format(system, 'ul')) for system in SYSTEMS ]]

    print("rho", scstats.pearsonr(xy.T[0], xy.T[1]))

    #pdb.set_trace()
    #XY.T[1:3] - XY.T[0]
    #XY.T[4:6] - XY.T[3]


    #metrics = ["rouge-l", "rouge-1", "rouge-2", "meteor", "bleu", "sim", "gold",]
    #systems = ["fastqa", "fastqa_ext", "snet.single", "snet.ensemble"]

    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    xlim = np.array([XY.T[0].min(), XY.T[0].max()])
    coeffs = np.polyfit(xy.T[0], xy.T[3], 1)
    # Plot y == x line
    plt.plot(xlim, xlim * coeffs[0] + coeffs[1], linestyle='--', linewidth=2, zorder=-1)
    for _xy in [xy, xy_lr, xy_ur]:
        plt.errorbar(_xy.T[0], _xy.T[3], xerr=[_xy.T[0]-_xy.T[1],_xy.T[2]-_xy.T[0]], yerr=[_xy.T[3]-_xy.T[4], _xy.T[5]-_xy.T[3]], capsize=2, alpha=0.5, linestyle='', marker="", zorder=-1)

    plt.scatter(xy_lr.T[0], xy_lr.T[3], color=colors, marker=">")
    #plt.scatter(xy_ll.T[0], xy_ll.T[3], color=colors, marker="<")
    plt.scatter(xy_ur.T[0], xy_ur.T[3], color=colors, marker="^")
    #plt.scatter(xy_ul.T[0], xy_ul.T[3], color=colors[ix], marker="<")
    pts = plt.scatter(xy.T[0], xy.T[3], 100, c=colors, marker="o")
    plt.xlabel("Human judgement")
    plt.ylabel("ROUGE-L")
    plt.tight_layout()
    plt.legend(handles=[mp.Patch(color=colors[i], label=LABELS.get(system, system)) for i, system in enumerate(SYSTEMS)])
    plt.savefig(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=str, default="lqual/lqual_bias.json", help="")
    parser.add_argument('-o', '--output', type=str, default="lqual_bias.pdf", help="An example trajectory for a task")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
