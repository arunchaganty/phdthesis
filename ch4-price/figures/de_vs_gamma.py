
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

    gammas = json.load(open(args.input_msmarco_gamma))
    gammas_= json.load(open(args.input_lqual_gamma))

    colors = cm.Dark2.colors

    xs = []
    for vs in [data["gold"]]: #data.values():
        for prompt, vvs in vs.items():
            for system, v in vvs.items():
                xs.append([gammas[system][prompt]['nu'], v])
    for vs in [data_["gold"]]: #data.values():
    #for vs in data.values():
        for prompt, vvs in vs.items():
            for system, v in vvs.items():
                xs.append([gammas_[system][prompt]['nu'], v])

    xs = np.array(xs)
    print(xs)

    plt.rc("font", size=14)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    plt.xlabel(r"Relative inter-annotator noise ($\gamma$)")
    plt.ylabel(r"Data efficiency")

    xlim = np.array([xs.T[0].min(), xs.T[0].max()])
    coeffs = np.polyfit(xs.T[0], xs.T[1], 1)
    plt.plot(xlim, xlim * coeffs[0] + coeffs[1], linestyle='--', linewidth=2, zorder=-1)

    plt.scatter(xs.T[0], xs.T[1])

    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-im', '--input-msmarco', type=str, default="msmarco/msmarco_table.json", help="")
    parser.add_argument('-il', '--input-lqual', type=str, default="lqual/lqual_table.json", help="")
    parser.add_argument('-img', '--input-msmarco-gamma', type=str, default="msmarco/msmarco_gamma.json", help="")
    parser.add_argument('-ilg', '--input-lqual-gamma', type=str, default="lqual/lqual_gamma.json", help="")
    parser.add_argument('-o', '--output', type=str, default="de_gamma.pdf", help="An example trajectory for a task")
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
