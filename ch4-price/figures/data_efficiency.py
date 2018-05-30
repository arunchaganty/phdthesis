#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots data efficiency
"""

from copy import copy
import  numpy as np 
import  matplotlib.pyplot as plt

NUS = {
    "msmarco": (0.08 + 0.24)/ 0.39,
    "dialog": (0.08 + 0.20)/ 0.19,
    "lqual": (0.11 + 0.22)/ 0.25,
    }

def do_plot(args):
    xs = np.linspace(0.05,0.95,10) + 0.5
    ys = np.linspace(0.05,0.95,10)
    zs = np.array([[(1-y**2/(1+x**2)) for x in xs] for y in ys])
    print(NUS)

    cm = copy(plt.cm.magma_r)

    plt.rc("font", size=18)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))
    fig, ax = plt.subplots()

    m = ax.imshow(zs, cmap=cm, origin="lower", aspect="auto", extent=[0.5, 1.5, 0, 1])
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\rho$")
    fig.colorbar(m, orientation="horizontal")

    for task in NUS:
        ax.axvline(x=NUS[task], label=task, linestyle=':', color="black")

    #plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(args.output, dpi=400)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--output', type=str, default='data_efficiency.pdf', help="Where to save output")
    parser.set_defaults(func=do_plot)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
