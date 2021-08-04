from ScriptParser import *
from Timer import *

import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert script to csv")
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file name')
    parser.add_argument('--sep', type=str, default=';', help='Field separator')
    args = parser.parse_args()

    s_parser = ScriptParser(args.filename, True)
    s_parser.parse()

    timer = Timer(s_parser)
    timer.set_deltas()

    for i,s in enumerate(s_parser.samples):
        start = s['time']
        end = s['T'] + start
        plt.plot([start, end], [i, i], 'b')

    plt.show()
