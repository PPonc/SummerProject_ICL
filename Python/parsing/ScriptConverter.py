from ScriptParser import *
from Timer import *

import argparse
import pandas as pd

class ScriptConverter:

    def __init__(self, filename, cpu = True, TC = Timer):
        self.parser = ScriptParser(filename, cpu)
        self.parser.parse()
        self.timer = TC(self.parser)
        self.timer.set_deltas()
        self.data = dict()

    def samples(self):
        return self.timer.samples

    def features(self):
        return self.parser.features

    def programs(self):
        return self.parser.programs

    def init_data(self):
        self.data = dict()
        for f in self.features():
            self.data[f] = []
        self.data['time'] = []
        self.data['T'] = []
        self.data['program'] = []

    def add_line(self, time, T, prog):
        for f in self.features():
            self.data[f].append(0)
        self.data['time'].append(time)
        self.data['T'].append(T)
        self.data['program'].append(prog)

    def add_sample(self, sample):
        self.add_line(sample['time'], sample['T'], sample['program'])
        self.data[sample['event']][-1] = sample['period']

    def to_dataframe(self):
        self.init_data()
        for sample in self.samples():
            self.add_sample(sample)
        return pd.DataFrame(self.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert script to csv")
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file name')
    parser.add_argument('--sep', type=str, default=';', help='Field separator')
    args = parser.parse_args()

    conv = ScriptConverter(args.filename)
    df = conv.to_dataframe()

    if args.output:
        df.to_csv(args.output, sep=args.sep, index=False)
    else:
        df.to_csv(f"{args.filename}.out", sep=args.sep, index=False)
