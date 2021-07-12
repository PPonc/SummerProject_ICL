from parse_report import read_csv
import pandas as pd
import argparse

def get_stats(samples, s = False):
    values = dict()
    print("time;program;event;period;")
    for sample in samples:
        name = sample['event']
        period = sample['period']
        if name in values.keys():
            values[name] += period if s else 1
        else:
            values[name] = period if s else 1
        print(f"{sample['time']};{sample['program']};{sample['event']};{sample['period']};")
    return values

def run(args):
    samples = read_csv(args.filename, args.cpu)
    print("{} Samples read".format(len(samples)))
    stats = get_stats(samples, args.sum)
    for k in stats.keys():
        print("{} \t\t{:,}".format(k, stats[k]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate perf measurements with energy")
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('--sum', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    run(args)
