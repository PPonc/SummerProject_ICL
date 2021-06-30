import pandas as pd
import numpy as np
import argparse

def to_loc(str, old = ',', new = '.'):
    return str.replace(old, new)

def load_csv(filename):
    file = open(filename, "r")

    values = dict()
    values['time'] = []
    values['T'] = []
    time = 0.0
    base_time = 0.0
    i = 1
    while(line := file.readline()):
        line = line.replace("\n", "")
        v = line.split(";")
        if len(v) >= 4:
            t = float(to_loc(v[0]))
            c = to_loc(v[1])
            c = float("NaN") if c == "<not counted>" else (c if c == "<not supported>" else float(c))
            n = v[3]
            if i == 1:
                base_time = t
            if t != time:
                values['T'].append(t - time)
                values['time'].append(t - base_time)
            if n in values.keys():
                values[n].append(c)
            else:
                values[n] = [c]
            time = t
        i+=1
    return values

def run(filename, sep, o_filename, y):
    val = load_csv(filename)
    df = pd.DataFrame.from_dict(val)
    print("Mean all {} ({})".format(df[y].mean(), df[y].std()))
    df = df.dropna()
    print("Mean all {} ({})".format(df[y].mean(), df[y].std()))
    print(df)
    df.to_csv(o_filename, sep=sep, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate perf measurements with energy")
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file name')
    parser.add_argument('--sep', type=str, default=';', help='Field separator')
    parser.add_argument('-y', type=str, default='power/energy-pkg/', help='Y feature')
    args = parser.parse_args()
    o_f = "{}.out".format(args.filename) if args.output == None else args.output
    run(args.filename, args.sep, o_f, args.y)
