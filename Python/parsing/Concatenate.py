from ScriptConverter import *

import argparse
import pandas as pd

import threading
import queue
from multiprocessing import Process

from math import ceil,floor

class Point1d:
    def __init__(self, start, end):
        self.start = start
        self.end = end

def collision(a_start, a_end, b_start, b_end):
    if a_start > b_end or a_end < b_start:
        return False
    return True

def is_b_after_a(a_start, a_end, b_start, b_end):
    return b_start > a_end

def collision_b_over_a(a_start, a_end, b_start, b_end):
    if a_start > b_end or a_end < b_start:
        return 0.0
    elif b_start < a_start:
        if a_end < b_end:
            return (a_end - a_start) / (b_end - b_start)
        else:
            return (b_end - a_start) / (b_end - b_start)
    else:
        if b_end <= a_end:
            return 1.0
        else:
            return (a_end - b_start) / (b_end - b_start)

def find_samples_unfiltered(times, width):
    samples = []

    for k in range(len(times)):
        t = times.loc[k, 'time']
        T = times.loc[k, 'T']
        start = int(floor(t / width))
        finish = int(floor((t+T) / width))
        if finish >= len(samples):
            while len(samples) < finish + 1:
                samples.append([])
        for i in range(start, finish + 1):
            samples[i].append(k)

    return samples, len(samples)

def concatenate_unfiltered(df, times, width):
    samples,N = find_samples_unfiltered(times, width)
    new_df = pd.DataFrame(columns = df.columns, index = range(N)).fillna(0)
    print("Sample indices precomputed")
    for i in range(N):
        print(f"concatenate {i + 1} / {N}, samples = {len(samples[i])}", end="\r")
        for k in samples[i]:
            t = times.loc[k, 'time']
            T = times.loc[k, 'T']
            if collision(width * i, width * (i + 1), t, t + T):
                coverage = collision_b_over_a(width * i, width * (i + 1), t, t + T)
                new_df.iloc[i, :] += coverage * df.iloc[k,:]
        new_df.loc[i, 'time'] = width * i
        new_df.loc[i, 'T'] = width
    return new_df


def find_samples_filtered(times, width, filter):
    samples = dict()

    for k in range(len(times)):
        t = times.loc[k, 'time']
        T = times.loc[k, 'T']
        f = times.loc[k, filter]
        start = int(floor(t / width))
        finish = int(floor((t+T) / width))
        if f not in samples.keys():
            samples[f] = []
        if finish >= len(samples[f]):
            while len(samples[f]) < finish + 1:
                samples[f].append([])
        for i in range(start, finish + 1):
            samples[f][i].append(k)

    N = 0
    for k in samples.keys():
        N += len(samples[k])

    return samples, N

def concatenate_filtered(df, times, width, filter):
    samples, N = find_samples_filtered(times, width, filter)
    new_df = pd.DataFrame(columns = df.columns, index = range(N)).fillna(0)
    print("Sample indices precomputed")
    idx = 0
    for k_idx,k in enumerate(samples.keys()):
        N_k = len(samples[k])
        for i in range(N_k):
            print(f"concatenate key: {k} {k_idx + 1} / {len(samples.keys())}, {i + 1} / {N_k}, samples = {len(samples[k][i])}", end="\r")
            for j in samples[k][i]:
                t = times.loc[j, 'time']
                T = times.loc[j, 'T']
                if collision(width * i, width * (i + 1), t, t + T):
                    coverage = collision_b_over_a(width * i, width * (i + 1), t, t + T)
                    new_df.iloc[idx, :] += coverage * df.iloc[j,:]
            new_df.loc[idx, 'time'] = width * i
            new_df.loc[idx, 'T'] = width
            new_df.loc[idx, filter] = k
            idx += 1
    return new_df


def concatenate(times, df, width, filter):
    if filter:
        return concatenate_filtered(df, times, width, filter)
    else:
        return concatenate_unfiltered(df, times, width)

def run_concatenation(df, width, filter = None):
    df = df.sort_values(["time", "T"], ascending=(True, True))
    print(df)

    times = df[['time', 'T', 'program', 'pid']]
    if not filter:
        df = df.drop(['program', 'pid'], axis=1)
    elif filter == "program":
        df = df.drop(['pid'], axis = 1)
    elif filter == "pid":
        pass
    else:
        raise RuntimeError(f"filter {filter} unknown.")

    return concatenate(times, df, width, filter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert script to csv")
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file name')
    parser.add_argument('--sep', type=str, default=';', help='Field separator')
    parser.add_argument('-d', '--duration', type=float, default=2.0, help='duation of each sample')
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--group-by', type=str, default=None)

    args = parser.parse_args()

    if args.group_by and args.group_by != "program" and args.group_by != "pid":
        raise RuntimeError(f"Cannot filter by {args.group_by}. Either program or pid.")

    if args.csv:
        print(f"Loading {args.filename} as a CSV file")
        df = pd.read_csv(args.filename, sep=args.sep)
    else:
        conv = ScriptConverter(args.filename)
        df = conv.to_dataframe()

    print("CSV loaded")
    df = run_concatenation(df, args.duration, args.group_by)
    print(df)

    if args.output:
        df.to_csv(args.output, sep=args.sep, index=False)
    else:
        df.to_csv(f"{args.filename}.conc", sep=args.sep, index=False)
