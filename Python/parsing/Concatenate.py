from ScriptConverter import *

import argparse
import pandas as pd

import threading
import queue
from multiprocessing import Process

from math import ceil

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

def iteration(new_df, idx, width, start_idx, old_df, len_df, times):
    working = True
    new_start_idx = start_idx
    for k in range(start_idx, len_df):
        t = times.loc[k, 'time']
        T = times.loc[k, 'T']
        if collision(width * idx, width * (idx + 1), t, t + T):
            if working and t + T < width * (idx + 1):
                new_start_idx = k + 1
            else:
                working = False
            coverage = collision_b_over_a(width * idx, width * (idx + 1), t, t + T)
            new_df.iloc[idx, :] += coverage * old_df.iloc[k,:]
        elif is_b_after_a(width * idx, width * (idx + 1), t, t + T):
            break
        elif working:
            start_idx = k
    new_df.loc[idx, 'time'] = width * idx
    new_df.loc[idx, 'T'] = width
    return new_start_idx

def run_worker(q, new_df, old_df, times, width):
    start_idx = 0
    len_old = len(old_df)
    while(True):
        idx = q.get()
        if idx == -1:
            return
        print(f"starting line {idx}")
        start_idx = iteration(new_df, idx, width, start_idx, old_df, len_old, times)
        print(f"line {idx} finished")


def create_workers(N, new_df, old_df, times, width, workers_nb = 8):
    print(f"running task with {workers_nb} workers")
    my_queue = queue.Queue()
    workers = list()
    for n in range(workers_nb):
        workers.append(Process(target=run_worker, args=(my_queue, new_df, old_df, times, width)))
        workers[-1].start()

    for i in range(N):
        my_queue.put(i)

    for i in range(workers_nb):
        my_queue.put(-1)

    for i in range(workers_nb):
        if workers[i]:
            workers[i].join()

def concatenate(df, width):
    df = df.sort_values(["time", "T"], ascending=(True, True))
    print(df)

    times = df[['time', 'T']]
    df = df.drop(['program'], axis=1)


    N = times.loc[len(times) - 1, 'time'] + times.loc[len(times) - 1, 'T']
    N = int(ceil(N / width))

    new_df = pd.DataFrame(columns = df.columns, index = range(N)).fillna(0)

    start_idx = 0
    for i in range(N):
        working = True
        for k in range(start_idx, len(df)):
            t = times.loc[k, 'time']
            T = times.loc[k, 'T']
            if collision(width * i, width * (i + 1), t, t + T):
                if working and t + T < width * (i + 1):
                    start_idx = k + 1
                else:
                    working = False
                coverage = collision_b_over_a(width * i, width * (i + 1), t, t + T)
                new_df.iloc[i, :] += coverage * df.iloc[k,:]
            elif is_b_after_a(width * i, width * (i + 1), t, t + T):
                break
            elif working:
                start_idx = k

        new_df.loc[i, 'time'] = width * i
        new_df.loc[i, 'T'] = width
        print(f"concatenate {i + 1}/{N} [start = {start_idx}, end = {k}]", end='\r')
    # create_workers(N, new_df, df, times, width, 8)
    return new_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert script to csv")
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file name')
    parser.add_argument('--sep', type=str, default=';', help='Field separator')
    parser.add_argument('-d', '--duration', type=float, default=2.0, help='duation of each sample')
    parser.add_argument('--csv', action='store_true')

    args = parser.parse_args()

    if args.csv:
        print(f"Loading {args.filename} as a CSV file")
        df = pd.read_csv(args.filename, sep=args.sep)
    else:
        conv = ScriptConverter(args.filename)
        df = conv.to_dataframe()

    print("CSV loaded")
    df = concatenate(df, args.duration)
    print(df)

    if args.output:
        df.to_csv(args.output, sep=args.sep, index=False)
    else:
        df.to_csv(f"{args.filename}.conc", sep=args.sep, index=False)
