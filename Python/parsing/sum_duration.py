import argparse
import pandas as pd
import matplotlib.pyplot as plt

duration = 0.100

def run(filename, duration, y):
    df = pd.read_csv(filename, sep=";")
    final_df = pd.DataFrame()
    final_df = final_df.reindex(['time', y])
    t_start = 0.0
    t_end = duration
    final_df = final_df.append({'time':0.0, y:0.0}, ignore_index=True)

    for i in range(df.shape[0]):
        l = df.iloc[i]
        while l['time'] > t_end:
            t_start = t_end
            t_end = t_end + duration
            final_df = final_df.append({'time': t_start, y: 0.0}, ignore_index=True)
        final_df.iloc[-1][y] += l[y]

    final_df.to_csv("{}.reduced".format(filename), sep=";", index=False)
    return final_df





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate perf measurements with energy")
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-d', '--duration', type=float, default=0.100, help='Duration')
    parser.add_argument('-y', type=str, default='power/energy-pkg/', help='Y value')
    args = parser.parse_args()
    df = run(args.filename, args.duration, args.y)
    plt.plot(df['time'], df[args.y])
    plt.show()
