import pandas as pd
import argparse
import matplotlib.pyplot as plt

def find_values(p_df, filter, includes = None):
    vals = []
    for df in p_df:
        for key in df[filter].unique():
            if not key in vals and (not includes or key in includes):
                vals.append(key)
    return vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert script to csv")
    parser.add_argument('-p', '--prediction', type=str)
    parser.add_argument('-t', '--training', type=str, default=None)
    parser.add_argument('--sep', type=str, default=';', help='Field separator')
    parser.add_argument('-y', type=str, default='power/energy-pkg/')
    parser.add_argument('--y-pred', type=str, default=None)
    parser.add_argument('--filter-by', type=str, default = None)
    parser.add_argument('--include', type=str, default=None)
    args = parser.parse_args()


    p_df = []
    for f_name in args.prediction.split(","):
        p_df.append(pd.read_csv(f_name, sep=args.sep))

    if args.y_pred:
        y_pred = args.y_pred
    else:
        y_pred = f"{args.y}-pred"

    if args.include:
        includes = args.include.split(',')
    else:
        includes = None

    if args.filter_by:
        assert(args.filter_by == 'program' or args.filter_by == 'pid')
        vals = find_values(p_df, args.filter_by, includes)
        for i,df in enumerate(p_df):
            tmp_df = df.sort_values('time', ascending=True)
            plt.step(tmp_df['time'], tmp_df[y_pred], label = 'Total')
            for key in vals:
                key_df = tmp_df.loc[tmp_df[args.filter_by] == key]
                plt.step(key_df['time'], key_df[y_pred], label = key)
            plt.title(f"Prediction n°{i}")
            plt.xlabel('time')
            plt.ylabel(f"{args.y}/{y_pred}")
            plt.legend()
            plt.show()

    if args.training:
        t_df = pd.read_csv(args.training, sep=args.sep)
        plt.step((t_df['time'] - t_df['time'][0])[1:], t_df[args.y][1:], label="reference")

    for i,df in enumerate(p_df):
        plt.step(df['time'] + df['T'], df[y_pred], label = f"prediction n°{i}")

    plt.xlabel('time')
    plt.ylabel(f"{args.y}/{y_pred}")
    plt.legend()
    plt.show()
