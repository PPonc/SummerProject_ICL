import pandas as pd
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert script to csv")
    parser.add_argument('-p', '--prediction', type=str)
    parser.add_argument('-t', '--training', type=str, default=None)
    parser.add_argument('--sep', type=str, default=';', help='Field separator')
    parser.add_argument('-y', type=str, default='power/energy-pkg/')
    parser.add_argument('--y-pred', type=str, default=None)
    args = parser.parse_args()

    p_df = pd.read_csv(args.prediction, sep=args.sep)
    t_df = pd.read_csv(args.training, sep=args.sep)

    if args.y_pred:
        y_pred = args.y_pred
    else:
        y_pred = f"{args.y}-pred"

    plt.step(p_df['time'], p_df[args.y])
    plt.step(t_df['time'], p_df[y_pred])
    plt.show()
