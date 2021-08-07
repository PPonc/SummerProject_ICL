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

    if args.prediction.find(",") == -1:
        p_df = pd.read_csv(args.prediction, sep=args.sep)
        single_pred = True
    else:
        p_df = []
        for f_name in args.prediction.split(","):
            p_df.append(pd.read_csv(f_name, sep=args.sep))
        single_pred = False
    if args.training:
        t_df = pd.read_csv(args.training, sep=args.sep)
        plt.step(t_df['time'], t_df[args.y], label="reference")

    if args.y_pred:
        y_pred = args.y_pred
    else:
        y_pred = f"{args.y}-pred"

    if single_pred:
        plt.step(p_df['time'], p_df[y_pred], label = "prediction")
    else:
        for i,df in enumerate(p_df):
            plt.step(df['time'], df[y_pred], label = f"prediction n°{i}")
    plt.xlabel('time')
    plt.ylabel(f"{args.y}/{y_pred}")
    plt.legend()
    plt.show()
