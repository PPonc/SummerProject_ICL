import pandas as pd
import matplotlib.pyplot as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


from utils import *


def K_fold_cross_val(df, features, y_param, K=10, plot=True, hist=True):
    kf = KFold(n_splits=K, shuffle=True)
    kf.get_n_splits(df)

    train_rmses = []
    test_rmses = []

    y_pred_test = []
    y_pred_train = []
    y_true_test = []
    y_true_train = []

    for train_index, test_index in kf.split(df):
        d_train = df.iloc[train_index]
        d_test = df.iloc[test_index]
        X_train,y_train = split_dataset_Xy(d_train, y_param, features)
        X_test,y_test = split_dataset_Xy(d_test, y_param, features)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = relu(model.predict(X_train))
        train_rmses.append(np.sqrt(mean_squared_error(y_train, y_pred)))

        y_pred_train.append(y_pred)
        y_true_train.append(y_train)

        y_pred = relu(model.predict(X_test))
        test_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        y_pred_test.append(y_pred)
        y_true_test.append(y_test)

    print("RMSE train {} ({})".format(np.mean(train_rmses), np.std(train_rmses)))
    print("RMSE test {} ({})".format(np.mean(test_rmses), np.std(test_rmses)))

    y_pred_train = np.concatenate(y_pred_train)
    y_true_train = np.concatenate(y_true_train)
    y_pred_test = np.concatenate(y_pred_test)
    y_true_test = np.concatenate(y_true_test)

    if plot:
        sm.qqplot_2samples(y_true_test, y_pred_test, xlabel="True value", ylabel="Predicted value")
        plt.title("QQ-plot 2 samples of validation dataset")
        plt.legend()
        plt.show()

        plt.scatter(y_true_train, y_pred_train, label="Training")
        plt.scatter(y_true_test, y_pred_test, label="Validation")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        plt.title("Plot of predicted value against the true value")
        plt.legend()
        plt.show()

        plt.scatter(y_true_train, y_pred_train - y_true_train, label="Training")
        plt.scatter(y_true_test, y_pred_test - y_true_test, label="Validation")
        plt.xlabel("True value")
        plt.ylabel("Error of predicted value")
        plt.title("Error of prediction against true value")
        plt.legend()
        plt.show()

        plot_relative_error(y_true_test, y_pred_test)
        plt.title("Histogram of relative errors")
        plt.xlabel("Relative error (%)")
        plt.show()
