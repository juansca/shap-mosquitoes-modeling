import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


def smoothing(xs):
    n = len(xs)
    for i in range(1, n - 1):
        xs[i] = sum([xs[j] for j in [i - 1, i, i + 1]]) / 3
    xs[0] = xs[1]
    xs[n - 1] = xs[n - 2]

    return xs

def load_data(filename='data/all.csv'):
    """Load the training data set."""

    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    n_rows, n_cols = data.shape
    assert n_cols > 3

    weeks = data[:, 1]
    ts = smoothing(data[:, 2])
    xs = data[:, 3:]

    return n_cols, weeks, ts, xs


def stats(xs, ts, model, n_splits=5):
    cv = TimeSeriesSplit(n_splits)
    scores = cross_val_score(model, xs, ts, cv=cv.split(xs), scoring='neg_mean_squared_error')

    scores = np.sqrt(-scores)

    mean = scores.mean()
    std_dev = scores.std()

    return scores, mean, std_dev


def print_stats(scores, mean, std_dev, title='Stats'):
    print()
    print(title)
    print('-' * len(title))
    print('')
    print('Model Scores: ', scores)
    print('Mean Score: ', mean)
    print('Standard Deviation of Score: ', std_dev)
