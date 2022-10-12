import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf


# plt.style.use('ggplot')


def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)


def segment_signal(data, window_size=90):
    _segments = np.empty((0, window_size, 3))
    _labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if (len(data["timestamp"][start:end]) == window_size):
            _segments = np.vstack([_segments, np.dstack([x, y, z])])
            _labels = np.append(_labels, stats.mode(data["activity"][start:end])[0][0])
    return _segments, _labels


def get_train_test_sample_data():
    # dataset = read_data('actitracker_raw.txt')
    dataset = read_data(
        '/media/christina/Data/MindXS/playground-data/time-series-noneeg/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
    dataset.dropna(axis=0, how='any', inplace=True)

    # dataset = dataset[:40000]

    dataset['x-axis'] = feature_normalize(dataset['x-axis'])
    dataset['y-axis'] = feature_normalize(dataset['y-axis'])
    dataset['z-axis'] = dataset['z-axis'].str.replace(';', '')
    dataset['z-axis'] = dataset['z-axis'].astype(float)
    dataset['z-axis'] = feature_normalize(dataset['z-axis'])

    # for activity, activity_data in dataset.groupby(by=['activity']):
    #     subset = activity_data[:180]
    #     plot_activity(activity, subset)

    # The window size used is 90, which equals 4.5 seconds of data, and as we are moving each time by 45 points,
    # the step size is equal to 2.25 seconds.
    # The label (activity) for each segment will be selected by the most frequent class label presented in that window.
    # The segment_signal will generate fixed-size segments and append each signal component along the third dimension
    # so that the input dimension will be [total segments, input width, and input channel]. We will reshape the
    # generated segments to have a height of 1 as we will perform one-dimensional convolution (depth-wise) over the
    # signal. Moreover, labels will be one hot encoded using get_dummies function available in Pandas package.

    segments, labels = segment_signal(dataset)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    reshaped_segments = segments.reshape(len(segments), 1, 90, 3)

    train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
    train_x = reshaped_segments[train_test_split]
    train_y = labels[train_test_split]
    test_x = reshaped_segments[~train_test_split]
    test_y = labels[~train_test_split]

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    trainx, trainy, testx, testy = get_train_test_sample_data()
    with open("train_x.npy", "wb") as f:
        np.save(f, trainx)
    with open("train_y.npy", "wb") as f:
        np.save(f, trainy)
    with open("test_x.npy", "wb") as f:
        np.save(f, testx)
    with open("test_y.npy", "wb") as f:
        np.save(f, testy)
