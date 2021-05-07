"""
Helper file to extract features from data

@author: Gati Aher
"""

import numpy as np
import pandas as pd

"""
For all the feature extractors:

Args:
    timeseries: ndarray (t x n x c)

Return:
    features: ndarray (f x n x c) where f is usually 1
"""

AXIS = 0


def MAV(timeseries):
    """
    The mean absolute value is one of the most commonly used values in sEMG signal analysis. The MAV feature is the average of the absolute values of the amplitude of the sEMG signal in the sliding window. It provides information about the muscle contraction level.
    """
    return np.mean(np.abs(timeseries), axis=AXIS)


def RMS(timeseries):
    """
    The root mean square represents the mean power of the sEMG signal, which reflects the activity of muscles.
    """
    return np.sqrt(np.mean(np.square(timeseries), axis=AXIS))


def SSC(timeseries):
    """
    Slope sign change indicates the frequency information of the sEMG signal.
    """
    return np.sum(1 * (np.diff(np.sign(timeseries), axis=AXIS) != 0), axis=AXIS)


def WL(timeseries):
    """
    Waveform length is the cumulative length of the sEMG signal waveform, which is related to waveform amplitude, frequency, and time and can be used to measure signal complexity.
    """
    return np.sum(np.abs(np.diff(timeseries, axis=AXIS)), axis=AXIS)


def HP_A(timeseries):
    """
    Hjorth activity parameter represents the signal power, the variance of a time function. This can indicate the surface of power spectrum in the frequency domain.
    """
    return np.var(timeseries, ddof=1, axis=AXIS)


def HP_M(timeseries):
    """
    Hjorth mobility parameter represents the mean frequency or the proportion of standard deviation of the power spectrum. This is defined as the square root of variance of the first derivative of the signal y(t) divided by variance of the signal y(t).
    """
    denom = HP_A(timeseries)
    return np.sqrt(HP_A(np.diff(timeseries, axis=AXIS))/denom)


def HP_C(timeseries):
    """
    Hjorth Complexity parameter represents the change in frequency. The parameter compares the signal's similarity to a pure sine wave, where the value converges to 1 if the signal is more similar.
    """
    denom = HP_M(timeseries)
    return HP_M(np.diff(timeseries, axis=AXIS))/denom


def get_feature_labels():
    """
    Return full list of feature labels

    NOTE: Take care to update when new features are added 
    """
    feature_labels = [
        "MAV",
        "RMS",
        "SSC",  # zero
        "WL",
        "HP_A",
        "HP_M",
        "HP_C"
    ]
    return feature_labels


def extract_features(data, only_return_labels=False, verbose=False, drop_constants=False):
    """
    Extract features from data

    n = samples
    t = time series points
    c = channels
    f = features

    Args:
        data: ndarray (n x t x c)
        verbose: default False

    Return:
        features: ndarray (f x n x c)
        feature_labels: ndarray (f)
    """

    feature_labels = [
        "MAV",
        "RMS",
        "SSC",  # zero
        "WL",
        "HP_A",
        "HP_M",
        "HP_C"
    ]

    if (only_return_labels):
        return feature_labels

    # rearrage data so that time points are the first dimension
    data_r = np.moveaxis(data, 1, 0)

    # add all the single value features first
    features_f = [
        MAV(data_r),
        RMS(data_r),
        SSC(data_r),  # zero
        WL(data_r),
        HP_A(data_r),
        HP_M(data_r),
        HP_C(data_r),
    ]

    # turn features in ndarray
    features = np.array(features_f)

    if (verbose):
        print("features.shape", features.shape)
        for i, f in enumerate(feature_labels):
            print(f)
            for c in range(features.shape[-1]):
                print("\tmin", min(features[i, :, c]))
                print("\tmax", max(features[i, :, c]))

    # remove any columns with constant values across all samples
    drop_idx = list(set(np.argwhere(np.std(features, axis=1) == 0)[:, 0]))
    print("WARNING: constant features", drop_idx)

    if(drop_constants):
        # reverse to pop multiple items without issues
        drop_idx.sort(reverse=True)
        if(drop_idx):
            features = np.delete(features, drop_idx, axis=0)
        for i in drop_idx:
            feature_labels.pop(i)

    return features, feature_labels
