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


######################
# FEATURE EXTRACTORS #
######################


def AFB(timeseries):
    """
    amplitude of the first burst
    """
    pass


def ApEn(timeseries):
    """
    approximate entropy
    """
    pass


def SampEn(timeseries):
    """
    sample entropy
    """
    pass


def AR(timeseries):
    """
    autoregressive model
    NOTE: returns 4 values
    """
    pass


def DAR(timeseries):
    """
    differencing autoregressive model
    NOTE: returns 4 values
    """
    pass


def BC(timeseries):
    """
    box counting dimension
    """
    pass


def CC(timeseries):
    """
    cepstrum/cepstral coefficients
    NOTE: returns 4 values
    """
    pass


def DCC(timeseries):
    """
    differencing cepstrum/cepstral coefficients
    NOTE: returns 4 values
    """
    pass


def CEA(timeseries):
    """
    critical exponent analysis
    """
    pass


def DAMV(timeseries):
    """
    difference absolute mean value
    """
    pass


def DASDV(timeseries):
    """
    difference absolute standard deviation value
    """
    pass


def DFA(timeseries):
    """
    detrended fluctuation analysis
    """
    pass


def DPR(timeseries):
    """
    maximum-to-minimum drop in power density ratio
    """
    pass


def FR(timeseries):
    """
    frequency ratio
    """
    pass


def HG(timeseries):
    """
    Higuchi's fractal dimension
    """
    pass


def HIST(timeseries):
    """
    histogram
    NOTE: returns 3 values
    """
    pass


def HP_A(timeseries):
    """
    Hjorth activity parameter: represents the signal power, the variance of a time function. This can indicate the surface of power spectrum in the frequency domain.
    """
    return np.var(timeseries, ddof=1, axis=AXIS)


def HP_M(timeseries):
    """
    Hjorth mobility parameter: represents the mean frequency or the proportion of standard deviation of the power spectrum. This is defined as the square root of variance of the first derivative of the signal y(t) divided by variance of the signal y(t).
    """
    denom = HP_A(timeseries)
    return np.sqrt(HP_A(np.diff(timeseries, axis=AXIS))/denom)


def HP_C(timeseries):
    """
    Hjorth Complexity parameter: represents the change in frequency. The parameter compares the signal's similarity to a pure sine wave, where the value converges to 1 if the signal is more similar.
    """
    denom = HP_M(timeseries)
    return HP_M(np.diff(timeseries, axis=AXIS))/denom


def IEMG(timeseries):
    """
    integrated EMG
    """
    pass


def KATZ(timeseries):
    """
    Katz's fractal dimension
    """
    pass


def KURT(timeseries):
    """
    kurtosis
    """
    pass


def SKEW(timeseries):
    """
    skewness
    """
    pass


def LD(timeseries):
    """
    log detector
    """
    pass


def DLD(timeseries):
    """
    differencing log detector
    """
    pass


def M2(timeseries):
    """
    second order moment
    """
    pass


def MAV(timeseries):
    """
    mean absolute value: one of the most commonly used values in sEMG signal analysis. The MAV feature is the average of the absolute values of the amplitude of the sEMG signal in the sliding window. It provides information about the muscle contraction level.
    """
    return np.mean(np.abs(timeseries), axis=AXIS)


def MAV1(timeseries):
    """
    modified mean absolute value type 1
    """
    pass


def MAV2(timeseries):
    """
    modified mean absolute value type 2
    """
    pass


def MAVS(timeseries):
    """
    mean absolute value slope
    """
    pass


def MAX(timeseries):
    """
    maximum amplitude
    """
    pass


def MDF(timeseries):
    """
    median frequency
    """
    pass


def MNF(timeseries):
    """
    mean frequency
    """
    pass


def MFL(timeseries):
    """
    maximum fractal length
    """
    pass


def MHW(timeseries):
    """
    multiple hamming windows
    NOTE: returns 3 values
    """
    pass


def MTW(timeseries):
    """
    trapezoidal windows
    NOTE: returns 3 values
    """
    pass


def MNP(timeseries):
    """
    mean power
    """
    pass


def TTP(timeseries):
    """
    total power
    """
    pass


def MYOP(timeseries):
    """
    myopulse percentage rate
    """
    pass


def OHM(timeseries):
    """
    power spectrum deformation
    """
    pass


def PKF(timeseries):
    """
    peak frequency
    """
    pass


def PSDFD(timeseries):
    """
    power spectral density fractal dimension
    """
    pass


def PSR(timeseries):
    """
    power spectrum ratio
    """
    pass


def RMS(timeseries):
    """
    root mean square: represents the mean power of the sEMG signal, which reflects the activity of muscles.
    """
    return np.sqrt(np.mean(np.square(timeseries), axis=AXIS))


def SM(timeseries):
    """
    spectral moment
    """
    pass


def SMR(timeseries):
    """
    signal-to-motion artefact ratio
    """
    pass


def SNR(timeseries):
    """
    signal-to-noise ratio
    """
    pass


def SSC(timeseries):
    """
    slope sign change: indicates the frequency information of the sEMG signal.
    """
    return np.sum(1 * (np.diff(np.sign(timeseries), axis=AXIS) != 0), axis=AXIS)


def SSI(timeseries):
    """
    simple square integral
    """
    pass


def TDPSD(timeseries):
    """
    time-dependent power spectrum descriptors
    NOTE: returns 6 values
    """
    pass


def TM(timeseries):
    """
    absolute temporal moment
    """
    pass


def DTM(timeseries):
    """
    differencing absolute temporal moment
    """
    pass


def VAR(timeseries):
    """
    variance
    """
    pass


def DVARV(timeseries):
    """
    differencing variance
    """
    pass


def VCF(timeseries):
    """
    variance of central frequency
    """
    pass


def VFD(timeseries):
    """
    variance fractal dimension
    """
    pass


def V(timeseries):
    """
    v-order
    """
    pass


def DV(timeseries):
    """
    differencing v-order
    """
    pass


def WAMP(timeseries):
    """
    Wilson amplitude
    """
    pass


def WL(timeseries):
    """
    waveform length: the cumulative length of the sEMG signal waveform, which is related to waveform amplitude, frequency, and time and can be used to measure signal complexity.
    """
    return np.sum(np.abs(np.diff(timeseries, axis=AXIS)), axis=AXIS)


def ZC(timeseries):
    """
    zero crossing
    """
    pass


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
        # "SSC",  # zero
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
        # SSC(data_r),  # zero
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
    if (drop_idx):
        print("WARNING: constant features", drop_idx)

    if(drop_constants):
        # reverse to pop multiple items without issues
        drop_idx.sort(reverse=True)
        if(drop_idx):
            features = np.delete(features, drop_idx, axis=0)
        for i in drop_idx:
            feature_labels.pop(i)

    return features, feature_labels


def moving_window(X, sampling_rate_Hz, window_span_sec, window_incr_sec, verbose=False):
    """
    Return a feature matrix of shape (f x (n x w) x c)

    Args:
        X: ndarray (n x t x c)

    Return
        features: (f x (n x w) x c)
        feature_labels: (f,)
        n_windows: (int) number of windows
    """
    # number of time points in window
    window_span = int(sampling_rate_Hz * window_span_sec)
    # number of time points to shift window by
    window_incr = int(sampling_rate_Hz * window_incr_sec)
    # number of analysis segments
    n_windows = int((X.shape[1] - window_span) // window_incr) + 1
    print("number of analysis segments:", n_windows)

    feature_labels = extract_features(None, only_return_labels=True)
    features = np.zeros(
        (len(feature_labels), X.shape[0] * n_windows, X.shape[2])
    )

    # sliding windows
    i = 0
    for e in range(window_span, X.shape[1], window_incr):
        s = e - window_incr
        fm, _ = extract_features(X[:, s:e, :])
        features[:, i:i+fm.shape[1], :] = fm
        if (verbose):
            print("\n----\n \t mean \t std ")
            for f, flabel in enumerate(feature_labels):
                for c in range(fm.shape[2]):
                    print(flabel + str(c), np.mean(features[f, i:i+X.shape[0], c]),
                          np.std(features[f, i:i+X.shape[0], c]))
        i += X.shape[0]

    # replace nans
    np.nan_to_num(features, copy=False)

    return features, feature_labels, n_windows
