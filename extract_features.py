"""
Helper file to extract features from data

@author: Gati Aher
"""

import numpy as np
import pandas as pd

from scipy import stats
import sampen
import math

import sys


def cmp(a, b):
    """
    Util function cmp (needed in Python 3)
    """
    return bool(a > b) - bool(a < b)


"""
For all the feature extractors:

Args:
    vector: ndarray (t, )

Return:
    features: ndarray (f x n x c) where f is usually 1
"""


def getAFB(vector):
    """
    amplitude of the first burst
    """
    pass


def getApEn(vector):
    """
    approximate entropy
    """
    pass


def getDCC(vector):
    """
    differencing cepstrum/cepstral coefficients
    NOTE: returns 4 values
    """
    pass


def getCEA(vector):
    """
    critical exponent analysis
    """
    pass


def getDAMV(vector):
    """
    difference absolute mean value
    """
    pass


def getDFA(vector):
    """
    detrended fluctuation analysis
    """
    pass


def getDPR(vector):
    """
    maximum-to-minimum drop in power density ratio
    """
    pass


def getFR(vector):
    """
    frequency ratio
    """
    pass


def getHG(vector):
    """
    Higuchi's fractal dimension
    """
    pass


def getKATZ(vector):
    """
    Katz's fractal dimension
    """
    pass


def getDLD(vector):
    """
    differencing log detector
    """
    pass


def getM2(vector):
    """
    second order moment
    """
    pass


def getMAVS(vector):
    """
    mean absolute value slope
    """
    pass


def getMAX(vector):
    """
    maximum amplitude
    """
    pass


def getMHW(vector):
    """
    multiple hamming windows
    NOTE: returns 3 values
    """
    pass


def getMTW(vector):
    """
    trapezoidal windows
    NOTE: returns 3 values
    """
    pass


def getOHM(vector):
    """
    power spectrum deformation
    """
    pass


def getPKF(vector):
    """
    peak frequency
    """
    pass


def getPSDFD(vector):
    """
    power spectral density fractal dimension
    """
    pass


def getPSR(vector):
    """
    power spectrum ratio
    """
    pass


def getSMR(vector):
    """
    signal-to-motion artefact ratio
    """
    pass


def getSNR(vector):
    """
    signal-to-noise ratio
    """
    pass


def getTDPSD(vector):
    """
    time-dependent power spectrum descriptors
    NOTE: returns 6 values
    """
    pass


def getDTM(vector):
    """
    differencing absolute temporal moment
    """
    pass


def getDVARV(vector):
    """
    differencing variance
    """
    pass


def getVCF(vector):
    """
    variance of central frequency
    """
    pass


def getVFD(vector):
    """
    variance fractal dimension
    """
    pass


def getV(vector):
    """
    v-order
    """
    pass


def getDV(vector):
    """
    differencing v-order
    """
    pass


###############
# IMPLEMENTED #
###############


def getSampEn(vector, m=2, r_multiply_by_sigma=.2):
    """
    sample entropy
    NOTE: returns 3 values
    """
    vector = np.asarray(vector)
    r = r_multiply_by_sigma * np.std(vector)
    try:
        results = sampen.sampen2(data=vector.tolist(), mm=m, r=r)
    except:
        return [0.0, 0.0, 0.0]

    results_SampEN = []
    for x in np.array(results)[:, 1]:
        if x is not None:
            results_SampEN.append(x)
        else:
            results_SampEN.append(-100.)

    return list(results_SampEN)


def getAR(vector, order=4):
    """
    autoregressive model
    NOTE: returns 5 values
    """
    # Using Levinson Durbin prediction algorithm, get autoregressive coefficients
    # Square signal
    vector = np.asarray(vector)
    R = [vector.dot(vector)]
    if R[0] == 0:
        return [1] + [0]*(order-2) + [-1]
    for i in range(1, order + 1):
        r = vector[i:].dot(vector[:-i])
        R.append(r)
    R = np.array(R)
    # step 2:
    AR = np.array([1, -R[1] / R[0]])
    E = R[0] + R[1] * AR[1]
    for k in range(1, order):
        if (E == 0):
            E = 10e-17
        alpha = - AR[:k + 1].dot(R[k + 1:0:-1]) / E
        AR = np.hstack([AR, 0])
        AR = AR + alpha * AR[::-1]
        E *= (1 - alpha ** 2)
    return AR


def getDAR(vector):
    """
    differencing autoregressive model
    NOTE: returns 5 values
    """
    # Get the first difference of the vector
    vector_diff = np.diff(vector)
    # Calculate the AR coefficient on it
    AR = getAR(vector_diff, order=4)
    if (len(AR) < 5):
        vals = np.zeros((5, ))
        vals[:-1] = getAR(vector_diff, order=4)
        return vals
    return AR


def getBC(vector):
    """
    box counting dimension
    """
    k_max = int(np.floor(np.log2(len(vector))))-1
    Nr = np.zeros(k_max)
    r = np.zeros(k_max)
    for k in range(0, k_max):
        r[k] = 2**(k+1)
        curve_box = int(np.floor(len(vector)/r[k]))
        box_r = np.zeros(curve_box)
        for i in range(curve_box):
            max_dat = np.max(vector[int(r[k]*i):int(r[k]*(i+1))])
            min_dat = np.min(vector[int(r[k]*i):int(r[k]*(i+1))])
            box_r[i] = np.ceil((max_dat-min_dat)/r[k])
        Nr[k] = np.sum(box_r)

    try:
        bc_poly = np.polyfit(np.log2(1/r), np.log2(Nr), 1)
        val = bc_poly[0]
    except:
        return 0
    return val


def getCC(vector, order=4):
    """
    cepstrum/cepstral coefficients
    NOTE: returns 4 values
    """
    AR = getAR(vector, order)
    cc = np.zeros(order+1)
    cc[0] = -1*AR[0]  # BUG: issue with this line
    if order > 2:
        for p in range(2, order+2):
            for l in range(1, p):
                cc[p-1] = cc[p-1]+(AR[p-1] * cc[p-2] * (1-(l/p)))

    return cc


def getDASDV(vector):
    """
    difference absolute standard deviation value
    """
    vector = np.asarray(vector)
    return np.lib.scimath.sqrt(np.mean(np.diff(vector)))


def getHIST(vector, bins=3):
    """
    histogram
    NOTE: returns 3 values
    """
    hist, bin_edges = np.histogram(vector, bins)
    return hist.tolist()


def getHP_A(vector):
    """
    Hjorth activity parameter: represents the signal power, the variance of a time function. This can indicate the surface of power spectrum in the frequency domain.
    """
    return np.var(vector, ddof=1)


def getHP_M(vector):
    """
    Hjorth mobility parameter: represents the mean frequency or the proportion of standard deviation of the power spectrum. This is defined as the square root of variance of the first derivative of the signal y(t) divided by variance of the signal y(t).
    """
    denom = getHP_A(vector)
    return np.sqrt(getHP_A(np.diff(vector))/denom)


def getHP_C(vector):
    """
    Hjorth Complexity parameter: represents the change in frequency. The parameter compares the signal's similarity to a pure sine wave, where the value converges to 1 if the signal is more similar.
    """
    denom = getHP_M(vector)
    return getHP_M(np.diff(vector))/denom


def getIQR(vector):
    """
    interquartile range
    NOTE: returns multiple values
    """
    vector = np.asarray(vector)
    vector.sort()
    return [vector[int(round(vector.shape[0]/4))], vector[int(round(vector.shape[0]*3/4))]]


def getIEMG(vector):
    """
    integrated EMG
    """
    vector = np.asarray(vector)
    return np.sum(np.abs(vector))


def getKURT(vector):
    """
    kurtosis
    """
    vector = np.asarray(vector)
    return stats.kurtosis(vector)


def getSKEW(vector):
    """
    skewness
    """
    vector = np.asarray(vector)
    return stats.skew(vector)


def getLD(vector):
    """
    log detector
    """
    vector = np.asarray(vector)
    return np.exp(np.mean(np.log(np.abs(vector)+1)))


def getMAV(vector):
    """
    mean absolute value: one of the most commonly used values in sEMG signal analysis. The MAV feature is the average of the absolute values of the amplitude of the sEMG signal in the sliding window. It provides information about the muscle contraction level.
    """
    return np.mean(np.abs(vector))


def getMMAV1(vector):
    """
    modified mean absolute value type 1
    """
    vector_array = np.array(vector)
    total_sum = 0.0
    for i in range(0, len(vector_array)):
        if((i+1) < 0.25*len(vector_array) or (i+1) > 0.75*len(vector_array)):
            w = 0.5
        else:
            w = 1.0
        total_sum += abs(vector_array[i]*w)
    return total_sum/len(vector_array)


def getMMAV2(vector):
    """
    modified mean absolute value type 2
    """
    total_sum = 0.0
    vector_array = np.array(vector)
    for i in range(0, len(vector_array)):
        if ((i + 1) < 0.25 * len(vector_array)):
            w = ((4.0 * (i + 1)) / len(vector_array))
        elif ((i + 1) > 0.75 * len(vector_array)):
            w = (4.0 * ((i + 1) - len(vector_array))) / len(vector_array)
        else:
            w = 1.0
        total_sum += abs(vector_array[i] * w)
    return total_sum / len(vector_array)


def getMDF(vector, fs=1000):
    """
    median frequency
    """
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0]/2))]
    # f = np.fft.fftfreq(vector.shape[-1])
    POW = spec * np.conj(spec)
    totalPOW = np.sum(POW)
    medfreq = 0
    for i in range(0, vector.shape[0]):
        if np.sum(POW[0:i]) > 0.5 * totalPOW:
            medfreq = fs*i/vector.shape[0]
            break
    return medfreq


def getMFL(vector):
    """
    maximum fractal length
    """
    try:
        val = np.log10(np.sum(abs(np.diff(vector))))
    except:
        return 0
    return val


def getMNF(vector, fs=500):
    """
    mean frequency
    """
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    f = np.fft.fftfreq(vector.shape[-1])*fs
    spec = spec[0:int(round(spec.shape[0]/2))]
    f = f[0:int(round(f.shape[0]/2))]
    POW = spec * np.conj(spec)

    return np.sum(POW*f)/sum(POW)


def getMNP(vector):
    """
    mean power
    """
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0]/2))]
    POW = spec*np.conj(spec)
    return np.sum(POW)/POW.shape[0]


def getMYOP(vector, threshold=1.0):
    """
    myopulse percentage rate
    """
    return np.sum(np.abs(vector) >= threshold)/float(vector.shape[0])


def getRMS(vector):
    """
    root mean square: represents the mean power of the sEMG signal, which reflects the activity of muscles.
    """
    vector = np.asarray(vector)
    return np.sqrt(np.mean(np.square(vector)))


def getSM(vector, order=2, fs=500):
    """
    spectral moment
    """
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0]/2))]
    f = np.fft.fftfreq(vector.shape[-1]) * fs
    f = f[0:int(round(f.shape[0] / 2))]
    POW = spec*np.conj(spec)
    return np.sum(POW.dot(np.power(f, order)))


def getSSC(vector, threshold=0.1):
    """
    slope sign change: indicates the frequency information of the sEMG signal.
    """
    # return np.sum(1 * (np.diff(np.sign(vector)) != 0)) # BUG: always returns 0
    vector = np.asarray(vector)
    slope_change = 0
    for i in range(1, len(vector)-1):
        get_x = (vector[i]-vector[i-1])*(vector[i]-vector[i+1])
        if(get_x >= threshold):
            slope_change += 1
    return slope_change


def getSSI(vector):
    """
    simple square integral
    """
    vector = np.asarray(vector)
    return np.sum(np.square(vector))


def getSTD(vector):
    vector = np.asarray(vector)
    return np.std(vector)


def getTM(vector, order=3):
    """
    absolute temporal moment
    """
    vector = np.asarray(vector)
    return np.abs(np.mean(np.power(vector, order)))


def getTTP(vector):
    """
    total power
    """
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0]/2))]
    POW = spec*np.conj(spec)
    return np.sum(POW)


def getVAR(vector):
    """
    variance
    """
    vector = np.asarray(vector)
    return np.square(np.std(vector))


def getWAMP(vector, threshold=0.1):
    """
    Wilson amplitude
    """
    vector = np.asarray(vector)
    wamp_decision = 0
    for i in range(1, len(vector)):
        get_x = abs(vector[i] - vector[i - 1])
        if (get_x >= threshold):
            wamp_decision += 1
    return wamp_decision


def getWL(vector):
    """
    waveform length: the cumulative length of the sEMG signal waveform, which is related to waveform amplitude, frequency, and time and can be used to measure signal complexity.
    """
    vector = np.asarray(vector)
    return np.sum(np.abs(np.diff(vector)))


def getZC(vector, threshold=0.1):
    """
    zero crossing
    """
    vector = np.asarray(vector)
    number_zero_crossing = 0
    current_sign = cmp(vector[0], 0)
    for i in range(0, len(vector)):
        if current_sign == -1:
            # We give a delta to consider that the zero was crossed
            if current_sign != cmp(vector[i], threshold):
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
        else:
            if current_sign != cmp(vector[i], -threshold):
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
    return number_zero_crossing


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
        "SampEn(1)",  # multiple
        "SampEn(2)",
        "SampEn(3)",
        "IQR(start)",  # multiple
        "IQR(end)",
        "HIST(1)",  # multiple
        "HIST(2)",
        "HIST(3)",
        # "DAR(1)", # always 1
        "DAR(2)",   # multiple
        "DAR(3)",
        "DAR(4)",
        "DAR(5)",
        # "AR(1)",  # always 1
        "AR(2)",    # multiple
        "AR(3)",
        "AR(4)",
        "AR(5)",
        # "CC(1)",  # always 1
        "CC(2)",    # multiple
        "CC(3)",
        "CC(4)",
        "CC(5)",
        "BC",
        "DASDV",
        "HP_A",
        "HP_M",
        "HP_C",
        "IEMG",
        "KURT",
        "SKEW",
        "LD",
        "MAV",
        "MMAV1",
        "MMAV2",
        # "MDF", # constant
        "MFL",
        "MNF",
        "MNP",
        # "MYOP", # constant
        "RMS",
        "SM",
        # "SSC", # constant
        "SSI",
        "STD",
        "TM",
        "TTP",
        "VAR",
        # "WAMP",
        "WL",
        # "ZC", # constant
    ]

    if (only_return_labels):
        return feature_labels

    # rearrage data so that time points are the last dimension
    data_r = np.moveaxis(data, 1, -1)

    # (n x c x f)
    res = []
    for n in range(data_r.shape[0]):
        print("\tfinished {}/{}".format(n, data_r.shape[0]))
        n_arr = []
        for c in range(data_r.shape[1]):
            c_arr = []

            # multiple value features
            c_arr.extend(getSampEn(data_r[n, c, :]))
            c_arr.extend(getIQR(data_r[n, c, :]))
            c_arr.extend(getHIST(data_r[n, c, :]))
            c_arr.extend(getDAR(data_r[n, c, :]))
            c_arr.extend(getAR(data_r[n, c, :]))
            c_arr.extend(getCC(data_r[n, c, :]))

            # single value features
            c_arr.append(getBC(data_r[n, c, :]))
            c_arr.append(getDASDV(data_r[n, c, :]))
            c_arr.append(getHP_A(data_r[n, c, :]))
            c_arr.append(getHP_M(data_r[n, c, :]))
            c_arr.append(getHP_C(data_r[n, c, :]))
            c_arr.append(getIEMG(data_r[n, c, :]))
            c_arr.append(getKURT(data_r[n, c, :]))
            c_arr.append(getSKEW(data_r[n, c, :]))
            c_arr.append(getLD(data_r[n, c, :]))
            c_arr.append(getMAV(data_r[n, c, :]))
            c_arr.append(getMMAV1(data_r[n, c, :]))
            c_arr.append(getMMAV2(data_r[n, c, :]))
            # c_arr.append(getMDF(data_r[n, c, :]))
            c_arr.append(getMFL(data_r[n, c, :]))
            c_arr.append(getMNF(data_r[n, c, :]))
            c_arr.append(getMNP(data_r[n, c, :]))
            # c_arr.append(getMYOP(data_r[n, c, :]))
            c_arr.append(getRMS(data_r[n, c, :]))
            c_arr.append(getSM(data_r[n, c, :]))
            # c_arr.append(getSSC(data_r[n, c, :]))
            c_arr.append(getSSI(data_r[n, c, :]))
            c_arr.append(getSTD(data_r[n, c, :]))
            c_arr.append(getTM(data_r[n, c, :]))
            c_arr.append(getTTP(data_r[n, c, :]))
            c_arr.append(getVAR(data_r[n, c, :]))
            # c_arr.append(getWAMP(data_r[n, c, :]))
            c_arr.append(getWL(data_r[n, c, :]))
            # c_arr.append(getZC(data_r[n, c, :]))

            c_narr = np.array(c_arr)
            np.nan_to_num(c_narr, copy=False)

            n_arr.append(c_narr)
        n_narr = np.array(n_arr)
        res.append(n_narr)

    # turn features into ndarray
    features = np.array(res)
    features = np.moveaxis(features, -1, 0)
    features = np.real(features)

    # remove known constant features (first AR coef always 1)
    # reverse to pop multiple items without issues
    drop_idx = [18, 13, 8]
    drop_idx = list(set(drop_idx))
    drop_idx.sort(reverse=True)
    if(drop_idx):
        features = np.delete(features, drop_idx, axis=0)

    # check for constant features
    drop_idx = []
    drop_flabels = []
    print("features.shape", features.shape)
    for i, f in enumerate(feature_labels):
        for c in range(features.shape[-1]):
            if (verbose):
                print("feature: ", f)
                print("\tmin", min(features[i, :, c]))
                print("\tmax", max(features[i, :, c]))
            if (min(features[i, :, c]) == max(features[i, :, c])):
                drop_idx.append(i)
                drop_flabels.append(f)

    # remove any columns with constant values across all samples
    if (drop_idx):
        print("WARNING: constant features", drop_flabels)
        print("WITH INDICIES:", drop_idx)

    if(drop_constants):
        # reverse to pop multiple items without issues
        drop_idx = list(set(drop_idx))
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
    w = 0
    for e in range(window_span, X.shape[1] + 1, window_incr):
        s = e - window_incr
        print("window {}/{}".format(w, n_windows))
        fm, _ = extract_features(X[:, s:e, :])
        features[:, i:i+fm.shape[1], :] = fm
        if (verbose):
            print("\n----\n \t mean \t std ")
            for f, flabel in enumerate(feature_labels):
                for c in range(fm.shape[2]):
                    print(flabel + str(c), np.mean(features[f, i:i+X.shape[0], c]),
                          np.std(features[f, i:i+X.shape[0], c]))
        w += 1
        i += X.shape[0]

    # replace nans
    np.nan_to_num(features, copy=False)

    return features, feature_labels, n_windows
