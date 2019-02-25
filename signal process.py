from numpy.fft import *
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import *
import gc
from sklearn.feature_selection import f_classif
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, uniform, norm
from scipy.stats import randint, poisson
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
from tqdm import *
import pywt
from statsmodels.robust import mad
import scipy
from scipy import signal
from scipy.signal import butter
import scipy.stats as ss
import statsmodels
import warnings


#Fast Fourier Transform denoising
def filter_signal(signal, threshold=1e8):
    """
    param signal:1D array
    """
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)



# 800,000 data points taken over 20 ms
# Grid operates at 50hz, 0.02 * 50 = 1, so 800k samples in 20 milliseconds will capture one complete cycle
n_samples = 800000

# Sample duration is 20 miliseconds
sample_duration = 0.02

# Sample rate is the number of samples in one second
# Sample rate will be 40mhz
sample_rate = n_samples * (1 / sample_duration)

def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff=1000, sample_rate=sample_rate):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    param x:1D array
    """
    
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    #sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')
    
    # scipy version 1.1.0
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig

def denoise_signal( x, wavelet='db4', level=1):
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest( coeff[-level] )

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )


import collections
import concurrent.futures
import glob
import json
import math
import multiprocessing as mp
import os

import numpy as np
import pywt
from scipy import signal
from scipy import stats
import tqdm


def rolling_window(arr, window):
    """Returns an array view that can be used to calculate rolling statistics.
    From http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html.
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def mad(x, axis=None):
    return np.mean(np.abs(x - np.mean(x, axis)), axis)


def wavelet_denoise(x, wavelet='db1', mode='hard'):

    # Extract approximate and detailed coefficients
    c_a, c_d = pywt.dwt(x, wavelet)

    # Determine the threshold
    sigma = 1 / 0.6745 * mad(np.abs(c_d))
    threshold = sigma * math.sqrt(2 * math.log(len(x)))

    # Filter the detail coefficients
    c_d_t = pywt.threshold(c_d, threshold, mode=mode)

    # Reconstruct the signal
    y = pywt.idwt(np.zeros_like(c_a), c_d_t, wavelet)

    return y


def peaks(x):
    y = wavelet_denoise(x)
    peaks, properties = signal.find_peaks(y)
    widths = signal.peak_widths(y, peaks)[0]
    prominences = signal.peak_prominences(y, peaks)[0]
    return {
        'count': peaks.size,
        'width_mean': widths.mean() if widths.size else -1.,
        'width_max': widths.max() if widths.size else -1.,
        'width_min': widths.min() if widths.size else -1.,
        'prominence_mean': prominences.mean() if prominences.size else -1.,
        'prominence_max': prominences.max() if prominences.size else -1.,
        'prominence_min': prominences.min() if prominences.size else -1.,
    }


def denoised_std(x):
    return np.std(wavelet_denoise(x))


def signal_entropy(x):

    y = wavelet_denoise(x)

    for i in range(3):
        max_pos = y.argmax()
        y[max_pos - 1000:max_pos + 1000] = 0.

    return stats.entropy(np.histogram(y, 15)[0])


def detail_coeffs_entropy(x, wavelet='db1'):

    c_a, c_d = pywt.dwt(x, wavelet)

    return stats.entropy(np.histogram(c_d, 15)[0])


def bucketed_entropy(x):

    y = wavelet_denoise(x)

    return {
        f'bucket_{i}': stats.entropy(np.histogram(bucket, 10)[0])
        for i, bucket in enumerate(np.split(y, 10))
    }


FUNCS = [
    np.mean,
    np.std,
    stats.kurtosis,
    peaks,
    denoised_std,
    signal_entropy,
    detail_coeffs_entropy,
    bucketed_entropy
]
