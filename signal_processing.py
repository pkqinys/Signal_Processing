import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft

def denoise(raw_data, delta_t):
    filter = lambda xval, freq2: 0 if abs(freq2) > 4000000000 else xval
    freq = np.fft.fftfreq(len(raw_data), delta_t)
    F_filtered = [filter(x, freq) for x, freq in zip(fft(raw_data), freq)]
    s_filtered = ifft(F_filtered)
    return s_filtered
