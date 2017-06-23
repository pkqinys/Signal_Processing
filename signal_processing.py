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


def extract_wave_feature(index, representative_type, file_name):
    """ 
    We return a single number to quantify features of the waveform at specified index. 
    Supported types of features are: 'peak'(after Fourier Transformation), 'nonft_peak',
    'peak_width', 'peak_location', 'peak_space' and 'variation'.
    :param index: index of a wave within the file_name.
    :param representative_type: the value you want to extract from the waveform.
    :param file_name: the absolute path of a csv file.
    :return: a quantified feature.
    """

    # Read all data between two time point
    all_data = pd.read_csv(file_name, names=np.arange(512))

    # Use max value as the representative
    if representative_type == 'peak':
        # Perform Fourier Transformation
        waveform_1 = all_data.ix[index]
        waveform_1_mod = pd.Series(np.sqrt(fft(waveform_1).real ** 2 + fft(waveform_1).imag ** 2))[6:-6].get_values()

        # Determine the max index and the threshold
        max_index = np.argmax(waveform_1_mod)
        max_mod = waveform_1_mod[max_index]
        max_log = math.log(max_mod, math.e)

        # Put the highest peak value into the list
        print(r_z, r_r, max_log)
        return max_log

    elif representative_type == 'nonft_peak':
        # Perform Fourier Transformation
        waveform_1 = all_data.ix[index][6:-6].get_values()

        # Determine the max index and the threshold
        max_index = np.argmax(waveform_1)
        maxp = waveform_1[max_index]

        return math.log(maxp, math.e)

    # Use the numbers of measurements around the max sin/cos density as the representative
    elif representative_type == 'peak_width':
        # Perform Fourier Transformation
        waveform_1 = all_data.ix[index]
        waveform_1_mod = pd.Series(np.sqrt(fft(waveform_1).real ** 2 + fft(waveform_1).imag ** 2))[6:-6].get_values()

        # Determine the max index and the threshold
        max_index = np.argmax(waveform_1_mod)
        max_mod = waveform_1_mod[max_index]
        floor = 0.5 * max_mod
        bucket = []

        # Propagate forward
        for i in np.arange(max_index, len(waveform_1_mod)):
            if waveform_1_mod[i] >= floor:
                bucket.append(waveform_1_mod[i])
            else:
                break

        # Propagate backward
        for i in np.arange(max_index - 1, 0, -1):
            if waveform_1_mod[i] >= floor:
                bucket.insert(0, waveform_1_mod[i])
            else:
                break

        # Put the peak width into the list
        print(r_z, r_r, len(bucket))
        return len(bucket)

    elif representative_type == 'peak_location':
        waveform_1 = all_data.ix[index]
        waveform_1_mod = pd.Series(np.sqrt(fft(waveform_1).real ** 2 + fft(waveform_1).imag ** 2))[6:-6].get_values()
        maxp = np.max(waveform_1_mod)
        return find_peaks(waveform_1_mod, limit=maxp * 0.9999999)

    elif representative_type == 'peak_space':
        waveform_1 = all_data.ix[index]
        waveform_1_mod = pd.Series(np.sqrt(fft(waveform_1).real ** 2 + fft(waveform_1).imag ** 2))[6:-6].get_values()
        maxp = np.max(waveform_1_mod)
        temp = find_peaks(waveform_1_mod, limit=maxp * 0.9999999)
        space = (temp[1] - temp[0]) / 1000.0
        print(r_z, r_r, space)
        return space

    elif representative_type == 'variation':
        waveform_1 = all_data.ix[index]
        waveform_1_mod = pd.Series(np.sqrt(fft(waveform_1).real ** 2 + fft(waveform_1).imag ** 2))[6:-6].get_values()
        maxp = np.max(waveform_1_mod)
        temp = find_peaks(waveform_1_mod, limit=maxp * 0.9999999)
        variation = np.var(waveform_1_mod[temp[0] + 50: temp[1] - 50]) * 1000000
        print(r_z, r_r, variation)
        return variation

    else:
        raise ValueError('Invalid representative type!')





