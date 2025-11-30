import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
import time
from datetime import datetime

# SDR initialization
sdr = RtlSdr()
sdr.sample_rate = 2.4e6 # Hz
sdr.center_freq = 52.0e6   # Hz
sdr.freq_correction = 60  # PPM
sdr.gain = 49.6

# Spectrogram/FFT config
fft_size = 1024
seconds_per_iter = 5
num_rows = 5000
interval_width = 5

csv_filename = "energy_peaks.csv"

# Write CSV header if file doesn't exist
try:
    with open(csv_filename, 'x') as f:
        f.write("timestamp,peak_value\n")
except FileExistsError:
    pass

for i in range(1,5):
    # 1. Sample data for 5 seconds
    x = sdr.read_samples(fft_size * num_rows)
    spectrogram = np.zeros((num_rows, fft_size))

    # 2. Compute spectrogram
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
    spect_df = pd.DataFrame(spectrogram)

    # 3. Compute mean vals over intervals
    num_intervals = int(num_rows / interval_width)
    mean_vals = np.zeros((num_intervals, 1))

    for i in range(0, num_rows, interval_width):
        mean_vals[int(i / interval_width)] = spect_df.iloc[i:i+interval_width].values.mean()

    noise_floor = np.sqrt(np.mean(mean_vals**2))
    normalized_mean_vals = mean_vals / noise_floor

    # 4. Compute difference (energy surge detector)
    d_mean_vals = np.zeros((num_intervals, 1))
    for i in range(0, num_intervals-1):
        d_mean_vals[i] = (normalized_mean_vals[i+1] - normalized_mean_vals[i]) / interval_width

    mean_change = np.percentile(d_mean_vals, 95)
    above_threshold = d_mean_vals[d_mean_vals > mean_change]

    # 5. Store peaks with timestamp
    timestamp = datetime.utcnow().isoformat()
    with open(csv_filename, "a") as f:
        for peak in above_threshold:
            f.write(f"{timestamp},{peak}\n")

    print(f"{len(above_threshold)} peaks recorded at {timestamp}")

    # Optional: plot (comment out in headless/production use)
    # plt.figure()
    # plt.plot(range(num_intervals), normalized_mean_vals)
    # plt.stem(np.where(d_mean_vals > mean_change)[0], above_threshold)
    # plt.show(block=False)
    # plt.pause(0.1)
    # plt.close()

    # 6. Loop immediately for next 5 seconds