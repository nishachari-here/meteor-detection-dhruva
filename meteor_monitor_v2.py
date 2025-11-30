import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
import threading
import queue
import time
from datetime import datetime

# Parameters
SAMPLE_RATE = 2.4e6  # Hz
CENTER_FREQ = 52.0e6 # Hz
FREQ_CORR = 60       # PPM
GAIN = 49.6
FFT_SIZE = 1024
PROCESS_SECONDS = 5
BUFFER_SECONDS = 20  # Size of rolling buffer (should be >= PROCESS_SECONDS)
CSV_FILENAME = "energy_peaks.csv"
NUM_RUNS = 5         # Number of processing intervals for testing

# Global buffer (thread-safe)
sample_buffer = queue.Queue(maxsize=int(SAMPLE_RATE * BUFFER_SECONDS))

stop_event = threading.Event()

# SDR Reader Thread
def sdr_reader():
    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.freq_correction = FREQ_CORR
    sdr.gain = GAIN

    samples_per_chunk = 2048  # Read in small chunks for responsiveness

    while not stop_event.is_set():
        x = sdr.read_samples(samples_per_chunk)
        # Push samples to buffer, discard if full
        for sample in x:
            try:
                sample_buffer.put_nowait(sample)
            except queue.Full:
                pass  # Buffer is full, drop oldest data

# Detection/Processing Thread
def processor():
    samples_needed = int(SAMPLE_RATE * PROCESS_SECONDS)
    if not pd.io.common.file_exists(CSV_FILENAME):
        with open(CSV_FILENAME, "w") as f:
            f.write("timestamp,peak_value\n")

    run_count = 0
    while run_count < NUM_RUNS:
        # Collect enough samples for PROCESS_SECONDS
        samples = []
        while len(samples) < samples_needed:
            try:
                samples.append(sample_buffer.get(timeout=1))
            except queue.Empty:
                if stop_event.is_set():
                    return
                continue  # Wait for samples

        x = np.array(samples)
        num_rows = int(len(x) // FFT_SIZE)
        spectrogram = np.zeros((num_rows, FFT_SIZE))

        # FFT
        for i in range(num_rows):
            spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*FFT_SIZE:(i+1)*FFT_SIZE])))**2)
        spect_df = pd.DataFrame(spectrogram)

        # Mean and noise normalization
        interval_width = 5
        mean_vals = []
        for i in range(0, num_rows, interval_width):
            mean_vals.append(spect_df.iloc[i:i+interval_width].values.mean())
        mean_vals = np.array(mean_vals).reshape(-1, 1)
        num_intervals = len(mean_vals)

        noise_floor = np.sqrt(np.mean(mean_vals**2))
        normalized_mean_vals = mean_vals / noise_floor

        d_mean_vals = np.zeros((num_intervals, 1))
        for i in range(0, num_intervals-1):
            d_mean_vals[i] = (normalized_mean_vals[i+1] - normalized_mean_vals[i]) / interval_width

        mean_change = np.percentile(d_mean_vals, 95)
        above_threshold = d_mean_vals[d_mean_vals > mean_change]
        timestamp = datetime.utcnow().isoformat()
        with open(CSV_FILENAME, "a") as f:
            for peak in above_threshold:
                f.write(f"{timestamp},{peak}\n")
        print(f"{len(above_threshold)} peaks recorded at {timestamp}")

        run_count += 1

    # Signal reader thread to stop
    stop_event.set()

# Launch threads
reader_thread = threading.Thread(target=sdr_reader, daemon=True)
processor_thread = threading.Thread(target=processor, daemon=True)

reader_thread.start()
processor_thread.start()

# Wait for processor thread to finish
processor_thread.join()
stop_event.set()
print("Test run complete.")