import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

def detect_spikes(trace, fs, threshold=-10.0, refractory_ms=2.0, prominence=5.0):
    # same implementation as before
    refractory_samples = int(refractory_ms * 1e-3 * fs)
    peaks, props = find_peaks(
        trace,
        height=threshold,
        distance=refractory_samples,
        prominence=prominence
    )
    return peaks, props

# import time

def choose_isi_gap(t_spike):
    """Return a data-driven burst gap (s) from the log-ISI histogram valley."""
    # t0 = time.time()
    isi = np.diff(t_spike)
    log_isi = np.log10(isi)
    xs = np.linspace(log_isi.min(), log_isi.max(), 1024)
    kde = gaussian_kde(log_isi)(xs)
    isi_gap = 10 ** xs[np.argmin(kde)]     # valley → threshold (s)
    # t1 = time.time()
    # time_diff = tend - t1
    # print(f"Time taken to compute ISI gap: {time_diff:.4f} seconds")
    return isi_gap

def split_into_bursts(t_spike, isi_gap=None):
    isi = np.diff(t_spike)
    if isi_gap is None:
        isi_gap = choose_isi_gap(t_spike)
    edges     = np.where(isi > isi_gap)[0]          # last spike of each burst
    starts    = np.insert(edges+1, 0, 0)
    stops     = np.append(edges, len(t_spike)-1)
    bursts    = [t_spike[s:e+1] for s, e in zip(starts, stops)]
    return bursts, isi_gap

def burst_metrics(bursts):
    """Return spikes/burst, inter-burst frequency, intra-burst frequencies."""
    # 3.1 spikes per burst
    n_spikes_pb = np.array([len(b) for b in bursts])

    # 3.2 inter-burst freq: first-spike times → IBIs
    t_burst   = np.array([b[0] for b in bursts])
    ibi       = np.diff(t_burst)           # s
    f_inter   = 1 / ibi                    # Hz

    # 3.3 intra-burst freq: ISIs inside each burst
    f_intra_lists = [1/np.diff(b) for b in bursts if len(b) > 1]

    return dict(n_spikes_per_burst=n_spikes_pb,
                inter_burst_freq=f_inter,
                intra_burst_freq_lists=f_intra_lists)


def extract_features(V_trace, dt):
    """
    Compute spike/burst features from a voltage trace.
    Returns a dict with keys: spike_count, mean_isi, cv_isi,
    n_bursts, burst_freq, mean_spikes_per_burst,
    mean_burst_duration, duty_cycle, regime
    """
    fs = 1.0 / dt
    peaks, props = detect_spikes(V_trace, fs)
    t_spike = peaks * dt
    spike_count = len(peaks)
    features = {"spike_count": spike_count}
    if spike_count == 0:
        features.update({
            "regime": "quiescent",
            "mean_isi": np.nan,
            "cv_isi": np.nan,
            "n_bursts": 0,
            "burst_freq": np.nan,
            "mean_spikes_per_burst": 0,
            "mean_burst_duration": 0,
            "duty_cycle": np.nan
        })
        return features

    # Inter-spike intervals
    isi = np.diff(t_spike)
    features["mean_isi"] = float(np.mean(isi))
    features["cv_isi"] = float(np.std(isi) / np.mean(isi))

    # Simple burst splitting: threshold = mean_isi * 2
    isi_gap = features["mean_isi"] * 2
    bursts = []
    current = [t_spike[0]]
    for dt_isi, t_next in zip(isi, t_spike[1:]):
        if dt_isi <= isi_gap:
            current.append(t_next)
        else:
            bursts.append(current)
            current = [t_next]
    bursts.append(current)

    n_bursts = len(bursts)
    features["n_bursts"] = n_bursts
    features["mean_spikes_per_burst"] = float(np.mean([len(b) for b in bursts]))

    # Burst durations and periods
    durations = [b[-1] - b[0] for b in bursts]
    features["mean_burst_duration"] = float(np.mean(durations))
    periods = []
    for i in range(n_bursts - 1):
        periods.append(bursts[i + 1][0] - bursts[i][0])
    features["burst_freq"] = float(1.0 / np.mean(periods)) if periods else np.nan
    features["duty_cycle"] = float(np.mean([d / p for d, p in zip(durations, periods)])) if periods else np.nan

    # Regime classification
    if n_bursts > 1:
        features["regime"] = "bursting"
    else:
        features["regime"] = "tonic"

    return features


def classify_regime(features):
    """Return the regime string from extracted features dict."""
    return features.get("regime", "unknown")


# Fourier transform to analyse frequency content of membrane potential
def fft_membrane_potential(neuron, dt):
    
    V = np.array(neuron.Vvalues)
    N = len(V)
    V_fft = np.fft.fft(V)
    freq = np.fft.fftfreq(N, d=dt)

    # Consider only positive frequencies
    mask = freq >= 0

    plt.figure(figsize=(12, 6))
    plt.plot(freq[mask], np.abs(V_fft[mask]), label="FFT amplitude")
    plt.title("Frequency Spectrum of Membrane Potential")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    dt = 5e-5
    # Read membrane potential data in csv
    csvpath = os.path.join(os.path.dirname(__file__), "SynapticNeuron_Plots/neuronV.csv")
    data = pd.read_csv(csvpath)
    time_axis = data.iloc[:, 0].values
    Vvalues = data.iloc[:, 1].values
    # time_axis = np.arange(0, len(Vvalues) * dt, dt)

    peaks, t_spike = detect_spikes(Vvalues, fs=1000, threshold=-10.0, refractory_ms=2.0, prominence=5.0)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(time_axis, Vvalues, label="Membrane Potential")
    axs[0].plot(time_axis[peaks], Vvalues[peaks], "x", label="Detected Spikes", color='red')

    isi_gap = choose_isi_gap(t_spike)
    bursts, isi_gap = split_into_bursts(t_spike, isi_gap=isi_gap)
    burst_metrics_data = burst_metrics(bursts)
    print(f"ISI gap: {isi_gap:.4f} s")
    print(f"Spikes per burst: {burst_metrics_data['n_spikes_per_burst']}")
    print(f"Inter-burst frequency: {burst_metrics_data['inter_burst_freq']}")
    print(f"Intra-burst frequencies: {burst_metrics_data['intra_burst_freq_lists']}")
    

    # for burst in bursts:
    #     axs[1].plot(burst, Vvalues[peaks][np.isin(peaks, burst)], "o", label="Burst", color='orange')
    # axs[1].set_title("Detected Spikes and Bursts")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Membrane Potential (mV)")
    # axs[1].legend()
    plt.grid(True)

    plt.show()
