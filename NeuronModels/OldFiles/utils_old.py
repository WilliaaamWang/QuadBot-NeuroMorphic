import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

def detect_spikes(trace, fs, threshold=-10.0, refractory_ms=2.0, prominence=5.0):
    refractory_samples = int(refractory_ms * 1e-3 * fs)
    peaks, props = find_peaks(
        trace,
        height=threshold,
        distance=refractory_samples,
        prominence=prominence
    )
    return peaks, props


# def choose_isi_gap(t_spike):
#     """Return a data-driven burst gap (s) from the log-ISI histogram valley."""
#     isi = np.diff(t_spike)
#     log_isi = np.log10(isi)
#     xs = np.linspace(log_isi.min(), log_isi.max(), 1024)
#     kde = gaussian_kde(log_isi)(xs)
#     isi_gap = 10 ** xs[np.argmin(kde)]     # valley → threshold (s)
#     return isi_gap

# def split_into_bursts(t_spike, isi_gap=None):
#     isi = np.diff(t_spike)
#     if isi_gap is None:
#         isi_gap = choose_isi_gap(t_spike)
#     edges     = np.where(isi > isi_gap)[0]          # last spike of each burst
#     starts    = np.insert(edges+1, 0, 0)
#     stops     = np.append(edges, len(t_spike)-1)
#     bursts    = [t_spike[s:e+1] for s, e in zip(starts, stops)]
#     return bursts, isi_gap

# def burst_metrics(bursts):
#     """Return spikes/burst, inter-burst frequency, intra-burst frequencies."""
#     # 3.1 spikes per burst
#     n_spikes_pb = np.array([len(b) for b in bursts])

#     # 3.2 inter-burst freq: first-spike times → IBIs
#     t_burst   = np.array([b[0] for b in bursts])
#     ibi       = np.diff(t_burst)           # s
#     f_inter   = 1 / ibi                    # Hz

#     # 3.3 intra-burst freq: ISIs inside each burst
#     f_intra_lists = [1/np.diff(b) for b in bursts if len(b) > 1]

#     return dict(n_spikes_per_burst=n_spikes_pb,
#                 inter_burst_freq=f_inter,
#                 intra_burst_freq_lists=f_intra_lists)


def extract_features(V_trace, dt, skip_bursts: int = 0, window_bursts: int | None = None):
    """Compute spike/burst features from a voltage trace.

    Parameters
    ----------
    V_trace : array_like
        Membrane potential trace.
    dt : float
        Simulation time step (s).
    skip_bursts : int, optional
        Number of initial bursts to discard when computing burst related
        features.  This allows ignoring initial transients.  Default is 0.
    window_bursts : int or None, optional
        If given, only this many bursts after ``skip_bursts`` are used for the
        feature calculations.  When ``None`` (default) the remainder of the
        trace is used.

    Returns
    -------
    dict
        Dictionary containing spike and burst statistics such as
        ``mean_spikes_per_burst`` and ``duty_cycle``.
    """
    fs = 1.0 / dt
    peaks, _ = detect_spikes(V_trace, fs)
    features = {"peaks": peaks}

    spike_count = len(peaks)
    features["spike_count"] = spike_count

    if spike_count == 0:
        features.update({
            "regime": "quiescent",
            "mean_isi": np.nan,
            "cv_isi": np.nan,
            "n_bursts": 0,
            "mean_spikes_per_burst": np.nan,
            # "mean_burst_duration": np.nan,
            "interburst_freq": np.nan,
            "intraburst_freq": np.nan,
            "duty_cycle": np.nan
        })
        return features

    t_spike = peaks * dt

    # 1) Inter-spike intervals for determining burst gap
    isi = np.diff(t_spike)
    if len(isi) > 0:
        mean_isi = float(np.mean(isi))
        cv_isi   = float(np.std(isi) / mean_isi)
    else:
        mean_isi = np.nan
        cv_isi   = np.nan
    features.update({
        "mean_isi": mean_isi,
        "cv_isi":   cv_isi
    })

    # 2) Simple burst splitting: threshold = mean_isi * 2
    isi_gap = features["mean_isi"] * 2 if not np.isnan(features["mean_isi"]) else 0.0
    bursts = []
    if len(t_spike) > 0:
        current = [t_spike[0]]
        for dt_isi, t_next in zip(isi, t_spike[1:]):
            if dt_isi <= isi_gap:
                current.append(t_next)
            else:
                bursts.append(current)
                current = [t_next]
        bursts.append(current)

    # 3) Apply skip/window to select bursts for analysis
    bursts = bursts[skip_bursts:]
    if window_bursts is not None:
        bursts = bursts[:window_bursts]

    n_bursts = len(bursts)
    features["n_bursts"] = n_bursts
    
    # Spike times from the bursts selected for analysis
    window_spikes = np.concatenate(bursts) if n_bursts > 0 else np.array([])

    # Early exit if there are fewer than 2 spikes in the analysis window
    if len(window_spikes) < 2:
        features.update({
            "spike_count": len(window_spikes),
            "regime": "quiescent",
            "n_bursts": 0,
            "mean_isi": np.nan,
            "cv_isi": np.nan,
            "mean_spikes_per_burst": np.nan,
            "interburst_freq": np.nan,
            "intraburst_freq": np.nan,
            "duty_cycle": np.nan,
        })
        return features

    # Inter-spike intervals in the selected window
    isi_window = np.diff(window_spikes)
    mean_isi = float(np.mean(isi_window))
    cv_isi = float(np.std(isi_window) / mean_isi)
    features["mean_isi"] = mean_isi
    features["cv_isi"] = cv_isi   
    """
    # 2) choose data-driven ISI gap via KDE
    # log_isi = np.log10(isi)
    # xs = np.linspace(log_isi.min(), log_isi.max(), 1024)
    # kde_vals = gaussian_kde(log_isi)(xs)
    # isi_gap = 10 ** xs[np.argmin(kde_vals)]

    # # 3) split into bursts
    # edges  = np.where(isi > isi_gap)[0]
    # starts = np.insert(edges + 1, 0, 0)
    # stops  = np.append(edges, len(t_spike)-1)
    # bursts2 = [t_spike[s:e+1] for s, e in zip(starts, stops)]

    # n_bursts2 = len(bursts2)
    # features["n_bursts2"] = n_bursts2
    """


    # 4) spikes per burst
    if n_bursts > 0:
        spikes_per_burst = np.array([len(b) for b in bursts])
        features["mean_spikes_per_burst"] = float(np.mean(spikes_per_burst))
    else:
        features["mean_spikes_per_burst"] = np.nan

    # spikes_per_burst2 = np.array([len(b) for b in bursts2])
    # features["mean_spikes_per_burst2"] = float(np.mean(spikes_per_burst2))

    # 5) inter-burst frequencies (Hz)
    if n_bursts > 0:
        t_burst = np.array([b[0] for b in bursts])
        ibi     = np.diff(t_burst)  # inter-burst intervals in s
        if len(ibi) > 0:
            interburst_freq = float(np.mean(1.0 / ibi))
        else:
            interburst_freq = np.nan
    else:
        interburst_freq = np.nan
        ibi = []
    features["interburst_freq"] = interburst_freq

    # 6) intra-burst frequencies (Hz)
    intra_lists = [1.0/np.diff(b) for b in bursts if len(b)>1]
    if intra_lists:
        all_intra = np.concatenate(intra_lists)
        intraburst_freq = float(np.mean(all_intra))
    else:
        intraburst_freq = np.nan
    features["intraburst_freq"] = intraburst_freq

    # 7) duty cycle = mean(duration/period) across bursts
    if n_bursts > 0:
        durations = np.array([b[-1] - b[0] for b in bursts])
        if len(ibi) > 0:
            ratios = durations[:-1] / ibi    # skip last burst (no next period)
            duty_cycle = float(np.mean(ratios))
        else:
            duty_cycle = np.nan
    else:
        duty_cycle = np.nan
    features["duty_cycle"] = duty_cycle

    # Burst durations and periods
    # durations = [b[-1] - b[0] for b in bursts]
    # features["mean_burst_duration"] = float(np.mean(durations))
    # periods = []
    # for i in range(n_bursts - 1):
    #     periods.append(bursts[i + 1][0] - bursts[i][0])
    # features["burst_freq"] = float(1.0 / np.mean(periods)) if periods else np.nan
    # features["duty_cycle"] = float(np.mean([d / p for d, p in zip(durations, periods)])) if periods else np.nan

    # Regime classification based on presence of long inter-spike gaps
    # if spike_count > 1 and not np.isnan(mean_isi):
    if len(isi_window) > 0 and not np.isnan(mean_isi):
        gap_thresh = mean_isi * 5.0
        # if np.any(isi > gap_thresh):
        if np.any(isi_window > gap_thresh):
            features["regime"] = "bursting"
        else:
            features["regime"] = "tonic_spiking"
    else:
        features["regime"] = "tonic_spiking"

    return features



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
    csvpath = os.path.join(os.path.dirname(__file__), "SynapticNeuron_Plots", "Varying_20.0s_I[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].csv")
    # csvpath = os.path.join(os.path.dirname(__file__), "SynapticNeuron_Plots", "neuronV.csv")
    data = pd.read_csv(csvpath)
    time_axis = data["time"].values
    Ivalues = data["I_ext"].values
    Vvalues = data["V"].values

    
    features = extract_features(Vvalues, dt)
    peaks = features["peaks"]
    print(f"Regime: {features['regime']}")
    print(f"Spike count: {features['spike_count']}")
    print(f"Mean ISI: {features['mean_isi']:.4f} s")
    print(f"CV ISI: {features['cv_isi']:.4f}")
    print(f"Number of bursts: {features['n_bursts']}")
    # print(f"Number of bursts (KDE): {features['n_bursts2']}")
    print(f"Mean spikes per burst: {features['mean_spikes_per_burst']:.4f}")
    # print(f"Mean spikes per burst 2: {features['mean_spikes_per_burst2']:.4f}")
    # print(f"Mean burst duration: {features['mean_burst_duration']:.4f} s")
    print(f"Duty cycle: {features['duty_cycle']:.4f}")
    # print(f"Duty cycle 2: {features['duty_cycle2']:.4f}")
    # print(f"Burst frequency: {features['burst_freq']:.4f} Hz")
    print(f"Interburst frequency: {features['interburst_freq']:.4f} Hz")
    print(f"Intraburst frequency: {features['intraburst_freq']:.4f} Hz")
    

    # Plot applied current & membrane potential with detected spikes in shared time axis
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(time_axis, Ivalues, label="Applied Current (I_ext)", color='orange')
    axs[0].set_ylabel("Current (nA)")
    axs[0].set_title("Applied Current and Membrane Potential")
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(time_axis, Vvalues, label="Membrane Potential (V)", color='blue')
    axs[1].plot(time_axis[peaks], Vvalues[peaks], "x", label="Detected Spikes", color='red')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Membrane Potential (mV)")
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()
