import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

def detect_spikes(trace, fs, threshold=-10.0, refractory_ms=2.0, prominence=5.0):
    """Detect spikes in voltage trace using peak detection.
    
    Parameters
    ----------
    trace : array_like
        Voltage trace
    fs : float
        Sampling frequency (Hz)
    threshold : float
        Minimum voltage for spike detection
    refractory_ms : float
        Refractory period in milliseconds
    prominence : float
        Minimum prominence for peaks
        
    Returns
    -------
    peaks : array
        Indices of detected spikes
    props : dict
        Properties of detected peaks
    """
    refractory_samples = int(refractory_ms * 1e-3 * fs)
    peaks, props = find_peaks(
        trace,
        height=threshold,
        distance=refractory_samples,
        prominence=prominence
    )
    return peaks, props


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
        features. This allows ignoring initial transients. Default is 0.
    window_bursts : int or None, optional
        If given, only this many bursts after ``skip_bursts`` are used for the
        feature calculations. When ``None`` (default) the remainder of the
        trace is used.

    Returns
    -------
    dict
        Dictionary containing spike and burst statistics.
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
        cv_isi = float(np.std(isi) / mean_isi) if mean_isi > 0 else np.nan
    else:
        mean_isi = np.nan
        cv_isi = np.nan
    
    features.update({
        "mean_isi": mean_isi,
        "cv_isi": cv_isi
    })

    # 2) Simple burst splitting: threshold = mean_isi * 2
    if not np.isnan(mean_isi):
        isi_gap = mean_isi * 2
    else:
        isi_gap = 0.0
        
    bursts = []
    if len(t_spike) > 0 and not np.isnan(mean_isi):
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

    # 4) spikes per burst
    if n_bursts > 0:
        spikes_per_burst = np.array([len(b) for b in bursts])
        features["mean_spikes_per_burst"] = float(np.mean(spikes_per_burst))
    else:
        features["mean_spikes_per_burst"] = np.nan

    # 5) inter-burst frequencies (Hz)
    if n_bursts > 1:
        t_burst = np.array([b[0] for b in bursts])
        ibi = np.diff(t_burst)  # inter-burst intervals in s
        if len(ibi) > 0:
            interburst_freq = float(np.mean(1.0 / ibi))
        else:
            interburst_freq = np.nan
    else:
        interburst_freq = np.nan
        ibi = []
    features["interburst_freq"] = interburst_freq

    # 6) intra-burst frequencies (Hz)
    intra_lists = [1.0/np.diff(b) for b in bursts if len(b) > 1]
    if intra_lists:
        all_intra = np.concatenate(intra_lists)
        intraburst_freq = float(np.mean(all_intra))
    else:
        intraburst_freq = np.nan
    features["intraburst_freq"] = intraburst_freq

    # 7) duty cycle = mean(duration/period) across bursts
    if n_bursts > 1 and len(ibi) > 0:
        durations = np.array([b[-1] - b[0] for b in bursts])
        # Only calculate duty cycle for bursts that have a following burst
        ratios = durations[:-1] / ibi
        duty_cycle = float(np.mean(ratios))
    else:
        duty_cycle = np.nan
    features["duty_cycle"] = duty_cycle

    # Regime classification based on presence of long inter-spike gaps
    if spike_count > 1 and not np.isnan(mean_isi):
        gap_thresh = mean_isi * 5.0
        if np.any(isi > gap_thresh):
            features["regime"] = "bursting"
        else:
            features["regime"] = "tonic_spiking"
    elif spike_count == 1:
        features["regime"] = "tonic_spiking"
    else:
        features["regime"] = "quiescent"

    return features