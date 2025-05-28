"""Parameter sweep for the quad-centre model."""

from __future__ import annotations

import os
import itertools
import time
from pathlib import Path
from typing import Generator as Gen, Iterable

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from numpy.random import default_rng

from synaptic_neuron import SynapticNeuron
from quad_centre import simulate_quadcentre
from utils import extract_features

# -----------------------------------------------------------------------------
# Hardware detection (mirrors hc_sweep.py)
# -----------------------------------------------------------------------------
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
    if HAS_TORCH:
        _GPU_NAME = torch.cuda.get_device_name(0)
        print(f"▶︎ [INFO] PyTorch sees CUDA device: {_GPU_NAME}")
    else:
        print("▶︎ [INFO] PyTorch present but no CUDA device visible — CPU only")
except ImportError:  # pragma: no cover - torch optional
    HAS_TORCH = False
    torch = None
    print("▶︎ [INFO] PyTorch not installed — CPU only build will be used")

if HAS_TORCH:
    confirm = input("Run sweeps on GPU? (y/n): ")
    if confirm.lower() != "y":
        HAS_TORCH = False
        print("▶︎ [INFO] Falling back to CPU")

# -----------------------------------------------------------------------------
# Default neuron parameters (same as hc_sweep)
# -----------------------------------------------------------------------------
DEFAULT_PARAMS = {
    'cap': 0.82, 'k': 250.0,
    'V0': -52.0, 'Vs0': -50.0, 'Vus0': -52.0,
    'g_f': 1.0, 'g_s': 0.5, 'g_us': 0.015,
    'tau_s': 4.3, 'tau_us': 278.0,
    'V_threshold': 20.0, 'V_peak': 20.0,
    'V_reset': -45.0, 'Vs_reset': 7.5, 'delta_Vus': 1.7,
    'Ve0': 0.0, 'Vi0': -90.0,
    'Ve_threshold': -20.0, 'Vi_threshold': -20.0,
    'tau_e': 50.0, 'tau_i': 50.0,
    'g_syn_e': 0.5, 'g_syn_i': 30.0
}

# Parameter sampling settings -------------------------------------------------
NUM_SAMPLES = 10
STD_FACTOR = 0.10
RNG_SEED = 42


def generate_param_ranges(n_samples: int = NUM_SAMPLES,
                          std_factor: float = STD_FACTOR,
                          seed: int = RNG_SEED):
    rng = default_rng(seed)
    ranges = {}
    neg_params = ["Vs0", "Vus0", "Vi_threshold", "Vi0"]
    for p in neg_params:
        mean = DEFAULT_PARAMS[p]
        sd = abs(mean) * std_factor
        samples = rng.normal(loc=mean, scale=sd, size=n_samples)
        ranges[p] = np.sort(samples)

    pos_params = [
        "g_us", "delta_Vus", "tau_us",
        "g_s", "tau_s", "g_syn_i", "tau_i",
    ]
    for p in pos_params:
        mean = DEFAULT_PARAMS[p]
        sd = abs(mean) * std_factor
        a = (0 - mean) / sd
        samples = truncnorm.rvs(a, np.inf, loc=mean, scale=sd,
                               size=n_samples, random_state=rng)
        ranges[p] = np.sort(samples)

    return ranges


PARAM_RANGES = generate_param_ranges()

PARAM_GROUPS = {
    'resting_potentials': ['Vs0', 'Vus0'],
    'ultraslow_dynamics': ['g_us', 'delta_Vus', 'tau_us'],
    'slow_dynamics': ['g_s', 'tau_s'],
    'synaptic': ['g_syn_i', 'tau_i', 'Vi_threshold', 'Vi0']
}

DT = 5e-5
RUNTIME = 15.0
I_EXT_AMPLITUDE = 5.0
I_EXT_START = 0.5

FEATURE_SKIP_BURSTS = 2
FEATURE_WINDOW_BURSTS = 2

TRACE_SAMPLES = 2
DEFAULT_GPU_BATCH = 400


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _auto_batch_size(bytes_per_sim: int, safety: float = 0.54,
                     hard_cap: int | None = None) -> int:
    if not HAS_TORCH:
        return 1
    free, _ = torch.cuda.mem_get_info()
    target = int(free * safety)
    bsz = max(target // bytes_per_sim, 1)
    if hard_cap:
        bsz = min(bsz, hard_cap)
    return bsz


def create_I_ext() -> np.ndarray:
    n_steps = int(RUNTIME / DT)
    I_ext = np.zeros(n_steps)
    start_idx = int(I_EXT_START / DT)
    I_ext[start_idx:] = I_EXT_AMPLITUDE
    return I_ext


def run_single_sim(params: dict, save_trace: bool = False) -> dict:
    neuron_params = DEFAULT_PARAMS.copy()
    neuron_params.update(params)

    A = SynapticNeuron(excitatory_Vin=None, inhibitory_Vin=None, **neuron_params)
    B = SynapticNeuron(excitatory_Vin=None, inhibitory_Vin=None, **neuron_params)
    C = SynapticNeuron(excitatory_Vin=None, inhibitory_Vin=None, **neuron_params)
    D = SynapticNeuron(excitatory_Vin=None, inhibitory_Vin=None, **neuron_params)

    I_ext = create_I_ext()
    empty = []

    A, B, C, D = simulate_quadcentre(
        A,
        B,
        C,
        D,
        # I_ext,
        # I_ext,
        # I_ext,
        # I_ext,
        np.array([I_ext,I_ext,I_ext,I_ext]),
        # empty,
        # empty,
        # empty,
        # empty,
        # empty,
        # empty,
        # empty,
        # empty,
        np.array([empty, empty, empty, empty]),
        np.array([empty, empty, empty, empty]),
        dt=DT,
        runtime=RUNTIME,
        plotter=False,
        same_start=False,
    )

    feats_A = extract_features(np.array(A.Vvalues), DT,
                               skip_bursts=FEATURE_SKIP_BURSTS,
                               window_bursts=FEATURE_WINDOW_BURSTS)
    feats_B = extract_features(np.array(B.Vvalues), DT,
                               skip_bursts=FEATURE_SKIP_BURSTS,
                               window_bursts=FEATURE_WINDOW_BURSTS)
    feats_C = extract_features(np.array(C.Vvalues), DT,
                               skip_bursts=FEATURE_SKIP_BURSTS,
                               window_bursts=FEATURE_WINDOW_BURSTS)
    feats_D = extract_features(np.array(D.Vvalues), DT,
                               skip_bursts=FEATURE_SKIP_BURSTS,
                               window_bursts=FEATURE_WINDOW_BURSTS)

    row = params.copy()
    for name, feats in zip("ABCD", [feats_A, feats_B, feats_C, feats_D]):
        row[f"regime_{name}"] = feats['regime']
        row[f"spike_count_{name}"] = feats['spike_count']
        row[f"mean_spikes_per_burst_{name}"] = feats['mean_spikes_per_burst']
        row[f"duty_cycle_{name}"] = feats['duty_cycle']
        row[f"interburst_freq_{name}"] = feats['interburst_freq']
        row[f"intraburst_freq_{name}"] = feats['intraburst_freq']
    if save_trace:
        row['V_trace_A'] = np.array(A.Vvalues)
        row['V_trace_B'] = np.array(B.Vvalues)
        row['V_trace_C'] = np.array(C.Vvalues)
        row['V_trace_D'] = np.array(D.Vvalues)
        row['time'] = np.arange(0, RUNTIME, DT)
    return row


# -----------------------------------------------------------------------------
# GPU batch processing (simplified from hc_sweep)
# -----------------------------------------------------------------------------

def run_gpu_batch(param_combos: list[dict],
                  trace_indices: Iterable[int] | None = None,
                  batch_size: int | None = None):
    if not HAS_TORCH:
        return None

    device = torch.device("cuda")
    total = len(param_combos)
    n_steps = int(RUNTIME / DT)

    trace_indices = sorted(set(trace_indices or []))
    keep_mask = torch.zeros(total, dtype=torch.bool)
    if trace_indices:
        keep_mask[trace_indices] = True

    bytes_per_trace = n_steps * 4 * 4  # four neurons
    est_bytes_per_sim = bytes_per_trace + 1024
    if batch_size is None:
        batch_size = _auto_batch_size(est_bytes_per_sim, hard_cap=DEFAULT_GPU_BATCH)
    print(f"▶︎ [GPU] processing {total} sims in chunks of {batch_size}")

    results = []
    time_cpu = np.arange(0, RUNTIME, DT, dtype=np.float32)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        slice_combos = param_combos[start:end]
        b = len(slice_combos)

        V_A = torch.full((b,), DEFAULT_PARAMS['V0'], device=device)
        V_B = torch.full((b,), DEFAULT_PARAMS['V0'] + 0.1, device=device)
        V_C = torch.full((b,), DEFAULT_PARAMS['V0'] + 0.2, device=device)
        V_D = torch.full((b,), DEFAULT_PARAMS['V0'] + 0.3, device=device)
        Vs_A = torch.full((b,), DEFAULT_PARAMS['Vs0'], device=device)
        Vs_B = torch.full((b,), DEFAULT_PARAMS['Vs0'], device=device)
        Vs_C = torch.full((b,), DEFAULT_PARAMS['Vs0'], device=device)
        Vs_D = torch.full((b,), DEFAULT_PARAMS['Vs0'], device=device)
        Vus_A = torch.full((b,), DEFAULT_PARAMS['Vus0'], device=device)
        Vus_B = torch.full((b,), DEFAULT_PARAMS['Vus0'], device=device)
        Vus_C = torch.full((b,), DEFAULT_PARAMS['Vus0'], device=device)
        Vus_D = torch.full((b,), DEFAULT_PARAMS['Vus0'], device=device)
        Si_A = torch.zeros((b,), device=device)
        Si_B = torch.zeros((b,), device=device)
        Si_C = torch.zeros((b,), device=device)
        Si_D = torch.zeros((b,), device=device)

        p_tensor = {}
        for p in PARAM_RANGES.keys():
            vals = [c.get(p, DEFAULT_PARAMS[p]) for c in slice_combos]
            p_tensor[p] = torch.tensor(vals, dtype=torch.float32, device=device)

        cap = torch.tensor(DEFAULT_PARAMS['cap'], device=device)
        k_const = torch.tensor(DEFAULT_PARAMS['k'], device=device)
        V0c = torch.tensor(DEFAULT_PARAMS['V0'], device=device)
        g_f = torch.tensor(DEFAULT_PARAMS['g_f'], device=device)
        V_thresh = torch.tensor(DEFAULT_PARAMS['V_threshold'], device=device)
        V_reset = torch.tensor(DEFAULT_PARAMS['V_reset'], device=device)
        Vs_reset = torch.tensor(DEFAULT_PARAMS['Vs_reset'], device=device)

        I_ext = torch.zeros(n_steps, device=device)
        I_ext[int(I_EXT_START/DT):] = I_EXT_AMPLITUDE

        traces_A = torch.zeros((b, n_steps), device=device)
        traces_B = torch.zeros((b, n_steps), device=device)
        traces_C = torch.zeros((b, n_steps), device=device)
        traces_D = torch.zeros((b, n_steps), device=device)

        for t in range(n_steps):
            inhib_A = 0.5 * (V_B + V_C)
            inhib_B = V_A
            inhib_C = 0.5 * (V_D + V_A)
            inhib_D = V_C

            dVs_A = k_const * (V_A - Vs_A) / p_tensor.get('tau_s')
            dVs_B = k_const * (V_B - Vs_B) / p_tensor.get('tau_s')
            dVs_C = k_const * (V_C - Vs_C) / p_tensor.get('tau_s')
            dVs_D = k_const * (V_D - Vs_D) / p_tensor.get('tau_s')
            dVus_A = k_const * (V_A - Vus_A) / p_tensor.get('tau_us')
            dVus_B = k_const * (V_B - Vus_B) / p_tensor.get('tau_us')
            dVus_C = k_const * (V_C - Vus_C) / p_tensor.get('tau_us')
            dVus_D = k_const * (V_D - Vus_D) / p_tensor.get('tau_us')

            Si_inf_A = torch.sigmoid(40 * (inhib_A - p_tensor.get('Vi_threshold')))
            Si_inf_B = torch.sigmoid(40 * (inhib_B - p_tensor.get('Vi_threshold')))
            Si_inf_C = torch.sigmoid(40 * (inhib_C - p_tensor.get('Vi_threshold')))
            Si_inf_D = torch.sigmoid(40 * (inhib_D - p_tensor.get('Vi_threshold')))
            dSi_A = k_const * (Si_inf_A - Si_A) / p_tensor.get('tau_i')
            dSi_B = k_const * (Si_inf_B - Si_B) / p_tensor.get('tau_i')
            dSi_C = k_const * (Si_inf_C - Si_C) / p_tensor.get('tau_i')
            dSi_D = k_const * (Si_inf_D - Si_D) / p_tensor.get('tau_i')

            I_inh_A = p_tensor.get('g_syn_i') * Si_A * (V_A - p_tensor.get('Vi0'))
            I_inh_B = p_tensor.get('g_syn_i') * Si_B * (V_B - p_tensor.get('Vi0'))
            I_inh_C = p_tensor.get('g_syn_i') * Si_C * (V_C - p_tensor.get('Vi0'))
            I_inh_D = p_tensor.get('g_syn_i') * Si_D * (V_D - p_tensor.get('Vi0'))

            dV_A = (k_const / cap) * (
                g_f * (V_A - V0c) ** 2
                - p_tensor.get('g_s') * (Vs_A - p_tensor.get('Vs0')) ** 2
                - p_tensor.get('g_us') * (Vus_A - p_tensor.get('Vus0')) ** 2
                + I_ext[t] - I_inh_A
            )
            dV_B = (k_const / cap) * (
                g_f * (V_B - V0c) ** 2
                - p_tensor.get('g_s') * (Vs_B - p_tensor.get('Vs0')) ** 2
                - p_tensor.get('g_us') * (Vus_B - p_tensor.get('Vus0')) ** 2
                + I_ext[t] - I_inh_B
            )
            dV_C = (k_const / cap) * (
                g_f * (V_C - V0c) ** 2
                - p_tensor.get('g_s') * (Vs_C - p_tensor.get('Vs0')) ** 2
                - p_tensor.get('g_us') * (Vus_C - p_tensor.get('Vus0')) ** 2
                + I_ext[t] - I_inh_C
            )
            dV_D = (k_const / cap) * (
                g_f * (V_D - V0c) ** 2
                - p_tensor.get('g_s') * (Vs_D - p_tensor.get('Vs0')) ** 2
                - p_tensor.get('g_us') * (Vus_D - p_tensor.get('Vus0')) ** 2
                + I_ext[t] - I_inh_D
            )

            V_A += dV_A * DT
            V_B += dV_B * DT
            V_C += dV_C * DT
            V_D += dV_D * DT
            Vs_A += dVs_A * DT
            Vs_B += dVs_B * DT
            Vs_C += dVs_C * DT
            Vs_D += dVs_D * DT
            Vus_A += dVus_A * DT
            Vus_B += dVus_B * DT
            Vus_C += dVus_C * DT
            Vus_D += dVus_D * DT
            Si_A += dSi_A * DT
            Si_B += dSi_B * DT
            Si_C += dSi_C * DT
            Si_D += dSi_D * DT

            spike_A = V_A >= V_thresh
            spike_B = V_B >= V_thresh
            spike_C = V_C >= V_thresh
            spike_D = V_D >= V_thresh
            if spike_A.any():
                V_A[spike_A] = V_reset
                Vs_A[spike_A] = Vs_reset
                Vus_A[spike_A] += p_tensor.get('delta_Vus')[spike_A]
            if spike_B.any():
                V_B[spike_B] = V_reset
                Vs_B[spike_B] = Vs_reset
                Vus_B[spike_B] += p_tensor.get('delta_Vus')[spike_B]
            if spike_C.any():
                V_C[spike_C] = V_reset
                Vs_C[spike_C] = Vs_reset
                Vus_C[spike_C] += p_tensor.get('delta_Vus')[spike_C]
            if spike_D.any():
                V_D[spike_D] = V_reset
                Vs_D[spike_D] = Vs_reset
                Vus_D[spike_D] += p_tensor.get('delta_Vus')[spike_D]

            traces_A[:, t] = V_A
            traces_B[:, t] = V_B
            traces_C[:, t] = V_C
            traces_D[:, t] = V_D

        VA_cpu = traces_A.cpu().numpy()
        VB_cpu = traces_B.cpu().numpy()
        VC_cpu = traces_C.cpu().numpy()
        VD_cpu = traces_D.cpu().numpy()

        for local_i, combo in enumerate(slice_combos):
            feats_A = extract_features(VA_cpu[local_i], DT,
                                       skip_bursts=FEATURE_SKIP_BURSTS,
                                       window_bursts=FEATURE_WINDOW_BURSTS)
            feats_B = extract_features(VB_cpu[local_i], DT,
                                       skip_bursts=FEATURE_SKIP_BURSTS,
                                       window_bursts=FEATURE_WINDOW_BURSTS)
            feats_C = extract_features(VC_cpu[local_i], DT,
                                       skip_bursts=FEATURE_SKIP_BURSTS,
                                       window_bursts=FEATURE_WINDOW_BURSTS)
            feats_D = extract_features(VD_cpu[local_i], DT,
                                       skip_bursts=FEATURE_SKIP_BURSTS,
                                       window_bursts=FEATURE_WINDOW_BURSTS)
            row = combo.copy()
            for name, f in zip('ABCD', [feats_A, feats_B, feats_C, feats_D]):
                row[f'regime_{name}'] = f['regime']
                row[f'spike_count_{name}'] = f['spike_count']
                row[f'mean_spikes_per_burst_{name}'] = f['mean_spikes_per_burst']
                row[f'duty_cycle_{name}'] = f['duty_cycle']
                row[f'interburst_freq_{name}'] = f['interburst_freq']
                row[f'intraburst_freq_{name}'] = f['intraburst_freq']

            if keep_mask[start + local_i]:
                row['V_trace_A'] = VA_cpu[local_i]
                row['V_trace_B'] = VB_cpu[local_i]
                row['V_trace_C'] = VC_cpu[local_i]
                row['V_trace_D'] = VD_cpu[local_i]
                row['time'] = time_cpu
            results.append(row)

        del V_A, V_B, V_C, V_D
        torch.cuda.empty_cache()

    return results


# -----------------------------------------------------------------------------
# Sweep routines
# -----------------------------------------------------------------------------

def single_param_sweep(param: str, values: np.ndarray, out_dir: Path,
                       batch_size: int | None = None) -> pd.DataFrame:
    combos = [{param: v} for v in values]
    n_total = len(combos)

    if TRACE_SAMPLES >= n_total:
        trace_indices = list(range(n_total))
    else:
        trace_indices = np.linspace(0, n_total - 1, TRACE_SAMPLES, dtype=int).tolist()

    if HAS_TORCH and n_total > 50:
        res = run_gpu_batch(combos, trace_indices=trace_indices, batch_size=batch_size)
    else:
        res = [run_single_sim(c, i in trace_indices) for i, c in enumerate(combos)]

    df = pd.DataFrame(res)
    df.to_csv(out_dir / f"{param}_sweep.csv", index=False)
    return df


def multi_param_sweep(name: str, params: list[str], out_dir: Path,
                      batch_size: int | None = None) -> pd.DataFrame:
    values = [PARAM_RANGES[p] for p in params]
    combos = [dict(zip(params, c)) for c in itertools.product(*values)]

    if HAS_TORCH and len(combos) > 50:
        res = run_gpu_batch(combos, batch_size=batch_size)
    else:
        res = [run_single_sim(c, False) for c in combos]

    df = pd.DataFrame(res)
    df.to_csv(out_dir / f"{name}_sweep.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    base = Path(__file__).resolve().parent / "qc_sweep"
    single_dir = base / "single_param"
    multi_dir = base / "multi_param"
    plots_dir = base / "plots"
    data_dir = base / "data"
    for d in [single_dir, multi_dir, plots_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for p, vals in PARAM_RANGES.items():
        single_param_sweep(p, vals, single_dir)

    for name, params in PARAM_GROUPS.items():
        multi_param_sweep(name, params, multi_dir)

    print("Done")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Runtime: {(time.time() - start)/60:.1f} min")
