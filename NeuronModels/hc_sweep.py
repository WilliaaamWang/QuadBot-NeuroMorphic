import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from datetime import datetime
import time
from tqdm import tqdm
import pickle
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import truncnorm
from numpy.random import default_rng, Generator

# Simulation parameters
DT = 5e-5
RUNTIME = 10.0  # seconds
I_EXT_AMPLITUDE = 5.0
I_EXT_START = 0.5  # seconds

SAME_START = True  # whether to start both neurons at the same voltage
INHIB_THRESH_FACTOR = 3/4  # factor for inhibitory threshold

# Config constants
TRACE_SAMPLES_SINGLE = 10  # number of traces to save in single parameter sweeps
TRACE_SAMPLES_MULTI = 20  # number of traces to save in multi-parameter sweeps
DEFAULT_GPU_BATCH = 800

# Feature extraction options
FEATURE_SKIP_BURSTS = 2       # discard initial transient bursts
FEATURE_WINDOW_BURSTS = 2     # analyse this many bursts after skipping

# Try to import parallelization libraries
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("▶︎ [INFO] joblib not available — will run sweeps serially on CPU")

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
    if HAS_TORCH:
        _GPU_NAME = torch.cuda.get_device_name(0)
        print(f"▶︎ [INFO] PyTorch sees CUDA — default device: {_GPU_NAME}")
    else:
        print("▶︎ [INFO] PyTorch present but **NO** CUDA device visible — CPU only")
except ImportError:
    HAS_TORCH = False
    torch = None
    print("▶︎ [INFO] PyTorch not installed — CPU only build will be used")

# Ask for confirmation before running on GPU
if HAS_TORCH:
    confirm = input("Running on GPU may be faster. Do you want to proceed? (y/n): ")
    if confirm.lower() != 'y':
        HAS_TORCH = False
        confirm = input("Proceed with CPU only? (y/n): ")
        if confirm.lower() != 'y':
            print("Exiting without running simulations.")
            exit(0)
        print("Proceeding with CPU only.")

# Import your modules
from synaptic_neuron import SynapticNeuron
from half_centre import simulate_halfcentre 
from utils import extract_features

# Default parameters from SynapticNeuron
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

# Parameter sweeps -----------------------------------------------------------
# Random sampling configuration ----------------------------------------------
NUM_SAMPLES   = 20     # number of samples per parameter
STD_FACTOR    = 0.10   # stdev factor relative to default value
RNG_SEED      = 42

def generate_param_ranges(n_samples: int = NUM_SAMPLES,
                          std_factor: float = STD_FACTOR,
                          seed: int = RNG_SEED,
                          rng: Generator | None = None) -> dict:
    """Return random parameter ranges centred on the defaults.

    Parameters that accept negative values (``Vs0``, ``Vus0``, ``Vi_threshold``
    and ``Vi0``) are drawn from a normal distribution. Parameters that must
    remain positive are drawn from a truncated normal distribution with a lower
    bound at zero. All distributions are centred at the default values.
    """
    rng = default_rng(seed) if rng is None else rng
    ranges = {}

    # Parameters allowed to go negative
    neg_params = ["Vs0", "Vus0", "Vi_threshold", "Vi0"]
    for p in neg_params:
        mean = DEFAULT_PARAMS[p]
        sd = abs(mean) * std_factor
        samples = rng.normal(loc=mean, scale=sd, size=n_samples)
        ranges[p] = np.sort(samples)

    # Parameters constrained to be positive
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

# Instantiate ranges for sweeps ------------------------------------------------
PARAM_RANGES = generate_param_ranges()

print("▶︎ [INFO] Parameter ranges generated:")
for param, values in PARAM_RANGES.items():
    print(f"  {param}: {len(values)} samples, "
          f"mean={np.mean(values):.3f}, "
          f"std={np.std(values):.3f}")

# Parameter groups for combined sweeps
PARAM_GROUPS = {
    'balance_point': ['Vs0', 'Vus0'],
    'ultraslow_dynamics': ['g_us', 'delta_Vus', 'tau_us'],
    'slow_dynamics': ['g_s', 'tau_s'],
    'synaptic': ['g_syn_i', 'tau_i', 'Vi_threshold', 'Vi0']
}

def _auto_batch_size(bytes_per_sim: int, safety: float = 0.54, hard_cap: int | None = None) -> int:
    """Return a batch size s.t. `bytes_per_sim * batch <= free_gpu * safety`."""
    if not HAS_TORCH:
        return 1
    free, _ = torch.cuda.mem_get_info()
    target = int(free * safety)
    bsz = max(target // bytes_per_sim, 1)
    if hard_cap:
        bsz = min(bsz, hard_cap)
    return bsz

def create_output_dirs():
    """Create directory structure for results."""
    base_dir = os.path.join(os.path.dirname(__file__), 'hc_sweep')
    dirs = {
        'base': base_dir,
        'single': os.path.join(base_dir, 'single_param'),
        'multi': os.path.join(base_dir, 'multi_param'),
        'plots': os.path.join(base_dir, 'plots'),
        'data': os.path.join(base_dir, 'data')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # Trace directories for single and multi sweeps
    trace_single = os.path.join(dirs['plots'], 'traces', 'single')
    trace_multi  = os.path.join(dirs['plots'], 'traces', 'multi')
    os.makedirs(trace_single, exist_ok=True)
    os.makedirs(trace_multi, exist_ok=True)

    dirs['trace_single'] = trace_single
    dirs['trace_multi'] = trace_multi
    return dirs

def create_I_ext():
    """Create external current array."""
    n_steps = int(RUNTIME / DT)
    I_ext = np.zeros(n_steps)
    start_idx = int(I_EXT_START / DT)
    I_ext[start_idx:] = I_EXT_AMPLITUDE
    return I_ext

def run_single_simulation(params_dict, save_trace=False):
    """Run a single half-center simulation with given parameters."""
    # Create neurons with specified parameters
    neuron_params = DEFAULT_PARAMS.copy()
    neuron_params.update(params_dict)

    # Create the diff in inhibitory threshold for half-centre neurons
    vi_thresh = neuron_params.pop('Vi_threshold', DEFAULT_PARAMS['Vi_threshold'])
    Vi_threshold_A = vi_thresh
    Vi_threshold_B = Vi_threshold_A * INHIB_THRESH_FACTOR
    
    neuronA = SynapticNeuron(
        excitatory_Vin=None, 
        inhibitory_Vin=None,
        Vi_threshold=Vi_threshold_A,
        **neuron_params
    )
    neuronB = SynapticNeuron(
        excitatory_Vin=None,
        inhibitory_Vin=None,
        Vi_threshold=Vi_threshold_B,
        **neuron_params
    )
    
    # Run simulation
    I_ext = create_I_ext()
    excit_ext = []
    inhib_ext = []
    
    neuronA, neuronB = simulate_halfcentre(
        neuronA, neuronB, 
        I_ext, I_ext,
        excit_ext, inhib_ext, excit_ext, inhib_ext,
        dt=DT, runtime=RUNTIME, 
        plotter=False, same_start=SAME_START
    )
    
    # Extract features from the steady-state window
    features_A = extract_features(
        np.array(neuronA.Vvalues), DT,
        skip_bursts=FEATURE_SKIP_BURSTS,
        window_bursts=FEATURE_WINDOW_BURSTS,
    )
    features_B = extract_features(
        np.array(neuronB.Vvalues), DT,
        skip_bursts=FEATURE_SKIP_BURSTS,
        window_bursts=FEATURE_WINDOW_BURSTS,
    )
    
    # Combine results
    result = params_dict.copy()
    result.update({
        'regime_A': features_A['regime'],
        'regime_B': features_B['regime'],
        'spike_count_A': features_A['spike_count'],
        'spike_count_B': features_B['spike_count'],
        'mean_spikes_per_burst_A': features_A['mean_spikes_per_burst'],
        'mean_spikes_per_burst_B': features_B['mean_spikes_per_burst'],
        'duty_cycle_A': features_A['duty_cycle'],
        'duty_cycle_B': features_B['duty_cycle'],
        'interburst_freq_A': features_A['interburst_freq'],
        'interburst_freq_B': features_B['interburst_freq'],
        'intraburst_freq_A': features_A['intraburst_freq'],
        'intraburst_freq_B': features_B['intraburst_freq']
    })
    
    if save_trace:
        result['V_trace_A'] = np.array(neuronA.Vvalues)
        result['V_trace_B'] = np.array(neuronB.Vvalues)
        result['time'] = np.arange(0, RUNTIME, DT)
    
    return result

def run_gpu_batch(param_combinations,
                  trace_indices: list[int] | None = None,
                  batch_size: int | None = None):
    """Simulate *all* `param_combinations` on the GPU.

    Parameters
    ----------
    param_combinations : list[dict]
        Each dict overrides one or more entries in ``DEFAULT_PARAMS``.
    trace_indices : list[int] | None
        Global indices **within** *param_combinations* whose full voltage
        traces should be returned and later saved / plotted. If *None*, no
        traces are kept.
    batch_size : int | None
        Fixed chunk size. If *None*, use `_auto_batch_size()` to stay within
        GPU VRAM limits.

    Returns
    -------
    list[dict]
        One result row *per* simulation. All scalar features are present for
        every row; the heavy ``V_trace_X`` arrays exist **only** for indices
        listed in ``trace_indices``.
    """
    if not HAS_TORCH:
        return None

    device = torch.device("cuda")
    total_sims = len(param_combinations)
    n_steps = int(RUNTIME / DT)

    # Determine which simulations need trace storage
    trace_indices = sorted(set(trace_indices or []))
    keep_trace_mask = torch.zeros(total_sims, dtype=torch.bool)
    if trace_indices:
        keep_trace_mask[trace_indices] = True

    # Decide batch size
    bytes_per_trace = n_steps * 4 * 2  # two float32 vectors
    est_bytes_per_sim = bytes_per_trace + 512  # scratchpad margin
    if batch_size is None:
        batch_size = _auto_batch_size(est_bytes_per_sim, hard_cap=DEFAULT_GPU_BATCH)
    print(f"▶︎ [GPU] processing {total_sims} sims in chunks of {batch_size} …")

    results: list[dict] = []
    time_cpu = np.arange(0, RUNTIME, DT, dtype=np.float32)

    # Process the workload slice-by-slice
    for start in range(0, total_sims, batch_size):
        end = min(start + batch_size, total_sims)
        slice_combos = param_combinations[start:end]
        b = len(slice_combos)  # slice size

        # Initialize state vectors
        V_A = torch.full((b,), DEFAULT_PARAMS['V0'], device=device)
        if SAME_START:
            V_B = torch.full((b,), DEFAULT_PARAMS['V0'], device=device)
        else:
            V_B = torch.full((b,), DEFAULT_PARAMS['V0'] + 0.1, device=device)
        Vs_A = torch.full((b,), DEFAULT_PARAMS['Vs0'], device=device)
        Vs_B = torch.full((b,), DEFAULT_PARAMS['Vs0'], device=device)
        Vus_A = torch.full((b,), DEFAULT_PARAMS['Vus0'], device=device)
        Vus_B = torch.full((b,), DEFAULT_PARAMS['Vus0'], device=device)
        Si_A = torch.zeros((b,), device=device)
        Si_B = torch.zeros((b,), device=device)

        # Prepare parameter tensors
        # Create tensors for all parameters that vary in this batch
        p_tensors = {}
        for p_name in PARAM_RANGES.keys():
            # Get values for this parameter across the batch
            values = [combo.get(p_name, DEFAULT_PARAMS[p_name]) for combo in slice_combos]
            p_tensors[p_name] = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Handle inhibitory thresholds
        if 'Vi_threshold' in p_tensors:
            Vi_threshold_A = p_tensors['Vi_threshold']
        else:
            Vi_threshold_A = torch.full((b,), DEFAULT_PARAMS['Vi_threshold'], device=device)
        Vi_threshold_B = Vi_threshold_A * INHIB_THRESH_FACTOR

        # Create constant tensors
        cap = DEFAULT_PARAMS['cap']
        k_const = DEFAULT_PARAMS['k']
        V0_const = DEFAULT_PARAMS['V0']
        g_f = DEFAULT_PARAMS['g_f']
        V_thresh = DEFAULT_PARAMS['V_threshold']
        V_peak = DEFAULT_PARAMS['V_peak']
        V_reset = DEFAULT_PARAMS['V_reset']
        Vs_reset = DEFAULT_PARAMS['Vs_reset']

        # External current
        I_ext = torch.zeros(n_steps, device=device)
        I_ext[int(I_EXT_START/DT):] = I_EXT_AMPLITUDE

        # Trace storage
        V_A_trace = torch.zeros((b, n_steps), device=device, dtype=torch.float32)
        V_B_trace = torch.zeros((b, n_steps), device=device, dtype=torch.float32)

        # Time integration (Euler method)
        for t in range(n_steps):
            # Mutual inhibition voltages
            inhib_A = V_B
            inhib_B = V_A

            # Get parameter values for this batch
            tau_s = p_tensors.get('tau_s', torch.full((b,), DEFAULT_PARAMS['tau_s'], device=device))
            tau_us = p_tensors.get('tau_us', torch.full((b,), DEFAULT_PARAMS['tau_us'], device=device))
            tau_i = p_tensors.get('tau_i', torch.full((b,), DEFAULT_PARAMS['tau_i'], device=device))
            g_s = p_tensors.get('g_s', torch.full((b,), DEFAULT_PARAMS['g_s'], device=device))
            g_us = p_tensors.get('g_us', torch.full((b,), DEFAULT_PARAMS['g_us'], device=device))
            g_syn_i = p_tensors.get('g_syn_i', torch.full((b,), DEFAULT_PARAMS['g_syn_i'], device=device))
            Vs0 = p_tensors.get('Vs0', torch.full((b,), DEFAULT_PARAMS['Vs0'], device=device))
            Vus0 = p_tensors.get('Vus0', torch.full((b,), DEFAULT_PARAMS['Vus0'], device=device))
            Vi0 = p_tensors.get('Vi0', torch.full((b,), DEFAULT_PARAMS['Vi0'], device=device))
            delta_Vus = p_tensors.get('delta_Vus', torch.full((b,), DEFAULT_PARAMS['delta_Vus'], device=device))

            # Vs & Vus derivatives
            dVs_A = k_const * (V_A - Vs_A) / tau_s
            dVs_B = k_const * (V_B - Vs_B) / tau_s
            dVus_A = k_const * (V_A - Vus_A) / tau_us
            dVus_B = k_const * (V_B - Vus_B) / tau_us

            # Synaptic gating
            Si_inf_A = torch.sigmoid(40 * (inhib_A - Vi_threshold_A))
            Si_inf_B = torch.sigmoid(40 * (inhib_B - Vi_threshold_B))
            dSi_A = k_const * (Si_inf_A - Si_A) / tau_i
            dSi_B = k_const * (Si_inf_B - Si_B) / tau_i

            # Inhibitory current
            I_inh_A = g_syn_i * Si_A * (V_A - Vi0)
            I_inh_B = g_syn_i * Si_B * (V_B - Vi0)

            # dV/dt
            dV_A = (k_const / cap) * (
                g_f * (V_A - V0_const) ** 2
                - g_s * (Vs_A - Vs0) ** 2
                - g_us * (Vus_A - Vus0) ** 2
                + I_ext[t] - I_inh_A
            )
            dV_B = (k_const / cap) * (
                g_f * (V_B - V0_const) ** 2
                - g_s * (Vs_B - Vs0) ** 2
                - g_us * (Vus_B - Vus0) ** 2
                + I_ext[t] - I_inh_B
            )

            # Euler update
            V_A += dV_A * DT
            V_B += dV_B * DT
            Vs_A += dVs_A * DT
            Vs_B += dVs_B * DT
            Vus_A += dVus_A * DT
            Vus_B += dVus_B * DT
            Si_A += dSi_A * DT
            Si_B += dSi_B * DT

            # Store traces BEFORE reset to capture spike peaks
            V_A_trace[:, t] = V_A
            V_B_trace[:, t] = V_B
            
            # Spike detection
            spike_A = V_A >= V_thresh
            spike_B = V_B >= V_thresh
            
            # For neurons that spiked, record the peak value in the trace
            if spike_A.any():
                V_A_trace[spike_A, t] = V_peak  # Record spike peak value
                V_A[spike_A] = V_reset
                Vs_A[spike_A] = Vs_reset
                Vus_A[spike_A] += delta_Vus[spike_A]
            if spike_B.any():
                V_B_trace[spike_B, t] = V_peak  # Record spike peak value
                V_B[spike_B] = V_reset
                Vs_B[spike_B] = Vs_reset
                Vus_B[spike_B] += delta_Vus[spike_B]

        # Move traces to CPU for feature extraction
        V_A_cpu = V_A_trace.cpu().numpy()
        V_B_cpu = V_B_trace.cpu().numpy()

        # Extract features and build results
        for local_i, combo in enumerate(slice_combos):
            global_i = start + local_i

            # Extract features
            feats_A = extract_features(
                V_A_cpu[local_i],
                DT,
                skip_bursts=FEATURE_SKIP_BURSTS,
                window_bursts=FEATURE_WINDOW_BURSTS,
            )
            feats_B = extract_features(
                V_B_cpu[local_i],
                DT,
                skip_bursts=FEATURE_SKIP_BURSTS,
                window_bursts=FEATURE_WINDOW_BURSTS,
            )

            # Build result dictionary
            row = combo.copy()
            row.update({
                'regime_A': feats_A['regime'],
                'regime_B': feats_B['regime'],
                'spike_count_A': feats_A['spike_count'],
                'spike_count_B': feats_B['spike_count'],
                'mean_spikes_per_burst_A': feats_A['mean_spikes_per_burst'],
                'mean_spikes_per_burst_B': feats_B['mean_spikes_per_burst'],
                'duty_cycle_A': feats_A['duty_cycle'],
                'duty_cycle_B': feats_B['duty_cycle'],
                'interburst_freq_A': feats_A['interburst_freq'],
                'interburst_freq_B': feats_B['interburst_freq'],
                'intraburst_freq_A': feats_A['intraburst_freq'],
                'intraburst_freq_B': feats_B['intraburst_freq']
            })

            # Add trace data if this index was selected for trace saving
            if global_i in trace_indices:
                row['V_trace_A'] = V_A_cpu[local_i]
                row['V_trace_B'] = V_B_cpu[local_i]
                row['time'] = time_cpu

            results.append(row)

        # Cleanup per-slice tensors
        del V_A, V_B, Vs_A, Vs_B, Vus_A, Vus_B, Si_A, Si_B
        del V_A_trace, V_B_trace
        torch.cuda.empty_cache()

    return results

def _plot_voltage_trace(result_dict: dict, param_name: str, label: str,
                        dirs: dict, sweep_type: str = "single"):
    """Draw the two-neuron voltage traces already stored in *result_dict*."""
    if "V_trace_A" not in result_dict:
        return  # nothing to save

    t = result_dict["time"]
    v_A = result_dict["V_trace_A"]
    v_B = result_dict["V_trace_B"]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(t, v_A, "b-", linewidth=0.7)
    axs[0].set_ylabel("Neuron A (mV)")
    axs[0].grid(alpha=0.3)
    # axs[0].set_ylim(-80, 30)

    axs[1].plot(t, v_B, "r-", linewidth=0.7)
    axs[1].set_ylabel("Neuron B (mV)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(alpha=0.3)
    # axs[1].set_ylim(-80, 30)

    fig.suptitle(f"{param_name} sweep – {label}")
    plt.tight_layout()
    plot_dir = os.path.join(dirs["plots"], "traces", sweep_type)
    os.makedirs(plot_dir, exist_ok=True)
    fname = f"{sweep_type}_{label}.png"
    plt.savefig(os.path.join(plot_dir, fname), dpi=150)
    plt.close(fig)

def single_param_sweep(param_name: str,
                       param_values: np.ndarray,
                       dirs: dict,
                       n_plot: int = TRACE_SAMPLES_SINGLE,
                       batch_size: int | None = None):
    """Sweep *one* parameter and collect scalar outputs."""
    print(f"\nSweeping {param_name} … ({len(param_values)} values)")

    # Build combinations
    param_combos = [{param_name: v} for v in param_values]
    n_total = len(param_combos)

    if n_plot >= n_total:
        trace_indices = list(range(n_total))
    else:
        trace_indices = np.linspace(0, n_total - 1, n_plot, dtype=int).tolist()

    # Choose execution path
    use_gpu = HAS_TORCH and n_total > 50

    if use_gpu:
        print("▶︎ [GPU] extracting features & (selected) traces in CUDA …")
        results = run_gpu_batch(param_combos,
                                trace_indices=trace_indices,
                                batch_size=batch_size)
    else:
        mode = "joblib-parallel" if HAS_JOBLIB else "serial"
        print(f"▶︎ [CPU] executing in {mode} mode …")
        if HAS_JOBLIB:
            from joblib import Parallel, delayed
            trace_set = set(trace_indices)
            results = Parallel(n_jobs=-1)(
                delayed(run_single_simulation)(combo, i in trace_set)
                for i, combo in enumerate(tqdm(param_combos, desc=param_name))
            )
        else:
            results = []
            trace_set = set(trace_indices)
            for i, combo in enumerate(tqdm(param_combos, desc=param_name)):
                results.append(run_single_simulation(combo, i in trace_set))

    # Plot only the kept traces
    for idx in trace_indices:
        res = results[idx]
        label = f"{param_name}_{res[param_name]:.3f}"
        _plot_voltage_trace(res, param_name, label, dirs, sweep_type="single")

    # Persist numeric results
    df = pd.DataFrame(results)
    csv_path = os.path.join(dirs['single'], f"{param_name}_sweep.csv")
    df.to_csv(csv_path, index=False)

    # Store traces (only the selected ones)
    h5_path = os.path.join(dirs['data'], f"{param_name}_traces.h5")
    with h5py.File(h5_path, "w") as f:
        for i, res in enumerate(results):
            if 'V_trace_A' in res:  # saved only for trace_indices
                grp = f.create_group(f"sim_{i}")
                grp.create_dataset("V_A", data=res['V_trace_A'])
                grp.create_dataset("V_B", data=res['V_trace_B'])
                grp.create_dataset("time", data=res['time'])
                grp.attrs[param_name] = res[param_name]

    # Overview metric plots
    plot_single_param_results(df, param_name, param_values, dirs)

    return df

def plot_single_param_results(df, param_name, param_values, dirs):
    """Plot results from single parameter sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Interburst frequency
    axes[0, 0].plot(df[param_name], df['interburst_freq_A'], 'b-o', label='Neuron A', markersize=4)
    axes[0, 0].plot(df[param_name], df['interburst_freq_B'], 'r--s', label='Neuron B', markersize=4)
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel('Interburst Frequency (Hz)')
    axes[0, 0].set_title('Interburst Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Intraburst frequency
    axes[0, 1].plot(df[param_name], df['intraburst_freq_A'], 'b-o', label='Neuron A', markersize=4)
    axes[0, 1].plot(df[param_name], df['intraburst_freq_B'], 'r--s', label='Neuron B', markersize=4)
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel('Intraburst Frequency (Hz)')
    axes[0, 1].set_title('Intraburst Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Duty cycle
    axes[1, 0].plot(df[param_name], df['duty_cycle_A'], 'b-o', label='Neuron A', markersize=4)
    axes[1, 0].plot(df[param_name], df['duty_cycle_B'], 'r--s', label='Neuron B', markersize=4)
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel('Duty Cycle')
    axes[1, 0].set_title('Duty Cycle')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Spikes per burst
    axes[1, 1].plot(df[param_name], df['mean_spikes_per_burst_A'], 'b-o', label='Neuron A', markersize=4)
    axes[1, 1].plot(df[param_name], df['mean_spikes_per_burst_B'], 'r--s', label='Neuron B', markersize=4)
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].set_ylabel('Mean Spikes per Burst')
    axes[1, 1].set_title('Spikes per Burst')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['plots'], f'{param_name}_single_sweep.png'), dpi=150)
    plt.close()

def multi_param_sweep(group_name, param_names, dirs,
                      n_plot: int = TRACE_SAMPLES_MULTI):
    """Sweep multiple parameters together and optionally save example traces."""
    print(f"\n――― Multi-parameter sweep: {group_name} ({len(param_names)} dims) ―――")
    
    # Create all combinations
    param_values = [PARAM_RANGES[p] for p in param_names]
    all_combos = list(itertools.product(*param_values))
    param_dicts = [dict(zip(param_names, combo)) for combo in all_combos]
    
    print(f"Total combinations: {len(param_dicts)}")
    
    n_total = len(param_dicts)

    if n_plot >= n_total:
        trace_indices = list(range(n_total))
    else:
        trace_indices = np.linspace(0, n_total - 1, n_plot, dtype=int).tolist()

    # Run simulations
    use_gpu = HAS_TORCH and n_total > 50
    
    if use_gpu:
        batch_size = 200
        print("▶︎ [GPU] extracting features & (selected) traces in CUDA …")
        results = run_gpu_batch(param_dicts, 
                                trace_indices=trace_indices,
                                batch_size=batch_size)
        if results is None:
            use_gpu = False

    if not use_gpu:
        trace_set = set(trace_indices)
        if HAS_JOBLIB:
            results = Parallel(n_jobs=-1)(
                delayed(run_single_simulation)(combo, i in trace_set)
                for i, combo in enumerate(tqdm(param_dicts, desc=group_name))
            )
        else:
            results = []
            for i, combo in enumerate(tqdm(param_dicts, desc=group_name)):
                results.append(run_single_simulation(combo, i in trace_set))
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(dirs['multi'], f'{group_name}_sweep.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as pickle for faster loading
    pkl_path = os.path.join(dirs['data'], f'{group_name}_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(df, f)

    # Plot a subset of example traces
    for idx in trace_indices:
        res = results[idx]
        label_vals = '_'.join(f'{p}{res[p]:.3f}' for p in param_names)
        _plot_voltage_trace(res, group_name, label_vals, dirs, sweep_type="multi")

    # Store traces to disk
    h5_path = os.path.join(dirs['data'], f'{group_name}_traces.h5')
    with h5py.File(h5_path, "w") as f:
        for i, res in enumerate(results):
            if 'V_trace_A' in res:
                grp = f.create_group(f"sim_{i}")
                grp.create_dataset("V_A", data=res['V_trace_A'])
                grp.create_dataset("V_B", data=res['V_trace_B'])
                grp.create_dataset("time", data=res['time'])
                for p in param_names:
                    grp.attrs[p] = res[p]

    # Plot results based on dimensionality
    if len(param_names) == 2:
        plot_2d_results(df, param_names, group_name, dirs)
    elif len(param_names) == 3:
        plot_3d_results(df, param_names, group_name, dirs)
    elif len(param_names) == 4:
        plot_4d_results(df, param_names, group_name, dirs)
    
    return df

def pairwise_sweep(group_name: str, param_names: list[str], dirs: dict,
                   n_plot: int = TRACE_SAMPLES_MULTI) -> list[pd.DataFrame]:
    """Run 2-D sweeps for every parameter pair in ``param_names``."""
    dfs: list[pd.DataFrame] = []
    for p1, p2 in itertools.combinations(param_names, 2):
        pair_label = f"{group_name}_{p1}_{p2}"
        df = multi_param_sweep(pair_label, [p1, p2], dirs, n_plot=n_plot)
        dfs.append(df)
    return dfs

def plot_2d_results(df, param_names, group_name, dirs):
    """Plot 2D parameter sweep results as heatmaps."""
    features = ['interburst_freq_A', 'intraburst_freq_A', 'duty_cycle_A']
    feature_labels = ['Interburst Frequency (Hz)', 'Intraburst Frequency (Hz)', 'Duty Cycle']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (feature, label) in enumerate(zip(features, feature_labels)):
        # Pivot data for heatmap
        pivot = df.pivot_table(
            values=feature, 
            index=param_names[1], 
            columns=param_names[0],
            aggfunc='mean'
        )
        
        im = axes[idx].imshow(pivot, aspect='auto', origin='lower', cmap='viridis')
        axes[idx].set_xlabel(param_names[0])
        axes[idx].set_ylabel(param_names[1])
        axes[idx].set_title(f'{label}\n{group_name}')
        
        # Set tick labels
        xticks = np.linspace(0, len(pivot.columns)-1, 5).astype(int)
        yticks = np.linspace(0, len(pivot.index)-1, 5).astype(int)
        axes[idx].set_xticks(xticks)
        axes[idx].set_yticks(yticks)
        axes[idx].set_xticklabels([f'{pivot.columns[i]:.3f}' for i in xticks], rotation=45)
        axes[idx].set_yticklabels([f'{pivot.index[i]:.3f}' for i in yticks])
        
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['plots'], f'{group_name}_2d_sweep.png'), dpi=150)
    plt.close()

def plot_3d_results(df, param_names, group_name, dirs):
    """Plot 3D parameter sweep results as slices."""
    # Fix the third parameter at its middle value
    fixed_param = param_names[2]
    fixed_value = PARAM_RANGES[fixed_param][len(PARAM_RANGES[fixed_param])//2]
    df_slice = df[np.isclose(df[fixed_param], fixed_value, rtol=0.01)]
    
    if len(df_slice) > 0:
        plot_2d_results(df_slice, param_names[:2], 
                       f'{group_name} ({fixed_param}={fixed_value:.3f})', dirs)

def plot_4d_results(df, param_names, group_name, dirs):
    """Plot 4D parameter sweep results as multiple 2D slices."""
    # Fix the last two parameters at their middle values
    fixed_params = param_names[2:]
    fixed_values = [PARAM_RANGES[p][len(PARAM_RANGES[p])//2] for p in fixed_params]
    
    df_slice = df.copy()
    for param, value in zip(fixed_params, fixed_values):
        df_slice = df_slice[np.isclose(df_slice[param], value, rtol=0.01)]
    
    if len(df_slice) > 0:
        fixed_str = ', '.join([f'{p}={v:.3f}' for p, v in zip(fixed_params, fixed_values)])
        plot_2d_results(df_slice, param_names[:2], 
                       f'{group_name} ({fixed_str})', dirs)

def _concat_and_store(dfs, dirs):
    """Concatenate a list of DataFrames and persist as Parquet."""
    if not dfs:
        return None
    big = pd.concat(dfs, ignore_index=True)
    pq_path = os.path.join(dirs["data"], "all_sweep_results.parquet")
    table = pa.Table.from_pandas(big)
    pq.write_table(table, pq_path, compression="zstd")
    print(f"\n✅ Aggregated {len(big)} rows → {pq_path}")
    return big

def main():
    """Run every sweep (single + grouped) and store artefacts."""
    dirs = create_output_dirs()

    # 1) SINGLE-PARAMETER SWEEPS
    single_dfs = []
    for p_name, p_vals in PARAM_RANGES.items():
        df = single_param_sweep(p_name, p_vals, dirs,
                                n_plot=TRACE_SAMPLES_SINGLE)
        single_dfs.append(df)

    # 2) MULTI-PARAMETER GROUP SWEEPS (pairwise)
    multi_dfs = []
    for grp, names in PARAM_GROUPS.items():
        if len(names) > 2:
            pair_dfs = pairwise_sweep(grp, names, dirs,
                                      n_plot=TRACE_SAMPLES_MULTI)
            multi_dfs.extend(pair_dfs)
        else:
            df = multi_param_sweep(grp, names, dirs,
                                   n_plot=TRACE_SAMPLES_MULTI)
            multi_dfs.append(df)

    # 3) Aggregation
    _concat_and_store(single_dfs + multi_dfs, dirs)

if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n🏁 Total runtime: {elapsed/60:.1f} min")