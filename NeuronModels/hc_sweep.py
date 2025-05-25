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
    bound at zero.  All distributions are centred at the default values.
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
    'resting_potentials': ['Vs0', 'Vus0'],
    'ultraslow_dynamics': ['g_us', 'delta_Vus', 'tau_us'],
    'slow_dynamics': ['g_s', 'tau_s'],
    'synaptic': ['g_syn_i', 'tau_i', 'Vi_threshold', 'Vi0']
}

# Simulation parameters
DT = 5e-5
RUNTIME = 10.0  # seconds
I_EXT_AMPLITUDE = 5.0
I_EXT_START = 0.5  # seconds

# Feature extraction options
FEATURE_SKIP_BURSTS = 2       # discard initial transient bursts
FEATURE_WINDOW_BURSTS = 2     # analyse this many bursts after skipping

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
    
    neuronA = SynapticNeuron(
        excitatory_Vin=None, 
        inhibitory_Vin=None,
        **neuron_params
    )
    neuronB = SynapticNeuron(
        excitatory_Vin=None,
        inhibitory_Vin=None,
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
        plotter=False, same_start=False
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

def run_gpu_batch(param_combinations):
    """Run batch of simulations on GPU using PyTorch."""
    if not HAS_TORCH:
        return None
    
    print(f"▶︎ [GPU]   launching batch of {len(param_combinations)} sims on CUDA …")
    device = torch.device('cuda')
    n_sims = len(param_combinations)
    n_steps = int(RUNTIME / DT)
    
    # Initialize state tensors for all simulations
    V_A = torch.full((n_sims,), DEFAULT_PARAMS['V0'], device=device)
    V_B = torch.full((n_sims,), DEFAULT_PARAMS['V0'], device=device) + 0.1  # offset
    Vs_A = torch.full((n_sims,), DEFAULT_PARAMS['Vs0'], device=device)
    Vs_B = torch.full((n_sims,), DEFAULT_PARAMS['Vs0'], device=device)
    Vus_A = torch.full((n_sims,), DEFAULT_PARAMS['Vus0'], device=device)
    Vus_B = torch.full((n_sims,), DEFAULT_PARAMS['Vus0'], device=device)
    Si_A = torch.zeros((n_sims,), device=device)
    Si_B = torch.zeros((n_sims,), device=device)
    
    # Create parameter tensors
    param_tensors = {}
    for param_name in PARAM_RANGES.keys():
        values = [combo.get(param_name, DEFAULT_PARAMS.get(param_name, 0)) 
                  for combo in param_combinations]
        param_tensors[param_name] = torch.tensor(values, dtype=torch.float32, device=device)
    
    # Constants
    cap = torch.tensor(DEFAULT_PARAMS['cap'], device=device)
    k = torch.tensor(DEFAULT_PARAMS['k'], device=device)
    V0 = torch.tensor(DEFAULT_PARAMS['V0'], device=device)
    g_f = torch.tensor(DEFAULT_PARAMS['g_f'], device=device)
    V_threshold = torch.tensor(DEFAULT_PARAMS['V_threshold'], device=device)
    V_reset = torch.tensor(DEFAULT_PARAMS['V_reset'], device=device)
    
    # Create I_ext tensor
    I_ext_tensor = torch.zeros(n_steps, device=device)
    I_ext_tensor[int(I_EXT_START/DT):] = I_EXT_AMPLITUDE
    
    # Storage for traces
    V_A_trace = torch.zeros((n_sims, n_steps), device=device)
    V_B_trace = torch.zeros((n_sims, n_steps), device=device)
    
    # Main simulation loop
    for t in range(n_steps):
        # Mutual inhibition
        inhib_A = V_B
        inhib_B = V_A
        
        # Update derivatives
        dVs_A = k * (V_A - Vs_A) / param_tensors.get('tau_s', torch.tensor(DEFAULT_PARAMS['tau_s'], device=device))
        dVs_B = k * (V_B - Vs_B) / param_tensors.get('tau_s', torch.tensor(DEFAULT_PARAMS['tau_s'], device=device))
        dVus_A = k * (V_A - Vus_A) / param_tensors.get('tau_us', torch.tensor(DEFAULT_PARAMS['tau_us'], device=device))
        dVus_B = k * (V_B - Vus_B) / param_tensors.get('tau_us', torch.tensor(DEFAULT_PARAMS['tau_us'], device=device))
        
        # Synaptic gating
        Si_inf_A = torch.sigmoid(40 * (inhib_A - param_tensors.get('Vi_threshold', torch.tensor(DEFAULT_PARAMS['Vi_threshold'], device=device))))
        Si_inf_B = torch.sigmoid(40 * (inhib_B - param_tensors.get('Vi_threshold', torch.tensor(DEFAULT_PARAMS['Vi_threshold'], device=device))))
        dSi_A = k * (Si_inf_A - Si_A) / param_tensors.get('tau_i', torch.tensor(DEFAULT_PARAMS['tau_i'], device=device))
        dSi_B = k * (Si_inf_B - Si_B) / param_tensors.get('tau_i', torch.tensor(DEFAULT_PARAMS['tau_i'], device=device))
        
        # Currents
        I_inh_A = param_tensors.get('g_syn_i', torch.tensor(DEFAULT_PARAMS['g_syn_i'], device=device)) * Si_A * (V_A - param_tensors.get('Vi0', torch.tensor(DEFAULT_PARAMS['Vi0'], device=device)))
        I_inh_B = param_tensors.get('g_syn_i', torch.tensor(DEFAULT_PARAMS['g_syn_i'], device=device)) * Si_B * (V_B - param_tensors.get('Vi0', torch.tensor(DEFAULT_PARAMS['Vi0'], device=device)))
        
        # Membrane potential derivatives
        dV_A = (k/cap) * (
            g_f * (V_A - V0)**2 
            - param_tensors.get('g_s', torch.tensor(DEFAULT_PARAMS['g_s'], device=device)) * (Vs_A - param_tensors.get('Vs0', torch.tensor(DEFAULT_PARAMS['Vs0'], device=device)))**2
            - param_tensors.get('g_us', torch.tensor(DEFAULT_PARAMS['g_us'], device=device)) * (Vus_A - param_tensors.get('Vus0', torch.tensor(DEFAULT_PARAMS['Vus0'], device=device)))**2
            + I_ext_tensor[t] - I_inh_A
        )
        dV_B = (k/cap) * (
            g_f * (V_B - V0)**2
            - param_tensors.get('g_s', torch.tensor(DEFAULT_PARAMS['g_s'], device=device)) * (Vs_B - param_tensors.get('Vs0', torch.tensor(DEFAULT_PARAMS['Vs0'], device=device)))**2
            - param_tensors.get('g_us', torch.tensor(DEFAULT_PARAMS['g_us'], device=device)) * (Vus_B - param_tensors.get('Vus0', torch.tensor(DEFAULT_PARAMS['Vus0'], device=device)))**2
            + I_ext_tensor[t] - I_inh_B
        )
        
        # Euler update
        V_A = V_A + dV_A * DT
        V_B = V_B + dV_B * DT
        Vs_A = Vs_A + dVs_A * DT
        Vs_B = Vs_B + dVs_B * DT
        Vus_A = Vus_A + dVus_A * DT
        Vus_B = Vus_B + dVus_B * DT
        Si_A = Si_A + dSi_A * DT
        Si_B = Si_B + dSi_B * DT
        
        # Spike detection and reset
        spike_A = V_A >= V_threshold
        spike_B = V_B >= V_threshold
        
        if spike_A.any():
            V_A[spike_A] = V_reset
            Vs_A[spike_A] = DEFAULT_PARAMS['Vs_reset']
            Vus_A[spike_A] += param_tensors.get('delta_Vus', torch.tensor(DEFAULT_PARAMS['delta_Vus'], device=device))[spike_A]
        
        if spike_B.any():
            V_B[spike_B] = V_reset
            Vs_B[spike_B] = DEFAULT_PARAMS['Vs_reset']
            Vus_B[spike_B] += param_tensors.get('delta_Vus', torch.tensor(DEFAULT_PARAMS['delta_Vus'], device=device))[spike_B]
        
        V_A_trace[:, t] = V_A
        V_B_trace[:, t] = V_B
    
    # Convert back to CPU and extract features
    V_A_cpu = V_A_trace.cpu().numpy()
    V_B_cpu = V_B_trace.cpu().numpy()
    
    results = []
    for i, combo in enumerate(param_combinations):
        features_A = extract_features(
            V_A_cpu[i], DT,
            skip_bursts=FEATURE_SKIP_BURSTS,
            window_bursts=FEATURE_WINDOW_BURSTS,
        )
        features_B = extract_features(
            V_B_cpu[i], DT,
            skip_bursts=FEATURE_SKIP_BURSTS,
            window_bursts=FEATURE_WINDOW_BURSTS,
        )
        
        result = combo.copy()
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
        results.append(result)
    
    return results

def _plot_voltage_trace(result_dict: dict, param_name: str, label: str, dirs: dict):
    """Draw the two‑neuron voltage traces already stored in *result_dict*."""
    if "V_trace_A" not in result_dict:
        return  # nothing to save

    t    = result_dict["time"]
    v_A  = result_dict["V_trace_A"]
    v_B  = result_dict["V_trace_B"]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(t, v_A, "b-")
    axs[0].set_ylabel("Neuron A (mV)")
    axs[0].grid(alpha=0.3)

    axs[1].plot(t, v_B, "r-")
    axs[1].set_ylabel("Neuron B (mV)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(alpha=0.3)

    fig.suptitle(f"{param_name} sweep – {label}")
    plt.tight_layout()
    fname = f"trace_{label}.png"
    plt.savefig(os.path.join(dirs["plots"], "traces", fname), dpi=150)
    plt.close(fig)

# def single_param_sweep(param_name, param_values, dirs):
#     """Sweep a single parameter."""
#     print(f"\nSweeping {param_name}...")
    
#     results = []
#     traces_to_save = [0, len(param_values)//2, len(param_values)-1]  # First, middle, last
    
#     # Create parameter combinations
#     param_combos = [{param_name: val} for val in param_values]
    
#     # Run simulations
#     if HAS_TORCH and len(param_combos) > 10:
#         # Use GPU for batch processing
#         batch_size = 100
#         for i in range(0, len(param_combos), batch_size):
#             batch = param_combos[i:i+batch_size]
#             batch_results = run_gpu_batch(batch)
#             if batch_results:
#                 results.extend(batch_results)
#             else:
#                 # Fallback to CPU
#                 for j, combo in enumerate(batch):
#                     save_trace = (i+j) in traces_to_save
#                     results.append(run_single_simulation(combo, save_trace))
#     else:
#         # Use CPU (parallel if available)
#         if HAS_JOBLIB:
#             # Parallel processing
#             results = Parallel(n_jobs=-1)(
#                 delayed(run_single_simulation)(combo, i in traces_to_save)
#                 for i, combo in enumerate(param_combos)
#             )
#         else:
#             # Serial processing
#             for i, combo in enumerate(tqdm(param_combos, desc=param_name)):
#                 results.append(run_single_simulation(combo, i in traces_to_save))
    
#     # Save results
#     df = pd.DataFrame(results)
#     csv_path = os.path.join(dirs['single'], f'{param_name}_sweep.csv')
#     df.to_csv(csv_path, index=False)
    
#     # Save traces with HDF5
#     h5_path = os.path.join(dirs['data'], f'{param_name}_traces.h5')
#     with h5py.File(h5_path, 'w') as f:
#         for i, result in enumerate(results):
#             if 'V_trace_A' in result:
#                 grp = f.create_group(f'sim_{i}')
#                 grp.create_dataset('V_A', data=result['V_trace_A'])
#                 grp.create_dataset('V_B', data=result['V_trace_B'])
#                 grp.create_dataset('time', data=result['time'])
#                 grp.attrs[param_name] = param_values[i]
    
#     # Plot results
#     plot_single_param_results(df, param_name, param_values, dirs)
    
#     return df


def single_param_sweep(param_name, param_values, dirs):
    """Sweep a *single* parameter and now save a voltage‑trace PNG for **every**
    simulation run (was previously just three)."""
    print(f"\nSweeping {param_name} … ({len(param_values)} values)")

    results = []

    # Build list of param‑dict combos
    param_combos = [{param_name: val} for val in param_values]

    # Decide execution path ----------------------------------------------------
    use_gpu_batches = HAS_TORCH and len(param_combos) > 50

    if use_gpu_batches:
        # GPU for features – then CPU rerun for traces so we still get PNGs
        batch_size = 200

        print("▶︎ [GPU]   using run_gpu_batch() for feature extraction …")
        for i in tqdm(range(0, len(param_combos), batch_size), desc="GPU batches"):
            batch = param_combos[i:i+batch_size]
            gpu_feats = run_gpu_batch(batch) or []
            if gpu_feats:
                print(f"Batch {i//batch_size + 1} processed on GPU, {len(gpu_feats)} results.")
                results.extend(gpu_feats)
            else:  # fallback – run on CPU one‑by‑one with traces
                for combo in batch:
                    print(f"Running fallback CPU simulation for {combo[param_name]:.3f}...")
                    res = run_single_simulation(combo, save_trace=True)
                    results.append(res)
            # In *any* case, generate traces PNGs on CPU so we have plots
            for combo in batch:
                label = f"{param_name}_{combo[param_name]:.3f}"
                trace_res = run_single_simulation(combo, save_trace=True)
                _plot_voltage_trace(trace_res, param_name, label, dirs)
    else:
        cpu_mode = "joblib‑parallel" if HAS_JOBLIB else "serial"
        print(f"▶︎ [CPU]   executing in {cpu_mode} mode …")
        # CPU route (parallel if joblib is available)
        if HAS_JOBLIB:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=-1)(
                delayed(run_single_simulation)(combo, True) for combo in tqdm(param_combos, desc=param_name)
            )
        else:
            for combo in tqdm(param_combos, desc=param_name):
                results.append(run_single_simulation(combo, True))
        # Draw PNGs directly from stored traces
        for res in results:
            label = f"{param_name}_{res[param_name]:.3f}"
            _plot_voltage_trace(res, param_name, label, dirs)

    # ---------------------------------------------------------------------
    # Save numeric results -------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = os.path.join(dirs["single"], f"{param_name}_sweep.csv")
    df.to_csv(csv_path, index=False)

    # HDF5 traces ----------------------------------------------------------
    h5_path = os.path.join(dirs["data"], f"{param_name}_traces.h5")
    with h5py.File(h5_path, "w") as f:
        for i, res in enumerate(results):
            if "V_trace_A" in res:
                grp = f.create_group(f"sim_{i}")
                grp.create_dataset("V_A", data=res["V_trace_A"])
                grp.create_dataset("V_B", data=res["V_trace_B"])
                grp.create_dataset("time", data=res["time"])
                grp.attrs[param_name] = res[param_name]

    # Overview metric plots (unchanged) -----------------------------------
    plot_single_param_results(df, param_name, param_values, dirs)

    return df

def plot_single_param_results(df, param_name, param_values, dirs):
    """Plot results from single parameter sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Interburst frequency
    axes[0, 0].plot(df[param_name], df['interburst_freq_A'], 'b-o', label='Neuron A')
    axes[0, 0].plot(df[param_name], df['interburst_freq_B'], 'r--s', label='Neuron B')
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel('Interburst Frequency (Hz)')
    axes[0, 0].set_title('Interburst Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Intraburst frequency
    axes[0, 1].plot(df[param_name], df['intraburst_freq_A'], 'b-o', label='Neuron A')
    axes[0, 1].plot(df[param_name], df['intraburst_freq_B'], 'r--s', label='Neuron B')
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel('Intraburst Frequency (Hz)')
    axes[0, 1].set_title('Intraburst Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Duty cycle
    axes[1, 0].plot(df[param_name], df['duty_cycle_A'], 'b-o', label='Neuron A')
    axes[1, 0].plot(df[param_name], df['duty_cycle_B'], 'r--s', label='Neuron B')
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel('Duty Cycle')
    axes[1, 0].set_title('Duty Cycle')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Spikes per burst
    axes[1, 1].plot(df[param_name], df['mean_spikes_per_burst_A'], 'b-o', label='Neuron A')
    axes[1, 1].plot(df[param_name], df['mean_spikes_per_burst_B'], 'r--s', label='Neuron B')
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].set_ylabel('Mean Spikes per Burst')
    axes[1, 1].set_title('Spikes per Burst')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['plots'], f'{param_name}_single_sweep.png'), dpi=150)
    plt.close()

def multi_param_sweep(group_name, param_names, dirs):
    """Sweep multiple parameters together."""
    print(f"\n――― Multi‑parameter sweep: {group_name}  ({len(param_names)} dims) ―――")
    
    # Create all combinations
    param_values = [PARAM_RANGES[p] for p in param_names]
    all_combos = list(itertools.product(*param_values))
    param_dicts = [dict(zip(param_names, combo)) for combo in all_combos]
    
    print(f"Total combinations: {len(param_dicts)}")
    
    # Run simulations
    results = []
    if HAS_TORCH and len(param_dicts) > 50:
        # Use GPU for batch processing
        batch_size = 200
        for i in tqdm(range(0, len(param_dicts), batch_size), desc="GPU batches"):
            batch = param_dicts[i:i+batch_size]
            batch_results = run_gpu_batch(batch)
            if batch_results:
                results.extend(batch_results)
            else:
                # Fallback
                for combo in batch:
                    results.append(run_single_simulation(combo, False))
    else:
        # Use CPU
        if HAS_JOBLIB:
            results = Parallel(n_jobs=-1)(
                delayed(run_single_simulation)(combo, False)
                for combo in tqdm(param_dicts, desc=group_name)
            )
        else:
            for combo in tqdm(param_dicts, desc=group_name):
                results.append(run_single_simulation(combo, False))
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(dirs['multi'], f'{group_name}_sweep.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as pickle for faster loading
    pkl_path = os.path.join(dirs['data'], f'{group_name}_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(df, f)
    
    # Plot results based on dimensionality
    if len(param_names) == 2:
        plot_2d_results(df, param_names, group_name, dirs)
    elif len(param_names) == 3:
        plot_3d_results(df, param_names, group_name, dirs)
    elif len(param_names) == 4:
        plot_4d_results(df, param_names, group_name, dirs)
    
    return df

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
    features = ['interburst_freq_A', 'intraburst_freq_A', 'duty_cycle_A']
    
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

def save_example_traces(param_dict, dirs, label: str = "example") -> None:
    """Run one HCO with *param_dict* and store voltage traces & PNG plot."""
    result = run_single_simulation(param_dict, save_trace=True)

    if "V_trace_A" not in result:
        print("Warning: trace not present – nothing to save.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    time = result["time"]

    axes[0].plot(time, result["V_trace_A"], "b-", linewidth=0.8)
    axes[0].set_ylabel("Neuron A Voltage (mV)")
    axes[0].set_title(f"Half‑Centre Oscillator – {label}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, result["V_trace_B"], "r-", linewidth=0.8)
    axes[1].set_ylabel("Neuron B Voltage (mV)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    # Parameter summary in footer
    if param_dict:
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in param_dict.items())
    else:
        param_str = "DEFAULT PARAMETER SET"
    fig.text(0.5, 0.02, param_str, ha="center", fontsize=9)

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    png_path = os.path.join(dirs["plots"], f"trace_{label}.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved example trace plot → {png_path}")

# ‑‑‑ aggregation helper ‑‑‑

def _concat_and_store(dfs, dirs):
    """Concatenate a list of DataFrames and persist as Parquet."""
    if not dfs:
        return None
    big = pd.concat(dfs, ignore_index=True)
    pq_path = os.path.join(dirs["data"], "all_sweep_results.parquet")
    table = pa.Table.from_pandas(big)
    pq.write_table(table, pq_path, compression="zstd")
    print(f"\n✅  Aggregated {len(big)} rows → {pq_path}")
    return big

# ‑‑‑ main orchestration ‑‑‑

def main():
    """Run every sweep (single + grouped) and store artefacts."""
    dirs = create_output_dirs()

    # 1) SINGLE‑PARAMETER SWEEPS
    single_dfs = []
    for p_name, p_vals in PARAM_RANGES.items():
        df = single_param_sweep(p_name, p_vals, dirs)
        single_dfs.append(df)

    # 2) MULTI‑PARAMETER GROUP SWEEPS
    multi_dfs = []
    for grp, names in PARAM_GROUPS.items():
        df = multi_param_sweep(grp, names, dirs)
        multi_dfs.append(df)

    # 3) Aggregation (optional but handy)
    _concat_and_store(single_dfs + multi_dfs, dirs)

    # 4) Example traces – default and an edge‑case variant
    save_example_traces({}, dirs, label="default_params")
    edgecase = {"g_syn_i": PARAM_RANGES["g_syn_i"][-1],
                "Vs0": PARAM_RANGES["Vs0"][0]}
    save_example_traces(edgecase, dirs, label="edge_case")


# -----------------------------------------------------------------------------
#  Script entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n🏁  Total runtime: {elapsed/60:.1f} min")
