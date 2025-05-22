import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# Attempt to import optional libraries for parallelization and GPU support
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
try:
    import torch
except ImportError:
    torch = None

# --- SynapticNeuron class (assumed to be available via import or defined above) ---
from synapticNeuronClass import SynapticNeuron
from half_centre import simulate_halfcentre
from utils import extract_features
# (Here we assume SynapticNeuron is already defined/imported with half-centre support)

# Toggle for GPU usage:
USE_GPU = False
if torch is not None and torch.cuda.is_available():
    print("CUDA is available, using GPU for simulation.")
    USE_GPU = True

# --- Parameter ranges for sweeps ---

# Define the range of values for each parameter to be swept.
# These ranges can be adjusted as needed. We choose values that cover a broad span
# around the default to observe changes in bursting behavior.
default_params = {
    'excitatory_Vin': None,
    'inhibitory_Vin': None,
    'cap': 1.0, 'k': 1.0,
    'V0': -52.0, 'Vs0': -56.0, 'Vus0': -60.0,
    'g_f': 1.5, 'g_s': 0.5, 'g_us': 0.015,
    'tau_s': 4.3, 'tau_us': 278.0,
    'V_threshold': -20.0, 'V_peak': 40.0,
    'V_reset': -60.0, 'Vs_reset': -65.0, 'delta_Vus': 1.7,
    'Ve0': 0.0, 'Vi0': -90.0,
    'Ve_threshold': -20.0, 'Vi_threshold': -20.0,
    'tau_e': 5.0, 'tau_i': 10.0,
    'g_syn_e': 10.0, 'g_syn_i': 30.0
}
param_values = {
    'Vs0': np.linspace(-60.0, -40.0, 20),        # default -50; sweep around ±10 mV
    'Vus0': np.linspace(-60.0, -40.0, 20),       # default -52; sweep around ±10 mV
    'g_us': np.linspace(0.005, 0.03, 20),        # default 0.015; sweep from lower to about 2x
    'delta_Vus': np.linspace(1.0, 2.5, 20),      # default 1.7; vary ±~40%
    'tau_us': np.linspace(100.0, 400.0, 20),     # default 278; vary shorter vs longer
    'g_s': np.linspace(0.1, 1.0, 20),           # default 0.5; from weaker to stronger slow conductance
    'tau_s': np.linspace(2.0, 10.0, 20),         # default 4.3; shorter vs longer slow time constant
    'g_syn_i': np.linspace(0.0, 60.0, 20),       # default 30; from no coupling to double strength
    'tau_i': np.linspace(20.0, 80.0, 20),        # default 50; vary inhibitory synapse time constant
    'Vi_threshold': np.linspace(-40.0, 0.0, 20), # default -20; inhibitory activation threshold
    'Vi0': np.linspace(-100.0, -70.0, 20)        # default -90; inhibitory reversal potential
}

# Group the parameters as specified for combined sweeps.
param_groups = {
    'Vs0_Vus0':      ['Vs0', 'Vus0'],
    'slow_gating':        ['g_s', 'tau_s'],
    'ultraslow_gating':  ['g_us', 'delta_Vus', 'tau_us'],
    'synapse':      ['g_syn_i', 'tau_i', 'Vi_threshold', 'Vi0']
}

# --- Simulation settings ---
simulation_time = 5.0  # total simulation time in ms (e.g., 5000 ms = 5 seconds)
dt = 5e-5                  # time step in ms (small dt for stability in stiff system)
burst_gap_threshold = 100.0  # gap (ms) between spikes to separate bursts (>=100ms = new burst)
steps = int(simulation_time / dt)  # number of simulation steps



# External input current
def make_I_ext(dt, T, amp=0.0, start=0.5):
    n = int(T/dt)
    I_ext = np.zeros(n)
    I_ext[int(start/dt):] = amp
    return I_ext

I_ext = make_I_ext(dt, simulation_time)

# --- Helper functions for feature computation ---

def analyze_spike_train(spike_times, total_time):
    """
    Given a list of spike times (in ms) for a single neuron, compute bursting features:
    - interburst_freq (Hz)
    - intraburst_freq (Hz)
    - spikes_per_burst
    - duty_cycle (fraction of time active)
    """
    if len(spike_times) == 0:
        # No spikes at all
        return 0.0, 0.0, 0.0, 0.0
    # Identify bursts by grouping spikes with intervals below threshold
    bursts = []
    current_burst = [spike_times[0]]
    for t in spike_times[1:]:
        if t - current_burst[-1] >= burst_gap_threshold:
            # Start of a new burst if gap is large
            bursts.append(current_burst)
            current_burst = [t]
        else:
            current_burst.append(t)
    bursts.append(current_burst)
    num_bursts = len(bursts)
    # Calculate interburst frequency (bursts per second)
    if num_bursts > 1:
        # Use burst start times to compute average period
        burst_starts = [b[0] for b in bursts]
        avg_period_ms = (burst_starts[-1] - burst_starts[0]) / (num_bursts - 1)
        interburst_freq = 1000.0 / avg_period_ms  # in Hz
    else:
        interburst_freq = 0.0  # Only one or zero bursts, no repeating cycle
    # Intraburst frequency: average spike frequency within bursts
    intraburst_freqs = []
    for burst in bursts:
        if len(burst) > 1:
            burst_duration = burst[-1] - burst[0]  # ms
            if burst_duration > 0:
                freq = (len(burst) - 1) * 1000.0 / burst_duration  # Hz
            else:
                freq = 0.0
        else:
            # Only one spike in the burst, define intraburst freq as 0 (no interval)
            freq = 0.0
        intraburst_freqs.append(freq)
    intraburst_freq = np.mean(intraburst_freqs) if intraburst_freqs else 0.0
    # Spikes per burst (average)
    spikes_per_burst = np.mean([len(b) for b in bursts])
    # Duty cycle: fraction of time active (bursting)
    duty_cycles = []
    if num_bursts > 1:
        # Compute duty for each complete burst-period cycle
        burst_starts = [b[0] for b in bursts]
        for i in range(num_bursts - 1):
            burst_duration = bursts[i][-1] - bursts[i][0]
            period = burst_starts[i+1] - burst_starts[i]
            duty_cycles.append(min(1.0, burst_duration / period))
        # For last burst, estimate duty cycle using simulation end as end of period
        last_burst_dur = bursts[-1][-1] - bursts[-1][0]
        last_period = total_time - bursts[-1][0]
        duty_cycles.append(min(1.0, last_burst_dur / last_period))
    elif num_bursts == 1:
        # If only one burst, use simulation end as the cycle boundary
        burst_duration = bursts[0][-1] - bursts[0][0]
        duty_cycles.append(min(1.0, burst_duration / total_time))
    else:
        duty_cycles.append(0.0)
    duty_cycle = np.mean(duty_cycles) if duty_cycles else 0.0
    return interburst_freq, intraburst_freq, spikes_per_burst, duty_cycle

# def simulate_single_neuron(params):
#     """
#     Simulate a single neuron with given parameter overrides.
#     Returns the bursting feature tuple.
#     """
#     # Initialize neuron with default parameters, overriding any given in params
#     neuron = SynapticNeuron(excitatory_Vin=None, inhibitory_Vin=None, **params)
#     spike_times = []
#     steps = int(simulation_time / dt)
#     for step in range(steps):
#         neuron.update_state(dt)
#         # Check if a spike occurred at this step by threshold crossing
#         # The SynapticNeuron class appends V_peak (threshold) to Vvalues when a spike occurs
#         if neuron.Vvalues and neuron.Vvalues[-1] >= neuron.V_threshold:
#             # Spike time = (step+1)*dt (ms) since index 0 corresponds to time dt
#             spike_times.append((step + 1) * dt)
#     # Analyze spike train to get features
#     return analyze_spike_train(spike_times, simulation_time)

# def simulate_half_centre(params):
#     """
#     Simulate a half-centre oscillator (two reciprocally inhibitory neurons) with given parameter overrides.
#     Assumes both neurons use the same parameter set for symmetry.
#     Returns bursting features for one neuron (they should be similar for the other in symmetric conditions).
#     """
#     # Initialize two neurons with mutual inhibitory input
#     # We'll set up such that each neuron has the other's membrane voltage as inhibitory input.
#     neuron1 = SynapticNeuron(excitatory_Vin=None, inhibitory_Vin=np.array([0.0]), **params)
#     neuron2 = SynapticNeuron(excitatory_Vin=None, inhibitory_Vin=np.array([0.0]), **params)
#     # Small difference in initial membrane potential to break symmetry (avoid perfect in-phase spiking)
#     neuron2.V += 0.1  # slight offset
#     spike_times1 = []
#     spike_times2 = []
#     steps = int(simulation_time / dt)
#     for step in range(steps):
#         # Update the inhibitory input for each neuron (use the partner's last V)
#         neuron1.update_inputs(inhibitory_Vin=np.array([neuron2.V]))
#         neuron2.update_inputs(inhibitory_Vin=np.array([neuron1.V]))
#         neuron1.update_state(dt)
#         neuron2.update_state(dt)
#         # Record spikes for neuron1 and neuron2
#         if neuron1.Vvalues and neuron1.Vvalues[-1] >= neuron1.V_threshold:
#             spike_times1.append((step + 1) * dt)
#         if neuron2.Vvalues and neuron2.Vvalues[-1] >= neuron2.V_threshold:
#             spike_times2.append((step + 1) * dt)
#     # Analyze spike trains (for symmetry, both should have similar metrics; use neuron1's metrics)
#     features1 = analyze_spike_train(spike_times1, simulation_time)
#     # (Optionally, one could verify neuron2's features similarly)
#     return features1

# If GPU is available, define a vectorized simulator for single neurons using PyTorch for speed
# def simulate_many_single_gpu(param_list, param_names):
#     """
#     Simulate many single neurons in parallel on GPU using PyTorch.
#     `param_list` is a list of parameter tuples corresponding to param_names.
#     Returns a list of feature tuples for each parameter combination.
#     """
#     # Convert parameter list to torch tensors
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     N = len(param_list)
#     # Prepare parameter tensors (default values expanded or specific values for each)
#     # We'll broadcast scalars and handle vector parameters
#     # Default parameters (from SynapticNeuron defaults) for reference:
#     default = {
#         'cap': 0.82, 'k': 250.0,
#         'V0': -52.0, 'Vs0': -50.0, 'Vus0': -52.0,
#         'g_f': 1.0, 'g_s': 0.5, 'g_us': 0.015,
#         'tau_s': 4.3, 'tau_us': 278.0,
#         'V_threshold': 20.0, 'V_peak': 20.0,
#         'V_reset': -45.0, 'Vs_reset': 7.5, 'delta_Vus': 1.7,
#         'Ve0': 0.0, 'Vi0': -90.0,
#         'Ve_threshold': -20.0, 'Vi_threshold': -20.0,
#         'tau_e': 50.0, 'tau_i': 50.0,
#         'g_syn_e': 0.5, 'g_syn_i': 30.0,
#         'I_ext': 0.0
#     }
#     # Initialize state tensors for N neurons
#     V  = torch.full((N,), default['V0'], dtype=torch.float32, device=device)
#     Vs = torch.full((N,), default['Vs0'], dtype=torch.float32, device=device)
#     Vus = torch.full((N,), default['Vus0'], dtype=torch.float32, device=device)
#     Se = torch.zeros((N,), dtype=torch.float32, device=device)
#     Si = torch.zeros((N,), dtype=torch.float32, device=device)
#     # Fixed scalar parameters (use torch.tensor on device)
#     cap = torch.tensor(default['cap'], device=device)
#     k   = torch.tensor(default['k'], device=device)
#     V0  = torch.tensor(default['V0'], device=device)
#     V_threshold = torch.tensor(default['V_threshold'], device=device)
#     V_peak = torch.tensor(default['V_peak'], device=device)
#     V_reset = torch.tensor(default['V_reset'], device=device)
#     Vs_reset = torch.tensor(default['Vs_reset'], device=device)
#     delta_Vus = torch.tensor(default['delta_Vus'], device=device)
#     Ve0 = torch.tensor(default['Ve0'], device=device)
#     Vi0 = torch.tensor(default['Vi0'], device=device)
#     Ve_thr = torch.tensor(default['Ve_threshold'], device=device)
#     Vi_thr = torch.tensor(default['Vi_threshold'], device=device)
#     # Parameter tensors (vectorized if varied, or scalar if not in param_names)
#     # Start from defaults and replace those in param_names
#     g_f  = torch.tensor(default['g_f'], device=device)   # not varied in sweeps
#     g_s  = torch.full((N,), default['g_s'], device=device)
#     g_us = torch.full((N,), default['g_us'], device=device)
#     tau_s  = torch.full((N,), default['tau_s'], device=device)
#     tau_us = torch.full((N,), default['tau_us'], device=device)
#     # Synaptic parameters (excitation not varied, inhibitory not used for single neuron)
#     g_syn_e = torch.tensor(default['g_syn_e'], device=device)
#     g_syn_i = torch.full((N,), default['g_syn_i'], device=device)
#     tau_e   = torch.tensor(default['tau_e'], device=device)
#     tau_i   = torch.full((N,), default['tau_i'], device=device)
#     Vi_threshold = torch.full((N,), default['Vi_threshold'], device=device)
#     # Override defaults with actual values from param_list for each run
#     for idx, combo in enumerate(param_list):
#         params = dict(zip(param_names, combo))
#         for pname, pval in params.items():
#             if pname == 'Vs0':
#                 Vs[idx] = pval
#             elif pname == 'Vus0':
#                 Vus[idx] = pval
#             elif pname == 'g_s':
#                 g_s[idx] = pval
#             elif pname == 'tau_s':
#                 tau_s[idx] = pval
#             elif pname == 'g_us':
#                 g_us[idx] = pval
#             elif pname == 'tau_us':
#                 tau_us[idx] = pval
#             elif pname == 'delta_Vus':
#                 delta_Vus = torch.tensor(pval, device=device)  # delta_Vus is global in class, treat as scalar (assuming same for all in a grouped sweep)
#             elif pname == 'g_syn_i':
#                 g_syn_i[idx] = pval
#             elif pname == 'tau_i':
#                 tau_i[idx] = pval
#             elif pname == 'Vi_threshold':
#                 Vi_threshold[idx] = pval
#             elif pname == 'Vi0':
#                 Vi0 = torch.tensor(pval, device=device)  # Vi0 considered global (if varied, assume same for all in group sweep)
#             elif pname == 'Vs_reset':
#                 Vs_reset = torch.tensor(pval, device=device)  # (not typically varied)
#             # Note: excitatory synapse params not varied in this context
#     # Prepare to record spike times for each neuron
#     spike_time_lists = [[] for _ in range(N)]
#     steps = int(simulation_time / dt)
#     # Precompute constant factors to speed loop (cast to float32 tensors)
#     coeff_e = g_syn_e  # constant scalar
#     coeff_i = g_syn_i  # vector for each neuron
#     # Simulation loop
#     for step in range(steps):
#         # Compute derivatives for all neurons (vectorized)
#         # Sigmoid gating for synapses (excitatory_Vin and inhibitory_Vin are constants set ~ -54 mV -> gating ~0)
#         # Here, for isolated neurons, excitatory_Vin = inhibitory_Vin = -54 (no external synaptic drive),
#         # so Se and Si will remain ~0. We include equations for completeness.
#         sigmoid_Ve = 1 / (1 + torch.exp(-40.0 * ((-54.0) - Ve_thr)))  # constant ~0
#         sigmoid_Vi = 1 / (1 + torch.exp(-40.0 * ((-54.0) - Vi_thr)))  # constant ~0 for single neuron
#         dSe = k * (sigmoid_Ve - Se) / tau_e
#         dSi = k * (sigmoid_Vi - Si) / tau_i
#         dVs = k * (V - Vs) / tau_s
#         dVus = k * (V - Vus) / tau_us
#         # Synaptic currents (none for isolated neuron except possibly self, but S ~0)
#         I_exc = coeff_e * Se * (V - Ve0)        # shape (N,)
#         I_inh = coeff_i * Si * (V - Vi0)        # shape (N,)
#         dV = k * (0.0 - I_exc - I_inh 
#                   + g_f * ((V - V0) ** 2) 
#                   - g_s * ((Vs - Vs_reset*0 + 0) ** 2)  # Correction: should use Vs0 in slow term
#                   - g_us * ((Vus - V0*0 + 0) ** 2)     # Correction: should use Vus0 in ultraslow term
#                   ) / cap
#         # (Note: The above two lines use placeholders "Vs_reset*0 + 0" etc. because we cannot easily vectorize Vs0 and Vus0 usage here if varied.
#         # For correctness, one should incorporate Vs0 and Vus0 similarly to g_s and g_us. For simplicity, we assume default V0 as baseline for those terms.)
#         # Euler update
#         V_new = V + dV * dt
#         # Determine which neurons spike this step
#         spiked_mask = V_new >= V_threshold
#         if spiked_mask.any():
#             # Handle spiking neurons: set to peak and update slow variables with spike derivatives
#             # Set those V to peak value
#             V[spiked_mask] = V_peak
#             # Recompute derivatives for spiking neurons at V_peak
#             # (Compute only for spiking ones to save computation)
#             # For simplicity, we'll compute for all and then mask (could optimize further)
#             dVs_spike = k * (V - Vs) / tau_s
#             dVus_spike = k * (V - Vus) / tau_us
#             dSe_spike = k * (sigmoid_Ve - Se) / tau_e   # essentially same as dSe
#             dSi_spike = k * (sigmoid_Vi - Si) / tau_i
#             # Update slow variables for spiked neurons using spike derivatives
#             Vs[spiked_mask] = Vs[spiked_mask] + dVs_spike[spiked_mask] * dt
#             Vus[spiked_mask] = Vus[spiked_mask] + dVus_spike[spiked_mask] * dt
#             Se[spiked_mask] = Se[spiked_mask] + dSe_spike[spiked_mask] * dt
#             Si[spiked_mask] = Si[spiked_mask] + dSi_spike[spiked_mask] * dt
#             # Log spike and apply reset
#             for idx in torch.nonzero(spiked_mask, as_tuple=False).cpu().numpy():
#                 spike_time_lists[int(idx)].append((step + 1) * dt)
#             V[spiked_mask]  = V_reset
#             Vs[spiked_mask] = Vs_reset
#             Vus[spiked_mask] += delta_Vus  # increment ultraslow var after spike
#         # Handle non-spiking neurons: normal Euler update (for those not spiking)
#         non_spiked = ~spiked_mask
#         if non_spiked.any():
#             V[non_spiked] = V_new[non_spiked]
#             Vs[non_spiked] = Vs[non_spiked] + dVs[non_spiked] * dt
#             Vus[non_spiked] = Vus[non_spiked] + dVus[non_spiked] * dt
#             Se[non_spiked] = Se[non_spiked] + dSe[non_spiked] * dt
#             Si[non_spiked] = Si[non_spiked] + dSi[non_spiked] * dt
#     # After simulation, compute features for each run
#     results = []
#     for times in spike_time_lists:
#         results.append(analyze_spike_train(times, simulation_time))
#     return results


def run_single(param, val):
    params = default_params.copy()
    params[param] = float(val)
    A = SynapticNeuron(**params)
    B = SynapticNeuron(**params)
    simulate_halfcentre(A, B, I_ext, I_ext, [], [], [], [], dt, simulation_time, False, True)
    feats = extract_features(np.array(A.Vvalues), dt)
    # Map utils output to desired order
    ib = feats['burst_freq']
    spb = feats['mean_spikes_per_burst']
    mbd = feats['mean_burst_duration']
    intra = (spb - 1) / mbd if mbd and spb>1 else np.nan
    return [val, ib, intra, spb, feats['duty_cycle']]

# Helper: run a group-param half-centre simulation (CPU)
def run_group_cpu(combo, plist):
    params = default_params.copy()
    for p,v in zip(plist, combo):
        params[p] = float(v)
    A = SynapticNeuron(**params)
    B = SynapticNeuron(**params)
    simulate_halfcentre(A, B, I_ext, I_ext, [], [], [], [], dt, simulation_time, False, False)
    feats = extract_features(np.array(A.Vvalues), dt)
    ib = feats['burst_freq']
    spb = feats['mean_spikes_per_burst']
    mbd = feats['mean_burst_duration']
    intra = (spb - 1) / mbd if mbd and spb>1 else np.nan
    return [*combo, ib, intra, spb, feats['duty_cycle']]
# GPU-accelerated grouped sweep

def run_group_gpu(plist, combos):
    # Prepare parameter tensors
    N = len(combos)
    device = torch.device('cuda')
    # Build tensors for each varying param
    p_tensors = {}
    for i,p in enumerate(plist):
        vals = [c[i] for c in combos]
        p_tensors[p] = torch.tensor(vals, dtype=torch.float32, device=device)
    # Initialize state tensors
    V_A = torch.full((N,), default_params['V0'], device=device)
    V_B = V_A.clone()
    Vs_A = torch.full((N,), default_params['Vs0'], device=device)
    Vs_B = Vs_A.clone()
    Vus_A = torch.full((N,), default_params['Vus0'], device=device)
    Vus_B = Vus_A.clone()
    Se_A = torch.zeros((N,), device=device)
    Se_B = Se_A.clone()
    Si_A = torch.zeros((N,), device=device)
    Si_B = Si_A.clone()
    # Constant parameters
    cap = torch.tensor(default_params['cap'], device=device)
    k0 = torch.tensor(default_params['k'], device=device)
    V0 = torch.tensor(default_params['V0'], device=device)
    Vs0 = torch.tensor(default_params['Vs0'], device=device)
    Vus0 = torch.tensor(default_params['Vus0'], device=device)
    g_f = torch.tensor(default_params['g_f'], device=device)
    # Synaptic neuron params
    g_s   = p_tensors.get('g_s', torch.tensor(default_params['g_s'], device=device))
    tau_s = p_tensors.get('tau_s', torch.tensor(default_params['tau_s'], device=device))
    g_us = p_tensors.get('g_us', torch.tensor(default_params['g_us'], device=device))
    delta_Vus = p_tensors.get('delta_Vus', torch.tensor(default_params['delta_Vus'], device=device))
    tau_us = p_tensors.get('tau_us', torch.tensor(default_params['tau_us'], device=device))
    g_syn_i = p_tensors.get('g_syn_i', torch.tensor(default_params['g_syn_i'], device=device))
    tau_i = p_tensors.get('tau_i', torch.tensor(default_params['tau_i'], device=device))
    Vi_thr = p_tensors.get('Vi_threshold', torch.tensor(default_params['Vi_threshold'], device=device))
    Vi0 = p_tensors.get('Vi0', torch.tensor(default_params['Vi0'], device=device))
    # Build external current tensor
    I = torch.zeros(steps, device=device)
    I[int(0.5/dt):] = 5.0
    # Record V_A traces
    V_A_trace = torch.zeros((N, steps), device=device)
    # Time-stepping
    for t in range(steps):
        inhib_A = V_B
        inhib_B = V_A
        # Slow & ultraslow
        dVs_A = k0 * (V_A - Vs_A) / tau_s
        dVs_B = k0 * (V_B - Vs_B) / tau_s
        dVus_A = k0 * (V_A - Vus_A) / tau_us
        dVus_B = k0 * (V_B - Vus_B) / tau_us
        # Gating (sigmoid)
        Si_inf_A = torch.sigmoid(40*(inhib_A - Vi_thr))
        Si_inf_B = torch.sigmoid(40*(inhib_B - Vi_thr))
        dSi_A = k0 * (Si_inf_A - Si_A) / tau_i
        dSi_B = k0 * (Si_inf_B - Si_B) / tau_i
        # Currents
        I_inh_A = g_syn_i * Si_A * (V_A - Vi0)
        I_inh_B = g_syn_i * Si_B * (V_B - Vi0)
        # Membrane deriv
        dV_A = (g_f*(V_A - V0)**2 - g_s*(Vs_A - Vs0)**2 - g_us*(Vus_A - Vus0)**2
                - I_inh_A + I[t]) * (k0/cap)
        dV_B = (g_f*(V_B - V0)**2 - g_s*(Vs_B - Vs0)**2 - g_us*(Vus_B - Vus0)**2
                - I_inh_B + I[t]) * (k0/cap)
        # Euler
        V_A = V_A + dV_A * dt
        V_B = V_B + dV_B * dt
        Vs_A = Vs_A + dVs_A * dt
        Vs_B = Vs_B + dVs_B * dt
        Vus_A = Vus_A + dVus_A * dt
        Vus_B = Vus_B + dVus_B * dt
        Si_A = Si_A + dSi_A * dt
        Si_B = Si_B + dSi_B * dt
        # Spiking/reset
        mask_A = V_A >= default_params['V_threshold']
        mask_B = V_B >= default_params['V_threshold']
        if mask_A.any():
            Vs_A[mask_A] = default_params['Vs_reset']
            V_A[mask_A] = default_params['V_reset']
            Vus_A[mask_A] += delta_Vus[mask_A]
        if mask_B.any():
            Vs_B[mask_B] = default_params['Vs_reset']
            V_B[mask_B] = default_params['V_reset']
            Vus_B[mask_B] += delta_Vus[mask_B]
        V_A_trace[:, t] = V_A
    # Extract features on CPU
    out = []
    V_A_cpu = V_A_trace.cpu().numpy()
    for i in range(N):
        feats = extract_features(V_A_cpu[i], dt)
        ib = feats['burst_freq']
        spb = feats['mean_spikes_per_burst']
        mbd = feats['mean_burst_duration']
        intra = (spb - 1)/mbd if mbd and spb>1 else np.nan
        out.append([*combos[i], ib, intra, spb, feats['duty_cycle']])
    return out

# --- Main sweeping loop ---

# We will iterate over each parameter group, perform sweeps, save results, and plot.
def sweeping():
    for group_name, params in param_groups.items():
        print(f"Running sweep for {group_name}: parameters {params}")
        # Prepare storage for results
        features_columns = ['interburst_freq', 'intraburst_freq', 'spikes_per_burst', 'duty_cycle']
        # Cartesian product of parameter values for this group
        value_lists = [param_values[p] for p in params]
        all_combinations = list(itertools.product(*value_lists))
        total_runs = len(all_combinations)
        # Estimate number of simulations for this group
        print(f"  Total combinations: {total_runs}")
        # Decide simulation method: half-centre vs single
        if group_name == 'group4_synaptic':
            model_type = 'halfcentre'
        else:
            model_type = 'single'
        # Run simulations (with parallelization or vectorization if applicable)
        results_data = []
        if model_type == 'single':
            if USE_GPU:
                # Use GPU vectorized simulation
                batch_results = simulate_many_single_gpu(all_combinations, params)
                for combo, feats in zip(all_combinations, batch_results):
                    results_data.append(list(combo) + list(feats))
            else:
                # Use parallel processing on CPU if available
                if Parallel is not None:
                    # run in parallel
                    parallel_out = Parallel(n_jobs=-1, prefer="threads")(
                        delayed(simulate_single_neuron)({p: combo[i] for i,p in enumerate(params)})
                        for combo in all_combinations
                    )
                    for combo, feats in zip(all_combinations, parallel_out):
                        results_data.append(list(combo) + list(feats))
                else:
                    # serial loop (if joblib not available)
                    for combo in all_combinations:
                        param_dict = {p: combo[i] for i,p in enumerate(params)}
                        feats = simulate_single_neuron(param_dict)
                        results_data.append(list(combo) + list(feats))
        else:  # half-centre model
            if Parallel is not None:
                parallel_out = Parallel(n_jobs=-1)(
                    delayed(simulate_half_centre)({p: combo[i] for i,p in enumerate(params)})
                    for combo in all_combinations
                )
                for combo, feats in zip(all_combinations, parallel_out):
                    results_data.append(list(combo) + list(feats))
            else:
                for combo in all_combinations:
                    param_dict = {p: combo[i] for i,p in enumerate(params)}
                    feats = simulate_half_centre(param_dict)
                    results_data.append(list(combo) + list(feats))
        # Convert results to DataFrame and save to CSV
        # Create subfolder 'hc_sweep' if it does not exist and save CSV there
        output_dir = os.path.join(os.path.dirname(__file__), 'hc_sweep')
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(results_data, columns=params + features_columns)
        
        csv_filename = os.path.join(output_dir, f"sweep_{group_name}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"  Saved results to {csv_filename}")

        # --- Plotting ---
        # Create plots for this group
        if len(params) == 1:
            # Single-parameter sweep: plot features vs parameter
            p = params[0]
            x_vals = df[p].values
            plt.figure(figsize=(6,4))
            # Plot each feature as a line
            plt.plot(x_vals, df['interburst_freq'], label='Interburst freq (Hz)')
            plt.plot(x_vals, df['intraburst_freq'], label='Intraburst freq (Hz)')
            plt.plot(x_vals, df['spikes_per_burst'], label='Spikes per burst')
            plt.plot(x_vals, df['duty_cycle'], label='Duty cycle')
            plt.xlabel(f"{p}")
            plt.ylabel("Feature value")
            plt.title(f"Sweep of {p}: Burst features")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sweep_{p}_1D.png"))
            plt.close()
        elif len(params) == 2:
            # Two-parameter sweep: contour plots for each feature
            p1, p2 = params
            X = value_lists[0]
            Y = value_lists[1]
            # Reshape feature data to 2D grids
            # (Assumes `all_combinations` were in nested order corresponding to X varying slow, Y fast)
            n1, n2 = len(X), len(Y)
            # Extract feature columns as 2D arrays
            f_inter = df['interburst_freq'].to_numpy().reshape(n1, n2)
            f_intra = df['intraburst_freq'].to_numpy().reshape(n1, n2)
            f_spikes= df['spikes_per_burst'].to_numpy().reshape(n1, n2)
            f_duty = df['duty_cycle'].to_numpy().reshape(n1, n2)
            # Plot contour for each feature
            fig, axes = plt.subplots(2, 2, figsize=(8,6))
            levels = 15  # number of contour levels
            cs = axes[0,0].contourf(X, Y, f_inter.T, levels=levels, cmap='viridis')
            axes[0,0].set_title('Interburst freq (Hz)')
            axes[0,0].set_xlabel(p1); axes[0,0].set_ylabel(p2)
            plt.colorbar(cs, ax=axes[0,0])
            cs = axes[0,1].contourf(X, Y, f_intra.T, levels=levels, cmap='viridis')
            axes[0,1].set_title('Intraburst freq (Hz)')
            axes[0,1].set_xlabel(p1); axes[0,1].set_ylabel(p2)
            plt.colorbar(cs, ax=axes[0,1])
            cs = axes[1,0].contourf(X, Y, f_spikes.T, levels=levels, cmap='plasma')
            axes[1,0].set_title('Spikes per burst')
            axes[1,0].set_xlabel(p1); axes[1,0].set_ylabel(p2)
            plt.colorbar(cs, ax=axes[1,0])
            cs = axes[1,1].contourf(X, Y, f_duty.T, levels=levels, cmap='plasma')
            axes[1,1].set_title('Duty cycle')
            axes[1,1].set_xlabel(p1); axes[1,1].set_ylabel(p2)
            plt.colorbar(cs, ax=axes[1,1])
            fig.suptitle(f"Sweep of {p1} vs {p2}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sweep_{p1}_{p2}_contour.png"))
            plt.close()
        else:
            # 3 or 4 parameters: plot a representative 2D slice (fix other params at mid-range)
            # Choose first two parameters as X-Y axes, fix others at median index
            p1, p2 = params[0], params[1]
            fixed_indices = []
            if len(params) > 2:
                # Fix parameter 3 (and 4) at middle value
                for j in range(2, len(params)):
                    fixed_indices.append(len(value_lists[j]) // 2)
            else:
                fixed_indices = []
            # Filter the DataFrame for the fixed values (approximately at mid of each fixed param range)
            df_slice = df.copy()
            for j, idx in enumerate(fixed_indices, start=2):
                fixed_val = value_lists[j][idx]
                df_slice = df_slice[np.isclose(df_slice[params[j]], fixed_val, rtol=1e-3)]
            if df_slice.empty:
                df_slice = df  # fallback: use full if filtering failed
            # Prepare grid for p1 vs p2 with fixed others
            X = value_lists[0]
            Y = value_lists[1]
            n1, n2 = len(X), len(Y)
            # We might need to reshape slice data; ensure it's sorted by p1,p2
            df_slice = df_slice.sort_values(by=[p1, p2])
            try:
                f_inter = df_slice['interburst_freq'].to_numpy().reshape(n1, n2)
            except Exception:
                # If reshape fails due to filtering (some combos missing), fall back to nearest approach
                f_inter = np.full((n1,n2), np.nan)
                for i,x in enumerate(X):
                    for j,y in enumerate(Y):
                        sub = df_slice[(df_slice[p1]==x) & (df_slice[p2]==y)]
                        f_inter[i,j] = sub['interburst_freq'].mean() if not sub.empty else np.nan
            # (Repeat for other features)
            f_intra = df_slice['intraburst_freq'].to_numpy().reshape(n1, n2) if df_slice.shape[0] == n1*n2 else np.copy(f_inter)*np.nan
            f_spikes= df_slice['spikes_per_burst'].to_numpy().reshape(n1, n2) if df_slice.shape[0] == n1*n2 else np.copy(f_inter)*np.nan
            f_duty  = df_slice['duty_cycle'].to_numpy().reshape(n1, n2) if df_slice.shape[0] == n1*n2 else np.copy(f_inter)*np.nan
            if np.isnan(f_intra).any():
                # In case we didn't populate properly due to missing combos, fill similarly
                for i,x in enumerate(X):
                    for j,y in enumerate(Y):
                        sub = df_slice[(df_slice[p1]==x) & (df_slice[p2]==y)]
                        if not sub.empty:
                            f_intra[i,j] = sub['intraburst_freq'].mean()
                            f_spikes[i,j] = sub['spikes_per_burst'].mean()
                            f_duty[i,j] = sub['duty_cycle'].mean()
            # Plot similar to 2-param case
            fig, axes = plt.subplots(2, 2, figsize=(8,6))
            cs = axes[0,0].contourf(X, Y, f_inter.T, levels=15, cmap='viridis')
            axes[0,0].set_title('Interburst freq (Hz)')
            axes[0,0].set_xlabel(p1); axes[0,0].set_ylabel(p2)
            plt.colorbar(cs, ax=axes[0,0])
            cs = axes[0,1].contourf(X, Y, f_intra.T, levels=15, cmap='viridis')
            axes[0,1].set_title('Intraburst freq (Hz)')
            axes[0,1].set_xlabel(p1); axes[0,1].set_ylabel(p2)
            plt.colorbar(cs, ax=axes[0,1])
            cs = axes[1,0].contourf(X, Y, f_spikes.T, levels=15, cmap='plasma')
            axes[1,0].set_title('Spikes per burst')
            axes[1,0].set_xlabel(p1); axes[1,0].set_ylabel(p2)
            plt.colorbar(cs, ax=axes[1,0])
            cs = axes[1,1].contourf(X, Y, f_duty.T, levels=15, cmap='plasma')
            axes[1,1].set_title('Duty cycle')
            axes[1,1].set_xlabel(p1); axes[1,1].set_ylabel(p2)
            plt.colorbar(cs, ax=axes[1,1])
            fix_info = ""
            if len(params) > 2:
                for j, idx in enumerate(fixed_indices, start=2):
                    fix_info += f", {params[j]}={value_lists[j][idx]}"
            fig.suptitle(f"{p1} vs {p2} (fixed{fix_info})")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sweep_{group_name}_{p1}_{p2}_slice.png"))
            plt.close()


base_dir = os.path.dirname(__file__)

# 1) Single-parameter sweeps
for param, vals in param_values.items():
    out_dir = os.path.join(base_dir, 'results', 'single_sweeps', param)
    os.makedirs(out_dir, exist_ok=True)
    rows = Parallel(n_jobs=-1)(delayed(run_single)(param, v) for v in vals)
    data = np.array(rows, float)
    # Save CSV
    csv_path = os.path.join(out_dir, f"{param}.csv")
    header = f"{param},interburst_freq,intraburst_freq,spikes_per_burst,duty_cycle"
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='')
    # Plotting: combined and individual (omitted for brevity)

# 2) Grouped-parameter sweeps
import itertools
for gname, plist in param_groups.items():
    ranges = [param_values[p] for p in plist]
    combos = list(itertools.product(*ranges))
    out_dir = os.path.join(base_dir, 'results', 'group_sweeps', gname)
    os.makedirs(out_dir, exist_ok=True)
    if USE_GPU:
        rows = run_group_gpu(plist, combos)
    else:
        rows = Parallel(n_jobs=-1)(delayed(run_group_cpu)(c, plist) for c in combos)
    data = np.array(rows, float)
    # Save CSV
    csv_path = os.path.join(out_dir, f"{gname}.csv")
    header = ','.join(plist + ['interburst_freq','intraburst_freq','spikes_per_burst','duty_cycle'])
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='')
    # Plotting: combined and slices (omitted for brevity)

print("All sweeps complete.")