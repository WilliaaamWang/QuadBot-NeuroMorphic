# """
# hc_sweep.py
# Author: <you>

# Batch–sweep utilities for the Synaptic-Neuron half-centre model.
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from itertools import product
# from tqdm import tqdm
# import pathlib
# import json

# from half_centre import SynapticNeuron, simulate_synapse   # ← your code

# # ----------------------------------------------------------------------
# # 1.  LOW-LEVEL ANALYSIS HELPERS
# # ----------------------------------------------------------------------

# def detect_spikes(v_trace, t_array, thresh=0.0):
#     """
#     Return the times where v_trace crosses 'thresh' on the upward swing
#     (very fast; good enough for integrate-and-fire traces that peak at V_peak).
#     """
#     over = v_trace > thresh
#     crossings = np.where(np.logical_and(over[1:], ~over[:-1]))[0] + 1
#     return t_array[crossings]

# def detect_bursts(spike_times, gap_factor=5.0):
#     """
#     Group spike_times into bursts using an ISI gap criterion:

#         • Compute all ISIs (diff of spike_times)
#         • Any ISI larger than (gap_factor × median_intraburst_ISI)
#           is treated as a burst boundary.

#     Returns
#     -------
#     bursts : list[list[float]]
#         Each inner list is the spike times of one burst.
#     """
#     if len(spike_times) < 2:
#         return []

#     isis = np.diff(spike_times)
#     med_isi = np.median(isis)
#     if med_isi == 0:                 # degenerate / no spikes
#         return []

#     gap_thresh = gap_factor * med_isi
#     burst_indices = np.where(isis > gap_thresh)[0]          # indices *before* the big gap
#     burst_edges = np.concatenate([[-1], burst_indices, [len(spike_times)-1]])

#     bursts = [ list(spike_times[ burst_edges[i]+1 : burst_edges[i+1]+1 ])
#                for i in range(len(burst_edges)-1) ]

#     return bursts

# def burst_metrics(bursts):
#     """
#     Compute inter-/intra-burst frequency, duty-cycle, CVs.
#     Returns a dict (keys will become CSV columns).
#     """
#     if len(bursts) < 2:
#         return { 'bursting_flag' : False }

#     # --- basic data ----------------------------------------------------
#     burst_periods = np.diff([b[0] for b in bursts])         # onset-to-onset
#     interburst_freq = 1.0 / np.mean(burst_periods)
#     interburst_cv   = np.std(burst_periods) / np.mean(burst_periods)

#     spikes_per_burst = [len(b) for b in bursts]
#     mean_spikes = np.mean(spikes_per_burst)

#     intraburst_ISIs = [np.diff(b) for b in bursts]
#     intraburst_ISIs = np.concatenate(intraburst_ISIs)
#     intraburst_freq = 1.0 / np.mean(intraburst_ISIs)
#     intraburst_cv   = np.std(intraburst_ISIs) / np.mean(intraburst_ISIs)

#     burst_durs = [b[-1] - b[0] for b in bursts]
#     duty_cycle = np.mean(burst_durs) * interburst_freq      # fraction 0-1

#     return dict(
#         bursting_flag   = True,
#         interburst_hz   = interburst_freq,
#         interburst_cv   = interburst_cv,
#         intraburst_hz   = intraburst_freq,
#         intraburst_cv   = intraburst_cv,
#         spikes_per_burst = mean_spikes,
#         duty_cycle      = duty_cycle
#     )

# # ----------------------------------------------------------------------
# # 2.  SINGLE–RUN WRAPPER
# # ----------------------------------------------------------------------

# def run_halfcentre(params,  *,
#                    dt      = 1e-4,
#                    runtime = 5.0,
#                    early_stop_silent = 0.5,
#                    early_stop_tonic  = 1.5,
#                    spike_thresh = 0.0):
#     """
#     Simulate one parameter set.

#     Parameters
#     ----------
#     params : dict
#         Keys are SynapticNeuron attributes to set (both neurons will share the same value
#         except 'Vi_threshold' or any suffix '_A' / '_B' for asymmetry).
#     early_stop_silent : float (seconds)
#         If no spikes detected for this long → abort & mark 'silent'.
#     early_stop_tonic : float (seconds)
#         If spikes detected but no burst gap for this long → abort & mark 'tonic'.
#     spike_thresh : float
#         Voltage threshold for spike detection.

#     Returns
#     -------
#     result_row : dict   (ready to append to DataFrame)
#     """

#     # ------------------------------------------------------------------
#     # build two neurons with the desired params
#     # allow asymmetric params by suffixing '_A' or '_B' in the param dict
#     # ------------------------------------------------------------------
#     common_kwargs = {}
#     neuronA_kwargs, neuronB_kwargs = {}, {}

#     for k, v in params.items():
#         if k.endswith('_A'):
#             neuronA_kwargs[k[:-2]] = v
#         elif k.endswith('_B'):
#             neuronB_kwargs[k[:-2]] = v
#         else:
#             common_kwargs[k] = v

#     neuronA = SynapticNeuron(**common_kwargs, **neuronA_kwargs)
#     neuronB = SynapticNeuron(**common_kwargs, **neuronB_kwargs)

#     # zero external drive by default
#     num_steps = int(runtime / dt)
#     I_ext_A = np.zeros(num_steps)
#     I_ext_B = np.zeros(num_steps)

#     t_array = np.arange(0, runtime, dt)

#     # ------------------------------------------------------------------
#     # simulate
#     # ------------------------------------------------------------------
#     simulate_synapse(neuronA, neuronB,
#                      I_ext_A, I_ext_B,
#                      [], [],             # no excitatory ext inputs
#                      [], [],
#                      dt, runtime, plotter=False, same_start=False)

#     # ------------------------------------------------------------------
#     # analyse spikes
#     # ------------------------------------------------------------------
#     spikes_A = detect_spikes(np.array(neuronA.Vvalues), t_array, spike_thresh)
#     spikes_B = detect_spikes(np.array(neuronB.Vvalues), t_array, spike_thresh)

#     # early classification  --------------------------------------------
#     if len(spikes_A) == 0 and len(spikes_B) == 0:
#         return {**params, 'mode':'silent'}

#     # check tonic (no big gaps at least early_stop_tonic) --------------
#     if len(spikes_A) > 1:
#         isiA = np.diff(spikes_A)
#         if np.max(isiA) < early_stop_tonic:
#             return {**params, 'mode':'tonic'}
#     if len(spikes_B) > 1:
#         isiB = np.diff(spikes_B)
#         if np.max(isiB) < early_stop_tonic:
#             return {**params, 'mode':'tonic'}

#     # otherwise attempt burst detection --------------------------------
#     bursts_A = detect_bursts(spikes_A)
#     bursts_B = detect_bursts(spikes_B)

#     # choose whichever neuron has more bursts (usually both identical)
#     bursts   = bursts_A if len(bursts_A) >= len(bursts_B) else bursts_B
#     metrics  = burst_metrics(bursts)

#     if not metrics['bursting_flag']:
#         return {**params, 'mode':'unclassified'}

#     # package results
#     result = {**params,
#               'mode':'bursting',
#               **metrics}

#     return result

# # ----------------------------------------------------------------------
# # 3.  PARAMETER-SWEEP ENGINE
# # ----------------------------------------------------------------------

# def grid_sweep(param_grid, *,
#                out_csv = 'sweep_results.csv',
#                dt      = 1e-4,
#                runtime = 5.0):
#     """
#     param_grid : dict
#         e.g. {'g_us': [0.01,0.015,0.02],
#               'Vi_threshold': [-40,-30,-20]}
#         Keys can have _A / _B suffix for asymmetric sweeps.
#     """
#     combos = list(product(*param_grid.values()))
#     keys   = list(param_grid.keys())

#     results = []
#     for combo in tqdm(combos, desc='sweeping'):
#         p = dict(zip(keys, combo))
#         res = run_halfcentre(p, dt=dt, runtime=runtime)
#         results.append(res)

#     df = pd.DataFrame(results)
#     df.to_csv(out_csv, index=False)
#     print(f"[saved] {out_csv}  ({len(df)} rows)")
#     return df

# # ----------------------------------------------------------------------
# # 4.  HEAT-MAP / CONTOUR PLOTTING
# # ----------------------------------------------------------------------

# def plot_heatmap(df, x_param, y_param, z_metric,
#                  title='', cmap='viridis', zlabel=None,
#                  mask_nonburst=True):
#     """
#     Simple wrapper around plt.contourf for quick inspection.
#     """
#     df_plot = df.copy()
#     if mask_nonburst:
#         df_plot = df_plot[df_plot['mode'] == 'bursting']

#     xv = np.sort(df_plot[x_param].unique())
#     yv = np.sort(df_plot[y_param].unique())
#     Z  = np.full((len(yv), len(xv)), np.nan)

#     # fill matrix
#     for _, row in df_plot.iterrows():
#         ix = np.where(xv == row[x_param])[0][0]
#         iy = np.where(yv == row[y_param])[0][0]
#         Z[iy, ix] = row[z_metric]

#     X, Y = np.meshgrid(xv, yv)

#     plt.figure(figsize=(8,6))
#     cs = plt.contourf(X, Y, Z, levels=20, cmap=cmap)
#     plt.colorbar(cs,label=zlabel or z_metric)
#     plt.xlabel(x_param)
#     plt.ylabel(y_param)
#     plt.title(title or f'{z_metric} vs {x_param}/{y_param}')
#     plt.show()

# # ----------------------------------------------------------------------
# # 5.  EXAMPLE USAGE
# # ----------------------------------------------------------------------

# if __name__ == '__main__':
#     # --- 2-D coarse sweep example  ------------------------------------
#     param_grid = {
#         'g_us'        : np.linspace(0.005, 0.03, 6),      # adaptation strength
#         'Vi_threshold': np.linspace(-40,   -15, 6)        # inhibitory threshold
#     }
#     df = grid_sweep(param_grid,
#                     out_csv='g_us_vs_ViThresh.csv',
#                     dt=1e-4, runtime=6.0)

#     # plot inter-burst frequency heat-map
#     plot_heatmap(df,
#                  x_param='g_us',
#                  y_param='Vi_threshold',
#                  z_metric='interburst_hz',
#                  zlabel='Inter-burst freq (Hz)',
#                  title='g_us vs Vi_threshold  →  Inter-burst frequency')
