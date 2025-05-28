"""Quad-centre oscillator model built from synaptic neurons.

This module extends the half-centre network to four neurons arranged as two
mutually inhibitory pairs with an additional cross-inhibition between one
neuron of each pair.  The topology is:

    A ↔ B      C ↔ D
      ↖   ↘  ↙   ↙
         ↘ ↖
           A ↔ C  (cross connection)

A demo is provided that runs a small simulation and plots the voltage traces of
all four neurons.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synaptic_neuron import SynapticNeuron
from utils import extract_features


# -----------------------------------------------------------------------------
#  Simulation
# -----------------------------------------------------------------------------

def simulate_quadcentre(
    neuronA: SynapticNeuron,
    neuronB: SynapticNeuron,
    neuronC: SynapticNeuron,
    neuronD: SynapticNeuron,
    I_ext_arrays: Iterable[np.ndarray],
    excit_exts: Iterable[Iterable[float] | np.ndarray],
    inhib_exts: Iterable[Iterable[float] | np.ndarray],
    *,
    dt: float = 1e-4,
    runtime: float = 5.0,
    plotter: bool = False,
    same_start: bool = True,
):
    """Simulate a quad-centre network.

    Parameters
    ----------
    neuronA, neuronB, neuronC, neuronD : SynapticNeuron
        Neuron instances representing the four nodes.
    I_ext_arrays : iterable of array_like
        External current arrays for each neuron.
    excit_exts, inhib_exts : iterable
        Extra excitatory / inhibitory voltage inputs per neuron.
    dt : float, optional
        Time step in seconds.
    runtime : float, optional
        Total simulation time in seconds.
    plotter : bool, optional
        If True, voltage traces are plotted at the end of the simulation.
    same_start : bool, optional
        If True, neurons B–D start from ``V0``; otherwise each is offset by
        ``0.1`` mV to break symmetry.
    """

    t_array = np.arange(0, runtime, dt)

    neurons = [neuronA, neuronB, neuronC, neuronD]
    prev_V = [n.V0 for n in neurons]
    if not same_start:
        prev_V = [neuronA.V0, neuronB.V0 + 0.1, neuronC.V0 + 0.2, neuronD.V0 + 0.3]
        for n, v in zip(neurons, prev_V):
            n.V = v
    else:
        for n in neurons:
            n.V = n.V0

    # Convert helpers
    excit_exts = [np.atleast_1d(e) for e in excit_exts]
    inhib_exts = [np.atleast_1d(i) for i in inhib_exts]

    for step, t in enumerate(t_array):
        I_A, I_B, I_C, I_D = (arr[step] for arr in I_ext_arrays)

        excit_A = excit_exts[0]
        excit_B = excit_exts[1]
        excit_C = excit_exts[2]
        excit_D = excit_exts[3]

        inhib_A = np.concatenate((inhib_exts[0], [prev_V[1], prev_V[2]]))
        inhib_B = np.concatenate((inhib_exts[1], [prev_V[0]]))
        inhib_C = np.concatenate((inhib_exts[2], [prev_V[3], prev_V[0]]))
        inhib_D = np.concatenate((inhib_exts[3], [prev_V[2]]))

        neuronA.update_inputs(I_A, excit_A, inhib_A)
        neuronB.update_inputs(I_B, excit_B, inhib_B)
        neuronC.update_inputs(I_C, excit_C, inhib_C)
        neuronD.update_inputs(I_D, excit_D, inhib_D)

        for n in neurons:
            n.update_state(dt)

        prev_V = [n.V for n in neurons]

    if plotter:
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        traces = [neuronA.Vvalues, neuronB.Vvalues, neuronC.Vvalues, neuronD.Vvalues]
        labels = ["Neuron A", "Neuron B", "Neuron C", "Neuron D"]
        for ax, v, lab in zip(axes, traces, labels):
            ax.plot(t_array, v)
            ax.set_ylabel(f"{lab} (mV)")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plot_dir = Path(__file__).with_suffix("").parent / "Quadcentre_Plots"
        plot_dir.mkdir(exist_ok=True)
        fname = plot_dir / "quadcentre_demo.png"
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved plot → {fname}")

    return neuronA, neuronB, neuronC, neuronD


# -----------------------------------------------------------------------------
#  Utilities
# -----------------------------------------------------------------------------

def save_qc_data(neurons: list[SynapticNeuron], dt: float, filename: str) -> None:
    """Save membrane potential traces of *neurons* to ``filename`` CSV."""
    data = {
        "Time": np.arange(len(neurons[0].Vvalues)) * dt,
        "Neuron A V": neurons[0].Vvalues,
        "Neuron B V": neurons[1].Vvalues,
        "Neuron C V": neurons[2].Vvalues,
        "Neuron D V": neurons[3].Vvalues,
    }
    pd.DataFrame(data).to_csv(filename, index=False)


def analyse_qc_features(dt: float, filename: str) -> None:
    """Load saved voltage traces and print burst features for each neuron."""
    data = pd.read_csv(filename)
    time_axis = data["Time"].values  # noqa: F841 – reserved for potential plots

    results = {}
    for lab in ["A", "B", "C", "D"]:
        trace = data[f"Neuron {lab} V"].values
        feats = extract_features(trace, dt)
        results[lab] = feats
        print(f"Neuron {lab} Regime: {feats['regime']}")
        print(f"Neuron {lab} Spike count: {feats['spike_count']}")
        print(f"Neuron {lab} Mean spikes/burst: {feats['mean_spikes_per_burst']}")
        print(f"Neuron {lab} Duty cycle: {feats['duty_cycle']}")
        print("-")

    for lab, feats in results.items():
        out = pd.DataFrame(feats)
        out.to_csv(filename.replace(".csv", f"_{lab}_features.csv"), index=False)


# -----------------------------------------------------------------------------
#  Demo
# -----------------------------------------------------------------------------

def demo() -> None:
    """Run a short demonstration simulation and plot the results."""
    dt = 5e-5
    runtime = 15.0

    steps = int(runtime / dt)
    I = np.zeros(steps)
    I[int(0.5 / dt) :] = 5.0

    neurons = [
        SynapticNeuron(excitatory_Vin=-52, inhibitory_Vin=-52, Vi_threshold=-20),
        SynapticNeuron(excitatory_Vin=-52, inhibitory_Vin=-52, Vi_threshold=-30),
        SynapticNeuron(excitatory_Vin=-52, inhibitory_Vin=-52, Vi_threshold=-20),
        SynapticNeuron(excitatory_Vin=-52, inhibitory_Vin=-52, Vi_threshold=-30),
    ]

    simulate_quadcentre(
        neurons[0], neurons[1], neurons[2], neurons[3],
        [I, I, I, I], [[], [], [], []], [[], [], [], []],
        dt=dt, runtime=runtime, plotter=True, same_start=False,
    )

    csv_dir = Path(__file__).with_suffix("").parent / "Quadcentre_Plots"
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / "quadcentre_demo.csv"
    save_qc_data(neurons, dt, str(csv_path))
    analyse_qc_features(dt, str(csv_path))


if __name__ == "__main__":
    demo()
