import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

if __package__ is None or __package__ == "":
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from synaptic_neuron import SynapticNeuron
    # from utils import fft_membrane_potential
else:
    from .synaptic_neuron import SynapticNeuron

# def sigmoid(x: np.array) -> np.array:
#     result = 1 / (1 + np.exp(-x))
#     return result

# def forward_euler(y, dt, dydt):
#     y_ret = y + dydt*dt
#     return y_ret

def save_plot(fig: plt.Figure, mode: str, runtime: float, level_I: float | list, outdir: str = None):
    """
    Save a matplotlib Figure to disk under a consistent naming scheme.
    mode    — one of 'baseline', 'excit', 'inhib', 'both', 'varying', etc.
    runtime — total simulated time in seconds.
    level_I — maximum or list of current levels.
    outdir  — optional output directory; defaults to 'SynapticNeuron_Plots' next to this script.
    """
    folder = outdir or os.path.join(os.path.dirname(__file__), "SynapticNeuron_Plots")
    os.makedirs(folder, exist_ok=True)
    fname = f"{mode.capitalize()}_{runtime}s_I{level_I}.png"
    full_path = os.path.join(folder, fname)
    fig.savefig(full_path)
    print(f"[plot saved] {full_path}")


def save_csv(df: pd.DataFrame, mode: str, runtime: float, level_I: float | list, outdir: str = None):
    """
    Save a DataFrame to CSV with a consistent naming scheme.
    """
    folder = outdir or os.path.join(os.path.dirname(__file__), "SynapticNeuron_Plots")
    os.makedirs(folder, exist_ok=True)
    fname = f"{mode.capitalize()}_{runtime}s_I{level_I}.csv"
    full_path = os.path.join(folder, fname)
    df.to_csv(full_path, index=False)
    print(f"[csv saved] {full_path}")


# Baseline simulation without excitatory/inhibitory inputs
def simulate_neuron_baseline(current_ext, dt, runtime):
    print("Simulating neuron WITHOUT linked EXCIT/INHIB...")
    time = np.arange(0, runtime, dt)
    # numsteps = int(runtime / dt)

    neuron = SynapticNeuron(None, None)

    excit_ext = np.full_like(time, neuron.V0)
    inhib_ext = np.full_like(time, neuron.V0)   
    
    for i, (_, I_ext) in enumerate(zip(time, current_ext)):
        # Pure excitatory input
        neuron.update_inputs(I_ext=I_ext, excitatory_Vin=excit_ext[i], inhibitory_Vin=inhib_ext[i])
        neuron.update_state(dt)

    # print(f"Max Se: {max(neuron.Sevalues)}")
    # print(f"Min Se: {min(neuron.Sevalues)}")
    # print(f"Max Si: {max(neuron.Sivalues)}")
    # print(f"Min Si: {min(neuron.Sivalues)}")

    # print(f"Max V: {max(neuron.Vvalues)}")
    # print(f"Min V: {min(neuron.Vvalues)}")
    
    # Build the plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time, current_ext)
    axes[0].set(title="Current Input", ylabel="I_ext (mA/nF)")

    axes[1].plot(time, neuron.Sevalues, label="Se")
    axes[1].plot(time, neuron.Sivalues, label="Si")
    axes[1].set(title="Se, Si", ylabel="Se/Si")
    axes[1].legend()

    axes[2].plot(time, excit_ext, label="Excitatory Vin")
    axes[2].plot(time, neuron.Ve_threshold * np.ones_like(time),
                 linestyle='dotted', alpha=0.75, label="Ve_th")
    axes[2].plot(time, inhib_ext, label="Inhibitory Vin")
    axes[2].plot(time, neuron.Vi_threshold * np.ones_like(time),
                 linestyle='dotted', alpha=0.75, label="Vi_th")
    axes[2].set(title="External Voltages", ylabel="V (mV)")
    axes[2].legend()

    axes[3].plot(time, neuron.Vvalues)
    axes[3].set(title="Membrane Potential", ylabel="V (mV)", xlabel="Time (s)")

    for ax in axes:
        ax.grid(True)
    fig.tight_layout()

    return fig, neuron

def simulate_neuron_excit(current_ext, dt, runtime):
    print("Simulating neuron with EXCITATORY inputs...")
    numsteps = int(runtime / dt)
    time = np.arange(0, runtime, dt)

    neuron = SynapticNeuron(None, None)
    
    # Excitatory input pulse
    amplitude = 100
    peak_time = 0.2
    decay_time = 0.1
    
    excit_ext = -52 + amplitude * np.exp(-((time - peak_time - 0.3*runtime)**2)/(2*decay_time**2))
    inhib_ext = np.full_like(time, neuron.V0)

    excit_ext = np.array(excit_ext)
    inhib_ext = np.array(inhib_ext)

    # print(f"Max excit: {max(excit_ext)}")
    # print(f"Min excit: {min(excit_ext)}")
    # print(f"Max inhib: {max(inhib_ext)}")
    # print(f"Min inhib: {min(inhib_ext)}")
    
    for i, (t, I_ext) in enumerate(zip(time, current_ext)):
        # Pure excitatory input
        neuron.update_inputs(I_ext=I_ext, excitatory_Vin=excit_ext[i], inhibitory_Vin=inhib_ext[i])
        neuron.update_state(dt)

    # print(f"Max Se: {max(neuron.Sevalues)}")
    # print(f"Min Se: {min(neuron.Sevalues)}")
    # print(f"Max Si: {max(neuron.Sivalues)}")
    # print(f"Min Si: {min(neuron.Sivalues)}")

    fig1, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time, current_ext)
    axes[0].set(title="Current Input", ylabel="I_ext (mA/nF)")

    axes[1].plot(time, neuron.Sevalues)
    axes[1].set(title="Se", ylabel="Se")

    axes[2].plot(time, excit_ext, label="Excitatory Vin")
    axes[2].plot(time, neuron.Ve_threshold * np.ones_like(time),
                 linestyle='dotted', alpha=0.75, label="Ve_th")
    axes[2].set(title="Excitatory Input", ylabel="V (mV)")
    axes[2].legend()

    axes[3].plot(time, neuron.Vvalues)
    axes[3].set(title="Membrane Potential", ylabel="V (mV)", xlabel="Time (s)")

    for ax in axes:
        ax.grid(True)
    fig1.tight_layout()

    # plt.close()

 
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(time, neuron.I_excitatory_values, label="I_excitatory")
    plt.title('I_excitatory')
    plt.xlabel('Time (s)')
    plt.ylabel('I_excitatory (mA/nF)')
    plt.grid()
    plt.legend()
    # plt.show()
    # plt.savefig(os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/Excit_{decay_time}decay_{runtime}s_{np.max(current_ext)}_current.png'))

    # plt.close()

    return fig1, fig2, neuron

def simulate_neuron_inhib(current_ext, dt, runtime):
    print("Simulating neuron with INHIBITORY inputs...")
    numsteps = int(runtime / dt)
    time = np.arange(0, runtime, dt)
    
    neuron = SynapticNeuron(None, None)
    
    # Inhibitory input pulse
    amplitude = 100
    peak_time = 0.2
    decay_time = 0.1

    excit_ext = np.full_like(time, neuron.V0)
    inhib_ext = -52 + amplitude * np.exp(-((time - peak_time - 0.3*runtime)**2)/(2*decay_time**2))

    excit_ext = np.array(excit_ext)
    inhib_ext = np.array(inhib_ext)

    # print(f"Max excit: {max(excit_ext)}")
    # print(f"Min excit: {min(excit_ext)}")
    # print(f"Max inhib: {max(inhib_ext)}")
    # print(f"Min inhib: {min(inhib_ext)}")

    for i, (t, I_ext) in enumerate(zip(time, current_ext)):
        # Pure inhibitory input
        neuron.update_inputs(I_ext=I_ext, excitatory_Vin=excit_ext[i], inhibitory_Vin=inhib_ext[i])
        neuron.update_state(dt)

    # print(f"Max Se: {max(neuron.Sevalues)}")
    # print(f"Min Se: {min(neuron.Sevalues)}")
    # print(f"Max Si: {max(neuron.Sivalues)}")
    # print(f"Min Si: {min(neuron.Sivalues)}")

    fig1, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time, current_ext)
    axes[0].set(title="Current Input", ylabel="I_ext (mA/nF)")

    axes[1].plot(time, neuron.Sivalues)
    axes[1].set(title="Si", ylabel="Si")

    axes[2].plot(time, inhib_ext, label="Inhibitory Vin")
    axes[2].plot(time, neuron.Vi_threshold * np.ones_like(time),
                 linestyle='dotted', alpha=0.75, label="Vi_th")
    axes[2].set(title="Inhibitory Input", ylabel="V (mV)")
    axes[2].legend()

    axes[3].plot(time, neuron.Vvalues)
    axes[3].set(title="Membrane Potential", ylabel="V (mV)", xlabel="Time (s)")

    for ax in axes:
        ax.grid(True)
    fig1.tight_layout()

    # plt.close()

    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(time, neuron.I_inhibitory_values, label="I_inhibitory")
    plt.title('I_inhibitory')
    plt.xlabel('Time (s)')
    plt.ylabel('I_inhibitory (mA/nF)')
    plt.grid()
    plt.legend()
    # plt.show()
    # plt.savefig(os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/Inhib_{decay_time}decay_{runtime}s_{np.max(current_ext)}_current.png'))

    # plt.close()

    return fig1, fig2, neuron

def simulate_neuron(current_ext, dt, runtime):
    print("Simulating neuron with BOTH excitatory and inhibitory inputs...")
    time = np.arange(0, runtime, dt)
    
    neuron = SynapticNeuron(None, None)
    
    # Inhibitory input pulse
    amplitude = 100
    peak_time = 0.2
    decay_time = 0.01

    excit_ext = -52 + amplitude * np.exp(-((time - peak_time - 0.2*runtime)**2)/(2*decay_time**2))

    inhib_ext = -52 + amplitude * np.exp(-((time - peak_time - 0.6*runtime)**2)/(2*decay_time**2))


    excit_ext = np.array(excit_ext)
    inhib_ext = np.array(inhib_ext)

    # print(f"Max excit: {max(excit_ext)}")
    # print(f"Min excit: {min(excit_ext)}")
    # print(f"Max inhib: {max(inhib_ext)}")
    # print(f"Min inhib: {min(inhib_ext)}")

    for i, (t, I_ext) in enumerate(zip(time, current_ext)):
        # Pure excitatory input
        neuron.update_inputs(I_ext=I_ext, excitatory_Vin=excit_ext[i], inhibitory_Vin=inhib_ext[i])
        neuron.update_state(dt)

    # print(f"Max Se: {max(neuron.Sevalues)}")
    # print(f"Min Se: {min(neuron.Sevalues)}")
    # print(f"Max Si: {max(neuron.Sivalues)}")
    # print(f"Min Si: {min(neuron.Sivalues)}")

    fig1, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time, current_ext)
    axes[0].set(title="Current Input", ylabel="I_ext (mA/nF)")

    axes[1].plot(time, neuron.Sevalues, label="Se")
    axes[1].plot(time, neuron.Sivalues, label="Si")
    axes[1].set(title="Se, Si", ylabel="Se/Si")
    axes[1].legend()

    axes[2].plot(time, excit_ext, label="Excitatory Vin")
    axes[2].plot(time, neuron.Ve_threshold * np.ones_like(time), linestyle='dotted', alpha=0.75)
    axes[2].plot(time, inhib_ext, label="Inhibitory Vin")
    axes[2].plot(time, neuron.Vi_threshold * np.ones_like(time), linestyle='dotted', alpha=0.75)
    axes[2].set(title="External Voltages", ylabel="V (mV)")
    axes[2].legend()

    axes[3].plot(time, neuron.Vvalues)
    axes[3].set(title="Membrane Potential", ylabel="V (mV)", xlabel="Time (s)")

    for ax in axes:
        ax.grid(True)
    fig1.tight_layout()

    # plt.show()
    # plt.close()

    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(time, neuron.I_excitatory_values, label="I_excitatory", color='red')
    plt.plot(time, neuron.I_inhibitory_values, label="I_inhibitory", color='blue')
    plt.title('I_excitatory & I_inhibitory')
    plt.xlabel('Time (s)')
    plt.ylabel('I (mA/nF)')
    plt.grid()
    plt.legend()
    # plt.show()
    # plt.close()

    return fig1, fig2, neuron

    # sigmoid_time = sigmoid(time)
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, sigmoid_time, label="Sigmoid")
    # plt.title('Sigmoid')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Sigmoid')
    # plt.grid()
    # plt.legend()
    # plt.savefig(os.path.join(os.path.dirname(__file__), 'synaptic_neuron_sigmoid.png'))
    # plt.show()


def simulate_modes():
    dt = 5e-5
    # runtimes = [5.0, 10.0, 15.0]
    runtimes = [5.0]
    amplitude = 5

    for runtime in runtimes:
        numsteps = int(runtime / dt)
        time = np.arange(0, runtime, dt)
        # Constant current input
        current_ext = np.zeros(numsteps)
        start_time = int(0.5 / dt)
        current_ext[start_time:] = amplitude

        "Simulate baseline"
        fig, neuron_baseline = simulate_neuron_baseline(current_ext, dt, runtime)
        save_plot(fig, "Baseline", runtime, amplitude)
        df = pd.DataFrame({'time': time, 'V': neuron_baseline.Vvalues})
        save_csv(df, "Baseline", runtime, amplitude)

        # neuron_baseline = simulate_neuron_baseline(current_ext, dt, runtime)
        # csvpath = os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/{runtime}s_I={amplitude}_baseline.csv')
        # data = pd.DataFrame({'time': time, 'V': neuron_baseline.Vvalues})
        # data.to_csv(csvpath, index=False)
        # print("Saved baseline data to", csvpath)


        "Simulate excitatory input"
        # fig1, fig2, neuron_excit = simulate_neuron_excit(current_ext, dt, runtime)
        # save_plot(fig1, "Excit", runtime, amplitude)
        # save_plot(fig2, "Excit_I", runtime, amplitude)
        # df = pd.DataFrame({'time': time, 'V': neuron_excit.Vvalues})
        # save_csv(df, "Excit", runtime, amplitude)

        # neuron_excit = simulate_neuron_excit(current_ext, dt, runtime)
        # csvpath = os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/{runtime}s_I={amplitude}_excit.csv')
        # data = pd.DataFrame({'time': time, 'V': neuron_excit.Vvalues})
        # data.to_csv(csvpath, index=False)
        # print("Saved excitatory data to", csvpath)

        """Simulate inhibitory input"""
        # fig1, fig2, neuron_inhib = simulate_neuron_inhib(current_ext, dt, runtime)
        # save_plot(fig1, "Inhib", runtime, amplitude)
        # save_plot(fig2, "Inhib_I", runtime, amplitude)
        # df = pd.DataFrame({'time': time, 'V': neuron_inhib.Vvalues})
        # save_csv(df, "Inhib", runtime, amplitude)

        # neuron_inhib = simulate_neuron_inhib(current_ext, dt, runtime)
        # csvpath = os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/{runtime}s_I={amplitude}_inhib.csv')
        # data = pd.DataFrame({'time': time, 'V': neuron_inhib.Vvalues})
        # data.to_csv(csvpath, index=False)
        # print("Saved inhibitory data to", csvpath)

        """Simulate both excitatory and inhibitory inputs"""
        # fig1, fig2, neuron_EI = simulate_neuron(current_ext, dt, runtime)
        # save_plot(fig1, "Both", runtime, amplitude)
        # save_plot(fig2, "Both_I", runtime, amplitude)
        # df = pd.DataFrame({'time': time, 'V': neuron_EI.Vvalues})
        # save_csv(df, "Both", runtime, amplitude)

        # neuron_EI = simulate_neuron(current_ext, dt, runtime)
        # csvpath = os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/{runtime}s_I={amplitude}_EI.csv')
        # data = pd.DataFrame({'time': time, 'V': neuron_EI.Vvalues})
        # data.to_csv(csvpath, index=False)
        # print("Saved EI data to", csvpath)

def simulate_varying_current():
    dt = 5e-5
    # runtimes = [5.0, 10.0, 15.0]
    runtimes = [20.0]
    amplitudes = [-10, 0, 10, 20]
    # amplitudes = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    for runtime in runtimes:
        numsteps    = int(runtime / dt)
        time        = np.arange(0, runtime, dt)
        seg_length  = numsteps // len(amplitudes)

        current_ext = np.zeros(numsteps)
        for idx, amp in enumerate(amplitudes):
            start = idx * seg_length
            # ensure last segment fills to end even if not divisible exactly
            end   = (idx + 1) * seg_length if idx < len(amplitudes) - 1 else numsteps
            current_ext[start:end] = amp

        fig, neuron = simulate_neuron_baseline(current_ext, dt, runtime)
        save_plot(fig, "varying", runtime, amplitudes)
        df = pd.DataFrame({"time": time, "I_ext": current_ext, "V": neuron.Vvalues})
        save_csv(df, "varying", runtime, amplitudes)
        # fig.show()



if __name__ == "__main__":
    simulate_modes()
    # simulate_varying_current()
