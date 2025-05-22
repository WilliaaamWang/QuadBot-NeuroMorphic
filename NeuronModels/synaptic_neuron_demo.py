import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from .synaptic_neuron import SynapticNeuron  # make sure synaptic_neuron.py is available in the same directory
# from utils import fft_membrane_potential

# def sigmoid(x: np.array) -> np.array:
#     result = 1 / (1 + np.exp(-x))
#     return result

# def forward_euler(y, dt, dydt):
#     y_ret = y + dydt*dt
#     return y_ret


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

    print(f"Max Se: {max(neuron.Sevalues)}")
    print(f"Min Se: {min(neuron.Sevalues)}")
    print(f"Max Si: {max(neuron.Sivalues)}")
    print(f"Min Si: {min(neuron.Sivalues)}")

    print(f"Max V: {max(neuron.Vvalues)}")
    print(f"Min V: {min(neuron.Vvalues)}")
    
    plt.figure(figsize=(12, 9))
    plt.subplot(4, 1, 1)
    plt.plot(time, current_ext, label="Current Input current_ext")
    plt.title('Current Input')
    plt.xlabel('Time (s)')
    plt.ylabel('I_ext (mA/nF)')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time, neuron.Sevalues, label="Se")
    plt.plot(time, neuron.Sivalues, label="Si")
    plt.title('Se, Si')
    plt.xlabel('Time (s)')
    plt.ylabel('Se, Si')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time, excit_ext, label="Excitatory Input voltage")
    plt.plot(time, neuron.Ve_threshold*np.ones_like(time),
                label="Excit threshold Ve_th",
                linestyle='dotted',
                alpha=0.75)
    plt.plot(time, inhib_ext, label="Inhibitory Input voltage")
    plt.plot(time, neuron.Vi_threshold*np.ones_like(time),
                label="Inhib threshold Vi_th",
                linestyle='dotted',
                alpha=0.75)
    plt.title('Excit/Inhib Input')
    plt.xlabel('Time (s)')
    plt.ylabel('V (mV)')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time, neuron.Vvalues, label="Membrane Potential")
    plt.title('Membrane Potential')
    plt.xlabel('Time (s)')
    plt.ylabel('V (mV)')
    plt.grid()
    plt.legend()

    plt.tight_layout()

    # fft_membrane_potential(neuron, dt)

    plotname = f'SynapticNeuron_Plots/Baseline_{runtime}s_{np.max(current_ext)}.png'
    print("Saving figure as", plotname)
    plt.savefig(os.path.join(os.path.dirname(__file__), plotname))
    # plotname = f'SynapticNeuron_Plots/Baseline_{runtime}s_{np.max(current_ext)}.png'
    # print("Saving figure as", plotname)
    # plt.savefig(os.path.join(os.path.dirname(__file__), plotname))
    # plt.show()
    
    plt.close()

    return neuron

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

    print(f"Max excit: {max(excit_ext)}")
    print(f"Min excit: {min(excit_ext)}")
    print(f"Max inhib: {max(inhib_ext)}")
    print(f"Min inhib: {min(inhib_ext)}")
    
    for i, (t, I_ext) in enumerate(zip(time, current_ext)):
        # Pure excitatory input
        neuron.update_inputs(I_ext=I_ext, excitatory_Vin=excit_ext[i], inhibitory_Vin=inhib_ext[i])
        neuron.update_state(dt)

    print(f"Max Se: {max(neuron.Sevalues)}")
    print(f"Min Se: {min(neuron.Sevalues)}")
    print(f"Max Si: {max(neuron.Sivalues)}")
    print(f"Min Si: {min(neuron.Sivalues)}")

    plt.figure(figsize=(12, 9))
    plt.subplot(4, 1, 1)
    plt.plot(time, current_ext, label="Current Input current_ext")
    plt.title('Current Input')
    plt.xlabel('Time (s)')
    plt.ylabel('I_ext (mA/nF)')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time, neuron.Sevalues, label="Se")
    plt.title('Se')
    plt.xlabel('Time (s)')
    plt.ylabel('Se')
    plt.grid()
    plt.legend()


    plt.subplot(4, 1, 3)
    plt.plot(time, excit_ext, label="Excitatory Input voltage")
    plt.plot(time, neuron.Ve_threshold*np.ones_like(time), 
             label="Excit threshold Ve_th", 
             linestyle='dotted', 
             alpha=0.75)
    plt.plot(time, inhib_ext, label="Inhibitory Input voltage")
    plt.plot(time, neuron.Vi_threshold*np.ones_like(time), 
             label="Inhib threshold Vi_th", 
             linestyle='dotted', 
             alpha=0.75)
    plt.title('Excitatory Input')
    plt.xlabel('Time (s)')
    plt.ylabel('V (mV)')
    plt.grid()
    plt.legend()


    plt.subplot(4, 1, 4)
    plt.plot(time, neuron.Vvalues, label="Membrane Potential")
    plt.title('Membrane Potential')
    plt.xlabel('Time (s)')
    plt.ylabel('V (mV)')
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plotname = f'SynapticNeuron_Plots/Excit_{decay_time}decay_{runtime}s_{np.max(current_ext)}.png'
    print("Saving figure as", plotname)
    plt.savefig(os.path.join(os.path.dirname(__file__), plotname))

    # plt.show()
    plt.close()
 
    plt.figure(figsize=(12, 6))
    plt.plot(time, neuron.I_excitatory_values, label="I_excitatory")
    plt.title('I_excitatory')
    plt.xlabel('Time (s)')
    plt.ylabel('I_excitatory (mA/nF)')
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/Excit_{decay_time}decay_{runtime}s_{np.max(current_ext)}_current.png'))

    plt.close()

    return neuron

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

    print(f"Max excit: {max(excit_ext)}")
    print(f"Min excit: {min(excit_ext)}")
    print(f"Max inhib: {max(inhib_ext)}")
    print(f"Min inhib: {min(inhib_ext)}")

    for i, (t, I_ext) in enumerate(zip(time, current_ext)):
        # Pure inhibitory input
        neuron.update_inputs(I_ext=I_ext, excitatory_Vin=excit_ext[i], inhibitory_Vin=inhib_ext[i])
        neuron.update_state(dt)

    print(f"Max Se: {max(neuron.Sevalues)}")
    print(f"Min Se: {min(neuron.Sevalues)}")
    print(f"Max Si: {max(neuron.Sivalues)}")
    print(f"Min Si: {min(neuron.Sivalues)}")

    plt.figure(figsize=(12, 9))
    plt.subplot(4, 1, 1)
    plt.plot(time, current_ext, label="Current Input current_ext")
    plt.title('Current Input')
    plt.xlabel('Time (s)')
    plt.ylabel('I_ext (mA/nF)')
    plt.grid()
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(time, neuron.Sivalues, label="Si")
    plt.title('Si')
    plt.xlabel('Time (s)')
    plt.ylabel('Si')
    plt.grid()
    plt.legend()


    plt.subplot(4, 1, 3)
    plt.plot(time, excit_ext, label="Excitatory Input voltage")
    plt.plot(time, neuron.Ve_threshold*np.ones_like(time), 
             label="Excit threshold Ve_th", linestyle='dotted',
             alpha=0.75)
    plt.plot(time, inhib_ext, label="Inhibitory Input voltage")
    plt.plot(time, neuron.Vi_threshold*np.ones_like(time), 
             label="Inhib threshold Vi_th", 
             linestyle='dotted',
             alpha=0.75)
    plt.title('Inhibitory Input')
    plt.xlabel('Time (s)')
    plt.ylabel('V (mV)')
    plt.grid()
    plt.legend()


    plt.subplot(4, 1, 4)
    plt.plot(time, neuron.Vvalues, label="Membrane Potential")
    plt.title('Membrane Potential')
    plt.xlabel('Time (s)')
    plt.ylabel('V (mV)')
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plotname = f'SynapticNeuron_Plots/Inhib_{decay_time}decay_{runtime}s_{np.max(current_ext)}.png'
    print("Saving figure as", plotname)
    plt.savefig(os.path.join(os.path.dirname(__file__), plotname))

    # plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(time, neuron.I_inhibitory_values, label="I_inhibitory")
    plt.title('I_inhibitory')
    plt.xlabel('Time (s)')
    plt.ylabel('I_inhibitory (mA/nF)')
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/Inhib_{decay_time}decay_{runtime}s_{np.max(current_ext)}_current.png'))

    plt.close()

    return neuron

def simulate_neuron(current_ext, dt, runtime):
    print("Simulating neuron with BOTH excitatory and inhibitory inputs...")
    numsteps = int(runtime / dt)
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

    plt.figure(figsize=(12, 9))
    # plt.subplot(5, 1, 1)
    plt.subplot(4, 1, 1)
    plt.plot(time, current_ext, label="Current Input current_ext")
    plt.title('Current Input')
    plt.xlabel('Time (s)')
    plt.ylabel('I_ext (mA/nF)')
    plt.grid()
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(time, neuron.Sevalues, label="Se")
    plt.plot(time, neuron.Sivalues, label="Si")
    plt.title('Se, Si')
    plt.xlabel('Time (s)')
    plt.ylabel('Se, Si')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time, excit_ext, label="Excitatory Input voltage")
    plt.plot(time, neuron.Ve_threshold*np.ones_like(time), 
             label="Excit threshold Ve_th", 
             linestyle='dotted',
             alpha=0.75)
    plt.plot(time, inhib_ext, label="Inhibitory Input voltage")
    plt.plot(time, neuron.Vi_threshold*np.ones_like(time), 
             label="Inhib threshold Vi_th", 
             linestyle='dotted',
             alpha=0.75)
    plt.title('External Excit/Inhib Input')
    plt.xlabel('Time (s)')
    plt.ylabel('V (mV)')
    plt.grid()
    plt.legend()


    plt.subplot(4, 1, 4)
    plt.plot(time, neuron.Vvalues, label="Membrane Potential")
    plt.title('Membrane Potential')
    plt.xlabel('Time (s)')
    plt.ylabel('V (mV)')
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plotname = f'SynapticNeuron_Plots/Both_{decay_time}decay_{runtime}s_{np.max(current_ext)}.png'
    print("Saving figure as", plotname)
    plt.savefig(os.path.join(os.path.dirname(__file__), plotname))

    # plt.show()
    plt.close()


    plt.figure(figsize=(12, 6))
    plt.plot(time, neuron.I_excitatory_values, label="I_excitatory", color='red')
    plt.plot(time, neuron.I_inhibitory_values, label="I_inhibitory", color='blue')
    plt.title('I_excitatory & I_inhibitory')
    plt.xlabel('Time (s)')
    plt.ylabel('I (mA/nF)')
    plt.grid()
    plt.legend()
    # plt.show()

    plotname = f'SynapticNeuron_Plots/Both_{decay_time}decay_{runtime}s_{np.max(current_ext)}_current.png'
    plt.savefig(os.path.join(os.path.dirname(__file__), plotname))
    plt.close()

    return neuron

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


def main():
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
        neuron_baseline = simulate_neuron_baseline(current_ext, dt, runtime)
        csvpath = os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/{runtime}s_I={amplitude}_baseline.csv')
        data = pd.DataFrame({'time': time, 'V': neuron_baseline.Vvalues})
        data.to_csv(csvpath, index=False)
        print("Saved baseline data to", csvpath)

        "Simulate excitatory input"
        neuron_excit = simulate_neuron_excit(current_ext, dt, runtime)
        csvpath = os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/{runtime}s_I={amplitude}_excit.csv')
        data = pd.DataFrame({'time': time, 'V': neuron_excit.Vvalues})
        data.to_csv(csvpath, index=False)
        print("Saved excitatory data to", csvpath)

        """Simulate inhibitory input"""
        neuron_inhib = simulate_neuron_inhib(current_ext, dt, runtime)
        csvpath = os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/{runtime}s_I={amplitude}_inhib.csv')
        data = pd.DataFrame({'time': time, 'V': neuron_inhib.Vvalues})
        data.to_csv(csvpath, index=False)
        print("Saved inhibitory data to", csvpath)

        """Simulate both excitatory and inhibitory inputs"""
        neuron_EI = simulate_neuron(current_ext, dt, runtime)
        csvpath = os.path.join(os.path.dirname(__file__), f'SynapticNeuron_Plots/{runtime}s_I={amplitude}_EI.csv')
        data = pd.DataFrame({'time': time, 'V': neuron_EI.Vvalues})
        data.to_csv(csvpath, index=False)
        print("Saved EI data to", csvpath)

    


if __name__ == "__main__":
    main()
