import numpy as np
import matplotlib.pyplot as plt
from synaptic_neuron import SynapticNeuron  # make sure synaptic_neuron.py is available in the same directory

def simulate_half_center(dt, runtime):
    numsteps = int(runtime / dt)
    time = np.arange(0, runtime, dt)
    
    # Create a constant external current input (after a short delay)
    current_ext = np.zeros(numsteps)
    current_ext[numsteps // 10:] = 5  # you can adjust the amplitude as needed

    # Instantiate two neurons
    neuronA = SynapticNeuron()
    neuronB = SynapticNeuron()

    # Introduce a slight difference in initial conditions to break symmetry.
    neuronA.V = -52
    neuronB.V = -51.9

    # Set a constant excitatory input voltage (baseline drive)
    baseline_excit = -54  # same as used in your single neuron model

    # Run the simulation over time.
    for i in range(numsteps):
        # For mutual inhibition, pass the other neuron's membrane potential to the inhibitory input.
        inhibitory_input_A = neuronB.V  # neuronB's voltage inhibits neuronA
        inhibitory_input_B = neuronA.V  # neuronA's voltage inhibits neuronB

        # Update the inputs for each neuron. Both get the same current and excitatory drive.
        neuronA.update_inputs(I_ext=current_ext[i],
                                excitatory_Vin=baseline_excit,
                                inhibitory_Vin=inhibitory_input_A)
        neuronB.update_inputs(I_ext=current_ext[i],
                                excitatory_Vin=baseline_excit,
                                inhibitory_Vin=inhibitory_input_B)
        
        # Update the states for both neurons using Euler integration.
        neuronA.update_state(dt)
        neuronB.update_state(dt)
        
    return time, neuronA, neuronB

def main():
    dt = 5e-5     # time step (seconds)
    runtime = 10.0  # total simulation time (seconds)
    
    # Run the simulation for the half-center model.
    time, neuronA, neuronB = simulate_half_center(dt, runtime)
    
    # Plot the membrane potentials of both neurons.
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Panel 1: Membrane Potentials
    axs[0].plot(time, neuronA.Vvalues, label="Neuron A V")
    axs[0].plot(time, neuronB.Vvalues, label="Neuron B V")
    axs[0].set_ylabel("V (mV)")
    axs[0].set_title("Membrane Potentials")
    axs[0].legend()
    axs[0].grid(True)
    
    # Panel 2: Excitatory Synaptic Variable (Se)
    axs[1].plot(time, neuronA.Sevalues, label="Neuron A Se")
    axs[1].plot(time, neuronB.Sevalues, label="Neuron B Se")
    axs[1].set_ylabel("Se")
    axs[1].set_title("Excitatory Synaptic Activation")
    axs[1].legend()
    axs[1].grid(True)
    
    # Panel 3: Inhibitory Synaptic Variable (Si)
    axs[2].plot(time, neuronA.Sivalues, label="Neuron A Si")
    axs[2].plot(time, neuronB.Sivalues, label="Neuron B Si")
    axs[2].set_ylabel("Si")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Inhibitory Synaptic Activation")
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
