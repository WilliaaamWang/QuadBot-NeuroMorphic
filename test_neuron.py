import numpy as np
import matplotlib.pyplot as plt
from mqif_neuron import MQIFNeuron
from synapse_model import Synapse

# Parameters
time_step = 0.1  # ms
total_time = 20  # ms
time_points = int(total_time / time_step)

# Initialize neurons
neuron_A = MQIFNeuron(neuron_id='A', i_ext=1.5)
neuron_B = MQIFNeuron(neuron_id='B', i_ext=2.5)
neuron_C = MQIFNeuron(neuron_id='C', i_ext=15.5)
neuron_D = MQIFNeuron(neuron_id='D', i_ext=20.5)

# Initialize synapses
synapse_A_B = Synapse(pre_neuron=neuron_A, post_neuron=neuron_B, g_syn=0.5, tau_syn=5)
synapse_B_C = Synapse(pre_neuron=neuron_B, post_neuron=neuron_C, g_syn=0.5, tau_syn=5)
synapse_C_D = Synapse(pre_neuron=neuron_C, post_neuron=neuron_D, g_syn=0.5, tau_syn=5)
synapse_D_A = Synapse(pre_neuron=neuron_D, post_neuron=neuron_A, g_syn=0.5, tau_syn=5)

# Initialize data containers
time = np.arange(0, total_time, time_step)
voltage_A = np.zeros_like(time)
voltage_B = np.zeros_like(time)
voltage_C = np.zeros_like(time)
voltage_D = np.zeros_like(time)

# Simulate neuron behavior
for i in range(time_points):
    t = i * time_step
    
    # Calculate synaptic currents
    i_syn_B = synapse_A_B.get_synaptic_current()
    i_syn_C = synapse_B_C.get_synaptic_current()
    i_syn_D = synapse_C_D.get_synaptic_current()
    i_syn_A = synapse_D_A.get_synaptic_current()
    
    # Update neurons
    neuron_A.update(time_step, i_syn_A)
    neuron_B.update(time_step, i_syn_B)
    neuron_C.update(time_step, i_syn_C)
    neuron_D.update(time_step, i_syn_D)
    
    # Store voltage data
    voltage_A[i] = neuron_A.v
    voltage_B[i] = neuron_B.v
    voltage_C[i] = neuron_C.v
    voltage_D[i] = neuron_D.v

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time, voltage_A, label='Neuron A Voltage')
plt.plot(time, voltage_B, label='Neuron B Voltage')
plt.plot(time, voltage_C, label='Neuron C Voltage')
plt.plot(time, voltage_D, label='Neuron D Voltage')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Neuron Voltages Over Time')
plt.legend()
plt.show()
