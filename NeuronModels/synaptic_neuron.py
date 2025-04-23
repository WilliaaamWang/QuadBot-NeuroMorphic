import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(x: np.array) -> np.array:
    result = 1 / (1 + np.exp(-x))
    return result

def forward_euler(y, dt, dydt):
    y_ret = y + dydt*dt
    return y_ret

class SynapticNeuron:
    def __init__(
        self,
        V0 = -52, Vs0 = -50, Vus0 = -52,
        tau_s = 4.3, tau_us = 278,
        g_f = 1.0, g_s = 0.5, g_us = 0.015,
        # V_threshold = 20, V_peak = 15,
        V_threshold = 20, V_peak = 20,
        V_reset = -45, Vs_reset = 7.5,
        delta_Vus = 1.7,
        # Synaptic parameters
        Ve0 = 0, Vi0 = -90,
        Ve_threshold = -20, Vi_threshold = -20,
        g_syn_e = 0.5, g_syn_i = 30,
        tau_e = 50, tau_i = 50,
        cap = 0.82, k = 250.0,
        I_ext=0.0, excitatory_Vin=None, inhibitory_Vin=None,):

        self.V0 = V0
        self.Vs0 = Vs0
        self.Vus0 = Vus0
        self.tau_s = tau_s
        self.tau_us = tau_us
        self.g_f = g_f
        self.g_s = g_s
        self.g_us = g_us
        self.V_threshold = V_threshold
        self.V_reset = V_reset
        self.Vs_reset = Vs_reset
        self.V_peak = V_peak
        self.delta_Vus = delta_Vus
        self.Ve0 = Ve0
        self.Vi0 = Vi0
        self.Ve_threshold = Ve_threshold
        self.Vi_threshold = Vi_threshold
        self.g_syn_e = g_syn_e
        self.g_syn_i = g_syn_i
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.cap = cap
        self.k = k

        # self.I_ext = I_ext
        self.excitatory_Vin = excitatory_Vin
        self.inhibitory_Vin = inhibitory_Vin

        self.V = V0
        self.Vs = Vs0
        self.Vus = Vus0
        self.Se = np.zeros_like(self.excitatory_Vin)
        self.Si = np.zeros_like(self.inhibitory_Vin)
        
        self.Vvalues = []
        self.Vsvalues = []
        self.Vusvalues = []
        self.Sevalues = []
        self.Sivalues = []
        self.I_excitatory_values = []
        self.I_inhibitory_values = []

        self.I_ext = I_ext
        self.excitatory_Vin = excitatory_Vin if excitatory_Vin is not None else Ve0
        self.inhibitory_Vin = inhibitory_Vin if inhibitory_Vin is not None else Vi0

    def compute_derivatives(self):
        """
        Se, Si corresponds to the probability of the synapse being open
        g_synapse = g_syn_e * Se OR g_syn_i * Si
        """
        
        dVs = self.k * (self.V - self.Vs) / self.tau_s
        dVus = self.k * (self.V - self.Vus) / self.tau_us

        sigmoid_Ve = sigmoid((self.excitatory_Vin - self.Ve_threshold)*40)
        sigmoid_Vi = sigmoid((self.inhibitory_Vin - self.Vi_threshold)*40)
        dSe = self.k * (sigmoid_Ve - self.Se) / self.tau_e
        dSi = self.k * (sigmoid_Vi - self.Si) / self.tau_i

        I_excitatory = self.g_syn_e * self.Se * (self.V - self.Ve0)
        I_inhibitory = self.g_syn_i * self.Si * (self.V - self.Vi0)
        
        dV = self.k * (self.I_ext - np.sum(I_excitatory) - np.sum(I_inhibitory) + self.g_f*((self.V-self.V0)**2) - self.g_s*((self.Vs-self.Vs0)**2) - self.g_us*((self.Vus-self.Vus0)**2) ) / self.cap

        self.I_excitatory_values.append(I_excitatory)
        self.I_inhibitory_values.append(I_inhibitory)

        return dV, dVs, dVus, dSe, dSi
    
    def update_inputs(self, I_ext = None, excitatory_Vin = None, inhibitory_Vin = None):
        if I_ext is not None:
            self.I_ext = I_ext
        if excitatory_Vin is not None:
            self.excitatory_Vin = excitatory_Vin
            #  print(self.excitatory_Vin)
        if inhibitory_Vin is not None:
            self.inhibitory_Vin = inhibitory_Vin

    # def update_state(self, dt):
    #     dV, dVs, dVus, dSe, dSi = self.compute_derivatives()
    #     V_new = forward_euler(self.V, dt, dV)
    #     Vs_new = forward_euler(self.Vs, dt, dVs)
    #     Vus_new = forward_euler(self.Vus, dt, dVus)
    #     Se_new = forward_euler(self.Se, dt, dSe)
    #     Si_new = forward_euler(self.Si, dt, dSi)

    #     if V_new >= self.V_threshold:
    #         self.V = self.V_reset
    #         self.Vs = self.Vs_reset
    #         self.Vus += self.delta_Vus
    #         self.Se = np.zeros_like(self.excitatory_Vin)
    #         self.Si = np.zeros_like(self.inhibitory_Vin)
    #     else:
    #         self.V = V_new
    #         self.Vs = Vs_new
    #         self.Vus = Vus_new
    #         self.Se = Se_new
    #         self.Si = Si_new

    #     self.Vvalues.append(self.V)
    #     # self.Vsvalues.append(self.Vs)
    #     # self.Vusvalues.append(self.Vus)
    #     self.Sevalues.append(self.Se)
    #     self.Sivalues.append(self.Si)

    def update_state(self, dt):
        dV, dVs, dVus, dSe, dSi = self.compute_derivatives()
        
        # First calculate value of V_new 
        V_new = forward_euler(self.V, dt, dV)

        if V_new >= self.V_threshold:
            # Set neuron voltage to fixed V_peak
            self.V = self.V_peak

            # # Recompute new derivatives based on V=V_peak
            # dV, dVs, dVus, dSe, dSi = self.compute_derivatives()

            # # Update values of Vs, Vus, Se, Si
            # Vs_new = forward_euler(self.Vs, dt, dVs)
            # Vus_new = forward_euler(self.Vus, dt, dVus)
            # Se_new = forward_euler(self.Se, dt, dSe)
            # Si_new = forward_euler(self.Si, dt, dSi)
            
            # Append peak value V_peak to Vvalues
            self.Vvalues.append(self.V)

            # Reset after spiking action
            self.V = self.V_reset
            self.Vs = self.Vs_reset
            self.Vus += self.delta_Vus
            self.Se = np.zeros_like(self.excitatory_Vin)
            self.Si = np.zeros_like(self.inhibitory_Vin)
        else:
            self.V = V_new

            Vs_new = forward_euler(self.Vs, dt, dVs)
            Vus_new = forward_euler(self.Vus, dt, dVus)
            Se_new = forward_euler(self.Se, dt, dSe)
            Si_new = forward_euler(self.Si, dt, dSi)

            self.Vs = Vs_new
            self.Vus = Vus_new
            self.Se = Se_new
            self.Si = Si_new

            self.Vvalues.append(self.V)

        # self.Vsvalues.append(self.Vs)
        # self.Vusvalues.append(self.Vus)
        self.Sevalues.append(self.Se)
        self.Sivalues.append(self.Si)


def simulate_neuron_excit(current_ext, dt, runtime):
    # runtime = 5.0
    numsteps = int(runtime / dt)
    time = np.arange(0, runtime, dt)

    neuron = SynapticNeuron()
    
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
        # neuron.update_inputs(I_ext=I_ext, excitatory_Vin=excit_ext[i], inhibitory_Vin=inhib_ext[i])
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
    if current_ext[-1] == 0:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/NoI_excit_{decay_time}decay_{runtime}s.png'))
    else:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/I_excit_{decay_time}decay_{runtime}s.png'))
    plt.show()
    # plt.close()
 
    plt.figure(figsize=(12, 6))
    plt.plot(time, neuron.I_excitatory_values, label="I_excitatory")
    plt.title('I_excitatory')
    plt.xlabel('Time (s)')
    plt.ylabel('I_excitatory (mA/nF)')
    plt.grid()
    plt.legend()
    # plt.show()
    if current_ext[-1] == 0:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/NoI_excit_{decay_time}decay_current_{runtime}s.png'))
    else:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/I_inhib_{decay_time}decay_current_{runtime}s.png'))
    plt.close()

    # sigmoid_time = sigmoid(time)
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, sigmoid_time, label="Sigmoid")
    # plt.title('Sigmoid')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Sigmoid')
    # plt.grid()
    # plt.legend()
    # plt.savefig(os.path.join(os.getcwd(), 'NeuronModels/synaptic_neuron_sigmoid.png'))
    # plt.show()

def simulate_neuron_inhib(current_ext, dt, runtime):
    # runtime = 5.0
    numsteps = int(runtime / dt)
    time = np.arange(0, runtime, dt)
    
    neuron = SynapticNeuron()
    
    # Inhibitory input pulse
    amplitude = 100
    peak_time = 0.2
    decay_time = 0.1

    excit_ext = np.full_like(time, neuron.V0)
    # excit_ext = amplitude * np.exp(-((time - peak_time - 0.6*runtime)**2)/(2*decay_time**2))

    inhib_ext = -52 + amplitude * np.exp(-((time - peak_time - 0.3*runtime)**2)/(2*decay_time**2))

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
    if current_ext[-1] == 0:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/NoI_inhib_{decay_time}decay_{runtime}s.png'))
    else:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/I_inhib_{decay_time}decay_{runtime}s.png'))
    plt.show()
    # plt.close()

    # Filter and print only the numerical values > 0 from each array in I_excitatory_values
    # filtered_values = [val for array in neuron.I_excitatory_values for val in np.atleast_1d(array) if val > 0]
    # print(filtered_values)

    plt.figure(figsize=(12, 6))
    plt.plot(time, neuron.I_inhibitory_values, label="I_inhibitory")
    plt.title('I_inhibitory')
    plt.xlabel('Time (s)')
    plt.ylabel('I_inhibitory (mA/nF)')
    plt.grid()
    plt.legend()
    # plt.show()
    if current_ext[-1] == 0:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/NoI_inhib_{decay_time}decay_{runtime}s_current.png'))
    else:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/I_inhib_{decay_time}decay_{runtime}s_current.png'))
    plt.close()

    # sigmoid_time = sigmoid(time)
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, sigmoid_time, label="Sigmoid")
    # plt.title('Sigmoid')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Sigmoid')
    # plt.grid()
    # plt.legend()
    # plt.savefig(os.path.join(os.getcwd(), 'NeuronModels/synaptic_neuron_sigmoid.png'))
    # plt.show()

def simulate_neuron(current_ext, dt, runtime):
    # runtime = 5.0
    numsteps = int(runtime / dt)
    time = np.arange(0, runtime, dt)
    
    neuron = SynapticNeuron()
    
    # Inhibitory input pulse
    amplitude = 100
    peak_time = 0.2
    decay_time = 0.01

    # excit_ext = np.full_like(time, neuron.Ve0)
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

    # plt.figure(figsize=(12, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(time, current_ext, label="Current Input current_ext")
    # plt.title('Current Input')
    # plt.xlabel('Time (s)')
    # plt.ylabel('I_ext (mA/nF)')
    # plt.grid()
    # plt.legend()
    
    plt.figure(figsize=(12, 9))
    plt.subplot(4, 1, 1)
    plt.plot(time, neuron.Sevalues, label="Se")
    plt.title('Se')
    plt.xlabel('Time (s)')


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
    if current_ext[-1] == 0:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/synaptic_neuron_{decay_time}decay_no_current.png'))
    else:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/synaptic_neuron_{decay_time}decay.png'))
    plt.show()
    # plt.close()

    # Filter and print only the numerical values > 0 from each array in I_excitatory_values
    # filtered_values = [val for array in neuron.I_excitatory_values for val in np.atleast_1d(array) if val > 0]
    # print(filtered_values)

    plt.figure(figsize=(12, 6))
    plt.plot(time, neuron.I_excitatory_values, label="I_excitatory", color='red')
    plt.plot(time, neuron.I_inhibitory_values, label="I_inhibitory", color='blue')
    plt.title('I_excitatory & I_inhibitory')
    plt.xlabel('Time (s)')
    plt.ylabel('I (mA/nF)')
    plt.grid()
    plt.legend()
    # plt.show()
    if current_ext[-1] == 0:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/synaptic_neuron_I_{decay_time}decay_no_current.png'))
    else:
        plt.savefig(os.path.join(os.getcwd(), f'NeuronModels/SynapticMQIF_Plots/synaptic_neuron_I_{decay_time}decay.png'))
    plt.close()

    # sigmoid_time = sigmoid(time)
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, sigmoid_time, label="Sigmoid")
    # plt.title('Sigmoid')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Sigmoid')
    # plt.grid()
    # plt.legend()
    # plt.savefig(os.path.join(os.getcwd(), 'NeuronModels/synaptic_neuron_sigmoid.png'))
    # plt.show()


def main():
    dt = 5e-5
    runtime = 15.0
    # runtime = 10.0
    numsteps = int(runtime / dt)
    time = np.arange(0, runtime, dt)
    # Constant current input
    amplitude = 5
    current_ext = np.zeros(numsteps)
    start_time = int(0.5/dt)
    # start_time = numsteps // 10
    current_ext[start_time:] = amplitude
    
    simulate_neuron_excit(current_ext, dt, runtime)
    # simulate_neuron_inhib(current_ext, dt, runtime)
    # simulate_neuron(current_ext, dt, runtime)

if __name__ == "__main__":
    main()