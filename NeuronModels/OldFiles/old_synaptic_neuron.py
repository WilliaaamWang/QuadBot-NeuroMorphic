import numpy as np
import os
import matplotlib.pyplot as plt

class SynapticNeuron():
    """
    MQIF variables
    ------------------
    V: Membrane potential
    Vs: Slow state variable
    Vus: Ultra-slow state variable
    I_ext: External current
    V0, Vs0, Vus0: Equilibrium values
    g_f, g_s, g_us: Neuron Conductances
    C: Capacitance
    tau_s: Time constant for slow state variable
    tau_us: Time constant for ultra-slow state variable
    V_threshold: Threshold potential
    V_reset: Reset potential
    Vs_reset: Reset potential for slow state variable
    delta_Vus: Change in ultra-slow state variable after spike
    k: Speed scaling factor. Multiplying the rate by k will make the simulation k times faster.
    
    Synapse variables
    ------------------
    Se: Excitatory synaptic weight
    Si: Inhibitory synaptic weight
    Ve: Excitatory Vin synaptic input(s)
    Vi: Inhibitory Vin synaptic input(s) e.g. [V_i_B, V_i_ext]
    Ve_threshold: Excitatory threshold
    Vi_threshold: Inhibitory threshold
    [Synaptic parameters]
    tau_e: Time constant for excitatory synapse
    tau_i: Time constant for inhibitory synapse
    Ve0: Equilibrium Excitatory voltage
    Vi0: Equilibrium Inhibitory voltage
    g_syn_e: Excitatory synapse conductance
    g_syn_i: Inhibitory synapse conductance
    Ie: Excitatory synaptic current
    Ii: Inhibitory synaptic current
    """

    def __init__(self, k=1.0):
        self.name = "Neuron"
        self.k = k

    def sigmoid(self, x):
        # result = np.where(
        #     x >= 0,
        #     1 / (1 + np.exp(-x)), # For x >= 0
        #     np.exp(x) / (1 + np.exp(x)) # For x < 0
        # )
        result = 1 / (1 + np.exp(-x))
        return result

    def Vs_dot(self, V, Vs, tau_s):
        return self.k * (V - Vs) / tau_s
    
    def Vus_dot(self, V, Vus, tau_us):
        return self.k * (V - Vus) / tau_us

    #! For half-centre synapse there are 2 inhibitory inputs for each neuron, make it an array of array
    def Se_dot(self, Se, Ve, Ve_threshold, tau_e) -> np.ndarray:
        # Non-linear function to control synaptic gating
        sigmoid_Ve = self.sigmoid((Ve - Ve_threshold)*40)
        return self.k * (sigmoid_Ve - Se) / tau_e
    
    def Si_dot(self, Si, Vi, Vi_threshold, tau_i) -> np.ndarray:
        # Non-linear function to control synaptic gating
        sigmoid_Vi = self.sigmoid((Vi - Vi_threshold)* 40)
        return self.k * (sigmoid_Vi - Si) / tau_i
    
    
    def V_dot(self, V, Vs, Vus, I_ext, # Current values
          V0, Vs0, Vus0, #Equilibrium values
          g_f, g_s, g_us, # Conductances
          C, #Capacitance
          Se, Si, # Synaptic weights
          Ve0, Vi0, # Equilibrium synaptic values
          g_syn_e, g_syn_i, # Synaptic conductances
          ):
        
        Ie = g_syn_e * Se * (Ve0 - V)
        Ii = g_syn_i * Si * (Vi0 - V)
        
        #! TODO
        return self.k * (I_ext + np.sum(Ie) + np.sum(Ii) + g_f*((V-V0)**2) - g_s*((Vs-Vs0)**2) - g_us*((Vus-Vus0)**2) ) / C

# (1) Forward Euler integration
def forward_euler(t, y, dt, dydt):
    t_ret = t + dt
    y_ret = y + dydt(t, y)*dt
    return t_ret, y_ret

# (2) Exponential Euler Integration FOR GATING VARIABLES
def exponent_euler(z, dt, z_inf, tau_z):
    """
    tau_z * dz/dt = z_inf(V) - z
    Assuming dt sufficiently small s.t. membrane potential
    V is constant over the interval
    => tau_z(V) and z_inf(V) can be treated as const
    => z(t+dt) = z_inf + (z(t) - z_inf) * exp(-dt/tau_z)
    z: any gating variable in conductance-based model
    """
    z_ret = z_inf + (z - z_inf) * np.exp(-dt/tau_z)
    return z_ret

# Runge-Kutta integration
def runge_kutta(t, y, dt, dydt):
    k1 = dt*dydt(t, y)
    k2 = dt*dydt(t + 0.5*dt, y + 0.5*k1)
    k3 = dt*dydt(t + 0.5*dt, y + 0.5*k2)
    k4 = dt*dydt(t + dt, y + k3)

    t_ret = t + dt
    y_ret = y + (k1 + 2*k2 + 2*k3 + k4)/6
    return t_ret, y_ret

# Simulation for a synaptic neuron
def simulate_euler(num_steps, dt, 
                  V0, Vs0, Vus0, I_ext, # State variables
                  Ve, Vi, # Synaptic inputs VECTORS excit/inhib Vin
                  Ve_threshold, Vi_threshold, # Synaptic thresholds
                  V_threshold = 20, V_reset = -45, Vs_reset = 7.5, delta_Vus = 1.7, # Thresholds
                  g_f = 1.0, g_s = 0.5, g_us = 0.015, # Conductances
                  tau_s = 4.3, tau_us = 278, # Time constants
                  C = 0.82,
                  Ve0 = 0, Vi0 = -90, # Equilibrium synaptic values
                  g_syn_e = 45, g_syn_i = 45, # Synaptic conductances
                  tau_e = 1, tau_i = 1, # Synaptic time constants
                  k = 250.0):
    """
        Function that generate arrays holding the values of each variable as a time-series.
    Integrates the ODEs and fire spikes.
    
    Parameters
    ----------
    num_steps, dt : INTEGER
        Governs the timebase of the simulation. num_steps denotes the total 
        number of updates steps for the whole simulation. dt is the time spacing 
        between these steps.
        
    V0, Vs0, Vus0 : INTEGER/FLOAT
        The initial/equilibrium values in the ODEs. This is passed into the 
        'v_dot' ODE function. V governs membrane potential. Vs governs the slow
        currents within the ODEs. Vus governs the ultraslow currents.
        
    I_ext : ARRAY
        Input current waveform.
        
    V_threshold, V_reset, VS_reset, delta_Vus : INTEGER/FLOAT
        Paramters governing the reset characteristics. When membrane potential 
        reaches V_threshold, a spike occurs, and the state variables are reset 
        using these paramters.
        
    g_f, g_s, g_us : FLOAT
        Paramters governing the associated conductances of each of the potential 
        differences that arise within the 'v_dot' ODE function.
        
    tau_s, tau_us : INTEGER/FLOAT
        Time constants governing the slow and ultraslow charactersitics, as set
        in the Vs and Vus ODEs.
    
    C : FLOAT
        Capactiance used in the V ODE.

    Ve0, Vi0 : FLOAT
        Equilibrium values for the excitatory and inhibitory synapses.
    
    g_syn_e, g_syn_i : FLOAT
        Conductances for the excitatory and inhibitory synapses.
    
    tau_e, tau_i : FLOAT
        Time constants for the excitatory and inhibitory synapses.

    Returns
    -------
    V_values, Vs_values, Vus_values : ARRAY
        Timeseries containing the values of each of the state variables at each
        time-step. Each element is separated by 'dt' milliseconds of time.

    Se_values, Si_values : ARRAY(each element [excit/inhib A, excit/inhib B, ...])
        Timeseries containing the values of each of the synaptic weights at each
        time-step. Each element is separated by 'dt' milliseconds of time.
        
    spikes : ARRAY
        Array of integers used to relay the number of spikes within a burst. 
        When a burst begins, the first spike is logged as '1', in an array element
        whose index corresponds to the time step at which it occured. The next
        spike is logged as '2', then '3', and so on until the burst ends and 
        the counter is reset. 
    
    time : ARRAY
        Array to keep track of the time in seconds. Useful for plotting.

    """
    # Time axis
    time = np.arange(0, num_steps*dt, dt)

    # Initialise arrays that store spike information
    spikes = np.zeros(num_steps)
    timer = 0
    num_in_burst = 0

    # Initialise arrays that store state variables
    V_values = np.zeros(num_steps)
    V_values[0] = V0

    Vs_values = np.zeros(num_steps)
    Vs_values[0] = Vs0

    Vus_values = np.zeros(num_steps)
    Vus_values[0] = Vus0

    #! TODO
    Se_values = np.zeros((num_steps, len(Ve))) # Excitatory input can be vector
    Si_values = np.zeros((num_steps, len(Vi))) # Inhibitory input can be vector

    # Initialise delay buffers for excitatory and inhibitory V_ins
    Ve_delayed = np.zeros(len(Ve))
    Vi_delayed = np.zeros(len(Vi))

    # Initialise instance of neuron
    neuron = SynapticNeuron(k)


    #! TODO: be mindful of delaying the excit/inhib inputs by 1 timestep
    # Perform forward Euler integration
    for i in range(0, num_steps-1):
        V_i, Vs_i, Vus_i = V_values[i], Vs_values[i], Vus_values[i]
        I_i = I_ext[i]
        Se_i, Si_i = Se_values[i], Si_values[i]

        # Update next step
        # Vs_new = Vs_i + neuron.Vs_dot(V_i, Vs_i, tau_s) * dt
        # Vus_new = Vus_i + neuron.Vus_dot(V_i, Vus_i, tau_us) * dt
        # Se_new = Se_i + neuron.Se_dot(Se_i, Ve_delayed, Ve_threshold, tau_e) * dt
        # Se_new = Se_i + neuron.Se_dot(Se_i, Ve, Ve_threshold, tau_e) * dt
        # Si_new = Si_i + neuron.Si_dot(Si_i, Vi_delayed, Vi_threshold, tau_i) * dt
        # Si_new = Si_i + neuron.Si_dot(Si_i, Vi, Vi_threshold, tau_i) * dt
        # V_new = V_i + neuron.V_dot(V_i, Vs_i, Vus_i, I_i, V0, Vs0, Vus0, g_f, g_s, g_us, C, Se_i, Si_i, Ve0, Vi0, g_syn_e, g_syn_i) * dt
        
        _, Vs_new = forward_euler(0, Vs_i, dt, lambda x, y: neuron.Vs_dot(V_i, Vs_i, tau_s))
        _, Vus_new = forward_euler(0, Vus_i, dt, lambda x, y: neuron.Vus_dot(V_i, Vus_i, tau_us))
        _, Se_new = forward_euler(0, Se_i, dt, lambda x, y: neuron.Se_dot(Se_i, Ve_delayed, Ve_threshold, tau_e))
        _, Si_new = forward_euler(0, Si_i, dt, lambda x, y: neuron.Si_dot(Si_i, Vi_delayed, Vi_threshold, tau_i))
        _, V_new = forward_euler(0, V_i, dt, lambda x, y: neuron.V_dot(V_i, Vs_i, Vus_i, I_i, V0, Vs0, Vus0, g_f, g_s, g_us, C, Se_i, Si_i, Ve0, Vi0, g_syn_e, g_syn_i))
        
        # _, Vs_new = runge_kutta(0, Vs_i, dt, lambda x, y: neuron.Vs_dot(V_i, Vs_i, tau_s))
        # _, Vus_new = runge_kutta(0, Vus_i, dt, lambda x, y: neuron.Vus_dot(V_i, Vus_i, tau_us))
        # _, Se_new = runge_kutta(0, Se_i, dt, lambda x, y: neuron.Se_dot(Se_i, Ve_delayed, Ve_threshold, tau_e))
        # _, Si_new = runge_kutta(0, Si_i, dt, lambda x, y: neuron.Si_dot(Si_i, Vi_delayed, Vi_threshold, tau_i))
        # _, V_new = runge_kutta(0, V_i, dt, lambda x, y: neuron.V_dot(V_i, Vs_i, Vus_i, I_i, V0, Vs0, Vus0, g_f, g_s, g_us, C, Se_i, Si_i, Ve0, Vi0, g_syn_e, g_syn_i))

        # print(Se_new, Si_new)
        # Update arrays
        if V_new > V_threshold:
            # Reset state variables after spike
            V_values[i+1] = V_reset
            Vs_values[i+1] = Vs_reset
            Vus_values[i+1] = Vus_new + delta_Vus
            Se_values[i+1] = np.zeros(len(Ve))
            Si_values[i+1] = np.zeros(len(Vi))

            # Log spike occurrence according to its number in burst
            num_in_burst += 1
            spikes[i] = num_in_burst
            timer = 0
        else:
            V_values[i+1] = V_new
            Vs_values[i+1] = Vs_new
            Vus_values[i+1] = Vus_new
            Se_values[i+1] = Se_new
            Si_values[i+1] = Si_new
            timer += 1

        Ve_delayed = Ve
        Vi_delayed = Vi
        # Check if burst has ended
        if timer > 0.04*num_steps:
            num_in_burst = 0
    
    # Perform Exponential Euler integration
    # for i in range(0, num_steps-1): 
    #     V_i, Vs_i, Vus_i = V_values[i], Vs_values[i], Vus_values[i]
    #     I_i = I_ext[i]
    #     Se_i, Si_i = Se_values[i], Si_values[i]

    #     # Update next step
    #     Vs_new = exponent_euler(Vs_i, dt, Vs0, tau_s)

    #     Vus_new = exponent_euler(Vus_i, dt, Vus0, tau_us)

    #     Se_new = exponent_euler(Se_i, dt, Ve0, tau_e)

    #     Si_new = exponent_euler(Si_i, dt, tau_i, )

    #     V_new = exponent_euler(V_i, dt, ) #! NEED working analytical sol for V

    #     # Update arrays
    #     if V_new > V_threshold:
    #         # Reset state variables after spike
    #         V_values[i+1] = V_reset
    #         Vs_values[i+1] = Vs_reset
    #         Vus_values[i+1] = Vus_new + delta_Vus
    #         Se_values[i+1] = np.zeros(len(Ve))
    #         Si_values[i+1] = np.zeros(len(Vi))

    #         # Log spike occurrence according to its number in burst
    #         num_in_burst += 1
    #         spikes[i] = num_in_burst
    #         timer = 0
    #     else:
    #         V_values[i+1] = V_new
    #         Vs_values[i+1] = Vs_new
    #         Vus_values[i+1] = Vus_new
    #         Se_values[i+1] = Se_new
    #         Si_values[i+1] = Si_new
    #         timer += 1

    #     Ve_delayed = Ve
    #     Vi_delayed = Vi
    #     # Check if burst has ended
    #     if timer > 0.04*num_steps:
    #         num_in_burst = 0

    # print(Se_values)
    # print(Si_values)
    return V_values, Vs_values, Vus_values, Se_values, Si_values, spikes, time, k, C

# Plotter
def plot_MQIF(time, I_ext, V, Vs, Vus, k, C, state_variables=False, phase_portrait=False):
    """
    Function for plotting a single simulation. Feed this function the generated 
    arrays from simulate_euler to view the dynamics.
    
    Parameters
    ----------
    time : ARRAY
        Array to keep track of the time in milliseconds. 
        
    I_ext : ARRAY
        Input current waveform.
        
    V, Vs, Vus : ARRAY
        Contains the values of each state variable at each time-step. V is the 
        most important here - membrane potential. The others are state variables
        used in the ODEs.

    state_variables : BOOLEAN
        Set to true if you also want to plot the state variables, Vs and Vus, 
        with respect to time. 
        
    phase_portrait : BOOLEAN
        Set to true if you also want to plot phase portraits of Vs against V and
        Vus against V.
    """
    plt.style.use("seaborn-darkgrid")
    
    fig1, axes = plt.subplots(3, 1, figsize=(15, 8))

    axes[0].plot(time, I_ext, color="blue")
    axes[0].set_title("External Current")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("I_ext (mA/nF)")

    axes[1].plot(time, V, color="red")
    axes[1].set_title(f"MQIF Membrane Potential, dt={time[1]-time[0]}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("V (mV)")

    for i in range(Se_values.shape[1]):
        axes[2].plot(time, Se_values[:, i], color="orange")
    axes[2].set_title("Se")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Se (mV)")
    # axes[2].plot(time, Si_values, color="purple")

    if state_variables:
        fig2, axes = plt.subplots(1, 2, figsize=(8, 3))

        axes[0].plot(time, Vs, color="green")
        axes[0].set_title("Vs")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Vs (mV)")

        axes[1].plot(time, Vus, color="purple")
        axes[1].set_title("Vus")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Vus (mV)")

        axes[0].set_title("MQIF STATE VARIABLES", loc="left")
        
        # plt.show()
    
    if phase_portrait:
        fig3, axes = plt.subplots(1, 2, figsize=(8, 3))

        axes[0].plot(V, Vs, color="green")
        axes[0].set_title("Vs against V")
        axes[0].set_xlabel("V (mV)")
        axes[0].set_ylabel("Vs (mV)")

        axes[1].plot(V, Vus, color="purple")
        axes[1].set_title("Vus against V")
        axes[1].set_xlabel("V (mV)")
        axes[1].set_ylabel("Vus (mV)")

        axes[0].set_title("MQIF PHASE PORTRAITS", loc="left")
        
    plt.tight_layout()
    plt.show()
    # Ensure the Plots directory exists in the script's directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(script_dir, "Plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    # fig1.savefig(os.path.join(plots_dir, f"SYNAPTIC_k{k}_dt{dt}_duration{runtime}s.png"))


if __name__ == "__main__":
    # Define simulation parameters
    dt = 1e-4
    runtime = 10 # Simulation time in s
    num_steps = int(runtime/dt)

    # Define input current
    amplitude = 5
    I_ext = np.zeros(num_steps)
    # start_index = num_steps // 6
    start_index = num_steps // 10
    # I_ext[start_index:start_index+int(num_steps/2)] = amplitude
    I_ext[start_index:num_steps] = amplitude

    # Neuron parameters
    V_threshold = 20
    V_reset = -45
    Vs_reset = 7.5
    delta_Vus = 1.7
    g_f = 1.0
    g_s = 0.5
    g_us = 0.015
    tau_s = 4.3
    tau_us = 278
    C = 0.82
    k = 250.0
    # Restoration and Regeneration Balance value
    V0 = -52
    Vs0 = -50
    Vus0 = -52

    # Synapse parameters
    # Ve = np.array([-54, -54, -54, -54])
    # Vi = np.array([-54, -54, -54, -54])
    Ve = np.array([-54, -54])
    Vi = np.array([-54, -54])
    Ve_threshold = -40
    Vi_threshold = -40
    Ve0 = 0
    Vi0 = -90
    g_syn_e = 0.5
    g_syn_i = 0.5
    tau_e = 1
    tau_i = 1
    
    # Run simulation
    V_MQIF, Vs, Vus, Se_values, Si_values, spikes, time, k, C = simulate_euler(num_steps, dt, V0, Vs0, Vus0, I_ext, Ve, Vi, Ve_threshold, Vi_threshold, V_threshold, V_reset, Vs_reset, delta_Vus, g_f, g_s, g_us, tau_s, tau_us, C, Ve0, Vi0, g_syn_e, g_syn_i, tau_e, tau_i, k)

    # Plot results
    plot_MQIF(time, I_ext, V_MQIF, Vs, Vus, k, C, state_variables=False, phase_portrait=False)

