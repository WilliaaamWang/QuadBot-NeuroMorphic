import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

# import seaborn as sns
# Set style
# sns.set_theme()

# Define neuron model
class MQIFNeuron():
    """
    All variables:
    V: Membrane potential
    Vs: Slow state variable
    Vus: Ultra-slow state variable
    I_ext: External current
    V0, Vs0, Vus0: Equilibrium values
    g_f, g_s, g_us: Neuron Conductances
    C: Capacitance
    tau_s: Time constant for slow state variable
    tau_us: Time constant for ultra-slow state variable
    k: Speed scaling factor. Multiplying the rate by k will make the simulation k times faster.
    """
    def __init__(self, k=1.0):
        self.name = "Neuron"
        self.k = k

    def Vs_dot(self, V, Vs, tau_s):
        return self.k * (V - Vs) / tau_s
    
    def Vus_dot(self, V, Vus, tau_us):
        return self.k * (V - Vus) / tau_us

    def V_dot(self, V, Vs, Vus, I_ext, # Current values
          V0, Vs0, Vus0, #Equilibrium values
          g_f, g_s, g_us, # Conductances
          C): #Capacitance
        
        return self.k * (I_ext + g_f*((V-V0)**2) - g_s*((Vs-Vs0)**2) - g_us*((Vus-Vus0)**2)) / C


# (1) Forward Euler integration
def forward_euler(t, y, dt, dydt):
    t_ret = t + dt
    y_ret = y + dydt(t, y)*dt
    return t_ret, y_ret

# (2) Backward Euler integration


# (3) Exponential Euler Integration FOR GATING VARIABLES
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

# Forward Euler integration simulation
def simulate_MQIF(num_steps, dt, V0, Vs0, Vus0, I_ext, # State variables
                  V_threshold = 20, V_reset = -45, Vs_reset = 7.5, delta_Vus = 1.7, # Thresholds
                  g_f = 1.0, g_s = 0.5, g_us = 0.015, # Conductances
                  tau_s = 4.3, tau_us = 278, # Time constants   
                  C = 0.82,
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

    Returns
    -------
    V_values, Vs_values, Vus_values : ARRAY
        Timeseries containing the values of each of the state variables at each
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


    # Simulation for a MQIF neuron
    for i in range(0, num_steps-1):
        V_i, Vs_i, Vus_i = V_values[i], Vs_values[i], Vus_values[i]
        I_i = I_ext[i]

        # Update next step -- Forward Euler
        Vs_new = Vs_i + MQIFNeuron(k).Vs_dot(V_i, Vs_i, tau_s) * dt
        Vus_new = Vus_i + MQIFNeuron(k).Vus_dot(V_i, Vus_i, tau_us) * dt
        V_new = V_i + MQIFNeuron(k).V_dot(V_i, Vs_i, Vus_i, I_i, V0, Vs0, Vus0, g_f, g_s, g_us, C) * dt
        
        
        # Update next step -- Exponential Euler

        # Vs_new = V_i + (Vs_i - V_i) * np.exp(-dt/tau_s)
        # Vus_new = V_i + (Vus_i - V_i) * np.exp(-dt/tau_us)
        # b = (I_i - g_s*((Vs_new - Vs0)**2) - g_us*((Vus_new - Vus0)**2)) / g_f
        # if b > 0:
        #     sqrt_b = np.sqrt(b)
        #     arctan_V = np.arctan((V_i - V0)/sqrt_b) + (g_f*sqrt_b/C) * dt
        #     V_new = V0 + sqrt_b*np.tan(arctan_V)
        # elif b < 0:
        #     d = -b
        #     sqrt_d = np.sqrt(d)
        #     Vexp = ((V_i - V0) - sqrt_d) / ((V_i - V0) + sqrt_d) * np.exp(2 * sqrt_d * g_f/C * dt)
        #     V_new = (V0 * (1 + Vexp) + sqrt_d * (Vexp - 1)) / (1 - Vexp)
        # else: # b = 0
        #     V_new = V0 + 1 / (1 / (V_i - V0) - (g_f/C) * dt)
        
        # Evaluate new vars except V

        # Update arrays
        if V_new > V_threshold:
            # Reset state variables after spike
            V_values[i+1] = V_reset
            Vs_values[i+1] = Vs_reset
            Vus_values[i+1] = Vus_new + delta_Vus

            # Log spike occurrence according to its number in burst
            num_in_burst += 1
            spikes[i] = num_in_burst
            timer = 0
        else:
            V_values[i+1] = V_new
            Vs_values[i+1] = Vs_new
            Vus_values[i+1] = Vus_new
            timer += 1

        # Check if burst has ended
        if timer > 0.04*num_steps:
            num_in_burst = 0
    
    return V_values, Vs_values, Vus_values, spikes, time, k, C

# Plotter
def plot_MQIF(time, I_ext, V, Vs, Vus, k, C, state_variables=False, phase_portrait=False):
    """
    Function for plotting a single simulation. Feed this function the generated 
    arrays from simulate_MQIF to view the dynamics.
    
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

    plt.figure(figsize=(8, 3))
    plt.plot(time, I_ext, color="blue")
    plt.title("External Current")
    plt.xlabel("Time (s)")
    plt.ylabel("I_ext (mA)")
    
    # fig1, axes = plt.subplots(4, 1, figsize=(19.2, 8.63))
    # fig1.suptitle(f"MQIF Neuron Simulation, k={k}, C={C}", fontsize=16)

    # axes[0].plot(time, I_ext, color="blue")
    # axes[0].set_title("External Current")
    # # axes[0].set_xlabel("Time (s)")
    # axes[0].set_ylabel("I_ext (mA/nF)")

    # axes[1].plot(time, V, color="red")
    # axes[1].set_title(f"MQIF Membrane Potential, dt={time[1]-time[0]}")
    # # axes[1].set_xlabel("Time (s)")
    # axes[1].set_ylabel("V (mV)")

    # axes[2].plot(time, Vs, color="green")
    # axes[2].set_title("Vs")
    # # axes[2].set_xlabel("Time (s)")
    # axes[2].set_ylabel("Vs (mV)")

    # axes[3].plot(time, Vus, color="purple")
    # axes[3].set_title("Vus")
    # axes[3].set_xlabel("Time (s)")
    # axes[3].set_ylabel("Vus (mV)")

    # if state_variables:
    #     fig2, axes = plt.subplots(1, 2, figsize=(8, 3))

    #     axes[0].plot(time, Vs, color="green")
    #     axes[0].set_title("Vs")
    #     axes[0].set_xlabel("Time (s)")
    #     axes[0].set_ylabel("Vs (mV)")

    #     axes[1].plot(time, Vus, color="purple")
    #     axes[1].set_title("Vus")
    #     axes[1].set_xlabel("Time (s)")
    #     axes[1].set_ylabel("Vus (mV)")

    #     axes[0].set_title("MQIF STATE VARIABLES", loc="left")
        
    #     # plt.show()
    
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "Plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    # fig1.savefig(os.path.join(plots_dir, f"MQIF_k{k}_dt{dt}_duration{runtime}s.png"))


if __name__ == "__main__":
    # Define simulation parameters
    dt = 1e-4
    runtime = 10 # Simulation time in s
    num_steps = int(runtime/dt)

    # Define input current
    amplitude = 5
    I_ext = np.zeros(num_steps)
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
    
    # Run simulation
    V_MQIF, Vs, Vus, spikes, time, k, C = simulate_MQIF(num_steps, dt, V0, Vs0, Vus0, I_ext, V_threshold, V_reset, Vs_reset, delta_Vus, g_f, g_s, g_us, tau_s, tau_us, C, k)

    print(time)
    print(V_MQIF)
    """
    # Compare with Simulink traces
    from scipy.io import loadmat
    
    sim_mqif_V = loadmat("Simulink Neurons/Core Neurons/mqif_V.mat")
    sim_mqif_Vs = loadmat("Simulink Neurons/Core Neurons/mqif_Vs.mat")
    sim_mqif_Vus = loadmat("Simulink Neurons/Core Neurons/mqif_Vus.mat")
    

    sim_V = sim_mqif_V["V"].flatten()
    sim_Vs = sim_mqif_Vs["Vs"].flatten()
    sim_Vus = sim_mqif_Vus["Vus"].flatten()
    sim_time = sim_mqif_V["time"].flatten()


    matlab_V = np.interp(time, sim_time, sim_V)
    matlab_Vs = np.interp(time, sim_time, sim_Vs)
    matlab_Vus = np.interp(time, sim_time, sim_Vus)
    # matlab_V = sim_V[:-1]
    # matlab_Vs = sim_Vs[:-1]
    # matlab_Vus = sim_Vus[:-1]


    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mae_V = mean_absolute_error(V_MQIF, matlab_V)
    rmse_V = np.sqrt(mean_squared_error(V_MQIF, matlab_V))
    max_diff_V = np.max(np.abs(V_MQIF - matlab_V))

    mae_Vs = mean_absolute_error(Vs, matlab_Vs)
    rmse_Vs = np.sqrt(mean_squared_error(Vs, matlab_Vs))
    max_diff_Vs = np.max(np.abs(Vs - matlab_Vs))

    mae_Vus = mean_absolute_error(Vus, matlab_Vus)
    rmse_Vus = np.sqrt(mean_squared_error(Vus, matlab_Vus))
    max_diff_Vus = np.max(np.abs(Vus - matlab_Vus))
    
    print(f"V: MAE={mae_V}, RMSE={rmse_V}, Max Diff={max_diff_V}")
    print(f"Vs: MAE={mae_Vs}, RMSE={rmse_Vs}, Max Diff={max_diff_Vs}")
    print(f"Vus: MAE={mae_Vus}, RMSE={rmse_Vus}, Max Diff={max_diff_Vus}")

    # Plot time series
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(time, matlab_V, label="Simulink", alpha=0.7)
    axes[0].plot(time, V_MQIF, label="Python Model", linestyle='--', alpha=0.7)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("V (mV)")
    axes[0].set_title("Simulink vs Python Model Output - V")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(time, matlab_Vs, label="Simulink", alpha=0.7)
    axes[1].plot(time, Vs, label="Python Model", linestyle='--', alpha=0.7)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Vs (mV)")
    axes[1].set_title("Simulink vs Python Model Output - Vs")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(time, matlab_Vus, label="Simulink", alpha=0.7)
    axes[2].plot(time, Vus, label="Python Model", linestyle='--', alpha=0.7)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Vus (mV)")
    axes[2].set_title("Simulink vs Python Model Output - Vus")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.show()

    # Plot the difference
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(time, V_MQIF - matlab_V, label="Difference")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Difference (V)")
    axes[0].set_title("Difference Between Simulink and Python Outputs - V")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(time, Vs - matlab_Vs, label="Difference")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Difference (Vs)")
    axes[1].set_title("Difference Between Simulink and Python Outputs - Vs")
    axes[1].grid()
    axes[1].legend()

    axes[2].plot(time, Vus - matlab_Vus, label="Difference")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Difference (Vus)")
    axes[2].set_title("Difference Between Simulink and Python Outputs - Vus")
    axes[2].grid()
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    """
    

    # Plot results
    plot_MQIF(time, I_ext, V_MQIF, Vs, Vus, k, C, state_variables=True, phase_portrait=False)
