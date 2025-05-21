import numpy as np

def sigmoid(x: np.array) -> np.array:
    result = 1 / (1 + np.exp(-x))
    return result

def forward_euler(y, dt, dydt):
    y_ret = y + dydt*dt
    return y_ret

class SynapticNeuron:
    def __init__(
        self,
        excitatory_Vin: np.ndarray,
        inhibitory_Vin: np.ndarray,
        # Neuron parameters
        cap: float = 0.82,
        k: float = 250.0,
        V0: float = -52.0, 
        Vs0: float = -50.0, 
        Vus0: float = -52.0,
        g_f: float = 1.0,
        g_s: float = 0.5,
        g_us: float = 0.015,
        tau_s: float = 4.3,
        tau_us: float = 278.0,
        V_threshold: float = 20.0, 
        V_peak: float = 20.0,
        V_reset: float = -45,
        Vs_reset: float = 7.5,
        delta_Vus: float = 1.7,
        # Synaptic parameters
        Ve0: float = 0,
        Vi0: float = -90,
        Ve_threshold: float = -20, 
        Vi_threshold: float = -20,
        tau_e: float = 50,
        tau_i: float = 50,
        g_syn_e: float = 0.5,
        g_syn_i: float = 30,
        I_ext=0.0
    ):
        # Neuron parameters
        self.cap = cap
        self.k = k
        self.V0 = V0
        self.Vs0 = Vs0
        self.Vus0 = Vus0
        self.g_f = g_f
        self.g_s = g_s
        self.g_us = g_us
        self.tau_s = tau_s
        self.tau_us = tau_us
        self.V_threshold = V_threshold
        self.V_peak = V_peak
        self.V_reset = V_reset
        self.Vs_reset = Vs_reset
        self.delta_Vus = delta_Vus

        # Synaptic parameters
        self.Ve0 = Ve0
        self.Vi0 = Vi0
        self.Ve_threshold = Ve_threshold
        self.Vi_threshold = Vi_threshold
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.g_syn_e = g_syn_e
        self.g_syn_i = g_syn_i

        self.I_ext = I_ext
        self.excitatory_Vin = np.atleast_1d(excitatory_Vin) if excitatory_Vin is not None else np.array(V0)
        self.inhibitory_Vin = np.atleast_1d(inhibitory_Vin) if inhibitory_Vin is not None else np.array(V0)
        self.Se = np.zeros_like(self.excitatory_Vin)
        self.Si = np.zeros_like(self.inhibitory_Vin)

        self.V = V0
        self.Vs = Vs0
        self.Vus = Vus0

        # Logged data
        self.Vvalues = []
        self.Vsvalues = []
        self.Vusvalues = []
        self.Sevalues = []
        self.Sivalues = []
        self.I_excitatory_values = []
        self.I_inhibitory_values = []

        # self.I_ext = I_ext
        # self.excitatory_Vin = excitatory_Vin if excitatory_Vin is not None else Ve0
        # self.inhibitory_Vin = inhibitory_Vin if inhibitory_Vin is not None else Vi0

    def compute_derivatives(self, log_currents=True):
        """
        Se, Si corresponds to the probability of the synapse being open
        g_synapse = g_syn_e * Se || g_syn_i * Si
        """

        dVs = self.k * (self.V - self.Vs) / self.tau_s
        dVus = self.k * (self.V - self.Vus) / self.tau_us

        sigmoid_Ve = sigmoid(40.0 * (self.excitatory_Vin - self.Ve_threshold))
        sigmoid_Vi = sigmoid(40.0 * (self.inhibitory_Vin - self.Vi_threshold))
        dSe = self.k * (sigmoid_Ve - self.Se) / self.tau_e
        dSi = self.k * (sigmoid_Vi - self.Si) / self.tau_i

        I_excitatory = np.sum(self.g_syn_e * self.Se * (self.V - self.Ve0))
        I_inhibitory = np.sum(self.g_syn_i * self.Si * (self.V - self.Vi0))
        
        dV = (
            self.g_f * (self.V - self.V0)**2
            - self.g_s * (self.Vs - self.Vs0)**2
            - self.g_us * (self.Vus - self.Vus0)**2
            + self.I_ext
            - I_excitatory
            - I_inhibitory
        ) * (self.k / self.cap)

        if log_currents:
            self.I_excitatory_values.append(I_excitatory)
            self.I_inhibitory_values.append(I_inhibitory)

        return dV, dVs, dVus, dSe, dSi
    
    def update_inputs(self, I_ext = None, excitatory_Vin = None, inhibitory_Vin = None):
        if I_ext is not None:
            self.I_ext = I_ext
        if excitatory_Vin is not None:
            self.excitatory_Vin = excitatory_Vin
        if inhibitory_Vin is not None:
            self.inhibitory_Vin = inhibitory_Vin

    def update_state(self, dt):
        dV, dVs, dVus, dSe, dSi = self.compute_derivatives()
        
        # First calculate value of V_new 
        V_new = forward_euler(self.V, dt, dV)

        if V_new >= self.V_threshold:
            # Set neuron voltage to fixed V_peak
            self.V = self.V_peak

            # Recompute new derivatives based on V=V_peak
            dV_spike, dVs_spike, dVus_spike, dSe_spike, dSi_spike = self.compute_derivatives(log_currents=False)

            # # Update values of Vs, Vus, Se, Si
            Vs_new = forward_euler(self.Vs, dt, dVs_spike)
            Vus_new = forward_euler(self.Vus, dt, dVus_spike)
            Se_new = forward_euler(self.Se, dt, dSe_spike)
            Si_new = forward_euler(self.Si, dt, dSi_spike)
            
            # Append peak value V_peak to Vvalues
            self.Vvalues.append(self.V)

            # Reset after spiking action
            self.V = self.V_reset
            self.Vs = self.Vs_reset
            self.Vus += self.delta_Vus
            # self.Se = np.zeros_like(self.excitatory_Vin)
            # self.Si = np.zeros_like(self.inhibitory_Vin)
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


    """
    def Vs_dot(self):
        return self.k * (self.V - self.Vs) / self.tau_s
    
    def Vus_dot(self):
        return self.k * (self.V - self.Vus) / self.tau_us

    #! For half-centre synapse there are 2 inhibitory inputs for each neuron, make it an array of array
    def Se_dot(self, excitatory_Vin) -> np.ndarray:
        # Non-linear function to control synaptic gating
        sigmoid_Ve = self.sigmoid((excitatory_Vin - self.Ve_threshold)*40)
        return self.k * (sigmoid_Ve - self.Se) / self.tau_e
    
    def Si_dot(self, inhibitory_Vin) -> np.ndarray:
        # Non-linear function to control synaptic gating
        sigmoid_Vi = self.sigmoid((inhibitory_Vin - self.Vi_threshold)* 40)
        return self.k * (sigmoid_Vi - self.Si) / self.tau_i
    
    def V_dot(self, I_ext):
        
        Ie = self.g_syn_e * self.Se * (self.Ve0 - self.V)
        Ii = self.g_syn_i * self.Si * (self.Vi0 - self.V)
        
        return self.k * (I_ext + np.sum(Ie) + np.sum(Ii) + self.g_f*((self.V-self.V0)**2) - self.g_s*((self.Vs-self.Vs0)**2) - self.g_us*((self.Vus-self.Vus0)**2)) / self.cap
    
    def update(self, dt, I_ext):
        new_Se = self.Se + dt * self.Se_dot(self.excitatory_Vin)
        new_Si = self.Si + dt * self.Si_dot(self.inhibitory_Vin)
        new_Vs = self.Vs + dt * self.Vs_dot()
        new_Vus = self.Vus + dt * self.Vus_dot()
        new_V = self.V + dt * self.V_dot(I_ext)
        return new_Se, new_Si, new_Vs, new_Vus, new_V
    """
