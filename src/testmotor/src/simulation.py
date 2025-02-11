# ---------------------------------------------------------------------------- #
#                                                                              #
# 	Module:       main.py                                                      #
# 	Author:       31194                                                        #
# 	Created:      1/17/2025, 2:11:42 AM                                        #
# 	Description:  V5 project                                                   #
#                                                                              #
# ---------------------------------------------------------------------------- #

# Library imports
# from vex import *
# # Brain should be defined by default
# brain=Brain()

# motor = Motor(Ports.PORT1, GearSetting.RATIO_18_1, False)

# brain.screen.print("Hello V5")

import matplotlib.pyplot as plt
import numpy as np


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
    
    def Vus_dot(self, V, V_us, tau_us):
        return self.k * (V - V_us) / tau_us

    def V_dot(self, V, Vs, Vus, I_ext, # Current values
          V0, Vs0, Vus0, #Equilibrium values
          g_f, g_s, g_us, # Conductances
          C): #Capacitance
        
        return self.k * (I_ext + g_f*((V-V0)**2) - g_s*((Vs-Vs0)**2) - g_us*((Vus-Vus0)**2)) / C

V0 = -52
Vs0 = -50
Vus0 = -52

V_threshold = 20
V_reset = -45
Vs_reset = 7.5
delta_Vus = 1.7
# Thresholds
g_f = 1.0
g_s = 0.5
g_us = 0.015
# Conductances
tau_s = 4.3
tau_us = 278
# Time constants   
C = 0.82
k = 250.0

dt = 1e-4
runtime = 5.0
num_steps = int(runtime/dt)

V_i = V0
Vs_i = Vs0
Vus_i = Vus0

V_values = []
Vs_values = []
Vus_values = []
motor_speed_data = []

for step in range(num_steps):
    t = step * dt
    if t < 1.0:
        I = 0.0
    else:
        I = 5.0
    
    Vs_new = Vs_i + MQIFNeuron(k).Vs_dot(V_i, Vs_i, tau_s) * dt
    Vus_new = Vus_i + MQIFNeuron(k).Vus_dot(V_i, Vus_i, tau_us) * dt
    V_new = V_i + MQIFNeuron(k).V_dot(V_i, Vs_i, Vus_i, I, V0, Vs0, Vus0, g_f, g_s, g_us, C) * dt

    if V_new > V_threshold:
        V_i = V_reset
        Vs_i = Vs_reset
        Vus_i = Vus_new + delta_Vus
    else:
        V_i = V_new
        Vs_i = Vs_new
        Vus_i = Vus_new
    
    # In this example, we map voltage V to motor spin velocity (percent).
    # Feel free to adjust the scaling factor as needed.
    motor_speed_percent = V_i * 0.5  # Scale factor
    # Constrain speed to avoid extremes (optional):
    if motor_speed_percent > 100:
        motor_speed_percent = 100
    elif motor_speed_percent < -100:
        motor_speed_percent = -100

    V_values.append(V_i)
    Vs_values.append(Vs_i)
    Vus_values.append(Vus_i)
    motor_speed_data.append(motor_speed_percent)


# Plot the 4 series in 4 subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axs[0].plot(V_values, label='V')
axs[0].set_title('Membrane Potential')
axs[1].plot(Vs_values, label='Vs')
axs[1].set_title('Slow State Variable')
axs[2].plot(Vus_values, label='Vus')
axs[2].set_title('Ultra-slow State Variable')
axs[3].plot(motor_speed_data, label='Motor Speed')
axs[3].set_title('Motor Speed')
plt.grid()
plt.tight_layout()
plt.show()



    # Command the motor to spin forward/backward according to V
#     motor.spin(DirectionType.FORWARD, motor_speed_percent, VelocityUnits.PERCENT)

#     wait(dt, SECONDS)
#     brain.screen.new_line()
#     brain.screen.print(motor_speed_percent)

# motor.stop()