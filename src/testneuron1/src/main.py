# ---------------------------------------------------------------------------- #
#                                                                              #
# 	Module:       main.py                                                      #
# 	Author:       31194                                                        #
# 	Created:      11/12/2024, 3:22:28 PM                                       #
# 	Description:  V5 project                                                   #
#                                                                              #
# ---------------------------------------------------------------------------- #

# Library imports
from vex import *

# Brain should be defined by default
brain=Brain()

brain.screen.print("Hello V5")

# Motor Declarations
Motor1 = Motor(Ports.PORT1, 1.0, True)
Motor1 = Motor(Ports.PORT10, 1.0, True)

# ---------------------------------------------------------------------------- #

# Define neuron model
#! TODO: Copy the neural model
class Neuron():
    SCALING = 1.0

    def __init__(self):
        self.name = "Neuron"

    def Vs_dot(self, V, Vs, tau_s):
        return self.SCALING * (V - Vs) / tau_s
    
    def Vus_dot(self, V, V_us, tau_us):
        return self.SCALING * (V - V_us) / tau_us

    def V_dot(self, V, Vs, Vus, I_ext, # Current values
          V0, Vs0, Vus0, #Intial values
          g_f, g_s, g_us, # Conductances
          C): #Capacitance
        
        return self.SCALING * (I_ext + g_f*((V-V0)**2) - g_s*((Vs-Vs0)**2) - g_us*((Vus-Vus0)**2)) / C




# Main function
def main():
    # METHOD 1: Rotate motor to ABSOLUTE POS 90 degrees
    brain.screen.new_line()
    brain.screen.print("Rotating to ABS 90 degrees")
    # Motor1.spin_to_position(rotation=90, units=DEGREES, velocity=100, units_v=RPM, wait=True)
    Motor1.spin_to_position(90, DEGREES, 10, PERCENT, True)
    wait(1, SECONDS)

    # METHOD 2: Rotate motor to RELATIVE POS 120 degrees
    brain.screen.new_line()
    brain.screen.print("Rotating by 120 degrees")
    # Motor1.spin_for(direction=FORWARD, amount=120, units=DEGREES, velocity=100, units_v=RPM, wait=True)
    Motor1.spin_for(FORWARD, 120, DEGREES, 10, PERCENT, True)
    wait(1, SECONDS)

    brain.screen.new_line()
    brain.screen.print("Rotating to ABS 180 degrees")
    Motor1.spin_to_position(180, DEGREES, 10, PERCENT, True)
    wait(1, SECONDS)

    # METHOD 3: Set velocity, direction and stop after 2 seconds
    brain.screen.new_line()
    brain.screen.print("Setting velocity to 50%")
    Motor1.set_velocity(50, PERCENT)
    Motor1.spin(direction=FORWARD)
    wait(2, SECONDS)
    Motor1.stop()

if __name__ == "__main__":
    main()