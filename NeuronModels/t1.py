import numpy as np
import matplotlib.pyplot as plt

# Example interburst frequency calculation:
def calculate_interburst_frequency(interburst_interval):
    if len(interburst_interval) == 0:
        return np.nan  # Use NaN when there's no burst
    else:
        # Example: frequency = 1/mean interval
        return 1.0 / np.mean(interburst_interval)

# Suppose you have a meshgrid of param1 and param2, and a corresponding 2D array for interburst frequency:
param1_values = np.linspace(0, 1, 50)
param2_values = np.linspace(0, 1, 50)
interburst_freq_grid = np.random.rand(50, 50)  # replace with your computed values

# For demonstration, set some random entries to NaN to mimic tonic spiking responses.
mask = np.random.rand(50, 50) < 0.3  # 30% of the points as tonic spiking
interburst_freq_grid[mask] = np.nan

# Plotting with a contour plot, for example:
plt.figure(figsize=(8, 6))
cp = plt.contourf(param1_values, param2_values, interburst_freq_grid, cmap="viridis")
plt.colorbar(cp, label="Interburst Frequency")
plt.xlabel("Param 1")
plt.ylabel("Param 2")
plt.title("Interburst Frequency Map (NaN indicates tonic spiking)")
plt.show()
