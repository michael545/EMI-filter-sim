import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import cumulative_trapezoid # Added import

from damped_sine import generate_damped_sine_data

Q_sim = 1.448  # Quality factor from damped_sine.py defaults
f0_sim_kHz = 20.03  # Resonant frequency in kHz from damped_sine.py defaults
Itarget_peak_sim = 100000.0  # Target peak current in Amperes (100 kA)
t_sim = 80.0  # Simulation duration in microseconds
sim_num_points = 10000000    # Number of points for the simulation

# --- Circuit Parameters ---
R_parallel = 20000
C_KEMET = 2e-6  # Example: 10 µF
V_clamp_kV = 2.0  # MOV clamping voltage in kV

def run_capacitor_simulation():
    """
    Runs the sim of applying the damped sine current to a capacitor
    and plots the results.
    """
    # 1. Generate the damped sine current waveform data
    time_array_us, current_array_A = generate_damped_sine_data(
        Q_param=Q_sim,
        f0_kHz_param=f0_sim_kHz,
        Itarget_peak_param=Itarget_peak_sim,
        t_max_sim=t_sim,
        num_points_sim=sim_num_points
    )

    time_array_s = time_array_us * 1e-6
    voltage_across_capacitor = np.zeros_like(current_array_A)

        # Ensure there are points to process
    if len(current_array_A) > 1:
        dt = time_array_s[1] - time_array_s[0]
        V_clamp_V = V_clamp_kV * 1000  # Convert kV to V for consistency

        for n in range(len(current_array_A) - 1):
            V_n = voltage_across_capacitor[n]
            
            # MOV clamping: If voltage >= clamp, halt further rise
            if V_n >= V_clamp_V:
                voltage_across_capacitor[n+1] = V_clamp_V
            else:
                dV_dt = (1 / C_KEMET) * (current_array_A[n] - V_n / R_parallel)
                voltage_across_capacitor[n+1] = V_n + dV_dt * dt
                # Ensure voltage doesn't exceed clamp due to discretization
                if voltage_across_capacitor[n+1] > V_clamp_V:
                    voltage_across_capacitor[n+1] = V_clamp_V

    # Convert to kV for output/plotting
    voltage_across_capacitor_kV = voltage_across_capacitor / 1000.0

    # --- Plotting Results ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:red'
    ax1.set_xlabel(f'Time (µs) - Total duration: {t_sim} µs')
    ax1.set_ylabel('Current (A)', color=color)
    ax1.plot(time_array_us, current_array_A, color=color, label='Input Current')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", ls="--", alpha=0.7)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Voltage (kV)', color=color)  # we already handled the x-label with ax1
    ax2.plot(time_array_us, voltage_across_capacitor_kV, color=color, linestyle='--', label='Capacitor Voltage (kV)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Capacitor Response to Damped Sine Current', fontsize=16)
    # Adding legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
    plt.show()

    print(f"Simulation complete. Max voltage: {np.max(voltage_across_capacitor_kV):.2f} kV")
    print(f"Min voltage: {np.min(voltage_across_capacitor_kV):.2f} kV")
    print(f"Max current: {np.max(current_array_A):.2f} A")
    print(f"Min current: {np.min(current_array_A):.2f} A")

if __name__ == "__main__":
    run_capacitor_simulation()
