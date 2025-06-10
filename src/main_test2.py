import numpy as np
import matplotlib.pyplot as plt

# This is assumed to be an external file providing the current waveform
# from damped_sine import generate_damped_sine_data

# --- Dummy function to replace external dependency for standalone execution ---
def generate_damped_sine_data(Q_param, f0_kHz_param, Itarget_peak_param, t_max_sim, num_points_sim):
    """Generates a sample damped sine wave for demonstration."""
    f0_Hz = f0_kHz_param * 1000
    omega0 = 2 * np.pi * f0_Hz
    alpha = omega0 / (2 * Q_param)
    t = np.linspace(0, t_max_sim * 1e-6, num_points_sim)
    raw_current = Itarget_peak_param * np.exp(-alpha * t) * np.sin(omega0 * t)
    return t * 1e6, raw_current # Return time in µs, current in A

# --- Simulation Parameters ---
Q_sim = 1.448
f0_sim_kHz = 20.03
Itarget_peak_sim = 100000.0
t_sim = 80.0
sim_num_points = 1000000

# --- Circuit Parameters ---
R_parallel = 20000.0  # High-value parallel resistor (Ohms).
C_KEMET = 2e-6        # Capacitor value (Farads).
# --- MOV Parameters ---
V_clamp_V = 2000.0    # MOV clamping voltage in Volts (2kV).
R_mov_clamping = 0.002 # MOV on-state resistance in Ohms (2 mΩ).

def run_capacitor_simulation():
    # 1. Generate the source current waveform
    time_array_us, current_array_A = generate_damped_sine_data(
        Q_param=Q_sim,
        f0_kHz_param=f0_sim_kHz,
        Itarget_peak_param=Itarget_peak_sim,
        t_max_sim=t_sim,
        num_points_sim=sim_num_points
    )
    
    time_array_s = time_array_us * 1e-6
    
    # --- Corrected Simulation Loop based on Current Source Model ---
    # Initialize arrays to store the results
    voltage_capacitor = np.zeros_like(current_array_A)
    current_capacitor = np.zeros_like(current_array_A)
    current_mov = np.zeros_like(current_array_A)
    
    if len(current_array_A) > 1:
        dt = time_array_s[1] - time_array_s[0]

        # Loop through each time step to solve the differential equation numerically
        for n in range(len(current_array_A) - 1):
            # Voltage on the capacitor from the previous step
            V_c_n = voltage_capacitor[n]
            
            # --- This is the core of the corrected physics ---
            # Model: I_source -> (C || R_parallel || R_mov)
            
            # Determine the current through the MOV based on the voltage
            I_mov_n = 0.0
            if V_c_n >= V_clamp_V:
                # Once voltage hits the clamp level, MOV turns on with low resistance
                I_mov_n = V_c_n / R_mov_clamping
            
            # Current through the high-value parallel resistor
            I_r_parallel_n = V_c_n / R_parallel

            # The net current available to charge the capacitor is the source current
            # minus the currents leaking through the two parallel resistive paths.
            I_net_for_C = current_array_A[n] - I_r_parallel_n - I_mov_n
            
            # The rate of change of voltage on the capacitor (dV/dt = I_c / C)
            dV_dt = I_net_for_C / C_KEMET
            
            # Calculate the voltage at the next step using Euler method
            V_c_next = V_c_n + dV_dt * dt
            
            voltage_capacitor[n+1] = V_c_next
            
            # Store the currents at this time step for plotting
            current_capacitor[n] = I_net_for_C
            current_mov[n] = I_mov_n

    # Ensure the last values are reasonable for plotting
    current_capacitor[-1] = current_capacitor[-2]
    current_mov[-1] = current_mov[-2]

    # Convert final voltage to kV for plotting and analysis
    voltage_capacitor_kV = voltage_capacitor / 1000.0
    
    # --- Post-simulation analysis and plotting ---
    print("\n--- Simulation Results ---")
    peak_cap_current = np.max(np.abs(current_capacitor))
    peak_mov_current = np.max(np.abs(current_mov))
    print(f"Peak Current into Capacitor: {peak_cap_current:.2f} A")
    print(f"Peak Current through MOV: {peak_mov_current:.2f} A")
    
    # Calculate dV/dt in V/μs for validation
    dV = np.diff(voltage_capacitor)
    dt_us = np.diff(time_array_us)
    dV_dt_Vus = dV / dt_us if len(dt_us) > 0 and not np.all(dt_us==0) else np.array([0])
    
    max_dv_dt = np.max(np.abs(dV_dt_Vus)) if len(dV_dt_Vus) > 0 else 0
    print(f"Max dV/dt: {max_dv_dt:.2f} V/μs")
    
    # Find and report violations
    violations = np.where(np.abs(dV_dt_Vus) > 150)[0]
    if len(violations) > 0:
        print(f"WARNING: {len(violations)} voltage rise rate violations detected!")
    else:
        print("SUCCESS: All voltage rises are within the 150V/μs limit.")
    
    # --- Plotting Results ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Currents
    ax1.plot(time_array_us, current_array_A, label='Source Current', color='blue', alpha=0.5)
    ax1.plot(time_array_us, current_capacitor, label=f'Capacitor Current (Peak: {peak_cap_current:.2f} A)', color='red', linestyle='--')
    ax1.plot(time_array_us, current_mov, label=f'MOV Current (Peak: {peak_mov_current/1000:.2f} kA)', color='orange')
    ax1.set_ylabel('Current (A)')
    ax1.set_title('Current Source Simulation with MOV Clamping')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Voltage and dV/dt
    ax2.plot(time_array_us, voltage_capacitor_kV, label='Capacitor Voltage', color='green')
    ax2.axhline(V_clamp_V / 1000.0, color='grey', linestyle=':', label=f'MOV Clamp Level ({V_clamp_V/1000.0} kV)')
    ax2.set_xlabel('Time (µs)')
    ax2.set_ylabel('Voltage (kV)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper left')
    
    ax3 = ax2.twinx() # Create a second y-axis for dV/dt
    ax3.plot(time_array_us[:-1], dV_dt_Vus, label='dV/dt', color='purple', alpha=0.6)
    ax3.axhline(150, color='r', linestyle='--', label='150V/μs limit')
    ax3.axhline(-150, color='r', linestyle='--')
    ax3.set_ylabel('dV/dt (V/µs)', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.legend(loc='upper right')
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_capacitor_simulation()
