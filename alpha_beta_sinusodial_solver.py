import numpy as np
from scipy.optimize import minimize
import time

# --- Goal Parameters ---
TARGET_FRONT_TIME_US = 8.0
TARGET_HALF_VALUE_TIME_US = 20.0

# --- User-specified starting point ---
INITIAL_GUESS = [1.4, 20.0]  # [Q, f₀ in kHz]

# --- Global Time Array for Calculation ---
t_max = 50.0
num_points = 1000000
t = np.linspace(0, t_max, num_points)

# --- Core Waveform and Analysis Functions (Unchanged) ---
# These helper functions are already optimized and correct.

def get_waveform_params(Q, f0_kHz):
    """Calculates alpha and omega_d from Q and f0."""
    if Q <= 0.5: return None, None
    f0_Hz = f0_kHz * 1000.0
    omega0_rad_s = 2 * np.pi * f0_Hz
    zeta = 1 / (2 * Q)
    alpha_per_us = zeta * omega0_rad_s / 1e6
    omega_d_rad_us = omega0_rad_s * np.sqrt(1 - zeta**2) / 1e6
    return alpha_per_us, omega_d_rad_us

def calculate_normalized_damped_sine(t_arr, alpha, omega_d):
    """Calculates a damped sine wave with its peak normalized to 1.0."""
    if alpha is None or omega_d is None or omega_d == 0: return np.zeros_like(t_arr)
    t_peak_unscaled = (np.arctan(omega_d / alpha)) / omega_d
    peak_unscaled = np.exp(-alpha * t_peak_unscaled) * np.sin(omega_d * t_peak_unscaled)
    if peak_unscaled == 0: return np.zeros_like(t_arr)
    return (1.0 / peak_unscaled) * np.exp(-alpha * t_arr) * np.sin(omega_d * t_arr)

def find_time_at_value(t_arr, current_arr, target_value, rising=True, start_index=0):
    """Finds the time when current crosses target_value using interpolation."""
    diff = current_arr[start_index:] - target_value
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    for idx_in_diff_array in sign_changes:
        actual_idx = start_index + idx_in_diff_array
        is_rising = diff[idx_in_diff_array] < 0 and diff[idx_in_diff_array + 1] >= 0
        is_falling = diff[idx_in_diff_array] > 0 and diff[idx_in_diff_array + 1] <= 0
        if (rising and is_rising) or (not rising and is_falling):
            c1, c2 = current_arr[actual_idx], current_arr[actual_idx + 1]
            t1, t2 = t_arr[actual_idx], t_arr[actual_idx + 1]
            if c2 == c1: return t1
            return t1 + (t2 - t1) * (target_value - c1) / (c2 - c1)
    return np.nan

# --- The Core of the New Approach: The Cost Function ---

def cost_function(params):
    """
    Calculates the sum of squared errors for a given [Q, f0] pair.
    The optimizer's goal is to drive the return value of this function to zero.
    """
    Q, f0_kHz = params
    
    alpha, omega_d = get_waveform_params(Q, f0_kHz)
    current = calculate_normalized_damped_sine(t, alpha, omega_d)
    
    if np.all(current == 0): return 1e12 # Return a huge cost for invalid waveforms

    idx_peak = np.argmax(current)
    t10 = find_time_at_value(t, current, 0.1, rising=True)
    t90 = find_time_at_value(t, current, 0.9, rising=True)
    t50_decay = find_time_at_value(t, current, 0.5, rising=False, start_index=idx_peak)

    if any(np.isnan([t10, t90, t50_decay])): return 1e12

    front_time = t90 - t10
    half_value_time = t50_decay
    
    # Calculate the squared errors
    error_front = (front_time - TARGET_FRONT_TIME_US)**2
    error_half = (half_value_time - TARGET_HALF_VALUE_TIME_US)**2
    
    return error_front + error_half

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting High-Precision Optimizer using SciPy ---")
    print(f"Initial Guess: Q = {INITIAL_GUESS[0]}, f₀ = {INITIAL_GUESS[1]} kHz")

    # Define bounds for the parameters to guide the optimizer
    # Q must be > 0.5 for an under-damped system. f0 must be positive.
    bounds = [(0.501, None), (1.0, None)] 

    start_time = time.time()

    # Call the optimizer
    result = minimize(
        fun=cost_function,          # The function to minimize
        x0=INITIAL_GUESS,           # The starting point
        method='L-BFGS-B',          # A powerful and efficient algorithm
        bounds=bounds,              # Parameter constraints
        options={
            'disp': True,           # Display convergence messages
            'ftol': 1e-15,          # Extremely tight function tolerance
            'gtol': 1e-10           # Extremely tight gradient tolerance
        }
    )
    
    end_time = time.time()
    print(f"\nOptimizer finished in {end_time - start_time:.4f} seconds.")
    print("-" * 50)

    if result.success:
        final_Q, final_f0 = result.x
        final_cost = result.fun

        print("✅ Optimization Successful!")
        print(f"   Final Cost (Sum of Squared Errors): {final_cost:e}")
        print("\n   Optimal Parameters found:")
        print(f"     Quality Factor (Q) = {final_Q:.6f}")
        print(f"     Frequency (f₀)     = {final_f0:.6f} kHz")

        # Final verification with the found parameters
        print("\n--- Verifying Timings with Optimal Parameters ---")
        alpha_opt, omega_d_opt = get_waveform_params(final_Q, final_f0)
        final_current = calculate_normalized_damped_sine(t, alpha_opt, omega_d_opt)
        idx_peak_final = np.argmax(final_current)
        t10_final = find_time_at_value(t, final_current, 0.1)
        t90_final = find_time_at_value(t, final_current, 0.9)
        t50_decay_final = find_time_at_value(t, final_current, 0.5, rising=False, start_index=idx_peak_final)
        
        front_time_final = t90_final - t10_final
        half_value_time_final = t50_decay_final

        print(f"   => Final Front Time:     {front_time_final:.6f} µs (Target: {TARGET_FRONT_TIME_US})")
        print(f"   => Final Half-Value Time:  {half_value_time_final:.6f} µs (Target: {TARGET_HALF_VALUE_TIME_US})")
    else:
        print(f"❌ Optimizer failed to converge.")
        print(f"   Message: {result.message}")