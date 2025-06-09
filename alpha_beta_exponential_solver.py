import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import root

# --- Global Time Array ---
# Defined globally so it doesn't need to be passed around constantly
t_max = 80.0  # µs
num_points = 1000000
t = np.linspace(0, t_max, num_points)
Itarget_peak = 100000.0  # 100 kA (used for calculations, doesn't affect timing ratios)

# --- Functions from your code (with minor improvements) ---
def calculate_current(t_arr, alpha, beta):
    """Calculates the double exponential current. Peak value is normalized later."""
    if beta <= alpha or alpha <= 0:
        return np.zeros_like(t_arr)

    tp_unscaled = (np.log(beta / alpha)) / (beta - alpha)
    Punscaled = np.exp(-alpha * tp_unscaled) - np.exp(-beta * tp_unscaled)
    if Punscaled == 0.0:
        return np.zeros_like(t_arr)
    I0 = 1.0 / Punscaled  # We work with a normalized peak of 1 for simplicity
    return I0 * (np.exp(-alpha * t_arr) - np.exp(-beta * t_arr))

def find_time_at_value(t_arr, current_arr, target_value, rising=True, start_index=0):
    """Finds the time when current crosses target_value using interpolation."""
    diff = current_arr[start_index:] - target_value
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    for idx_in_diff_array in sign_changes:
        actual_idx = start_index + idx_in_diff_array
        # Check for the correct crossing direction
        is_rising_crossing = diff[idx_in_diff_array] < 0 and diff[idx_in_diff_array + 1] >= 0
        is_falling_crossing = diff[idx_in_diff_array] > 0 and diff[idx_in_diff_array + 1] <= 0

        if (rising and is_rising_crossing) or (not rising and is_falling_crossing):
            c1, c2 = current_arr[actual_idx], current_arr[actual_idx + 1]
            t1, t2 = t_arr[actual_idx], t_arr[actual_idx + 1]
            if c2 == c1: return t1
            # Linear interpolation to find the exact time
            return t1 + (t2 - t1) * (target_value - c1) / (c2 - c1)
    return np.nan

# --- Solver Implementation ---

def error_function(params, target_front_time, target_duration):
    """
    Objective function for the solver.
    Calculates the difference between the actual and target timings.
    """
    alpha, beta = params
    
    # Constraints check
    if beta <= alpha or alpha <= 0:
        return [1e6, 1e6] # Return a large error for invalid parameters

    # Calculate waveform (normalized to peak of 1)
    current = calculate_current(t, alpha, beta)
    
    # In case of invalid parameters leading to zero current
    if np.all(current == 0):
        return [1e6, 1e6]
        
    idx_peak = np.argmax(current)

    # Find t10, t90, and t50
    t10 = find_time_at_value(t, current, 0.1, rising=True)
    t90 = find_time_at_value(t, current, 0.9, rising=True, start_index=np.where(t>=t10)[0][0] if not np.isnan(t10) else 0)
    t50_decay = find_time_at_value(t, current, 0.5, rising=False, start_index=idx_peak)

    if any(np.isnan([t10, t90, t50_decay])):
        return [1e6, 1e6] # Return large error if timings can't be found

    # Calculate actual front time and duration
    front_time_actual = t90 - t10
    duration_actual = t50_decay - t10

    # Calculate errors
    error1 = front_time_actual - target_front_time
    error2 = duration_actual - target_duration
    
    return [error1, error2]

def solve_for_alpha_beta(target_front_time=8.0, target_duration=20.0):
    """
    Solves for the optimal alpha and beta parameters.
    """
    # Initial guess for alpha and beta
    initial_guess = [0.06, 0.12] 
    
    print("Solving for optimal alpha and beta...")
    
    # Use scipy.optimize.root to find the roots of the error function
    solution = root(error_function, initial_guess, args=(target_front_time, target_duration), method='hybr')
    
    if solution.success:
        optimal_alpha, optimal_beta = solution.x
        print(f"Solver successful!")
        print(f"  Optimal alpha: {optimal_alpha:.6f} µs⁻¹")
        print(f"  Optimal beta:  {optimal_beta:.6f} µs⁻¹")
        return optimal_alpha, optimal_beta
    else:
        print(f"Solver failed to find a solution. Reason: {solution.message}")
        # Fallback to initial parameters if solver fails
        return 0.061596, 0.122203

# --- Main Execution ---

# 1. Find the optimal parameters first
alpha_opt, beta_opt = solve_for_alpha_beta(target_front_time=8.0, target_duration=20.0)

# 2. Use the found parameters to set up the interactive plot

# --- Functions for Plotting (from your original code) ---
def get_display_current(t_arr, alpha, beta, I_target_peak_val):
    if beta <= alpha or alpha <= 0:
        return np.zeros_like(t_arr)
    tp_unscaled = (np.log(beta / alpha)) / (beta - alpha)
    Punscaled = np.exp(-alpha * tp_unscaled) - np.exp(-beta * tp_unscaled)
    if Punscaled == 0.00:
        return np.zeros_like(t_arr)
    I0 = I_target_peak_val / Punscaled
    return I0 * (np.exp(-alpha * t_arr) - np.exp(-beta * t_arr))


# --- Set up the Figure and Axes ---
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

# --- Initial Calculation and Plotting with Optimal Values ---
initial_current = get_display_current(t, alpha_opt, beta_opt, Itarget_peak)
line, = ax.plot(t, initial_current, lw=2)

# --- Plot Formatting ---
ax.set_title('Interactive 8/20 µs Double Exponential Waveform (100kA Peak)')
ax.set_xlabel('Time t (µs)')
ax.set_ylabel('Current I(t) (A)')
ax.grid(True, which="both", ls="--")
ax.set_xlim([0, t_max])
ax.set_ylim([np.min(initial_current) * 1.1, np.max(initial_current) * 1.1])


# --- Add Interactive Sliders ---
ax_alpha = plt.axes([0.15, 0.20, 0.65, 0.03])
ax_beta = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_Itarget = plt.axes([0.15, 0.10, 0.65, 0.03])

slider_alpha = Slider(ax=ax_alpha, label='α (µs⁻¹)', valmin=0.01, valmax=0.5, valinit=alpha_opt, valfmt='%0.6f')
slider_beta = Slider(ax=ax_beta, label='β (µs⁻¹)', valmin=0.02, valmax=2.0, valinit=beta_opt, valfmt='%0.6f')
slider_Itarget = Slider(ax=ax_Itarget, label='Target Peak (kA)', valmin=1.0, valmax=200.0, valinit=Itarget_peak / 1000.0, valfmt='%0.1f kA')


# --- Timing Annotations (placeholders and lines) ---
t10_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
t90_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
tf_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
t_peak_text = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
I_peak_actual_text = ax.text(0.02, 0.75, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
t_half_decay_text = ax.text(0.02, 0.70, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
duration_text = ax.text(0.02, 0.65, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

vline_t10 = ax.axvline(0, color='cyan', linestyle=':', lw=1, label='t₁₀')
vline_t90 = ax.axvline(0, color='magenta', linestyle=':', lw=1, label='t₉₀')
vline_t_peak = ax.axvline(0, color='green', linestyle='--', lw=1, label='Peak')
vline_t_half_decay = ax.axvline(0, color='orange', linestyle=':', lw=1, label='t₅₀')
ax.legend(handles=[vline_t10, vline_t90, vline_t_peak, vline_t_half_decay])

def update_timing_annotations(current_data):
    if np.all(current_data == 0):
        vals = (np.nan,) * 7
    else:
        idx_peak = np.argmax(current_data)
        I_peak_actual = current_data[idx_peak]
        t_peak = t[idx_peak]
        t10 = find_time_at_value(t, current_data, 0.1 * I_peak_actual, rising=True)
        t90 = find_time_at_value(t, current_data, 0.9 * I_peak_actual, rising=True, start_index=np.where(t>=t10)[0][0] if not np.isnan(t10) else 0)
        t_half_decay = find_time_at_value(t, current_data, 0.5 * I_peak_actual, rising=False, start_index=idx_peak)
        
        tf = t90 - t10 if not (np.isnan(t90) or np.isnan(t10)) else np.nan
        duration = t_half_decay - t10 if not (np.isnan(t_half_decay) or np.isnan(t10)) else np.nan
        vals = [t10, t90, tf, t_peak, I_peak_actual, t_half_decay, duration]

    t10_text.set_text(f't₁₀: {vals[0]:.2f} µs' if not np.isnan(vals[0]) else 't₁₀: N/A')
    t90_text.set_text(f't₉₀: {vals[1]:.2f} µs' if not np.isnan(vals[1]) else 't₉₀: N/A')
    tf_text.set_text(f'Front Time: {vals[2]:.2f} µs' if not np.isnan(vals[2]) else 'Front Time: N/A')
    t_peak_text.set_text(f'Time to Peak: {vals[3]:.2f} µs' if not np.isnan(vals[3]) else 't_peak: N/A')
    I_peak_actual_text.set_text(f'Actual Peak: {vals[4]/1000:.2f} kA' if not np.isnan(vals[4]) else 'Peak: N/A')
    t_half_decay_text.set_text(f't₅₀: {vals[5]:.2f} µs' if not np.isnan(vals[5]) else 't₅₀: N/A')
    duration_text.set_text(f'Duration (t₅₀-t₁₀): {vals[6]:.2f} µs' if not np.isnan(vals[6]) else 'Duration: N/A')

    vline_t10.set_xdata([vals[0]]*2 if not np.isnan(vals[0]) else [0,0])
    vline_t90.set_xdata([vals[1]]*2 if not np.isnan(vals[1]) else [0,0])
    vline_t_peak.set_xdata([vals[3]]*2 if not np.isnan(vals[3]) else [0,0])
    vline_t_half_decay.set_xdata([vals[5]]*2 if not np.isnan(vals[5]) else [0,0])

# --- Update Function for Sliders ---
def update(val):
    alpha_val = slider_alpha.val
    beta_val = slider_beta.val
    Itarget_val = slider_Itarget.val * 1000

    current_data = get_display_current(t, alpha_val, beta_val, Itarget_val)
    line.set_ydata(current_data)
    
    # Update y-axis limits
    max_current = np.max(current_data)
    ax.set_ylim([-0.1 * max_current, max_current * 1.1])

    update_timing_annotations(current_data)
    fig.canvas.draw_idle()

# --- Final Setup ---
slider_alpha.on_changed(update)
slider_beta.on_changed(update)
slider_Itarget.on_changed(update)

# Initial call to set annotations
update_timing_annotations(initial_current)

plt.show()