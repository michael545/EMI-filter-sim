import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Initial IEC Standard Parameters ---
Q_init = 1.448
f0_init = 20.03  # in kHz
Itarget_peak = 100000.0  # 100 kA

# --- Global Time Array ---
t_max = 80# µs (reduced for better view of the 8/20 pulse)
num_points = 10000000
t = np.linspace(0, t_max, num_points)

# --- Core Waveform Calculation Functions ---

def get_waveform_params(Q, f0_kHz):
    """Calculates alpha and omega_d from Q and f0."""
    if Q <= 0.5: # System is critically damped or overdamped, no oscillation
        return None, None
    
    f0_Hz = f0_kHz * 1000.0
    omega0_rad_s = 2 * np.pi * f0_Hz
    
    zeta = 1 / (2 * Q)
    
    # Frequencies need to be in rad/µs for a time array in µs
    alpha_per_us = zeta * omega0_rad_s / 1e6
    omega_d_rad_us = omega0_rad_s * np.sqrt(1 - zeta**2) / 1e6
    
    return alpha_per_us, omega_d_rad_us

def calculate_damped_sine(t_arr, alpha, omega_d, I_target_peak):
    """Calculates the scaled damped sine wave current."""
    if alpha is None or omega_d is None or omega_d == 0:
        return np.zeros_like(t_arr)

    # To scale the peak correctly, we must first find the time of the unscaled peak
    # The peak of e^(-at)sin(bt) occurs when its derivative is 0.
    # d/dt = -a*e^(-at)sin(bt) + b*e^(-at)cos(bt) = 0  => tan(bt) = b/a
    t_peak_unscaled = (np.arctan(omega_d / alpha)) / omega_d
    
    # Now find the value of the unscaled peak
    peak_unscaled = np.exp(-alpha * t_peak_unscaled) * np.sin(omega_d * t_peak_unscaled)
    
    if peak_unscaled == 0:
        return np.zeros_like(t_arr)
        
    # Calculate scaling factor
    I0 = I_target_peak / peak_unscaled
    
    # Return the scaled current waveform
    return I0 * np.exp(-alpha * t_arr) * np.sin(omega_d * t_arr)

def generate_damped_sine_data(Q_param, f0_kHz_param, Itarget_peak_param, t_max_sim, num_points_sim):
    """
    Generates the time array and current array for a damped sine wave.
    
    Args:
        Q_param (float): Quality factor.
        f0_kHz_param (float): Resonant frequency in kHz.
        Itarget_peak_param (float): Target peak current in Amperes.
        t_max_sim (float): Maximum time for the simulation in µs.
        num_points_sim (int): Number of points for the time array.
        
    Returns:
        tuple: (numpy.ndarray, numpy.ndarray)
               t_signal: Time array in µs.
               current_signal: Current array in Amperes.
    """
    t_signal = np.linspace(0, t_max_sim, num_points_sim)
    alpha, omega_d = get_waveform_params(Q_param, f0_kHz_param)
    current_signal = calculate_damped_sine(t_signal, alpha, omega_d, Itarget_peak_param)
    return t_signal, current_signal

def find_time_at_value(t_arr, current_arr, target_value, rising=True, start_index=0):
    """Finds the time when current crosses target_value using interpolation."""
    diff = current_arr[start_index:] - target_value
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    for idx_in_diff_array in sign_changes:
        actual_idx = start_index + idx_in_diff_array
        is_rising_crossing = diff[idx_in_diff_array] < 0 and diff[idx_in_diff_array + 1] >= 0
        is_falling_crossing = diff[idx_in_diff_array] > 0 and diff[idx_in_diff_array + 1] <= 0

        if (rising and is_rising_crossing) or (not rising and is_falling_crossing):
            c1, c2 = current_arr[actual_idx], current_arr[actual_idx + 1]
            t1, t2 = t_arr[actual_idx], t_arr[actual_idx + 1]
            if c2 == c1: return t1
            return t1 + (t2 - t1) * (target_value - c1) / (c2 - c1)
    return np.nan

# --- Initial Plot Setup ---
# fig, ax = plt.subplots(figsize=(12, 8)) # MOVED
# plt.subplots_adjust(left=0.1, bottom=0.35) # MOVED

# alpha_init_val, omega_d_init_val = get_waveform_params(Q_init, f0_init) # MOVED
# initial_current = calculate_damped_sine(t, alpha_init_val, omega_d_init_val, Itarget_peak) # MOVED
# line, = ax.plot(t, initial_current, lw=2, color='red') # MOVED

# ax.set_title('IEC 61000-4-5 Damped Sine Wave (8/20 µs Current Surge)') # MOVED
# ax.set_xlabel('Time t (µs)') # MOVED
# ax.set_ylabel('Current I(t) (A)') # MOVED
# ax.grid(True, which="both", ls="--") # MOVED
# ax.axhline(-33000, color='green', linestyle='--', lw=1.5, label='-33kA Threshold') # MOVED
# ax.set_xlim([0, t_max]) # MOVED
# # Adjust y-limits to ensure the new line and the waveform are visible # MOVED
# initial_max_current = np.max(initial_current) if np.any(initial_current) else Itarget_peak # MOVED
# lower_y_limit = min(-0.3 * Itarget_peak, -33000 - 0.1 * abs(-33000)) # MOVED
# upper_y_limit = 1.1 * initial_max_current # MOVED
# ax.set_ylim([lower_y_limit, upper_y_limit]) # MOVED

# # --- Sliders for Q and f0 --- # MOVED
# ax_Q = plt.axes([0.15, 0.20, 0.65, 0.03]) # MOVED
# ax_f0 = plt.axes([0.15, 0.15, 0.65, 0.03]) # MOVED
# ax_Itarget = plt.axes([0.15, 0.10, 0.65, 0.03]) # MOVED

# slider_Q = Slider(ax=ax_Q, label='Q Factor', valmin=0.5, valmax=15, valinit=Q_init, valfmt='%0.3f') # MOVED
# slider_f0 = Slider(ax=ax_f0, label='f₀ (kHz)', valmin=10.0, valmax=50.0, valinit=f0_init, valfmt='%0.2f kHz') # MOVED
# slider_Itarget = Slider(ax=ax_Itarget, label='Target Peak (kA)', valmin=1.0, valmax=200.0, valinit=Itarget_peak / 1000.0, valfmt='%0.1f kA') # MOVED

# # --- Timing Annotations --- # MOVED
# t10_text = ax.text(0.65, 0.95, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8)) # MOVED
# t90_text = ax.text(0.65, 0.90, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8)) # MOVED
# tf_text = ax.text(0.65, 0.85, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue')) # MOVED
# t_half_decay_text = ax.text(0.65, 0.80, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue')) # MOVED
# I_peak_actual_text = ax.text(0.65, 0.75, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8)) # MOVED

# vline_t10 = ax.axvline(0, color='cyan', linestyle=':', lw=1) # MOVED
# vline_t90 = ax.axvline(0, color='magenta', linestyle=':', lw=1) # MOVED
# vline_t_half_decay = ax.axvline(0, color='orange', linestyle=':', lw=1) # MOVED

# def update_timing_annotations(current_data): # MOVED
#     if np.all(current_data == 0): # MOVED
#         vals = (np.nan,) * 5 # MOVED
#     else: # MOVED
#         idx_peak = np.argmax(current_data) # MOVED
#         I_peak_actual = current_data[idx_peak] # MOVED
#         t10 = find_time_at_value(t, current_data, 0.1 * I_peak_actual, rising=True) # MOVED
#         t90 = find_time_at_value(t, current_data, 0.9 * I_peak_actual, rising=True) # MOVED
#         # Per standard, half-value time is the duration from t=0 to 50% decay # MOVED
#         t_half_decay = find_time_at_value(t, current_data, 0.5 * I_peak_actual, rising=False, start_index=idx_peak) # MOVED
        
#         tf = t90 - t10 if not (np.isnan(t90) or np.isnan(t10)) else np.nan # MOVED
#         vals = [t10, t90, tf, I_peak_actual, t_half_decay] # MOVED

#     t10_text.set_text(f't₁₀: {vals[0]:.2f} µs' if not np.isnan(vals[0]) else 't₁₀: N/A') # MOVED
#     t90_text.set_text(f't₉₀: {vals[1]:.2f} µs' if not np.isnan(vals[1]) else 't₉₀: N/A') # MOVED
#     tf_text.set_text(f'Front Time (t₉₀-t₁₀): {vals[2]:.2f} µs' if not np.isnan(vals[2]) else 'Front Time: N/A') # MOVED
#     I_peak_actual_text.set_text(f'Actual Peak: {vals[3]/1000:.2f} kA' if not np.isnan(vals[3]) else 'Peak: N/A') # MOVED
#     t_half_decay_text.set_text(f'Time to 50% Decay: {vals[4]:.2f} µs' if not np.isnan(vals[4]) else 't₅₀: N/A') # MOVED

#     vline_t10.set_xdata([vals[0]]*2 if not np.isnan(vals[0]) else [0,0]) # MOVED
#     vline_t90.set_xdata([vals[1]]*2 if not np.isnan(vals[1]) else [0,0]) # MOVED
#     vline_t_half_decay.set_xdata([vals[4]]*2 if not np.isnan(vals[4]) else [0,0]) # MOVED

# # --- Update Function for Sliders --- # MOVED
# def update(val): # MOVED
#     Q_val = slider_Q.val # MOVED
#     f0_val = slider_f0.val # MOVED
#     Itarget_val = slider_Itarget.val * 1000 # MOVED

#     alpha, omega_d = get_waveform_params(Q_val, f0_val) # MOVED
#     current_data = calculate_damped_sine(t, alpha, omega_d, Itarget_val) # MOVED
    
#     line.set_ydata(current_data) # MOVED
    
#     max_current_val = np.max(current_data) if np.any(current_data) else Itarget_val # MOVED
#     # Adjust y-limits to ensure the new line and the waveform are visible # MOVED
#     dynamic_lower_y_limit = min(-0.1 * max_current_val, -33000 - 0.1 * abs(-33000)) # MOVED
#     dynamic_upper_y_limit = max_current_val * 1.1 # MOVED
#     ax.set_ylim([dynamic_lower_y_limit, dynamic_upper_y_limit]) # MOVED
    
#     update_timing_annotations(current_data) # MOVED
#     fig.canvas.draw_idle() # MOVED

# # --- Final Setup --- # MOVED
# slider_Q.on_changed(update) # MOVED
# slider_f0.on_changed(update) # MOVED
# slider_Itarget.on_changed(update) # MOVED

# update_timing_annotations(initial_current) # MOVED
# plt.show() # MOVED

if __name__ == "__main__":
    # This 't' is for the interactive plot when run directly.
    # The 'generate_damped_sine_data' function creates its own time array.
    t_interactive = np.linspace(0, t_max, num_points) # Use existing global t_max and num_points

    # --- Initial Plot Setup ---
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.35)

    alpha_init_val, omega_d_init_val = get_waveform_params(Q_init, f0_init)
    initial_current_interactive = calculate_damped_sine(t_interactive, alpha_init_val, omega_d_init_val, Itarget_peak)
    line, = ax.plot(t_interactive, initial_current_interactive, lw=2, color='red')

    ax.set_title('IEC 61000-4-5 Damped Sine Wave (8/20 µs Current Surge) - Interactive')
    ax.set_xlabel('Time t (µs)')
    ax.set_ylabel('Current I(t) (A)')
    ax.grid(True, which="both", ls="--")
    ax.axhline(-33000, color='green', linestyle='--', lw=1.5, label='-33kA Threshold')
    ax.set_xlim([0, t_max])
    initial_max_current = np.max(initial_current_interactive) if np.any(initial_current_interactive) else Itarget_peak
    lower_y_limit = min(-0.3 * Itarget_peak, -33000 - 0.1 * abs(-33000))
    upper_y_limit = 1.1 * initial_max_current
    ax.set_ylim([lower_y_limit, upper_y_limit])
    ax.legend()


    # --- Sliders for Q and f0 ---
    ax_Q = plt.axes([0.15, 0.20, 0.65, 0.03])
    ax_f0 = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_Itarget_slider = plt.axes([0.15, 0.10, 0.65, 0.03]) # Renamed to avoid conflict

    slider_Q = Slider(ax=ax_Q, label='Q Factor', valmin=0.5, valmax=15, valinit=Q_init, valfmt='%0.3f')
    slider_f0 = Slider(ax=ax_f0, label='f₀ (kHz)', valmin=10.0, valmax=50.0, valinit=f0_init, valfmt='%0.2f kHz')
    slider_Itarget = Slider(ax=ax_Itarget_slider, label='Target Peak (kA)', valmin=1.0, valmax=200.0, valinit=Itarget_peak / 1000.0, valfmt='%0.1f kA')

    # --- Timing Annotations (for interactive plot) ---
    # Note: These annotations use the global 't_interactive' for find_time_at_value
    # Ensure 't' used in find_time_at_value within update_timing_annotations_interactive is t_interactive
    t10_text = ax.text(0.65, 0.95, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    t90_text = ax.text(0.65, 0.90, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    tf_text = ax.text(0.65, 0.85, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
    t_half_decay_text = ax.text(0.65, 0.80, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
    I_peak_actual_text = ax.text(0.65, 0.75, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    vline_t10 = ax.axvline(0, color='cyan', linestyle=':', lw=1)
    vline_t90 = ax.axvline(0, color='magenta', linestyle=':', lw=1)
    vline_t_half_decay = ax.axvline(0, color='orange', linestyle=':', lw=1)

    def update_timing_annotations_interactive(current_data_interactive): # Renamed
        if np.all(current_data_interactive == 0):
            vals = (np.nan,) # This will be moved
        else:
            idx_peak = np.argmax(current_data_interactive)
            I_peak_actual = current_data_interactive[idx_peak]
            # Pass t_interactive to find_time_at_value
            t10 = find_time_at_value(t_interactive, current_data_interactive, 0.1 * I_peak_actual, rising=True)
            t90 = find_time_at_value(t_interactive, current_data_interactive, 0.9 * I_peak_actual, rising=True)
            t_half_decay = find_time_at_value(t_interactive, current_data_interactive, 0.5 * I_peak_actual, rising=False, start_index=idx_peak)
            tf = t90 - t10 if not (np.isnan(t90) or np.isnan(t10)) else np.nan
            vals = [t10, t90, tf, I_peak_actual, t_half_decay]

        t10_text.set_text(f't₁₀: {vals[0]:.2f} µs' if not np.isnan(vals[0]) else 't₁₀: N/A')
        t90_text.set_text(f't₉₀: {vals[1]:.2f} µs' if not np.isnan(vals[1]) else 't₉₀: N/A')
        tf_text.set_text(f'Front Time (t₉₀-t₁₀): {vals[2]:.2f} µs' if not np.isnan(vals[2]) else 'Front Time: N/A')
        I_peak_actual_text.set_text(f'Actual Peak: {vals[3]/1000:.2f} kA' if not np.isnan(vals[3]) else 'Peak: N/A')
        t_half_decay_text.set_text(f'Time to 50% Decay: {vals[4]:.2f} µs' if not np.isnan(vals[4]) else 't₅₀: N/A')

        vline_t10.set_xdata([vals[0]]*2 if not np.isnan(vals[0]) else [0,0])
        vline_t90.set_xdata([vals[1]]*2 if not np.isnan(vals[1]) else [0,0])
        vline_t_half_decay.set_xdata([vals[4]]*2 if not np.isnan(vals[4]) else [0,0])

    def update_interactive(val): # Renamed
        Q_val = slider_Q.val
        f0_val = slider_f0.val
        Itarget_val = slider_Itarget.val * 1000

        alpha, omega_d = get_waveform_params(Q_val, f0_val)
        # Use t_interactive for the plot
        current_data_interactive = calculate_damped_sine(t_interactive, alpha, omega_d, Itarget_val)
        
        line.set_ydata(current_data_interactive)
        
        max_current_val = np.max(current_data_interactive) if np.any(current_data_interactive) else Itarget_val
        dynamic_lower_y_limit = min(-0.1 * max_current_val if max_current_val > 0 else -0.1 * Itarget_val, -33000 - 0.1 * abs(-33000))
        dynamic_upper_y_limit = max_current_val * 1.1 if max_current_val > 0 else Itarget_val * 0.1
        if dynamic_lower_y_limit >= dynamic_upper_y_limit: # Ensure sensible limits
             dynamic_upper_y_limit = dynamic_lower_y_limit + Itarget_val * 0.1 # Add a small gap
        ax.set_ylim([dynamic_lower_y_limit, dynamic_upper_y_limit])
        
        update_timing_annotations_interactive(current_data_interactive)
        fig.canvas.draw_idle()

    slider_Q.on_changed(update_interactive)
    slider_f0.on_changed(update_interactive)
    slider_Itarget.on_changed(update_interactive)

    update_timing_annotations_interactive(initial_current_interactive)
    plt.show()