import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Initial IEC Standard Parameters ---
Q_init = 5
f0_init = 20.03  # in kHz
Itarget_peak = 100000.0  # 100 kA

# --- Global Time Array ---
t_max = 40 # µs (reduced for better view of the 8/20 pulse)
num_points = 1000000
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
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

alpha_init_val, omega_d_init_val = get_waveform_params(Q_init, f0_init)
initial_current = calculate_damped_sine(t, alpha_init_val, omega_d_init_val, Itarget_peak)
line, = ax.plot(t, initial_current, lw=2, color='red')

ax.set_title('IEC 61000-4-5 Damped Sine Wave (8/20 µs Current Surge)')
ax.set_xlabel('Time t (µs)')
ax.set_ylabel('Current I(t) (A)')
ax.grid(True, which="both", ls="--")
ax.set_xlim([0, t_max])
ax.set_ylim([-0.15 * Itarget_peak, 1.1 * Itarget_peak]) # ax.set_ylim([-0.15 * max_current, 1.1 * max_current])

# --- Sliders for Q and f0 ---
ax_Q = plt.axes([0.15, 0.20, 0.65, 0.03])
ax_f0 = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_Itarget = plt.axes([0.15, 0.10, 0.65, 0.03])

slider_Q = Slider(ax=ax_Q, label='Q Factor', valmin=0.5, valmax=15, valinit=Q_init, valfmt='%0.3f') #slider_Q = Slider(ax=ax_Q, label='Q Factor', valmin=0.5, valmax=15.0, valinit=Q_init, valfmt='%0.3f')
slider_f0 = Slider(ax=ax_f0, label='f₀ (kHz)', valmin=10.0, valmax=50.0, valinit=f0_init, valfmt='%0.2f kHz')
slider_Itarget = Slider(ax=ax_Itarget, label='Target Peak (kA)', valmin=1.0, valmax=200.0, valinit=Itarget_peak / 1000.0, valfmt='%0.1f kA')

# --- Timing Annotations ---
t10_text = ax.text(0.65, 0.95, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
t90_text = ax.text(0.65, 0.90, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
tf_text = ax.text(0.65, 0.85, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
t_half_decay_text = ax.text(0.65, 0.80, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
I_peak_actual_text = ax.text(0.65, 0.75, '', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

vline_t10 = ax.axvline(0, color='cyan', linestyle=':', lw=1)
vline_t90 = ax.axvline(0, color='magenta', linestyle=':', lw=1)
vline_t_half_decay = ax.axvline(0, color='orange', linestyle=':', lw=1)

def update_timing_annotations(current_data):
    if np.all(current_data == 0):
        vals = (np.nan,) * 5
    else:
        idx_peak = np.argmax(current_data)
        I_peak_actual = current_data[idx_peak]
        t10 = find_time_at_value(t, current_data, 0.1 * I_peak_actual, rising=True)
        t90 = find_time_at_value(t, current_data, 0.9 * I_peak_actual, rising=True)
        # Per standard, half-value time is the duration from t=0 to 50% decay
        t_half_decay = find_time_at_value(t, current_data, 0.5 * I_peak_actual, rising=False, start_index=idx_peak)
        
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

# --- Update Function for Sliders ---
def update(val):
    Q_val = slider_Q.val
    f0_val = slider_f0.val
    Itarget_val = slider_Itarget.val * 1000

    alpha, omega_d = get_waveform_params(Q_val, f0_val)
    current_data = calculate_damped_sine(t, alpha, omega_d, Itarget_val)
    
    line.set_ydata(current_data)
    
    max_current = np.max(current_data) if np.any(current_data) else Itarget_val
    ax.set_ylim([-0.1 * max_current, max_current * 1.1])
    
    update_timing_annotations(current_data)
    fig.canvas.draw_idle()

# --- Final Setup ---
slider_Q.on_changed(update)
slider_f0.on_changed(update)
slider_Itarget.on_changed(update)

update_timing_annotations(initial_current)
plt.show()