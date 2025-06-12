import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Component Values ---
R_source = 50.0     # Fixed 50 Ohm internal resistance of the generator
Ra_init = 50.0      # Ohms
C1_init = 2.2e-6    # Farads (2.2 µF)
L_init = 470e-6     # Henrys (470 µH)
C2_init = 4.7e-6    # Farads (4.7 µF)

# --- Set up the Figure and Axes ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.4)

# --- Frequency Range ---
frequencies = np.logspace(0, 7, 10000)
omega = 2 * np.pi * frequencies

# --- Transfer Function Calculation ---
def calculate_H(omega_range, R_source, Ra, C1, L, C2):
    """
    Calculates the transfer function H(jω) = V_out / V_source.
    
    The circuit model is:
    V_source -> R_source -> Node1 -> L -> Node2 (V_out)
                             |               |
                           Z_shunt1          Z_shunt2
                             |               |
                            GND             GND
    
    where Z_shunt1 = Ra || C1 and Z_shunt2 = C2.
    """
    s = 1j * omega_range
    
    # Define impedances of individual components
    Z_L = s * L
    
    # Add a small epsilon to prevent division by zero at DC if C is zero
    epsilon = 1e-12 
    
    # Impedance of the second shunt element (C2)
    Z_shunt2 = 1 / (s * C2 + epsilon)
    
    # Impedance of the first shunt element (Ra || C1)
    Z_C1 = 1 / (s * C1 + epsilon)
    Z_shunt1 = (Ra * Z_C1) / (Ra + Z_C1 + epsilon)
    
    # Using the transfer function derived from nodal analysis:
    # H = (1/R_source) / [ (1+Z_L/Z_shunt2)*(1/R_source + 1/Z_shunt1) + 1/Z_shunt2 ]
    term1 = 1 + Z_L / Z_shunt2
    term2 = (1 / R_source) + (1 / Z_shunt1)
    term3 = 1 / Z_shunt2
    
    denominator = term1 * term2 + term3
    H = (1 / R_source) / (denominator + epsilon)
    
    return H

# --- Initial Calculation and Plotting ---
initial_H = calculate_H(omega, R_source, Ra_init, C1_init, L_init, C2_init)
initial_magnitude_db = 20 * np.log10(np.maximum(np.abs(initial_H), 1e-12))

# Plot the initial response curve
line, = ax.plot(frequencies, initial_magnitude_db, linewidth=2, color='blue')

# --- Plot Formatting ---
ax.set_xscale('log')
ax.set_title('Frequency Response of asymetric Pi  with 50 Ohm R_source')
ax.set_xlabel('Frequency (f) [Hz]')
ax.set_ylabel('Magnitude |H(jω)| [dB]')
ax.grid(True, which="both", ls="--")
ax.set_xlim([frequencies[0], frequencies[-1]])
ax.set_ylim([-120, 10])

# --- Initial Annotations ---
idx_3db_init = np.argmin(np.abs(initial_magnitude_db - (-3)))
freq_at_3db_init = frequencies[idx_3db_init]
idx_50db_init = np.argmin(np.abs(initial_magnitude_db - (-50)))
freq_at_50db_init = frequencies[idx_50db_init]

line_3db_v = ax.axvline(x=freq_at_3db_init, color='r', linestyle='--')
line_3db_h = ax.axhline(y=-3, color='r', linestyle='--')
line_50db_v = ax.axvline(x=freq_at_50db_init, color='purple', linestyle=':')
line_50db_h = ax.axhline(y=-50, color='purple', linestyle=':')

legend = ax.legend(
    [line, line_3db_v, line_50db_v],
    [
        'Frequency Response',
        f'at -3dB: {freq_at_3db_init:,.2f} Hz',
        f'at -50dB: {freq_at_50db_init:,.2f} Hz'
    ],
    loc='lower left'
)

# --- Add Interactive Sliders ---
ax_Ra = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_C1 = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_L  = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_C2 = plt.axes([0.25, 0.10, 0.65, 0.03])

slider_Ra = Slider(ax=ax_Ra, label='Ra (Ω)', valmin=1, valmax=1000, valinit=Ra_init, valstep=1)
slider_C1 = Slider(ax=ax_C1, label='C1 (µF)', valmin=0.1, valmax=10, valinit=C1_init * 1e6)
slider_L  = Slider(ax=ax_L,  label='L (µH)', valmin=1, valmax=1000, valinit=L_init * 1e6)
slider_C2 = Slider(ax=ax_C2, label='C2 (µF)', valmin=0.1, valmax=10, valinit=C2_init * 1e6)

# --- Update Function ---
def update(val):
    # Get current values from the sliders
    Ra = slider_Ra.val
    C1 = slider_C1.val * 1e-6  # µF to F
    L  = slider_L.val * 1e-6   # µH to H
    C2 = slider_C2.val * 1e-6  # µF to F

    # Recalculate the frequency response with the fixed R_source
    new_H = calculate_H(omega, R_source, Ra, C1, L, C2)
    new_magnitude_db = 20 * np.log10(np.maximum(np.abs(new_H), 1e-12))
    line.set_ydata(new_magnitude_db)

    # Recalculate and update annotation points
    idx_3db = np.argmin(np.abs(new_magnitude_db - (-3)))
    freq_at_3db = frequencies[idx_3db]
    idx_50db = np.argmin(np.abs(new_magnitude_db - (-50)))
    freq_at_50db = frequencies[idx_50db]
    
    line_3db_v.set_xdata([freq_at_3db, freq_at_3db])
    line_50db_v.set_xdata([freq_at_50db, freq_at_50db])
    
    # Update legend text
    legend.get_texts()[1].set_text(f'at -3dB: {freq_at_3db:,.2f} Hz')
    legend.get_texts()[2].set_text(f'at -50dB: {freq_at_50db:,.2f} Hz')
    
    fig.canvas.draw_idle()

# Register the update function with each slider
slider_Ra.on_changed(update)
slider_C1.on_changed(update)
slider_L.on_changed(update)
slider_C2.on_changed(update)

plt.show()