import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

R_inrush_init = 1000   # 
C_init = 4.7e-6         #

# --- Set up the Figure and Axes ---
#
fig, ax = plt.subplots(figsize=(10, 8))
# Adjust the main plot to make space for the sliders at the bottom
plt.subplots_adjust(left=0.1, bottom=0.3)

# --- Frequency Range ---
# Create a logarithmically spaced frequency vector from 1 Hz to 100 MHz.
frequencies = np.logspace(0, 8, 10000)
omega = 2 * np.pi * frequencies

# --- Initial Calculation and Plotting ---
# Calculate the initial frequency response using the initial values
# For a classical RC low-pass filter, H(jω) = 1 / (1 + jωRC)
# Magnitude |H(jω)| = 1 / sqrt(1 + (ωRC)^2)
# Here, R is R_inrush_init and C is C_init. R_init_init is not used.
initial_magnitude = 1 / np.sqrt(1 + (omega * R_inrush_init * C_init)**2)
initial_magnitude_db = 20 * np.log10(initial_magnitude)

# Plot the initial response curve. We store the line object to update it later.
line, = ax.plot(frequencies, initial_magnitude_db, linewidth=2)

# --- Plot Formatting ---
ax.set_xscale('log')
ax.set_title('Amplitudni odziv vezja z kondenzatorjem')
ax.set_xlabel('f [Hz]')
ax.set_ylabel('Amplituda |H(omega)| [dB]')
ax.grid(True, which="both", ls="--")
ax.set_xlim([frequencies[0], frequencies[-1]])
ax.set_ylim([-80, 5])

# --- Initial Annotations ---
idx_3db_init = np.argmin(np.abs(initial_magnitude_db - (-3)))
freq_at_3db_init = frequencies[idx_3db_init]
idx_50db_init = np.argmin(np.abs(initial_magnitude_db - (-50)))
freq_at_50db_init = frequencies[idx_50db_init]

# premice za -3dB in -50dB
line_3db_v = ax.axvline(x=freq_at_3db_init, color='r', linestyle='--')
line_3db_h = ax.axhline(y=-3, color='r', linestyle='--')
line_50db_v = ax.axvline(x=freq_at_50db_init, color='purple', linestyle=':')
line_50db_h = ax.axhline(y=-50, color='purple', linestyle=':')

# legenda se updata dinamicno.
legend = ax.legend(
    [line, line_3db_v, line_50db_v],
    [
        'Freq. odziv',
        f'pri -3dB: {freq_at_3db_init:,.2f} Hz',
        f'pri -50dB: {freq_at_50db_init:,.2f} Hz'
    ]
)

# --- Add Interactive Sliders ---
# Define axes for the sliders
ax_R_inrush = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_C = plt.axes([0.15, 0.05, 0.65, 0.03]) # Axis for Capacitor slider


# Create a linear slider for R_inrush
slider_R_inrush = Slider(
    ax=ax_R_inrush,
    label='R_inrush (Ω)',
    valmin=0.1,
    valmax=100.0,
    valinit=R_inrush_init,
    valfmt='%1.1f'
)

# Create a linear slider for C
slider_C = Slider(
    ax=ax_C,
    label='C (µF)',
    valmin=1.0,          # 1µF
    valmax=6.0,          # 6µF
    valinit=C_init * 1e6, # Initial value from C_init in µF
    valfmt='%1.1f µF'    # Format for the slider value display
)

# --- Update Function ---
def update(val):
    # Get current values from the sliders
    R_inrush = slider_R_inrush.val
    C = slider_C.val / 1e6 # Get C from slider and convert µF to F

    # Recalculate the frequency response for a classical RC low-pass filter
    # Here, R is R_inrush and C is C from sliders. R_init is not used.
    new_magnitude = 1 / np.sqrt(1 + (omega * R_inrush * C)**2)
    new_magnitude_db = 20 * np.log10(new_magnitude)
    line.set_ydata(new_magnitude_db)

    # Recalculate annotation points
    idx_3db = np.argmin(np.abs(new_magnitude_db - (-3)))
    freq_at_3db = frequencies[idx_3db]
    idx_50db = np.argmin(np.abs(new_magnitude_db - (-50)))
    freq_at_50db = frequencies[idx_50db]

    # Safely update annotation lines
    xmin, xmax = ax.get_xlim()
    if xmin <= freq_at_3db <= xmax:
        line_3db_v.set_xdata([freq_at_3db, freq_at_3db])
        line_3db_v.set_visible(True)
    else:
        line_3db_v.set_visible(False)

    if xmin <= freq_at_50db <= xmax:
        line_50db_v.set_xdata([freq_at_50db, freq_at_50db])
        line_50db_v.set_visible(True)
    else:
        line_50db_v.set_visible(False)

    # Update legend text
    legend.get_texts()[1].set_text(f'pri -3dB: {freq_at_3db:,.2f} Hz')
    legend.get_texts()[2].set_text(f'pri -50dB: {freq_at_50db:,.2f} Hz')

    # Redraw the figure
    fig.canvas.draw_idle()

# Register the update functions
slider_R_inrush.on_changed(update)
slider_C.on_changed(update)

plt.show()
