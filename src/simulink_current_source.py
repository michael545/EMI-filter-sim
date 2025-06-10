import numpy as np
from damped_sine import generate_damped_sine_data

# Parameters for the damped sine wave
Q_param = 1.448
f0_kHz_param = 20.03
Itarget_peak_param = 100000.0  # 100 kA
t_max_sim_us = 60.0  # Maximum time for the simulation in µs
num_points_sim = 100    # Number of points for the time array

def generate_and_save_arrays():
    time_array_us, current_array_A = generate_damped_sine_data(
        Q_param=Q_param,
        f0_kHz_param=f0_kHz_param,
        Itarget_peak_param=Itarget_peak_param,
        t_max_sim=t_max_sim_us,
        num_points_sim=num_points_sim
    )

    time_array_str = "[" + ", ".join(map(str, time_array_us)) + "]"
    current_array_str = "[" + ", ".join(map(str, current_array_A)) + "]"
    
    output_file_path = "c:\\Users\\micha\\code\\EMI-filter-sim\\signals\\simulink_input_arrays.txt"
    try:
        with open(output_file_path, 'w') as f:
            f.write(time_array_str + "\n")
            f.write(current_array_str + "\n")
        print(f"Successfully saved arrays to {output_file_path}")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    generate_and_save_arrays()
