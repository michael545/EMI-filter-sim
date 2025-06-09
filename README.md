# EMI Filter Design and Simulation for UL1283 Compliance

This repository contains Python scripts for designing and simulating an Electromagnetic Interference (EMI) filter. The primary goal is to create a filter that meets the requirements of the UL1283 standard and is robust enough to withstand Surge Protective Device (SPD) testing, specifically the 8/20µs 100kA current waveform.

## Project Overview

The main script (`main.py`) utilizes `numpy` for numerical calculations and `matplotlib` for plotting the frequency response of an RC (Resistor-Capacitor) filter. It features interactive sliders that allow  dynamically adjustment of the values of the main resistor (R_init), an inrush current limiting resistor (R_inrush), and the capacitor (C). This helps in visualizing how component value changes affect the filter's performance, particularly its attenuation characteristics.

The simulation calculates and displays:
- The magnitude of the filter's transfer function in decibels (dB).
- Key performance indicators such as the frequencies at which the attenuation reaches -3dB and -50dB.

## Filter Design Considerations

The filter design aims to:
1.  **Provide effective EMI suppression:** Attenuate unwanted high-frequency noise to comply with electromagnetic compatibility (EMC) standards.
2.  **Adhere to UL1283:** This standard outlines safety requirements for EMI filters, covering aspects like construction, materials, and performance under various conditions. 
3.  **Withstand SPD Testing (8/20µs 100kA):** The filter components, particularly the capacitor and any series elements, must be chosen to handle the high energy and peak current of the 8/20µs impulse current waveform, which is a standard test for SPDs. This involves considering:
    *   **Capacitor Voltage Rating and dV/dt:** The capacitor must withstand the peak voltage and the rapid rate of voltage change during a surge event.
    *   **Resistor Power Rating and Pulse Withstand Capability:** Resistors, especially the inrush limiting resistor, must be able to dissipate the energy from surge currents without failing.
    *   **Overall Circuit Layout and Creepage/Clearance Distances:** To prevent arcing and ensure safety under high voltage stress.

## Scripts

*   `amplitudni_odziv.py`:.
*   `osem_20_testiranje.py`: 

## Usage

To run the simulation:
1.  Python (3.12.7) with `numpy` and `matplotlib`.
2.  Execute the `amplitudni_odziv.py` script:
    ```bash
    python amplitudni_odziv.py
    ```
3.  Use the sliders in the matplotlib window to adjust component values and observe the changes in the frequency response.

## Future Development

*   Incorporate more detailed models for components, including parasitic elements.
*   Expand the simulation to include other filter topologies (e.g., LC, Pi filters).
*   Integrate specific test parameters from UL1283 and SPD testing standards into the simulation or analysis.
*   Automated tests to verify compliance with standard requirements.

## Disclaimer

This project is for educational and simulation purposes. Real-world filter design and testing for UL1283 compliance and SPD robustness require Hardcore engineering, deep understanding of electromagnetics, adherence to safety regulations, and physical prototyping and testing in accredited labs.
