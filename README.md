# Investigating the impact of motion and associated B0 changes on oxygenation sensitive MRI through realistic simulations

Abstract submitted to ISMRM 2023


Analysis was performed in Python 3.8.12.


## Steps for repeating the analysis:

1) Extract the multi-slice acquisition scheme with `Scan_order.py`. A raw file in ismrmrd format is needed.
2) Perform simulations by running `Simulate_Motion_Whole_Dataset.py`, e.g. with the following command: `python -u ./Simulate_Motion_Whole_Dataset.py --config_path ./config/config_run_rigid_mild.yaml`.
3) Analyse the outcome of the simulations with `Main_analysis.py`.  


## Illustration of the motion simulation procedure:
![Simulation_overview](/SimulationOverview.png?raw=true "Overview of motion simulation")