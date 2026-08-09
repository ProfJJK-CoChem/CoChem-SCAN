#!/usr/bin/env python3
"""
CoChem-SCAN Stage 1.0: Ingestion & Pre-Flight (v2.0)
Standardizes spectral data, maps Dead Zones, checks hardware limits, 
and captures explicit user constraints (Temperature & Top-K).
"""

import os
import sys
import json
import shutil
import numpy as np

class Colors:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg: str, status: str = "info") -> None:
    if status == "success":
        print(f"  {Colors.OKGREEN}✅ {msg}{Colors.ENDC}")
    elif status == "warning":
        print(f"  {Colors.WARNING}⚠️ {msg}{Colors.ENDC}")
    elif status == "fail":
        print(f"  {Colors.FAIL}❌ {msg}{Colors.ENDC}")
    else:
        print(f"  {Colors.OKCYAN}➡️ {msg}{Colors.ENDC}")

def load_registry():
    config_path = "cochem_system_config.json"
    if not os.path.exists(config_path):
        print_status("Master registry missing. Run Stage 0.0 first.", "fail")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)

def pre_flight_check(workspace_path: str, min_gb_required: float = 1.0) -> bool:
    print_status(f"Executing Pre-Flight Integrity Check on {workspace_path}...")
    total, used, free = shutil.disk_usage(workspace_path)
    free_gb = free / (1024**3)
    
    if free_gb < min_gb_required:
        print_status(f"Insufficient disk space! {free_gb:.2f} GB free. Required: {min_gb_required} GB.", "fail")
        return False
    print_status(f"Disk space verified: {free_gb:.2f} GB available.", "success")
    return True

def map_dead_zones(spectrum_file: str, workspace_path: str, noise_multiplier: float = 1.5):
    """
    SCAN-14: Accurate rolling baseline / percentile noise floor estimation in dead zone mapping.
    Prevents weak absorption bands from being misclassified as dead zones.
    """
    print_status(f"Ingesting experimental spectrum: {spectrum_file}...")
    if not os.path.exists(spectrum_file):
        # SCAN-13: Raise explicit FileNotFoundError rather than silently masking missing file
        raise FileNotFoundError(f"Spectrum file '{spectrum_file}' not found. Please provide a valid spectrum.")

    try:
        data = np.loadtxt(spectrum_file)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        freqs = data[:, 0]
        intensities = data[:, 1]
    except Exception as e:
        print_status(f"Failed to parse spectrum: {str(e)}", "fail")
        return None

    # SCAN-14: Robust rolling noise floor estimation
    # Uses 10th percentile over sliding windows to avoid baseline tilt artifacts
    window_size = max(5, len(intensities) // 20)
    noise_floors = []
    for idx in range(0, len(intensities), window_size):
        sub = intensities[idx:idx+window_size]
        if len(sub) > 0:
            noise_floors.append(np.percentile(sub, 10))
            
    base_noise = np.median(noise_floors) if noise_floors else np.percentile(intensities, 10)
    noise_floor = max(base_noise * noise_multiplier, 0.01)
    
    dead_zones = []
    in_dead_zone = False
    start_freq = None
    
    for f, i in zip(freqs, intensities):
        if i <= noise_floor and not in_dead_zone:
            in_dead_zone = True
            start_freq = f
        elif i > noise_floor and in_dead_zone:
            in_dead_zone = False
            dead_zones.append((float(start_freq), float(f)))
            
    if in_dead_zone:
        dead_zones.append((float(start_freq), float(freqs[-1])))

    dead_zone_path = os.path.join(workspace_path, "experimental_dead_zones.json")
    with open(dead_zone_path, "w") as f:
        json.dump({"noise_floor_threshold": float(noise_floor), "dead_zones_cm-1": dead_zones}, f, indent=4)
        
    np.save(os.path.join(workspace_path, "exp_freqs.npy"), freqs)
    np.save(os.path.join(workspace_path, "exp_intensities.npy"), intensities)
    
    print_status(f"Mapped {len(dead_zones)} Dead Zones (Noise Floor: {noise_floor:.4f}).", "success")
    return dead_zone_path

def set_user_constraints(workspace: str, temperature: float = 298.15, top_k: int = 5):
    """Establishes the user-defined boundaries for the Negotiator Loop."""
    constraint_path = os.path.join(workspace, "user_constraints.json")
    constraints = {
        "boltzmann_temperature_K": temperature,
        "top_k_retention": top_k
    }
    with open(constraint_path, "w") as f:
        json.dump(constraints, f, indent=4)
    print_status(f"Locked constraints: {temperature} K, retaining Top {top_k} species.", "success")

def main():
    print(f"\n{Colors.HEADER}{Colors.BOLD}--- CoChem-SCAN: Stage 1.0 Ingestion (v2.0) ---{Colors.ENDC}")
    
    config = load_registry()
    scan_cfg = config.get("scan_engine", {})
    workspace = scan_cfg.get("workspace_path")
    
    if not workspace:
        print_status("SCAN Workspace not defined in registry.", "fail")
        sys.exit(1)
        
    if not pre_flight_check(workspace):
        sys.exit(1)
        
    test_spectrum = os.path.join(workspace, "input_spectrum.txt")
    
    # SCAN-13: Explicit check for synthetic mode via argument or missing spectrum handling
    if not os.path.exists(test_spectrum):
        if "--synthetic" in sys.argv:
            print_status("Creating synthetic spectrum file for testing...", "warning")
            freq_grid = np.linspace(500, 4000, 1000)
            int_grid = np.sin(freq_grid / 100.0)**2 * 50.0
            np.savetxt(test_spectrum, np.column_stack((freq_grid, int_grid)))
        else:
            print_status(f"Spectrum file '{test_spectrum}' does not exist. Pass --synthetic to generate a test spectrum.", "fail")
            sys.exit(1)
    
    map_dead_zones(test_spectrum, workspace)
    set_user_constraints(workspace, temperature=298.15, top_k=5)
    
    print(f"{Colors.HEADER}{Colors.BOLD}-----------------------------------------------{Colors.ENDC}\n")

if __name__ == "__main__":
    main()