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
import time
import threading
import h5py
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
from filelock import FileLock

_pes_thread_locks = {}
_pes_global_lock = threading.Lock()

def _get_thread_lock(path: Path) -> threading.RLock:
    canonical = str(path.resolve())
    with _pes_global_lock:
        if canonical not in _pes_thread_locks:
            _pes_thread_locks[canonical] = threading.RLock()
        return _pes_thread_locks[canonical]

class PESStore:
    """
    Standardized HDF5 PES Store Manager (§8C).
    Interfaces cochem_state.h5 /pes/ datasets with 512-point chunking & gzip level 4 compression.
    Groups:
      - /pes/grid: coordinates, internal_coords, atomic_numbers, exp_freqs, exp_intensities, etc.
      - /pes/fit: energies, gradients, fit_coefficients, model_type
      - /pes/uncertainty: variance, retier_flags
    """
    def __init__(self, h5_path: Union[str, Path]):
        self.h5_path = Path(h5_path)

    def _locked_op(self, func, *args, **kwargs):
        """Thread-safe and process-safe FileLock execution wrapper with backoff retries."""
        thread_lock = _get_thread_lock(self.h5_path)
        lock_path = str(self.h5_path) + ".lock"
        file_lock = FileLock(lock_path, timeout=30.0)
        with thread_lock:
            with file_lock:
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(0.05 * (2 ** attempt))

    def init_store(self) -> None:
        """Initializes the /pes/ HDF5 group structure and provenance tags."""
        return self._locked_op(self._init_store_impl)

    def _init_store_impl(self) -> None:
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.h5_path, "a") as f:
            pes = f.require_group("pes")
            pes.attrs["provenance_tag"] = "[D]"
            grid = pes.require_group("grid")
            grid.attrs["provenance_tag"] = "[D]"
            fit = pes.require_group("fit")
            fit.attrs["provenance_tag"] = "[D]"
            unc = pes.require_group("uncertainty")
            unc.attrs["provenance_tag"] = "[E]"

    def _create_lossless_dataset(self, group: h5py.Group, name: str, data: np.ndarray, tag: str = "[D]") -> h5py.Dataset:
        """Creates a dataset with 512-pt chunking, gzip level 4, shuffle filter, and Fletcher32 checksums."""
        if name in group:
            del group[name]
        data = np.asarray(data)
        if data.size == 0 or (data.ndim > 0 and data.shape[0] == 0):
            dset = group.create_dataset(name, data=data)
        elif data.ndim == 0:
            dset = group.create_dataset(name, data=data)
        else:
            N = data.shape[0]
            chunk_dim = min(512, N)
            chunks = (chunk_dim,) + data.shape[1:]
            dset = group.create_dataset(
                name,
                data=data,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True
            )
        dset.attrs["provenance_tag"] = tag
        return dset

    def save_experimental_spectrum(self, freqs: np.ndarray, intensities: np.ndarray) -> None:
        """Saves experimental frequency and intensity arrays under /pes/grid with tag [M]."""
        return self._locked_op(self._save_experimental_spectrum_impl, freqs, intensities)

    def _save_experimental_spectrum_impl(self, freqs: np.ndarray, intensities: np.ndarray) -> None:
        freqs = np.asarray(freqs, dtype=np.float64)
        intensities = np.asarray(intensities, dtype=np.float64)
        self._init_store_impl()
        with h5py.File(self.h5_path, "a") as f:
            grid = f.require_group("pes/grid")
            self._create_lossless_dataset(grid, "exp_freqs", freqs, tag="[M]")
            self._create_lossless_dataset(grid, "exp_intensities", intensities, tag="[M]")

    def load_experimental_spectrum(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Loads experimental spectrum arrays from /pes/grid if present."""
        return self._locked_op(self._load_experimental_spectrum_impl)

    def _load_experimental_spectrum_impl(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.h5_path.exists():
            return None, None
        with h5py.File(self.h5_path, "r") as f:
            if "pes/grid/exp_freqs" in f and "pes/grid/exp_intensities" in f:
                return np.array(f["pes/grid/exp_freqs"]), np.array(f["pes/grid/exp_intensities"])
        return None, None

    def save_grid_points(self, coordinates: Optional[np.ndarray] = None,
                         energies: Optional[np.ndarray] = None,
                         gradients: Optional[np.ndarray] = None,
                         variances: Optional[np.ndarray] = None,
                         retier_flags: Optional[np.ndarray] = None) -> None:
        """
        Saves or updates PES grid coordinates, fitted energies, gradients, uncertainties, and retier_flags.
        Enforces strict dimension length matching across primary dataset dimensions.
        """
        shapes = {}
        for name, arr in [
            ("coordinates", coordinates),
            ("energies", energies),
            ("gradients", gradients),
            ("variances", variances),
            ("retier_flags", retier_flags)
        ]:
            if arr is not None:
                a = np.asarray(arr)
                shapes[name] = a.shape[0] if a.ndim > 0 else 1

        if shapes:
            lengths = list(shapes.values())
            if len(set(lengths)) > 1:
                details = ", ".join(f"{k}={v}" for k, v in shapes.items())
                raise ValueError(f"Dimension mismatch in save_grid_points primary lengths: {details}")

        return self._locked_op(
            self._save_grid_points_impl,
            coordinates=coordinates,
            energies=energies,
            gradients=gradients,
            variances=variances,
            retier_flags=retier_flags
        )

    def _save_grid_points_impl(self, coordinates: Optional[np.ndarray] = None,
                               energies: Optional[np.ndarray] = None,
                               gradients: Optional[np.ndarray] = None,
                               variances: Optional[np.ndarray] = None,
                               retier_flags: Optional[np.ndarray] = None) -> None:
        self._init_store_impl()
        with h5py.File(self.h5_path, "a") as f:
            grid = f.require_group("pes/grid")
            fit = f.require_group("pes/fit")
            unc = f.require_group("pes/uncertainty")

            if coordinates is not None:
                self._create_lossless_dataset(grid, "coordinates", np.asarray(coordinates, dtype=np.float64), tag="[D]")
            if energies is not None:
                self._create_lossless_dataset(fit, "energies", np.asarray(energies, dtype=np.float64), tag="[E]")
            if gradients is not None:
                self._create_lossless_dataset(fit, "gradients", np.asarray(gradients, dtype=np.float64), tag="[E]")
            if variances is not None:
                self._create_lossless_dataset(unc, "variance", np.asarray(variances, dtype=np.float64), tag="[E]")
            if retier_flags is not None:
                self._create_lossless_dataset(unc, "retier_flags", np.asarray(retier_flags, dtype=np.uint8), tag="[D]")

    def load_pes_data(self) -> Dict[str, np.ndarray]:
        """Loads all PES datasets from /pes/ into a dictionary."""
        return self._locked_op(self._load_pes_data_impl)

    def _load_pes_data_impl(self) -> Dict[str, np.ndarray]:
        if not self.h5_path.exists():
            return {}
        res = {}
        with h5py.File(self.h5_path, "r") as f:
            if "pes/grid/coordinates" in f:
                res["coordinates"] = np.array(f["pes/grid/coordinates"])
            if "pes/fit/energies" in f:
                res["energies"] = np.array(f["pes/fit/energies"])
            if "pes/fit/gradients" in f:
                res["gradients"] = np.array(f["pes/fit/gradients"])
            if "pes/uncertainty/variance" in f:
                res["variance"] = np.array(f["pes/uncertainty/variance"])
            if "pes/uncertainty/retier_flags" in f:
                res["retier_flags"] = np.array(f["pes/uncertainty/retier_flags"])
        return res

class Colors:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg: str, status: str = "info") -> None:
    symbols = {"success": "[OK]", "warning": "[WARN]", "fail": "[FAIL]", "info": "[INFO]"}
    sym = symbols.get(status, "[INFO]")
    color = Colors.OKGREEN if status == "success" else (
        Colors.WARNING if status == "warning" else (
            Colors.FAIL if status == "fail" else Colors.OKCYAN
        )
    )
    try:
        print(f"  {color}{sym} {msg}{Colors.ENDC}")
    except UnicodeEncodeError:
        print(f"  {sym} {msg}")

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
        
    # SCAN-02: Store experimental spectrum in cochem_state.h5 via PESStore (§8C)
    store = PESStore(os.path.join(workspace_path, "cochem_state.h5"))
    store.save_experimental_spectrum(freqs, intensities)
    
    print_status(f"Mapped {len(dead_zones)} Dead Zones (Noise Floor: {noise_floor:.4f}). Archived to PESStore (cochem_state.h5).", "success")
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