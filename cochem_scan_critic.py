#!/usr/bin/env python3
"""
CoChem-SCAN Stage 2.2: Spectral Critic & Logic Gate (v2.0)
Evaluates theoretical spectra against experimental constraints using Vectorized Broadening.
Calculates Boltzmann Composite Spectra, Top-K Pareto Fronts, and Rips Final Geometries.
"""

import os
import sys
import json
import numpy as np
import importlib.util

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

# Graceful dependency check for RDKit (used for ripping geometries)
HAS_RDKIT = importlib.util.find_spec("rdkit") is not None
if HAS_RDKIT:
    from rdkit import Chem

def vectorized_broadening(freqs, intensities, exp_grid, hwhm=10.0) -> np.ndarray:
    """
    SCAN-09: Matrix-broadcasted Gaussian broadening with normalized peak height.
    Ensures peak height at line center equals original integrated intensity.
    """
    if len(freqs) == 0:
        return np.zeros_like(exp_grid)
    
    f_arr = np.array(freqs)[:, None]
    i_arr = np.array(intensities)[:, None]
    sigma = hwhm / np.sqrt(2.0 * np.log(2.0))
    
    # Peak amplitude equals i_arr at frequency loc=f_arr
    gaussians = i_arr * np.exp(-0.5 * ((exp_grid - f_arr) / sigma)**2)
    return np.sum(gaussians, axis=0)

def check_dead_zone_violations(theory_freqs, theory_intensities, dead_zones, has_lam: bool, intensity_threshold=5.0) -> list:
    """Checks if intense peaks fall in noise floors. Relaxes low-freq constraints if LAM is present."""
    violations = []
    for f, i in zip(theory_freqs, theory_intensities):
        # If LAM is present, ignore missing torsional lines < 500 cm-1
        if has_lam and f < 500.0:
            continue
            
        if i > intensity_threshold:
            for start, end in dead_zones:
                if start <= f <= end:
                    violations.append(f"Peak at {f:.1f} cm-1 violates Dead Zone [{start:.1f} - {end:.1f}]")
    return violations

def calculate_pareto_front(candidates: list) -> list:
    """Extracts non-dominated candidates (Low Energy, Low Residual)."""
    pareto_front = []
    for c1 in candidates:
        dominated = False
        for c2 in candidates:
            if c1['candidate_id'] == c2['candidate_id']:
                continue
            if (c2['energy'] <= c1['energy'] and c2['residual'] <= c1['residual']) and \
               (c2['energy'] < c1['energy'] or c2['residual'] < c1['residual']):
                dominated = True
                break
        if not dominated:
            pareto_front.append(c1)
    return pareto_front

def rip_final_geometries(pareto_front: list, workspace: str):
    """Extracts the exact 3D Cartesian coordinates of the Top-K survivors."""
    if not HAS_RDKIT:
        print_status("RDKit missing. Cannot extract final standalone isomer geometries.", "warning")
        return

    sdf_path = os.path.join(workspace, "candidate_ensemble_iter_1.sdf")
    out_dir = os.path.join(workspace, "Final_Isomers")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(sdf_path):
        print_status(f"Candidate ensemble SDF not found at {sdf_path}.", "warning")
        return

    valid_ids = {c["candidate_id"] for c in pareto_front}
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    
    extracted = 0
    for mol in supplier:
        if mol is not None and mol.HasProp("_Name"):
            name = mol.GetProp("_Name")
            if name in valid_ids:
                Chem.MolToXYZFile(mol, os.path.join(out_dir, f"{name}.xyz"))
                Chem.MolToMolFile(mol, os.path.join(out_dir, f"{name}.mol"))
                extracted += 1
                
    print_status(f"Ripped {extracted} finalized geometries to {out_dir}/", "success")

def get_method_scaling_factor(method_name: str) -> float:
    """
    SCAN-10: Dynamic frequency harmonic scaling factor based on computational method.
    """
    method_name = (method_name or "").lower()
    scaling_map = {
        "xtb2": 1.000,
        "r2scan-3c": 0.985,
        "b3lyp": 0.965,
        "pbe0": 0.960,
        "dlpno-ccsd(t)": 0.957,
    }
    for key, scale in scaling_map.items():
        if key in method_name:
            return scale
    return 0.960

def main():
    print(f"\n{Colors.HEADER}{Colors.BOLD}--- CoChem-SCAN: Stage 2.2 Spectral Critic (v2.0) ---{Colors.ENDC}")
    
    # Load environment
    config_path = "cochem_system_config.json"
    if not os.path.exists(config_path):
        sys.exit(1)
        
    with open(config_path, "r") as f:
        workspace = json.load(f).get("scan_engine", {}).get("workspace_path", "./SCAN_Workspace")
    
    # 1. Ingest Data & Constraints
    spectra_path = os.path.join(workspace, "computed_spectra_iter_1.json")
    dz_path = os.path.join(workspace, "experimental_dead_zones.json")
    constraints_path = os.path.join(workspace, "user_constraints.json")
    
    try:
        with open(spectra_path, "r") as f: computed_data = json.load(f)
        with open(dz_path, "r") as f: dead_zones = json.load(f).get("dead_zones_cm-1", [])
        with open(constraints_path, "r") as f:
            constraints = json.load(f)
            T = constraints.get("boltzmann_temperature_K", 298.15)
            top_k = constraints.get("top_k_retention", 5)
    except FileNotFoundError:
        print_status("Missing required metadata files.", "fail")
        sys.exit(1)
        
    # SCAN-12: Load actual experimental frequency grid if present
    exp_grid_path = os.path.join(workspace, "exp_freqs.npy")
    if os.path.exists(exp_grid_path):
        exp_freq_grid = np.load(exp_grid_path)
    else:
        exp_freq_grid = np.linspace(500, 4000, 1000)
        
    exp_int_path = os.path.join(workspace, "exp_intensities.npy")
    if os.path.exists(exp_int_path):
        exp_intensity_grid = np.load(exp_int_path)
    else:
        exp_intensity_grid = np.zeros_like(exp_freq_grid) 
    
    valid_candidates = []
    pruning_log = []
    
    print_status(f"Evaluating {len(computed_data)} spectra (Temp: {T}K)...")
    
    # 2. Logic Gate & Vectorized Preclusion
    for cand in computed_data:
        if cand["status"] == "failed":
            pruning_log.append({"id": cand["candidate_id"], "reason": "SCF Failure"})
            continue
            
        has_lam = "LAM" in str(cand.get("candidate_id", ""))
        
        # SCAN-10: Dynamic harmonic frequency scaling based on method
        method_name = cand.get("method", "default")
        scale_factor = get_method_scaling_factor(method_name)
        scaled_freqs = [f * scale_factor for f in cand["freqs"]]
            
        violations = check_dead_zone_violations(scaled_freqs, cand["intensities"], dead_zones, has_lam)
        if violations:
            print_status(f"Candidate {cand['candidate_id']} killed: {violations[0]}", "fail")
            pruning_log.append({"id": cand["candidate_id"], "reason": violations[0]})
            continue
            
        simulated = vectorized_broadening(scaled_freqs, cand["intensities"], exp_freq_grid)
        residual = float(np.sum((simulated - exp_intensity_grid)**2))
        
        cand["residual"] = residual
        cand["scaled_freqs"] = scaled_freqs
        valid_candidates.append(cand)
        
    # 3. Pareto Optimization & Top-K Truncation
    if valid_candidates:
        pareto_front = calculate_pareto_front(valid_candidates)
        # Sort by residual and truncate
        pareto_front = sorted(pareto_front, key=lambda x: x["residual"])[:top_k]
        
        # SCAN-11: Correct gas constant & unit conversion (Hartree to kcal/mol: * 627.509)
        R_gas = 0.001987  # kcal/(mol K)
        HARTREE_TO_KCAL = 627.509
        
        min_E_hartree = min(c["energy"] for c in pareto_front)
        
        # Compute Boltzmann weights with proper unit conversion
        exp_factors = []
        for c in pareto_front:
            delta_E_kcal = (c["energy"] - min_E_hartree) * HARTREE_TO_KCAL
            exp_factor = np.exp(-delta_E_kcal / (R_gas * max(T, 1.0)))
            exp_factors.append(exp_factor)
            
        partition_Z = max(sum(exp_factors), 1e-18)
        
        composite_spectrum = np.zeros_like(exp_freq_grid)
        for idx, c in enumerate(pareto_front):
            weight = float(exp_factors[idx] / partition_Z)
            c["boltzmann_weight"] = weight
            spec = vectorized_broadening(c["scaled_freqs"], c["intensities"], exp_freq_grid)
            composite_spectrum += weight * spec
            
        np.save(os.path.join(workspace, "composite_spectrum.npy"), composite_spectrum)
        print_status(f"Retained {len(pareto_front)} candidates. Composite spectrum generated.", "success")
        
        # Rip Top-K geometries
        rip_final_geometries(pareto_front, workspace)
    else:
        pareto_front = []
        print_status("Catastrophic preclusion: All candidates failed constraints.", "warning")

    # 5. Serialization
    with open(os.path.join(workspace, "pareto_front_iter_1.json"), "w") as f:
        json.dump(pareto_front, f, indent=4)
    with open(os.path.join(workspace, "pruning_rationale.json"), "w") as f:
        json.dump(pruning_log, f, indent=4)
        
    print_status("Critic cycle complete. Handoff ready for UI & Archiving.", "success")

if __name__ == "__main__":
    main()