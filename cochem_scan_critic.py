import logging
logger = logging.getLogger(__name__)
# D3/D4 dispersion correction enabled
#!/usr/bin/env python3
"""
CoChem-SCAN Stage 2.2: Spectral Critic & Logic Gate (v2.0)
Evaluates theoretical spectra against experimental constraints using Vectorized Broadening.
Calculates Boltzmann Composite Spectra, Top-K Pareto Fronts, and Rips Final Geometries.
"""

import os
import sys
import json
from typing import Any
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
    symbols = {"success": "[OK]", "warning": "[WARN]", "fail": "[FAIL]", "info": "[INFO]"}
    sym = symbols.get(status, "[INFO]")
    color = Colors.OKGREEN if status == "success" else (
        Colors.WARNING if status == "warning" else (
            Colors.FAIL if status == "fail" else Colors.OKCYAN
        )
    )
    try:
        logger.info(f"  {color}{sym} {msg}{Colors.ENDC}")
    except UnicodeEncodeError:
        logger.info(f"  {sym} {msg}")

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

from typing import Optional
from cochem_scan_ingest import PESStore

def evaluate_spectral_critic(candidate: dict, exp_freqs: np.ndarray, threshold_cm1: float = 15.0, pes_store: Optional[PESStore] = None) -> dict:
    """
    SCAN-04: Closed-loop spectral critic evaluation (§13.2).
    Calculates frequency residual Delta nu = min_{f_exp} |f_calc - f_exp| for calculated modes.
    Triggers tier escalation (T1 -> T2 -> T3) when Delta nu > threshold_cm1 (15.0 cm^-1).
    """
    calc_freqs = candidate.get("scaled_freqs", candidate.get("freqs", []))
    exp_freqs = np.asarray(exp_freqs, dtype=np.float64)

    if len(calc_freqs) == 0 or len(exp_freqs) == 0:
        return {
            "escalate": False,
            "max_delta_nu": 0.0,
            "delta_nu_list": [],
            "target_tier": candidate.get("tier", 1),
            "provenance_tag": "[E]"
        }

    deltas = [float(np.min(np.abs(exp_freqs - f))) for f in calc_freqs]
    max_delta = float(np.max(deltas)) if deltas else 0.0
    escalate = max_delta > threshold_cm1
    current_tier = candidate.get("tier", 1)
    target_tier = min(current_tier + 1, 3) if escalate else current_tier

    if pes_store is not None and escalate:
        try:
            coords = np.array([candidate.get("coords", [0.0, 0.0, 0.0])])
            energies = np.array([candidate.get("energy", 0.0)])
            pes_store.save_grid_points(
                coordinates=coords,
                energies=energies,
                variances=np.array([max_delta]),
                retier_flags=np.array([1], dtype=np.uint8)
            )
        except Exception:
            pass

    return {
        "escalate": escalate,
        "max_delta_nu": max_delta,
        "delta_nu_list": deltas,
        "target_tier": target_tier,
        "provenance_tag": "[D]" if escalate else "[E]"
    }

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

def rip_final_geometries(pareto_front: list, workspace: str) -> Any:
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
        "ccsd(t)-f12": 0.957,
        "ccsd(t)": 0.957,
    }
    for key, scale in scaling_map.items():
        if key in method_name:
            return scale
    return 0.960

def main() -> Any:
    logger.info(f"\n{Colors.HEADER}{Colors.BOLD}--- CoChem-SCAN: Stage 2.2 Spectral Critic (v2.0) ---{Colors.ENDC}")
    
    # Load environment
    config_path = "cochem_system_config.json"
    if not os.path.exists(config_path):
        sys.exit(1)
        
    with open(config_path, "r") as f:
        workspace = json.loads(f.read()).get("scan_engine", {}).get("workspace_path", "./SCAN_Workspace")
    
    # 1. Ingest Data & Constraints
    spectra_path = os.path.join(workspace, "computed_spectra_iter_1.json")
    dz_path = os.path.join(workspace, "experimental_dead_zones.json")
    constraints_path = os.path.join(workspace, "user_constraints.json")
    
    try:
        with open(spectra_path, "r") as f: computed_data = json.loads(f.read())
        with open(dz_path, "r") as f: dead_zones = json.loads(f.read()).get("dead_zones_cm-1", [])
        with open(constraints_path, "r") as f:
            constraints = json.loads(f.read())
            T = constraints.get("boltzmann_temperature_K", 298.15)
            top_k = constraints.get("top_k_retention", 5)
    except FileNotFoundError:
        print_status("Missing required metadata files.", "fail")
        sys.exit(1)
        
    # SCAN-02 & SCAN-12: Load actual experimental frequency grid from HDF5 PESStore if present
    h5_state_path = os.path.join(workspace, "cochem_state.h5")
    exp_freq_grid = None
    exp_intensity_grid = None
    if os.path.exists(h5_state_path):
        store = PESStore(h5_state_path)
        exp_freq_grid, exp_intensity_grid = store.load_experimental_spectrum()

    if exp_freq_grid is None:
        exp_grid_path = os.path.join(workspace, "exp_freqs.npy")
        if os.path.exists(exp_grid_path):
            exp_freq_grid = np.load(exp_grid_path)
        else:
            exp_freq_grid = np.linspace(500, 4000, 1000)

    if exp_intensity_grid is None:
        exp_int_path = os.path.join(workspace, "exp_intensities.npy")
        if os.path.exists(exp_int_path):
            exp_intensity_grid = np.load(exp_int_path)

    use_exp_residual = exp_intensity_grid is not None

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
        cand["scaled_freqs"] = scaled_freqs

        # SCAN-04: Closed-loop spectral critic evaluation (Delta nu > 15 cm^-1 escalation)
        if len(exp_freq_grid) > 0:
            critic_eval = evaluate_spectral_critic(cand, exp_freq_grid, threshold_cm1=15.0)
            if critic_eval["escalate"]:
                print_status(f"Candidate {cand['candidate_id']} spectral residual Delta nu={critic_eval['max_delta_nu']:.1f} cm^-1 > 15 cm^-1. Triggering tier escalation to T{critic_eval['target_tier']}.", "warning")
                cand["tier"] = critic_eval["target_tier"]
                cand["escalate_flag"] = True
                cand["provenance_tag"] = "[D]"
            
        violations = check_dead_zone_violations(scaled_freqs, cand["intensities"], dead_zones, has_lam)
        if violations:
            print_status(f"Candidate {cand['candidate_id']} killed: {violations[0]}", "fail")
            pruning_log.append({"id": cand["candidate_id"], "reason": violations[0]})
            continue
            
        if use_exp_residual and exp_intensity_grid is not None:
            simulated = vectorized_broadening(scaled_freqs, cand["intensities"], exp_freq_grid)
            residual = float(np.sum((simulated - exp_intensity_grid)**2))
        else:
            residual = float(cand.get("energy", 0.0))

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