import logging
logger = logging.getLogger(__name__)
#!/usr/bin/env python3
"""
CoChem-SCAN Stage 2.0: The Structural Generator (v2.0)
Utilizes RDKit to mutate molecular seeds. 
Hardened with Parallel Embedding, Strict Seeding, LAM flags, and Valency catching.
"""

import os
import sys
import json
import importlib.util
from typing import Optional, List, Dict, Union, Any
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

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

if not importlib.util.find_spec("rdkit"):
    print_status("RDKit is not installed in this environment. Please run: pip install rdkit", "fail")
    sys.exit(1)

from rdkit import Chem
from rdkit.Chem import AllChem

def build_fragment_library(workspace_path: str) -> str:
    """
    SCAN-06: Expanded reaction fragment library covering broader chemical functional group isomer space.
    """
    lib_path = os.path.join(workspace_path, "fragment_library.json")
    fragment_data = {
        "carbonyl_stretch_1700": {"name": "Carbonyl (Ketone/Aldehyde)", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1]=O"},
        "hydroxyl_stretch_3300": {"name": "Hydroxyl (Alcohol/Phenol)", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1][O][H]"},
        "methyl_rock_1450": {"name": "Methyl Group (LAM)", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1][CH3]"},
        "amine_stretch_3400": {"name": "Primary Amine", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1][NH2]"},
        "nitro_stretch_1550": {"name": "Nitro Group", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1][N+](=O)[O-]"},
        "ether_stretch_1100": {"name": "Methoxy Ether", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1]O[CH3]"},
        "carboxyl_stretch_1750": {"name": "Carboxylic Acid", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1]C(=O)O"},
        "halogen_chloro_750": {"name": "Chloro Substituent", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1]Cl"}
    }
    with open(lib_path, "w") as f:
        json.dump(fragment_data, f, indent=4)
    return lib_path

def isolate_and_embed(smiles: str, name: str, max_uff_energy: float = 500.0) -> dict:
    """
    Worker function for parallel structure embedding.
    SCAN-07: Checks UFF convergence, uses maxIters=1000, and falls back to MMFF94.
    SCAN-19: Refined single-bonded methyl rotor SMARTS pattern for LAM detection.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"status": "failed", "reason": "SMILES parse error"}
            
        mol = Chem.AddHs(mol)
        
        # Enforce strict stochastic seeding for reproducibility
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 
        
        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            if res != 0:
                return {"status": "failed", "reason": "Embedding failed"}
            
        # SCAN-07: Robust force field optimization with UFF maxIters=1000 & MMFF94 fallback
        uff_res = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        energy = 9999.0
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            if ff is not None:
                energy = ff.CalcEnergy()
        except Exception as err:
            conf = mol.GetConformer()
            coords = conf.GetPositions()
            dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            np.fill_diagonal(dist_matrix, 1.0)
            energy = float(np.sum(1.0 / (dist_matrix**6)))
            
        if uff_res != 0 or energy > max_uff_energy:
            # Fallback to MMFF94 optimization
            try:
                mmff_res = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                if mmff_props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
                    if ff is not None:
                        energy = ff.CalcEnergy()
            except Exception as err:
                conf = mol.GetConformer()
                coords = conf.GetPositions()
                dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
                np.fill_diagonal(dist_matrix, 1.0)
                energy = float(np.sum(1.0 / (dist_matrix**6)))

        if energy > max_uff_energy:
            return {"status": "failed", "reason": f"Steric clash (Energy={energy:.1f})"}
            
        # SCAN-19: Refined single-bonded methyl rotor SMARTS for Large Amplitude Motion
        lam_smarts = Chem.MolFromSmarts("[CH3]-[!#1]")
        has_lam = mol.HasSubstructMatch(lam_smarts) if lam_smarts is not None else False
        
        mol.SetProp("_Name", name)
        mol.SetProp("HAS_LAM", str(has_lam))
        
        return {"status": "success", "mol_block": Chem.MolToMolBlock(mol), "has_lam": has_lam, "smiles": smiles}
    except Exception as e:
        return {"status": "failed", "reason": str(e)}

from cochem_scan_ingest import PESStore
from pathlib import Path

def interpolate_pes_grid(grid_coords: list = None, grid_energies: np.ndarray = None, query_points: np.ndarray = None, pes_store: Optional[Union[PESStore, str, Path]] = None) -> np.ndarray:
    """
    SCAN-02: Computes exact Potential Energy Surface (PES) grid interpolation across potential landscapes.
    Connects to PESStore HDF5 datasets under /pes/grid and /pes/fit (§8C).
    """
    if pes_store is not None:
        if not isinstance(pes_store, PESStore):
            pes_store = PESStore(pes_store)
        data = pes_store.load_pes_data()
        if "coordinates" in data and grid_coords is None:
            coords_arr = data["coordinates"]
            if coords_arr.ndim == 2:
                grid_coords = [np.unique(coords_arr[:, i]) for i in range(coords_arr.shape[1])]
            else:
                grid_coords = [coords_arr]
        if "energies" in data and grid_energies is None:
            grid_energies = data["energies"]

    from scipy.interpolate import RegularGridInterpolator, RBFInterpolator
    query_points = np.asarray(query_points, dtype=np.float64)
    if isinstance(grid_coords, list) and all(isinstance(c, np.ndarray) for c in grid_coords):
        try:
            interpolator = RegularGridInterpolator(grid_coords, grid_energies, bounds_error=False, fill_value=None)
            return interpolator(query_points)
        except Exception:
            pass

    if isinstance(grid_coords, list):
        grid_flat = np.array(np.meshgrid(*grid_coords, indexing='ij')).T.reshape(-1, len(grid_coords))
    else:
        grid_flat = np.asarray(grid_coords, dtype=np.float64)

    rbf = RBFInterpolator(grid_flat, np.asarray(grid_energies).ravel())
    return rbf(query_points)

def ensure_seed_structure(seed_path: str) -> None:
    """
    SCAN-08: Ensures a valid seed file exists. Creates default benzene seed if absent.
    """
    if not os.path.exists(seed_path):
        os.makedirs(os.path.dirname(seed_path), exist_ok=True)
        print_status(f"Seed file '{seed_path}' not found. Generating default toluene/benzene seed...", "warning")
        mol = Chem.MolFromSmiles("Cc1ccccc1")  # Toluene seed
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        mol.SetProp("_Name", "Seed_Toluene")
        writer = Chem.SDWriter(seed_path)
        writer.write(mol)
        writer.close()
        print_status(f"Generated default seed at '{seed_path}'.", "success")

def generate_candidate_ensemble(seed_path: str, missing_features: list, frag_lib_path: str, workspace: str) -> Any:
    # SCAN-08: Check seed file existence before reading
    ensure_seed_structure(seed_path)
    
    print_status(f"Ingesting seed for mutation: {os.path.basename(seed_path)}")
    with open(frag_lib_path, "r") as f:
        frag_lib = json.loads(f.read())
        
    supplier = Chem.SDMolSupplier(seed_path, removeHs=False)
    seed_mol = next(supplier, None)
    if seed_mol is None:
        raise ValueError(f"Could not load seed molecule from {seed_path}")
        
    unique_smiles = set()
    raw_candidates = []
    
    for feature in missing_features:
        if feature in frag_lib:
            rule = frag_lib[feature]
            print_status(f"Applying mutation: Injecting {rule['name']} (Resolves {feature})")
            rxn = AllChem.ReactionFromSmarts(rule["reaction_smarts"])
            products = rxn.RunReactants((seed_mol,))
            
            for product_tuple in products:
                candidate = product_tuple[0]
                try:
                    # Catch valency saturation crashes gracefully
                    Chem.SanitizeMol(candidate)
                    smi = Chem.MolToSmiles(candidate, isomericSmiles=True)
                    if smi not in unique_smiles:
                        unique_smiles.add(smi)
                        raw_candidates.append(smi)
                except ValueError:
                    continue  # Bypass unphysical valency states

    if not raw_candidates:
        # Include original seed if no mutations matched
        raw_candidates.append(Chem.MolToSmiles(seed_mol, isomericSmiles=True))

    print_status(f"Generated {len(raw_candidates)} topologically unique candidates. Embedding in parallel...")
    
    valid_mols = []
    lam_count = 0
    
    # ProcessPoolExecutor for CPU-bound Embedding Tasks
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, 8)) as executor:
        futures = [executor.submit(isolate_and_embed, smi, f"SCAN_Cand_{i}") for i, smi in enumerate(raw_candidates)]
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "success":
                valid_mols.append(res)
                if res["has_lam"]: lam_count += 1
                
    print_status(f"Embedded {len(valid_mols)} viable isomers ({lam_count} flagged as LAM systems).", "success")
    
    out_path = os.path.join(workspace, "candidate_ensemble_iter_1.sdf")
    with open(out_path, "w") as f:
        for m in valid_mols:
            f.write(m["mol_block"] + "$$$$\n")
    print_status(f"Ensemble exported to: {out_path}", "success")

def main() -> Any:
    logger.info(f"\n{Colors.HEADER}{Colors.BOLD}--- CoChem-SCAN: Stage 2.0 Structural Generator (v2.0) ---{Colors.ENDC}")
    
    config_path = "cochem_system_config.json"
    if not os.path.exists(config_path):
        sys.exit(1)
        
    with open(config_path, "r") as f:
        workspace = json.loads(f.read()).get("scan_engine", {}).get("workspace_path", "./SCAN_Workspace")
        
    seed_file = os.path.join(workspace, "benzene_seed.sdf") 
    frag_lib = build_fragment_library(workspace)
    simulated_missing_features = ["carbonyl_stretch_1700", "methyl_rock_1450"]
    
    generate_candidate_ensemble(seed_file, simulated_missing_features, frag_lib, workspace)

if __name__ == "__main__":
    main()