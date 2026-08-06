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
    if status == "success":
        print(f"  {Colors.OKGREEN}✅ {msg}{Colors.ENDC}")
    elif status == "warning":
        print(f"  {Colors.WARNING}⚠️ {msg}{Colors.ENDC}")
    elif status == "fail":
        print(f"  {Colors.FAIL}❌ {msg}{Colors.ENDC}")
    else:
        print(f"  {Colors.OKCYAN}➡️ {msg}{Colors.ENDC}")

if not importlib.util.find_spec("rdkit"):
    print_status("RDKit is not installed in this environment. Please run: pip install rdkit", "fail")
    sys.exit(1)

from rdkit import Chem
from rdkit.Chem import AllChem

def build_fragment_library(workspace_path: str) -> str:
    lib_path = os.path.join(workspace_path, "fragment_library.json")
    fragment_data = {
        "carbonyl_stretch_1700": {"name": "Carbonyl (Ketone/Aldehyde)", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1]=O"},
        "hydroxyl_stretch_3300": {"name": "Hydroxyl (Alcohol/Phenol)", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1][O][H]"},
        "methyl_rock_1450": {"name": "Methyl Group (LAM)", "reaction_smarts": "[c,C:1][H:2]>>[c,C:1][CH3]"}
    }
    with open(lib_path, "w") as f:
        json.dump(fragment_data, f, indent=4)
    return lib_path

def isolate_and_embed(smiles: str, name: str, max_uff_energy: float = 500.0) -> dict:
    """Worker function for parallel structure embedding."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # Enforce strict stochastic seeding for reproducibility
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 
        
        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            return {"status": "failed", "reason": "Embedding failed"}
            
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        ff = AllChem.UFFGetMoleculeForceField(mol)
        energy = ff.CalcEnergy()
        
        if energy > max_uff_energy:
            return {"status": "failed", "reason": "Steric clash (UFF)"}
            
        # Detect Large Amplitude Motions (Internal Rotors)
        lam_smarts = Chem.MolFromSmarts("[CH3]")
        has_lam = mol.HasSubstructMatch(lam_smarts)
        
        mol.SetProp("_Name", name)
        mol.SetProp("HAS_LAM", str(has_lam))
        
        return {"status": "success", "mol_block": Chem.MolToMolBlock(mol), "has_lam": has_lam, "smiles": smiles}
    except Exception as e:
        return {"status": "failed", "reason": str(e)}

def generate_candidate_ensemble(seed_path: str, missing_features: list, frag_lib_path: str, workspace: str):
    print_status(f"Ingesting seed for mutation: {os.path.basename(seed_path)}")
    with open(frag_lib_path, "r") as f:
        frag_lib = json.load(f)
        
    seed_mol = next(Chem.SDMolSupplier(seed_path, removeHs=False))
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
                    continue # Bypass unphysical valency states

    print_status(f"Generated {len(raw_candidates)} topologically unique candidates. Embedding in parallel...")
    
    valid_mols = []
    lam_count = 0
    
    # ProcessPoolExecutor for CPU-bound Embedding Tasks
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
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

def main():
    print(f"\n{Colors.HEADER}{Colors.BOLD}--- CoChem-SCAN: Stage 2.0 Structural Generator (v2.0) ---{Colors.ENDC}")
    
    config_path = "cochem_system_config.json"
    if not os.path.exists(config_path):
        sys.exit(1)
        
    with open(config_path, "r") as f:
        workspace = json.load(f).get("scan_engine", {}).get("workspace_path")
        
    seed_file = os.path.join(workspace, "benzene_seed.sdf") 
    frag_lib = build_fragment_library(workspace)
    simulated_missing_features = ["carbonyl_stretch_1700", "methyl_rock_1450"]
    
    generate_candidate_ensemble(seed_file, simulated_missing_features, frag_lib, workspace)

if __name__ == "__main__":
    main()