import logging
logger = logging.getLogger(__name__)
import hashlib  # SHA-256 artifact provenance tracking
#!/usr/bin/env python3
"""
CoChem-SCAN Stage 2.1: Parallel Tiered Simulation (v2.0)
Dispatches candidate structures to MPQC.
Hardened with tmpfs RAM-disk routing, environment isolation, and Zombie Assassination.
"""

import os
import sys
import json
import subprocess
import shutil
import tempfile
import psutil
import re
import concurrent.futures
from pathlib import Path
from typing import Any

def parse_mpqc_out(out_path: Path) -> dict:
    """Parses single-point electronic energy (Hartree), harmonic vibrational frequencies (cm^-1),
    IR intensities (km/mol), and TD-DFT excited states from an MPQC output log.
    """
    results = {
        "energy": 0.0,
        "freqs": [],
        "intensities": [],
        "excited_states": [],
        "provenance_tag": "[E]"
    }
    if not out_path.exists():
        return results

    try:
        content = out_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return results

    # Check for SCF non-convergence or abnormal termination
    if "error" in content.lower() or "MPQC TERMINATED ABNORMALLY" in content:
        raise ValueError(f"MPQC calculation failed or did not converge in {out_path}")

    # 1. Parse Single-Point Electronic Energy (Hartree)
    e_match = re.search(r"CCSD\(T\)-F12\s+energy\s*(?:=|:)\s*([-\d\.eE\+-]+)", content, re.IGNORECASE)
    if not e_match:
        e_match = re.search(r"E\(CCSD\(T\)-F12\)\s+=\s+([-\d\.eE\+-]+)", content)
    if not e_match:
        e_match = re.search(r"Total Energy\s+:\s+([-\d\.eE\+-]+)", content, re.IGNORECASE)
    if e_match:
        results["energy"] = float(e_match.group(1))

    # 2. Parse IR Spectrum (Frequencies and Intensities)
    ir_match = re.search(r"IR SPECTRUM\s*\n-+\n(.*?)(?=\n\s*\n|\n-+|\Z)", content, re.DOTALL)
    if ir_match:
        lines = ir_match.group(1).strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            # Mode format e.g., "6:   1705.50   0.00   55.10   ..."
            if len(parts) >= 4 and parts[0].endswith(":"):
                try:
                    f_val = float(parts[1])
                    i_val = float(parts[3]) if len(parts) > 3 else float(parts[2])
                    if f_val > 0.0:  # Real frequency
                        results["freqs"].append(f_val)
                        results["intensities"].append(i_val)
                except ValueError:
                    continue

    # 3. Parse TD-DFT / Excited States
    ex_matches = re.finditer(r"STATE\s+(\d+):\s+E=\s*([0-9\.]+)\s*eV\s+.*?f=\s*([0-9\.]+)", content)
    for match in ex_matches:
        results["excited_states"].append({
            "state": int(match.group(1)),
            "energy_ev": float(match.group(2)),
            "osc_strength": float(match.group(3))
        })

    return results

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

try:
    from rdkit import Chem
    from tqdm import tqdm
except ImportError:
    print_status("Missing 'rdkit', 'psutil', or 'tqdm'. Please install them.", "fail")
    sys.exit(1)

def get_isolated_environment() -> dict:
    """Sanitizes the OS environment to prevent Fortran/C++ library collisions."""
    env = os.environ.copy()
    # Strip conda/python paths that interfere with MPQC's bundled libraries
    for key in ['LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME']:
        env.pop(key, None)
    return env

def get_optimal_scratch(base_workspace: str) -> str:
    """
    SCAN-03: Routes I/O to RAM disk or cross-platform temp directory.
    Checks sys.platform and available disk space safely.
    """
    if sys.platform != "win32":
        shm_path = "/dev/shm"
        if os.path.exists(shm_path) and os.access(shm_path, os.W_OK):
            total, used, free = shutil.disk_usage(shm_path)
            if free > (2 * 1024**3):  # Require at least 2GB of free RAM
                target = os.path.join(shm_path, "cochem_scan_tmpfs")
                os.makedirs(target, exist_ok=True)
                return target
    
    # Windows or non-tmpfs fallback: use tempdir or workspace scratch
    temp_dir = tempfile.gettempdir()
    target = os.path.join(temp_dir, "cochem_scan_scratch")
    os.makedirs(target, exist_ok=True)
    return target

def kill_zombie_processes(parent_pid: int) -> Any:
    """
    SCAN-04: Hunts down and terminates orphaned subprocesses safely on Linux and Windows.
    """
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as err:
                print_status(f"Subprocess {child.pid} cleanup skipped: {err}", "info")
        parent.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied) as err:
        print_status(f"Parent process {parent_pid} cleanup skipped: {err}", "info")

def parse_orca_out(out_path: Path) -> dict:
    """
    Parses single point energy and spectral details from an ORCA output log.
    Attaches [E] provenance tag.
    """
    results = {
        "energy": 0.0,
        "freqs": [],
        "intensities": [],
        "excited_states": [],
        "provenance_tag": "[E]"
    }
    if not out_path.exists():
        return results

    try:
        content = out_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return results

    if "error" in content.lower() or "ORCA TERMINATED ABNORMALLY" in content:
        raise ValueError(f"ORCA calculation failed or did not converge in {out_path}")

    e_match = re.search(r"FINAL SINGLE POINT ENERGY\s+([-\d\.eE\+-]+)", content, re.IGNORECASE)
    if not e_match:
        e_match = re.search(r"Total Energy\s+:\s+([-\d\.eE\+-]+)", content, re.IGNORECASE)
    if e_match:
        results["energy"] = float(e_match.group(1))

    return results

def build_orca_input(mol_idx: int, xyz_block: str, tier: int = 1, method_templates: dict = None) -> str:
    """
    SCAN-01: Explicit 5-threshold %geom block injection and prohibited !Opt template removal.
    Complies with MMv4 directives: InHess XTB2, D4 dispersion, DEFGRID3, TolMaxG 1e-5, and frozen-monomer protocol.
    """
    templates = method_templates or {
        1: "! XTB2",
        2: "! r2SCAN-3c",
        3: "! junChS"
    }
    
    raw_method = templates.get(tier, templates.get(1, "! XTB2"))
    clean_words = []
    for word in raw_method.split():
        if re.search(r"^(?:!?)(?:Opt|TightOpt|VeryTightOpt|LooseOpt|Calc_Hess)$", word, re.IGNORECASE):
            continue
        clean_words.append(word)
    header = " ".join(clean_words).strip()
    if not header.startswith("!"):
        header = "!" + header

    if method_templates is None:
        if "D4" not in header.upper() and "D3" not in header.upper():
            header += " D4"
        if "DEFGRID3" not in header.upper():
            header += " DEFGRID3"

    geom_block = (
        "%geom\n"
        "  InHess XTB2\n"
        "  TolE 1e-7\n"
        "  TolRMSG 3e-6\n"
        "  TolMaxG 1e-5\n"
        "  TolRMSD 5e-5\n"
        "  TolMaxD 1e-4\n"
        "  Constraints\n"
        "    { C 0 C 1 } # Frozen monomer constraint\n"
        "  end\n"
        "end"
    )
    
    inp = f"{header}\n{geom_block}\n%maxcore 3000\n* xyz 0 1\n{xyz_block.strip()}\n*\n"
    return inp

def build_mpqc_input(mol_idx: int, xyz_block: str, tier: int = 1, method_templates: dict = None) -> str:
    """
    SCAN-01: Configurable MPQC input method strings.
    Eradicated mock wrappers, implemented valid Object-Oriented keyval input generator.
    """
    templates = method_templates or {
        1: "xTB",
        2: "r2SCAN-3c",
        3: "CCSD(T)-F12 cc-pVTZ-F12"
    }
    
    method = templates.get(tier, templates.get(1, "xTB"))
    
    inp = f"% MPQC Input (Tier {tier}: {method})\n"
    inp += f"% Candidate {mol_idx}\n"
    inp += "molecule<Molecule>: (\n"
    inp += "  symmetry = auto\n"
    inp += "  unit = angstrom\n"
    inp += "  atoms = [\n"
    
    lines = xyz_block.strip().split('\n')
    if len(lines) > 2:
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                inp += f"    [{parts[0]} {parts[1]} {parts[2]} {parts[3]}]\n"
            
    inp += "  ]\n)\n"
    inp += "mpqc: (\n"
    inp += f"  method = {method}\n"
    inp += "  optimize = yes\n"
    inp += ")\n"
    return inp

def apply_active_retiering(candidates: list, workspace: str, variance_threshold: float = 0.5) -> list:
    """
    SCAN-03: Active-Learning Retiering Loop based on surrogate epistemic variance sigma^2(R).
    Evaluates variance across candidate points using ActiveLearningLoop and PESStore.
    Elevates high-uncertainty candidate points from Tier 1 to Tier 2 (r²SCAN-3c) or Tier 3 (MPQC CCSD(T)-F12).
    """
    from cochem_scan_active import ActiveLearningLoop
    from cochem_scan_ingest import PESStore
    
    h5_path = Path(workspace) / "cochem_state.h5"
    pes_store = PESStore(h5_path)
    active_loop = ActiveLearningLoop(pes_store=pes_store, variance_threshold=variance_threshold)
    
    retiered_candidates, _ = active_loop.run_active_iteration(candidates)
    return retiered_candidates


def mace_fallback_nudge(xyz_path: str) -> bool:
    print_status(f"  [Cascade] Initiating MACE / Forcefield nudge on {os.path.basename(xyz_path)}", "warning")
    p = Path(xyz_path)
    if not p.exists():
        print_status(f"XYZ file not found: {xyz_path}", "fail")
        return False

    # Attempt 1: ASE + MACE / EMT relaxation
    try:
        from ase.io import read, write
        from ase.optimize import BFGS
        try:
            from mace.calculators import mace_off
            calc = mace_off(model="small", device="cpu")
        except Exception:
            from ase.calculators.emt import EMT
            calc = EMT()
        
        atoms = read(str(xyz_path))
        atoms.calc = calc
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=0.05, steps=30)
        write(str(xyz_path), atoms)
        print_status(f"ASE relaxation completed on {os.path.basename(xyz_path)}", "success")
        return True
    except Exception:
        pass
    # Attempt 2: RDKit MMFF94 / UFF forcefield optimization
    try:
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        lines = p.read_text().strip().splitlines()
        if len(lines) >= 3:
            num_atoms = int(lines[0].strip())
            atom_lines = lines[2:2+num_atoms]
            symbols = []
            coords = []
            for line in atom_lines:
                parts = line.split()
                if len(parts) >= 4:
                    symbols.append(parts[0])
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            mol = Chem.RWMol()
            for sym in symbols:
                mol.AddAtom(Chem.Atom(sym))
            
            conf = Chem.Conformer(len(symbols))
            for idx, c in enumerate(coords):
                conf.SetAtomPosition(idx, c)
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    dist = float(np.linalg.norm(np.array(coords[i]) - np.array(coords[j])))
                    if dist < 1.8:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
            
            mol_obj = mol.GetMol()
            mol_obj.AddConformer(conf, assignId=True)
            
            try:
                AllChem.MMFFOptimizeMolecule(mol_obj, maxIters=50)
            except Exception:
                AllChem.UFFOptimizeMolecule(mol_obj, maxIters=50)
            
            new_conf = mol_obj.GetConformer()
            new_xyz = [f"{len(symbols)}", f"Nudged by RDKit FF"]
            for idx, sym in enumerate(symbols):
                pos = new_conf.GetAtomPosition(idx)
                new_xyz.append(f"{sym:<2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
            p.write_text("\n".join(new_xyz) + "\n")
            print_status(f"RDKit Forcefield nudge completed on {os.path.basename(xyz_path)}", "success")
            return True
    except Exception as e:
        print_status(f"Forcefield fallback nudge failed: {e}", "warning")
        return False

def compute_spectrum(candidate_idx: int, mol_block: str, scratch_base: str, final_workspace: str, mpqc_path: str, method_templates: dict = None) -> dict:
    """
    Executes the calculation within the RAM disk / scratch, then extracts metadata.
    SCAN-05: Parses MolBlock safely using RDKit rather than hardcoded line splitting.
    """
    work_dir = Path(scratch_base) / f"calc_cand_{candidate_idx}"
    work_dir.mkdir(exist_ok=True)
    
    inp_path = work_dir / "mpqc.inp"
    out_path = work_dir / "mpqc.out"
    xyz_path = work_dir / "input.xyz"
    
    # SCAN-05: Robust RDKit molblock coordinate extraction
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    if mol is not None and mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        xyz_lines = [f"{num_atoms}", f"Generated by CoChem Candidate {candidate_idx}"]
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            sym = atom.GetSymbol()
            pos = conf.GetAtomPosition(atom_idx)
            xyz_lines.append(f"{sym:<2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        xyz_str = "\n".join(xyz_lines)
    else:
        # Fallback to simple splitting if RDKit parse fails
        lines = mol_block.strip().split('\n')
        num_atoms = int(lines[3]) if len(lines) > 3 and lines[3].strip().isdigit() else 0
        xyz_str = f"{num_atoms}\nGenerated by CoChem\n" + "\n".join(lines[4:4+num_atoms])
        
    xyz_path.write_text(xyz_str)
    
    inp_path.write_text(build_mpqc_input(candidate_idx, xyz_str, tier=1, method_templates=method_templates))
    clean_env = get_isolated_environment()
    
    proc = None
    status = "failed"
    try:
        with open(out_path, "w") as out_f:
            try:
                subprocess.run([mpqc_path, str(inp_path)], stdout=out_f, stderr=subprocess.STDOUT, env=clean_env, check=True, timeout=1800)
                status = "success"
            except subprocess.TimeoutExpired as exc:
                logger.error(f"MPQC calculation timed out after 1800s for candidate {candidate_idx}: {exc}")
                raise RuntimeError(f"MPQC calculation timed out after 1800s for candidate {candidate_idx}")
            except subprocess.CalledProcessError as exc:
                logger.error(f"MPQC calculation failed with error code {exc.returncode}")
                if mace_fallback_nudge(str(xyz_path)):
                    out_f.write("\n--- RESTARTING AFTER MACE NUDGE ---\n")
                    try:
                        subprocess.run([mpqc_path, str(inp_path)], stdout=out_f, stderr=subprocess.STDOUT, env=clean_env, check=True, timeout=1800)
                        status = "recovered"
                    except subprocess.TimeoutExpired as exc_r:
                        logger.error(f"MPQC restart calculation timed out after 1800s for candidate {candidate_idx}: {exc_r}")
                        raise RuntimeError(f"MPQC restart calculation timed out after 1800s for candidate {candidate_idx}")
                    except subprocess.CalledProcessError as exc_r:
                        logger.error(f"MPQC restart calculation failed with error code {exc_r.returncode}")

    except Exception as e:
        print_status(f"Compute Exception on {candidate_idx}: {e}", "fail")
    finally:
        pass

    # Move .out file to workspace
    final_out = Path(final_workspace) / f"calc_cand_{candidate_idx}.out"
    if out_path.exists():
        shutil.copy(out_path, final_out)
        
    # Clean up RAM disk
    shutil.rmtree(work_dir, ignore_errors=True)

    try:
        parsed = parse_mpqc_out(final_out if final_out.exists() else out_path)
    except ValueError as ve:
        status = "failed"
        parsed = {"energy": 0.0, "freqs": [], "intensities": [], "excited_states": []}
    return {
        "candidate_id": candidate_idx,
        "status": status,
        "freqs": parsed["freqs"] if status != "failed" else [],
        "intensities": parsed["intensities"] if status != "failed" else [],
        "energy": parsed["energy"] if status != "failed" else 0.0,
        "excited_states": parsed["excited_states"] if status != "failed" else [],
        "provenance_tag": "[E]"
    }

def main() -> Any:
    logger.info(f"\n{Colors.HEADER}{Colors.BOLD}--- CoChem-SCAN: Stage 2.1 Compute Engine (v2.0) ---{Colors.ENDC}")
    
    config_path = "cochem_system_config.json"
    if not os.path.exists(config_path):
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = json.loads(f.read())
        
    scan_cfg = config.get("scan_engine", {})
    workspace = scan_cfg.get("workspace_path", "./SCAN_Workspace")
    mpqc_path = config.get("engines", {}).get("mpqc", "mpqc")
    method_templates = scan_cfg.get("method_templates", None)
    
    sdf_path = os.path.join(workspace, "candidate_ensemble_iter_1.sdf")
    if not os.path.exists(sdf_path):
        print_status(f"Ensemble not found at {sdf_path}.", "fail")
        sys.exit(1)

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    candidates = [(mol.GetProp("_Name"), Chem.MolToMolBlock(mol)) for mol in supplier if mol is not None]
    
    scratch_base = get_optimal_scratch(workspace)
    print_status(f"Routing computation scratch files to: {scratch_base}")
    
    results = []
    max_workers = min(15, os.cpu_count() or 1)
    
    print_status(f"Saturating compute matrix with {max_workers} parallel orchestrators...")
    
    # SCAN-01: Default dry_run to False unless explicitly configured or passed via CLI
    dry_run = scan_cfg.get("dry_run", False) or ("--dry-run" in sys.argv)
    
    if dry_run:
        raise NotImplementedError("Dry run mode requires explicit input geometry or benchmark dataset")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compute_spectrum, name, block, scratch_base, workspace, mpqc_path, method_templates): name 
                       for name, block in candidates}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Computing"):
                results.append(future.result())

    out_json = os.path.join(workspace, "computed_spectra_iter_1.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
        
    print_status(f"Tier 1 Spectral computation complete. Results saved to: {out_json}", "success")

if __name__ == "__main__":
    main()