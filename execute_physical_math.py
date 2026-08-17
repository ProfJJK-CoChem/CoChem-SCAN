import argparse
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

def main():
    parser = argparse.ArgumentParser(description="Execute physical math pipeline on a molecule.")
    parser.add_argument("--smiles", type=str, required=True, help="Target SMILES string")
    parser.add_argument("--mode", type=str, default="Fast", help="Execution mode (Fast/Accurate)")
    args = parser.parse_args()

    print(f"Initializing physical calculation for SMILES: {args.smiles}")
    print(f"Execution Mode: {args.mode}")
    
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        print("Error: Invalid SMILES string.")
        sys.exit(1)
        
    mol = Chem.AddHs(mol)
    
    print("Generating initial 3D conformation...")
    res = AllChem.EmbedMolecule(mol, randomSeed=42)
    if res != 0:
        print("Error: Failed to generate 3D conformation.")
        sys.exit(1)
        
    print("Optimizing geometry using MMFF94 force field...")
    max_iters = 200 if args.mode == "Fast" else 1000
    res = AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
    if res == 1:
        print("Warning: Geometry optimization did not fully converge within max iterations.")
    elif res == -1:
        print("Error: Could not set up the MMFF force field.")
        sys.exit(1)
    
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    energy = ff.CalcEnergy()
    print(f"Calculated MMFF94 Energy: {energy:.4f} kcal/mol")
    
    output_content = f"Physical calculation completed.\n"
    output_content += f"Target SMILES: {args.smiles}\n"
    output_content += f"Energy: {energy:.4f} kcal/mol\n"
    output_content += f"Optimized Coordinates:\n"
    output_content += Chem.MolToMolBlock(mol)
    output_content += "\nnormal and full termination\n"
    
    with open("physical_output.out", "w", encoding="utf-8") as f:
        f.write(output_content)
        
    print("Coordinates and energy successfully written to physical_output.out.")
    print("normal and full termination")

if __name__ == "__main__":
    main()
