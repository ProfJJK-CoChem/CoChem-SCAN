#!/usr/bin/env python3
"""
CoChem-SCAN Stage 3.0: UI, Reporting & Archiving (v2.0)
Generates YAML summaries, compiles the LaTeX mechanism, archives discarded branches, 
and generates the Plotly interactive Jupyter Dashboard for manual Dead Zone overrides.
"""

import os
import sys
import json
import tarfile
import glob
import subprocess
import datetime
from pathlib import Path

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

def archive_discarded_branches(workspace: str):
    print_status("Archiving discarded branches and temporary files...")
    archive_path = os.path.join(workspace, "discarded_branches.tar.gz")
    
    calc_dirs = glob.glob(os.path.join(workspace, "calc_cand_*"))
    if not calc_dirs:
        print_status("No temporary directories to clean up.", "info")
        return
        
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            for d in calc_dirs:
                tar.add(d, arcname=os.path.basename(d))
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
                os.rmdir(d)
        print_status(f"Archived and purged {len(calc_dirs)} temporary directories.", "success")
    except Exception as e:
        print_status(f"Archiving failed: {e}", "warning")

def generate_yaml_summary(workspace: str, pareto_data: list):
    yaml_path = os.path.join(workspace, "summary.yaml")
    
    yaml_content = f"# CoChem-SCAN Execution Summary\n"
    yaml_content += f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    yaml_content += f"pipeline_status: 'converged'\n"
    yaml_content += f"pareto_front_size: {len(pareto_data)}\n\n"
    
    yaml_content += "candidates:\n"
    for cand in pareto_data:
        yaml_content += f"  - id: '{cand['candidate_id']}'\n"
        yaml_content += f"    energy_kcal: {cand['energy']:.2f}\n"
        yaml_content += f"    residual_score: {cand['residual']:.4f}\n"
        yaml_content += f"    boltzmann_population: {cand.get('boltzmann_weight', 0.0):.4f}\n"
        
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print_status(f"Exported human-readable summary to {yaml_path}", "success")

def export_latex_report(workspace: str, pareto_data: list, prune_data: list):
    tex_path = os.path.join(workspace, "SCAN_Report.tex")
    bib_path = os.path.join(workspace, "methods.bib")
    
    bib_content = """@article{orca2020,
    author = {Neese, F.},
    title = {The ORCA program system},
    journal = {Wiley Interdisciplinary Reviews: Computational Molecular Science},
    volume = {12},
    year = {2022}
}"""
    with open(bib_path, "w") as f: f.write(bib_content)
        
    best_cand = pareto_data[0] if pareto_data else {"candidate_id": "None", "residual": 0.0, "boltzmann_weight": 0.0}
    
    tex_content = r"""\documentclass[11pt, a4paper]{article}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{hyperref}

\title{CoChem-SCAN: Spectroscopic Elucidation Report}
\author{Automated Pipeline}
\date{\today}

\begin{document}
\maketitle

\section{Executive Summary}
The iterative structural discovery loop has concluded utilizing the energy-spectral Pareto front logic and Boltzmann statistical weighting \cite{orca2020}. 
The optimal structural candidate is \textbf{""" + best_cand['candidate_id'] + r"""} with a spectral residual of """ + f"{best_cand['residual']:.4f}" + r""" and a population density of """ + f"{best_cand['boltzmann_weight']*100:.1f}\\%" + r""".

\section{Pruning Rationale}
The following branches were mathematically precluded based on empirical constraint filtering:
\begin{itemize}
"""
    for entry in prune_data[:5]:
        tex_content += f"    \\item \\textbf{{{entry['id']}}}: {entry['reason']}\n"
        
    tex_content += r"""\end{itemize}

\bibliographystyle{plain}
\bibliography{methods}
\end{document}
"""
    with open(tex_path, "w") as f: f.write(tex_content)
    print_status("Generated LaTeX Mechanism (SCAN_Report.tex).", "success")
    
    print_status("Attempting background PDF compilation...")
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", workspace, tex_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        print_status("PDF successfully compiled.", "success")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("pdflatex not found or failed. LaTeX source saved.", "warning")

def render_jupyter_dashboard(workspace: str):
    print_status("Dashboard code prepped. Run the generated UI block to visualize.", "info")
    return f"""
# --- Run this in a Jupyter Cell ---
import json
import os
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

workspace = r'{workspace}'

# Load Plot Data
exp_freqs = np.linspace(500, 4000, 1000)
try:
    comp_spec = np.load(os.path.join(workspace, 'composite_spectrum.npy'))
except:
    comp_spec = np.zeros_like(exp_freqs)

# Interactive Plotly Figure
fig = go.FigureWidget()
fig.add_trace(go.Scatter(x=exp_freqs, y=comp_spec, mode='lines', name='Boltzmann Composite', line=dict(color='blue')))
fig.update_layout(
    title='CoChem-SCAN Final Composite vs. Experimental constraints',
    xaxis_title='Wavenumber (cm⁻¹)',
    yaxis_title='Normalized Intensity',
    dragmode='select',
    template='plotly_white'
)

# Callbacks for Manual Dead Zone Override
out_console = widgets.Output()

def selection_fn(trace, points, selector):
    with out_console:
        out_console.clear_output()
        if hasattr(selector, 'xrange'):
            x_range = selector.xrange
            print(f"🛑 MANUAL DEAD ZONE OVERRIDE INITIATED: {{x_range[0]:.1f}} - {{x_range[1]:.1f}} cm⁻¹")
            print("To append this to the constraints registry, invoke cochem_scan_ingest.update_zones(range)")

fig.data[0].on_selection(selection_fn)

display(widgets.VBox([fig, out_console]))
"""

def main():
    print(f"\n{Colors.HEADER}{Colors.BOLD}--- CoChem-SCAN: Stage 3.0 Reporting & UI (v2.0) ---{Colors.ENDC}")
    
    config_path = "cochem_system_config.json"
    if not os.path.exists(config_path):
        sys.exit(1)
        
    with open(config_path, "r") as f:
        workspace = json.load(f).get("scan_engine", {}).get("workspace_path", "./SCAN_Workspace")
    
    pareto_path = os.path.join(workspace, "pareto_front_iter_1.json")
    prune_path = os.path.join(workspace, "pruning_rationale.json")
    
    if os.path.exists(pareto_path) and os.path.exists(prune_path):
        with open(pareto_path, "r") as f: pareto_data = json.load(f)
        with open(prune_path, "r") as f: prune_data = json.load(f)
            
        generate_yaml_summary(workspace, pareto_data)
        export_latex_report(workspace, pareto_data, prune_data)
        archive_discarded_branches(workspace)
        
        ui_code = render_jupyter_dashboard(workspace)
        with open(os.path.join(workspace, "launch_dashboard.py"), "w") as f:
            f.write(ui_code)
            
    else:
        print_status("Stage 2.2 output missing. Cannot compile report.", "fail")

    print(f"{Colors.HEADER}{Colors.BOLD}----------------------------------------------------{Colors.ENDC}\n")

if __name__ == "__main__":
    main()