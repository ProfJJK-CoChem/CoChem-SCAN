#!/usr/bin/env python3
"""
CoChem Setup Phase: SCAN Module Integration
Registers the CoChem-SCAN (Spectroscopic Candidate Analysis Network) into the master registry.
"""

import os
import sys
import json
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

def check_dependency(module_name: str) -> bool:
    """Checks if a python module is available in the current silo."""
    return importlib.util.find_spec(module_name) is not None

def integrate_scan_module():
    print(f"\n{Colors.HEADER}{Colors.BOLD}--- CoChem-SCAN: Micro-Silo Integration ---{Colors.ENDC}")
    
    config_path = "cochem_system_config.json"
    
    # 1. Authority Check
    if not os.path.exists(config_path):
        print_status(f"Master registry '{config_path}' not found. Run Stage 0 setup first.", "fail")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        try:
            config = json.load(f)
            print_status("Loaded master cochem_system_config.json", "success")
        except json.JSONDecodeError:
            print_status("JSON Decode Error in config.", "fail")
            sys.exit(1)

    # 2. Dependency Audit
    required_deps = ["networkx", "scipy", "h5py", "molsym"]
    missing_deps = []
    
    print_status("Auditing SCAN dependencies...")
    for dep in required_deps:
        if check_dependency(dep):
            print_status(f"Dependency '{dep}' found.", "success")
        else:
            print_status(f"Dependency '{dep}' missing.", "warning")
            missing_deps.append(dep)
            
    if missing_deps:
        print_status(f"Please install missing dependencies: pip install {' '.join(missing_deps)}", "fail")
        sys.exit(1)

    # 3. Provisioning Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scan_workspace = os.path.join(base_dir, "SCAN_Workspace")
    scan_registry_dir = os.path.join(scan_workspace, "Registry")
    scan_logs_dir = os.path.join(scan_workspace, "Logs")
    
    for directory in [scan_workspace, scan_registry_dir, scan_logs_dir]:
        os.makedirs(directory, exist_ok=True)
    print_status(f"Provisioned SCAN directories at: {scan_workspace}", "success")

    # 4. Registry Update
    scan_block = {
        "status": "online",
        "workspace_path": scan_workspace,
        "registry_db_path": os.path.join(scan_registry_dir, "scan_state.sqlite3"),
        "fragment_library_path": os.path.join(scan_workspace, "fragment_library.json"),
        "dependencies_verified": True
    }
    
    config["scan_engine"] = scan_block
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print_status("SCAN Engine successfully registered in cochem_system_config.json", "success")
    print(f"{Colors.HEADER}{Colors.BOLD}-------------------------------------------{Colors.ENDC}\n")

if __name__ == "__main__":
    integrate_scan_module()