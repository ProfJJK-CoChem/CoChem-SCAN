# **CoChem-SCAN: Massive Parallel Torsional Screening**

## **Overview**

**CoChem-SCAN** is the exploratory module designed for high-throughput, parallel mapping of torsional energy barriers and multidimensional Potential Energy Surfaces (PES) within the CoChem project.

When analyzing floppy molecular ensembles, simple 1D optimization is insufficient. SCAN orchestrates relaxed surface scans using ORCA's OpenMPI backbone. Because running thousands of concurrent quantum calculations can easily crash or freeze a system, SCAN acts as an automated compute governor and process supervisor.

---

## **Scientific & Technical Trade-offs**

* **RAM Disk Scratch Routing:** Quantum calculations generate gigabytes of temporary integral files (`.tmp`). To protect physical SSDs from wear (TBW limits) and accelerate I/O, SCAN routes all intermediate ORCA files directly to the Linux RAM disk (`/dev/shm`). If a molecular matrix exceeds available memory bounds, SCAN calculates estimated scratch space prior to run triggers and skips jobs violating the 10% safety margin.
* **Zombie Process Assassin:** If a calculation is canceled (e.g. Jupyter kernel interrupt) during a 32-core OpenMPI scan, child processes often detach and spin indefinitely, locking CPU threads. SCAN uses process tree-traversal watchdogs to identify and terminate orphaned Fortran and C++ workers.

---

## **File Topology & Core Scripts**

SCAN consists of the following key Python scripts:

1. **[cochem_setup_scan.py](file:///d:/GitHub-Repo/CoChem-SCAN/cochem_setup_scan.py)** (Micro-Silo Integration):
   * Verifies dependencies (`networkx`, `scipy`, `h5py`, `molsym`) and registers the SCAN workspace settings inside the system configuration registry.

2. **[cochem_scan_ingest.py](file:///d:/GitHub-Repo/CoChem-SCAN/cochem_scan_ingest.py)** & **[cochem_scan_construct.py](file:///d:/GitHub-Repo/CoChem-SCAN/cochem_scan_construct.py)**:
   * Coordinates input file parsing and builds coordinate scan grid configurations.

3. **[cochem_scan_structural_generator.py](file:///d:/GitHub-Repo/CoChem-SCAN/cochem_scan_structural_generator.py)**:
   * Generates relaxed candidate conformer coordinates along defined dihedral angles.

4. **[cochem_scan_compute.py](file:///d:/GitHub-Repo/CoChem-SCAN/cochem_scan_compute.py)** (Parallel Compute Watchdog):
   * Dispatches optimization geometries to ORCA, routes temp scratch files to `tmpfs`, and implements the Zombie Process Assassin process reaper.

5. **[cochem_scan_critic.py](file:///d:/GitHub-Repo/CoChem-SCAN/cochem_scan_critic.py)** & **[cochem_scan_report.py](file:///d:/GitHub-Repo/CoChem-SCAN/cochem_scan_report.py)**:
   * Appraises conformer energies, implements Bayesian probability matches against experimental spectra, and generates output metrics.

---

## **Workflow & How to Run**

To execute a parallel torsional grid screening:

1. **Register the SCAN Module**:
   Ensure `cochem_system_config.json` is initialized in the working directory, and run:
   ```bash
   python cochem_setup_scan.py
   ```

2. **Run Torsional Computations & Watchdogs**:
   Start the parallel ORCA dispatcher to process coordinate ensembles:
   ```bash
   python cochem_scan_compute.py
   ```

3. **Compile Bayesian Conformer Fit Reports**:
   Evaluate predicted spectrum traces and compile report files:
   ```bash
   python cochem_scan_report.py
   ```