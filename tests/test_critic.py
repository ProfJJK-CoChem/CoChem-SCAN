"""
Unit tests for CoChem-SCAN Stage 2.2 Spectral Critic and Logic Gate.
"""

import os
import tempfile
import pytest
import numpy as np
from cochem_scan_critic import vectorized_broadening, check_dead_zone_violations, calculate_pareto_front, get_method_scaling_factor

def test_vectorized_broadening_normalization():
    # Step = 1.0 cm-1 includes 1500.0 exactly
    exp_grid = np.linspace(1000, 2000, 1001)
    freqs = [1500.0]
    intensities = [85.0]
    
    spec = vectorized_broadening(freqs, intensities, exp_grid, hwhm=10.0)
    
    # Peak amplitude at f=1500 should equal input intensity 85.0
    peak_val = np.max(spec)
    assert np.isclose(peak_val, 85.0, rtol=1e-3)

def test_check_dead_zone_violations():
    freqs = [1700.0, 3300.0]
    intensities = [50.0, 10.0]
    dead_zones = [(1650.0, 1750.0)]
    
    # No LAM -> Should detect violation at 1700 cm-1
    v = check_dead_zone_violations(freqs, intensities, dead_zones, has_lam=False)
    assert len(v) == 1
    
    # Low frequency line with LAM -> Violation ignored if < 500 cm-1
    v_lam = check_dead_zone_violations([400.0], [50.0], [(350.0, 450.0)], has_lam=True)
    assert len(v_lam) == 0

def test_pareto_front_calculation():
    candidates = [
        {"candidate_id": "c1", "energy": -500.0, "residual": 10.0},
        {"candidate_id": "c2", "energy": -490.0, "residual": 5.0},
        {"candidate_id": "c3", "energy": -480.0, "residual": 20.0},  # Dominated by c1 & c2
    ]
    pareto = calculate_pareto_front(candidates)
    ids = {c["candidate_id"] for c in pareto}
    assert "c1" in ids and "c2" in ids
    assert "c3" not in ids

def test_method_scaling_factor():
    assert get_method_scaling_factor("xtb2") == 1.000
    assert get_method_scaling_factor("r2scan-3c") == 0.985
    assert get_method_scaling_factor("dlpno-ccsd(t)") == 0.957
