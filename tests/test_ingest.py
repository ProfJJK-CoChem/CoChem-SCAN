"""
Unit tests for CoChem-SCAN Stage 1.0 Ingestion and Dead Zone Mapping.
"""

import os
import tempfile
import pytest
import numpy as np
from cochem_scan_ingest import map_dead_zones, set_user_constraints, pre_flight_check

def test_pre_flight_check():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert pre_flight_check(tmpdir, min_gb_required=0.001)

def test_map_dead_zones():
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "test_spectrum.txt")
        freqs = np.linspace(500, 4000, 100)
        # Create a spectrum with a clear dip (dead zone) in the middle
        intensities = np.full(100, 10.0)
        intensities[40:60] = 0.001
        
        np.savetxt(spec_path, np.column_stack((freqs, intensities)))
        
        dz_path = map_dead_zones(spec_path, tmpdir)
        assert os.path.exists(dz_path)
        assert os.path.exists(os.path.join(tmpdir, "exp_freqs.npy"))

def test_missing_spectrum_file_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = os.path.join(tmpdir, "non_existent.txt")
        with pytest.raises(FileNotFoundError):
            map_dead_zones(missing_path, tmpdir)
