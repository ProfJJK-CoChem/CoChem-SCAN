import logging
logger = logging.getLogger(__name__)
"""
Unit tests for CoChem-SCAN Stage 2.0 Structural Generator.
"""

import os
import tempfile
import pytest
from cochem_scan_construct import build_fragment_library, isolate_and_embed, generate_candidate_ensemble

def test_build_fragment_library() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lib_path = build_fragment_library(tmpdir)
        assert os.path.exists(lib_path)

def test_isolate_and_embed_benzene() -> None:
    res = isolate_and_embed("c1ccccc1", "Benzene")
    assert res["status"] == "success"
    assert "mol_block" in res

def test_generate_candidate_ensemble() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        seed_path = os.path.join(tmpdir, "seed.sdf")
        lib_path = build_fragment_library(tmpdir)
        
        generate_candidate_ensemble(seed_path, ["carbonyl_stretch_1700"], lib_path, tmpdir)
        
        out_sdf = os.path.join(tmpdir, "candidate_ensemble_iter_1.sdf")
        assert os.path.exists(out_sdf)

def test_interpolate_pes_grid() -> None:
    import numpy as np
    from cochem_scan_construct import interpolate_pes_grid
    
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    grid_coords = [x, y]
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid_energies = X**2 + Y**2
    
    query = np.array([[0.5, 0.5]])
    val = interpolate_pes_grid(grid_coords, grid_energies, query)
    assert np.isclose(val[0], 0.5, atol=1e-2)

def test_interpolate_pes_grid_with_pes_store() -> None:
    import numpy as np
    from cochem_scan_ingest import PESStore
    from cochem_scan_construct import interpolate_pes_grid
    
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "cochem_state.h5")
        store = PESStore(h5_path)
        
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        X, Y = np.meshgrid(x, y, indexing='ij')
        coords = np.column_stack([X.ravel(), Y.ravel()])
        energies = (X**2 + Y**2).ravel()
        
        store.save_grid_points(coordinates=coords, energies=energies)
        
        query = np.array([[0.5, 0.5]])
        val = interpolate_pes_grid(query_points=query, pes_store=store)
        assert np.isclose(val[0], 0.5, atol=1e-2)

