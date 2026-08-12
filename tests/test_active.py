import logging
logger = logging.getLogger(__name__)
"""
Unit tests for CoChem-SCAN SCAN-03 Active Learning & Dynamic Retiering Loop (cochem_scan_active.py).
"""

import os
import tempfile
import pytest
import numpy as np
import h5py
from cochem_scan_ingest import PESStore
from cochem_scan_active import ActiveLearningLoop

def test_active_learning_epistemic_variance() -> None:
    loop = ActiveLearningLoop(variance_threshold=0.5)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    # Test ensemble variance
    ensemble_preds = np.array([
        [-100.0, -99.0, -98.0],
        [-100.1, -98.8, -99.0],
        [-100.2, -99.2, -98.5],
    ])
    vars_ens = loop.compute_epistemic_variance(coords, ensemble_predictions=ensemble_preds)
    assert len(vars_ens) == 3
    assert np.all(vars_ens >= 0.0)

def test_retier_candidates_promotion() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "cochem_state.h5")
        store = PESStore(h5_path)
        loop = ActiveLearningLoop(pes_store=store, variance_threshold=0.5)

        candidates = [
            {"candidate_id": "c1", "coords": [0.0, 0.0], "energy": -500.0, "tier": 1},
            {"candidate_id": "c2", "coords": [1.0, 1.0], "energy": -490.0, "tier": 1},
            {"candidate_id": "c3", "coords": [5.0, 5.0], "energy": -480.0, "tier": 2},
        ]
        variances = np.array([0.1, 0.8, 0.6]) # c2 and c3 exceed 0.5 threshold

        retiered = loop.retier_candidates(candidates, variances)
        
        # c1 stays at Tier 1
        assert retiered[0]["tier"] == 1
        assert retiered[0]["retier_flag"] is False

        # c2 promoted from Tier 1 -> Tier 2
        assert retiered[1]["tier"] == 2
        assert retiered[1]["retier_flag"] is True
        assert retiered[1]["provenance_tag"] == "[E]"

        # c3 promoted from Tier 2 -> Tier 3
        assert retiered[2]["tier"] == 3
        assert retiered[2]["retier_flag"] is True

        # Check HDF5 storage of variance and retier_flags
        with h5py.File(h5_path, "r") as f:
            assert "pes/uncertainty/variance" in f
            assert "pes/uncertainty/retier_flags" in f
            flags = np.array(f["pes/uncertainty/retier_flags"])
            assert np.array_equal(flags, np.array([0, 1, 1], dtype=np.uint8))

def test_full_active_iteration() -> None:
    loop = ActiveLearningLoop(variance_threshold=0.3)
    candidates = [
        {"candidate_id": "c1", "coords": [0.0, 0.0], "tier": 1},
        {"candidate_id": "c2", "coords": [10.0, 10.0], "tier": 1},
    ]
    retiered, variances = loop.run_active_iteration(candidates)
    assert len(retiered) == 2
    assert len(variances) == 2

def test_active_learning_3d_coordinates_flattening() -> None:
    candidates = [
        {"candidate_id": "c1", "coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), "energy": -10.0, "tier": 1},
        {"candidate_id": "c2", "coords": np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]]), "energy": -12.0, "tier": 1}
    ]
    loop = ActiveLearningLoop(variance_threshold=0.3)
    retiered, vars_calc = loop.run_active_iteration(candidates)
    assert vars_calc.ndim == 1
    assert len(vars_calc) == 2
    assert isinstance(retiered[0]["epistemic_variance"], float)
