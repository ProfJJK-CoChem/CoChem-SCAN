import logging
logger = logging.getLogger(__name__)
"""
Unit tests for CoChem-SCAN Stage 1.0 Ingestion, Dead Zone Mapping, and PESStore HDF5 (§8C).
"""

import os
import tempfile
from typing import Any
import pytest
import h5py
import numpy as np
from cochem_scan_ingest import map_dead_zones, set_user_constraints, pre_flight_check, PESStore

def test_pre_flight_check() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        assert pre_flight_check(tmpdir, min_gb_required=0.001)

def test_map_dead_zones() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "test_spectrum.txt")
        freqs = np.linspace(500, 4000, 100)
        intensities = np.full(100, 10.0)
        intensities[40:60] = 0.001
        
        np.savetxt(spec_path, np.column_stack((freqs, intensities)))
        
        dz_path = map_dead_zones(spec_path, tmpdir)
        assert os.path.exists(dz_path)
        
        # Verify HDF5 PESStore archive
        h5_path = os.path.join(tmpdir, "cochem_state.h5")
        assert os.path.exists(h5_path)
        with h5py.File(h5_path, "r") as f:
            assert "pes/grid/exp_freqs" in f
            assert "pes/grid/exp_intensities" in f
            dset = f["pes/grid/exp_freqs"]
            assert dset.compression == "gzip"
            assert dset.compression_opts == 4
            assert dset.shuffle is True
            assert dset.fletcher32 is True
            assert dset.attrs["provenance_tag"] == "[M]"

def test_missing_spectrum_file_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = os.path.join(tmpdir, "non_existent.txt")
        with pytest.raises(FileNotFoundError):
            map_dead_zones(missing_path, tmpdir)

def test_pes_store_full_interface() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "cochem_state.h5")
        store = PESStore(h5_path)
        store.init_store()
        
        coords = np.random.randn(600, 3)
        energies = np.random.randn(600)
        variances = np.random.rand(600)
        retier_flags = np.zeros(600, dtype=np.uint8)
        retier_flags[::2] = 1

        store.save_grid_points(coords, energies, variances=variances, retier_flags=retier_flags)

        with h5py.File(h5_path, "r") as f:
            assert "pes/grid/coordinates" in f
            assert "pes/fit/energies" in f
            assert "pes/uncertainty/variance" in f
            assert "pes/uncertainty/retier_flags" in f

            # Verify 512-pt chunking for N=600 dataset
            coord_dset = f["pes/grid/coordinates"]
            assert coord_dset.chunks[0] == 512
            assert coord_dset.compression == "gzip"
            assert coord_dset.compression_opts == 4
            assert coord_dset.shuffle is True
            assert coord_dset.fletcher32 is True

            # Verify provenance tags
            assert f["pes"].attrs["provenance_tag"] == "[D]"
            assert f["pes/fit/energies"].attrs["provenance_tag"] == "[E]"
            assert f["pes/uncertainty/variance"].attrs["provenance_tag"] == "[E]"

def test_pes_store_empty_array_saving() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "cochem_state.h5")
        store = PESStore(h5_path)
        empty = np.array([])
        store.save_grid_points(empty, empty)
        data = store.load_pes_data()
        assert data["coordinates"].size == 0
        assert data["energies"].size == 0

def test_pes_store_dimension_validation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "cochem_state.h5")
        store = PESStore(h5_path)
        coords = np.random.randn(10, 3)
        energies = np.random.randn(5)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            store.save_grid_points(coords, energies)

def test_pes_store_concurrency_safety() -> None:
    from concurrent.futures import ThreadPoolExecutor
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "cochem_state.h5")
        store = PESStore(h5_path)
        def worker_write(i) -> Any:
            c = np.random.randn(5, 3)
            e = np.random.randn(5)
            store.save_grid_points(c, e)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_write, i) for i in range(12)]
            for f in futures:
                f.result()
        data = store.load_pes_data()
        assert "coordinates" in data
