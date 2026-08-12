import logging
logger = logging.getLogger(__name__)
"""
Unit tests for CoChem-SCAN Stage 3.0 UI, Reporting & Archiving.
"""

import os
import tempfile
import pytest
from cochem_scan_report import generate_yaml_summary, sanitize_latex, export_latex_report, archive_discarded_branches

def test_sanitize_latex() -> None:
    assert sanitize_latex("SCAN_Cand_1") == "SCAN\\_Cand\\_1"
    assert sanitize_latex("100% & 50#") == "100\\% \\& 50\\#"

def test_generate_yaml_summary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        pareto_data = [{"candidate_id": "SCAN_Cand_0", "energy": -500.0, "residual": 1.234, "boltzmann_weight": 0.95}]
        generate_yaml_summary(tmpdir, pareto_data)
        assert os.path.exists(os.path.join(tmpdir, "summary.yaml"))

def test_archive_discarded_branches() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        calc_dir = os.path.join(tmpdir, "calc_cand_0")
        os.makedirs(calc_dir, exist_ok=True)
        with open(os.path.join(calc_dir, "dummy.txt"), "w") as f:
            f.write("test")
            
        archive_discarded_branches(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "discarded_branches.tar.gz"))
        assert not os.path.exists(calc_dir)
