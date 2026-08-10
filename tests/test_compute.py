"""
Unit tests for CoChem-SCAN SCAN-01 (cochem_scan_compute.py).
Verifies explicit 5-threshold %geom block injection and [E] provenance tagging.
"""

import re
from pathlib import Path
import pytest
from cochem_scan_compute import build_orca_input, parse_orca_out

def test_build_orca_input_5_threshold_geom_block():
    xyz = "O 0.0 0.0 0.0\nH 0.0 0.7 0.6\nH 0.0 -0.7 0.6\n"
    
    # Test Tier 1, Tier 2, Tier 3
    for tier in [1, 2, 3]:
        inp = build_orca_input(1, xyz, tier=tier)
        
        # Must contain explicit %geom block
        assert "%geom" in inp
        assert "TolE 1e-7" in inp
        assert "TolMaxG 1e-5" in inp
        assert "TolRMSG 3e-6" in inp
        assert "TolMaxD 1e-4" in inp
        assert "TolRMSD 5e-5" in inp
        
        # Prohibit default !Opt keywords
        assert "!Opt" not in inp
        assert "! Opt" not in inp
        assert not re.search(r"!\s*Opt\b", inp, re.IGNORECASE)

def test_build_orca_input_strips_passed_opt_templates():
    xyz = "C 0.0 0.0 0.0\n"
    custom_templates = {
        1: "! XTB2 Opt Freq",
        2: "! r2SCAN-3c TightOpt VeryTightOpt LooseOpt Freq"
    }
    inp1 = build_orca_input(1, xyz, tier=1, method_templates=custom_templates)
    assert "%geom" in inp1
    assert "TolE 1e-7" in inp1
    assert "TolMaxG 1e-5" in inp1
    assert not re.search(r"!\s*Opt\b", inp1, re.IGNORECASE)

    inp2 = build_orca_input(1, xyz, tier=2, method_templates=custom_templates)
    header = inp2.splitlines()[0]
    assert "TightOpt" not in header
    assert "VeryTightOpt" not in header
    assert "LooseOpt" not in header
    assert header == "! r2SCAN-3c Freq"

def test_parse_orca_out_provenance_tag(tmp_path):
    out_file = tmp_path / "dummy.out"
    out_file.write_text("FINAL SINGLE POINT ENERGY   -76.4321000\nORCA TERMINATED NORMALLY\n")
    res = parse_orca_out(out_file)
    assert res["provenance_tag"] == "[E]"
    assert res["energy"] == pytest.approx(-76.4321)

def test_apply_active_retiering(tmp_path):
    from cochem_scan_compute import apply_active_retiering
    candidates = [
        {"candidate_id": "c1", "coords": [0.0, 0.0], "tier": 1},
        {"candidate_id": "c2", "coords": [10.0, 10.0], "tier": 1},
    ]
    retiered = apply_active_retiering(candidates, str(tmp_path), variance_threshold=0.3)
    assert len(retiered) == 2
    assert (tmp_path / "cochem_state.h5").exists()

