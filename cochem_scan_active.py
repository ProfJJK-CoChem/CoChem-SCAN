#!/usr/bin/env python3
"""
CoChem-SCAN Stage 2.3: Active Learning & Retiering Loop (v4.0)
Calculates surrogate model epistemic variance sigma^2(R) across candidate grid points (§13.2).
Promotes points with sigma^2(R) > theta_retier to higher budget/method tiers (T1 -> T2 -> T3).
Stores uncertainty and retier flags in PESStore (cochem_state.h5) under /pes/uncertainty.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from cochem_scan_ingest import PESStore

class ActiveLearningLoop:
    """
    Active Learning Loop for Potential Energy Surface sampling (§13.2).
    Evaluates surrogate model epistemic variance sigma^2(R) across candidate grid points
    and retiers high-uncertainty points to higher computational tiers.
    """
    def __init__(self, pes_store: Optional[PESStore] = None, variance_threshold: float = 0.5):
        """
        :param pes_store: PESStore instance for archiving grid, variance, and retier flags
        :param variance_threshold: Epistemic variance threshold (theta_retier) for tier promotion
        """
        self.pes_store = pes_store
        self.variance_threshold = variance_threshold

    def compute_epistemic_variance(self, candidate_coords: np.ndarray, ensemble_predictions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        SCAN-03: Calculates surrogate model epistemic variance sigma^2(R) across candidate grid points.
        
        :param candidate_coords: Array of candidate coordinates [N_points, N_dim]
        :param ensemble_predictions: Optional matrix of model predictions [N_models, N_points]
        :return: Array of epistemic variances sigma^2(R) [N_points]
        """
        candidate_coords = np.asarray(candidate_coords, dtype=np.float64)
        if candidate_coords.ndim == 1:
            candidate_coords = candidate_coords.reshape(-1, 1)

        # If explicit surrogate ensemble predictions are provided: compute variance across models
        if ensemble_predictions is not None:
            ens = np.asarray(ensemble_predictions, dtype=np.float64)
            if ens.ndim == 2:
                return np.var(ens, axis=0)

        # If PESStore contains training grid points, compute RBF distance-based epistemic variance
        if self.pes_store is not None:
            pes_data = self.pes_store.load_pes_data()
            if "coordinates" in pes_data and len(pes_data["coordinates"]) > 0:
                train_coords = pes_data["coordinates"]
                train_flat = train_coords.reshape(train_coords.shape[0], -1)
                cand_flat = candidate_coords.reshape(candidate_coords.shape[0], -1)

                variances = []
                for pt in cand_flat:
                    dists = np.linalg.norm(train_flat - pt, axis=-1)
                    min_dist = np.min(dists)
                    # Variance scales with distance to nearest sampled point: sigma^2(R) = 1 - exp(-min_dist^2 / 2.0)
                    var = 1.0 - np.exp(- (min_dist ** 2) / 2.0)
                    variances.append(var)
                return np.array(variances, dtype=np.float64)

        # Default fallback: coordinate variance relative to centroid
        cand_flat = candidate_coords.reshape(candidate_coords.shape[0], -1)
        centroid = np.mean(cand_flat, axis=0)
        variances = np.sum((cand_flat - centroid) ** 2, axis=-1)
        return np.asarray(variances, dtype=np.float64)

    def retier_candidates(self, candidates: List[Dict], variances: np.ndarray) -> List[Dict]:
        """
        SCAN-03: Promotes points with sigma^2(R) > theta_retier to higher budget/method tiers (T1 -> T2 -> T3).
        
        :param candidates: List of candidate dictionaries (each containing 'tier', 'candidate_id', etc.)
        :param variances: Array of epistemic variances per candidate
        :return: Updated list of candidates with elevated tiers and retier flags
        """
        retiered = []
        retier_flags = []
        
        for cand, var in zip(candidates, variances):
            c = dict(cand)
            current_tier = c.get("tier", 1)
            var_arr = np.asarray(var)
            var_val = float(var_arr.mean()) if var_arr.size > 1 else float(var_arr.item())
            c["epistemic_variance"] = var_val

            if var_val > self.variance_threshold:
                # Promote to next tier up to T3 (Tier 1: xTB, Tier 2: r²SCAN-3c, Tier 3: junChS / MPQC CCSD(T)-F12)
                new_tier = min(current_tier + 1, 3)
                c["tier"] = new_tier
                c["retier_flag"] = True
                c["provenance_tag"] = "[E]"
                retier_flags.append(1)
            else:
                c["retier_flag"] = False
                c["provenance_tag"] = c.get("provenance_tag", "[D]")
                retier_flags.append(0)

            retiered.append(c)

        # Save uncertainty variance and retier flags to HDF5 store if connected
        if self.pes_store is not None:
            coords_list = []
            energies_list = []
            for idx, c in enumerate(candidates):
                coords_list.append(c.get("coords", [idx]))
                energies_list.append(c.get("energy", 0.0))
            
            self.pes_store.save_grid_points(
                coordinates=np.array(coords_list),
                energies=np.array(energies_list),
                variances=np.asarray(variances, dtype=np.float64),
                retier_flags=np.array(retier_flags, dtype=np.uint8)
            )

        return retiered

    def run_active_iteration(self, candidates: List[Dict], ensemble_predictions: Optional[np.ndarray] = None) -> Tuple[List[Dict], np.ndarray]:
        """
        Executes a complete active learning evaluation and retiering step across candidates.
        """
        coords = np.array([c.get("coords", [i]) for i, c in enumerate(candidates)], dtype=np.float64)
        variances = self.compute_epistemic_variance(coords, ensemble_predictions=ensemble_predictions)
        retiered_candidates = self.retier_candidates(candidates, variances)
        return retiered_candidates, variances
