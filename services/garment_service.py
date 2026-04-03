from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Dict, List

import numpy as np
import torch

from ..core.config import settings
from ..models.schemas import (
    BodyMeasurements,
    GarmentInfo,
    GarmentListResponse,
    GarmentMeshRequest,
    GarmentMeshResponse,
)

logger = logging.getLogger(__name__)


class GarmentService:
    """Service for loading the garment catalog and generating garment meshes
    via SMPLicit, with size control through z_cut manipulation."""

    def __init__(self) -> None:
        self._catalog: List[Dict] = []
        self._catalog_by_id: Dict[str, Dict] = {}
        self._smplicit_layer = None
        self._load_catalog()
        self._load_smplicit()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _load_catalog(self) -> None:
        """Load the garment catalog JSON produced by fit_SMPLicit.py."""
        catalog_path = settings.smplicit.garment_catalog_path
        if not catalog_path.exists():
            logger.warning(
                "Garment catalog not found at %s. "
                "Run fit_SMPLicit.py first to generate it. "
                "The garment list will be empty.",
                catalog_path,
            )
            return

        with open(catalog_path, "r") as f:
            self._catalog = json.load(f)

        self._catalog_by_id = {g["id"]: g for g in self._catalog}
        logger.info("Loaded %d garments from catalog", len(self._catalog))

    def _load_smplicit(self) -> None:
        """Load the SMPLicit layer for garment reconstruction."""
        try:
            import warnings
            import SMPLicit as smplicit_pkg
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
                warnings.filterwarnings("ignore", message=".*weight_norm.*")
                self._smplicit_layer = smplicit_pkg.SMPLicit()
                if torch.cuda.is_available():
                    self._smplicit_layer = self._smplicit_layer.cuda()
            logger.info("SMPLicit layer loaded successfully")
        except Exception as e:
            logger.error("Failed to load SMPLicit: %s", e)
            self._smplicit_layer = None

    def reload_catalog(self) -> int:
        """Reload the catalog from disk. Returns the number of garments loaded."""
        self._load_catalog()
        return len(self._catalog)

    # ------------------------------------------------------------------
    # List garments
    # ------------------------------------------------------------------
    def list_garments(self) -> GarmentListResponse:
        """Return all available garments from the catalog."""
        garments = [
            GarmentInfo(
                id=g["id"],
                source_image=g["source_image"],
                cloth_name=g["cloth_name"],
                model_id=g["model_id"],
            )
            for g in self._catalog
        ]
        return GarmentListResponse(garments=garments)

    # ------------------------------------------------------------------
    # Generate garment mesh
    # ------------------------------------------------------------------
    def generate_garment_mesh(self, req: GarmentMeshRequest) -> GarmentMeshResponse:
        """Generate a garment mesh on the requested body with size control."""
        if self._smplicit_layer is None:
            raise RuntimeError(
                "SMPLicit is not loaded. Check that all dependencies "
                "are installed and model files exist."
            )

        garment = self._catalog_by_id.get(req.garment_id)
        if garment is None:
            raise ValueError(f"Garment '{req.garment_id}' not found in catalog")

        # Build the Z vector with size offset applied to z_cut
        z_cut = np.array(garment["z_cut"], dtype=np.float32)
        z_style = np.array(garment["z_style"], dtype=np.float32)

        # Apply size offset to z_cut (controls garment fit/size)
        z_cut_adjusted = z_cut + req.size_offset
        z_full = np.concatenate([z_cut_adjusted, z_style])

        model_id = garment["model_id"]

        # Get SMPL beta from body measurements
        from .smpl_service import get_smpl_service
        smpl_svc = get_smpl_service()
        measurements = BodyMeasurements(
            sex=req.sex,
            height_cm=req.height_cm,
            weight_kg=req.weight_kg,
            bust_cm=req.bust_cm,
            waist_cm=req.waist_cm,
            hip_cm=req.hip_cm,
            bicep_cm=req.bicep_cm,
            arm_length_cm=req.arm_length_cm,
            leg_length_cm=req.leg_length_cm,
        )
        beta = smpl_svc.measurements_to_betas(measurements).astype(np.float64)

        # Default T-pose
        pose = np.zeros(72, dtype=np.float64)

        # Reconstruct garment mesh via SMPLicit
        # Returns [smpl_mesh, garment_mesh]
        meshes = self._smplicit_layer.reconstruct(
            model_ids=[model_id],
            Zs=[z_full],
            pose=pose,
            beta=beta,
        )

        # meshes[0] = SMPL body, meshes[1] = garment
        garment_mesh = meshes[1]

        vertices_flat = garment_mesh.vertices.reshape(-1).tolist()
        faces_flat = garment_mesh.faces.reshape(-1).tolist()

        return GarmentMeshResponse(
            vertices=vertices_flat,
            faces=faces_flat,
            garment_id=req.garment_id,
            cloth_name=garment["cloth_name"],
            size_offset=req.size_offset,
        )


@lru_cache(maxsize=1)
def get_garment_service() -> GarmentService:
    """Provide singleton garment service instance."""
    return GarmentService()
