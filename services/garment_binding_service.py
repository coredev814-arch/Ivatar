"""Garment binding service: upload GLB garments, bind them to the SMPL
body surface, and deform them at runtime with size control."""

from __future__ import annotations

import json
import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from ..core.config import settings
from ..mesh_ops.binding import compute_binding, reconstruct_from_binding
from ..models.schemas import (
    BodyMeasurements,
    GarmentBindingResponse,
    GarmentCatalogEntry,
    GarmentCatalogResponse,
    GarmentCategory,
    GarmentDeformRequest,
    GarmentDeformResponse,
    GarmentUploadResponse,
    Sex,
)

logger = logging.getLogger(__name__)

_CATALOG_FILE = "catalog.json"


class GarmentBindingService:
    def __init__(self) -> None:
        self._storage = settings.garment.garment_storage_path
        self._cache = settings.garment.binding_cache_path
        self._storage.mkdir(parents=True, exist_ok=True)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._catalog: dict[str, dict] = {}
        self._load_catalog()

    # ------------------------------------------------------------------
    # Catalog persistence
    # ------------------------------------------------------------------
    def _catalog_path(self) -> Path:
        return self._storage / _CATALOG_FILE

    def _load_catalog(self) -> None:
        path = self._catalog_path()
        if path.exists():
            with open(path, "r") as f:
                entries = json.load(f)
            self._catalog = {e["garment_id"]: e for e in entries}
            logger.info("Loaded %d garments from catalog", len(self._catalog))

    def _save_catalog(self) -> None:
        with open(self._catalog_path(), "w") as f:
            json.dump(list(self._catalog.values()), f, indent=2)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload_garment(
        self,
        file_bytes: bytes,
        filename: str,
        category: GarmentCategory,
    ) -> GarmentUploadResponse:
        garment_id = uuid.uuid4().hex[:12]
        cat_dir = self._storage / category.value
        cat_dir.mkdir(parents=True, exist_ok=True)

        glb_path = cat_dir / f"{garment_id}.glb"
        glb_path.write_bytes(file_bytes)

        # Load to get vertex/face counts
        mesh = self._load_glb(glb_path)

        self._catalog[garment_id] = {
            "garment_id": garment_id,
            "filename": filename,
            "category": category.value,
            "glb_path": str(glb_path.relative_to(self._storage)),
        }
        self._save_catalog()

        logger.info("Uploaded garment %s (%s) -> %s", garment_id, filename, glb_path)
        return GarmentUploadResponse(
            garment_id=garment_id,
            filename=filename,
            category=category,
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
        )

    # ------------------------------------------------------------------
    # Binding
    # ------------------------------------------------------------------
    def compute_binding(
        self,
        garment_id: str,
        sex: Sex,
    ) -> GarmentBindingResponse:
        entry = self._catalog.get(garment_id)
        if entry is None:
            raise ValueError(f"Garment '{garment_id}' not found")

        glb_path = self._storage / entry["glb_path"]
        garment_mesh = self._load_glb(glb_path)

        # Generate I-pose SMPL body for binding
        from .smpl_service import get_smpl_service

        smpl_svc = get_smpl_service()
        body_verts, body_faces, body_joints = smpl_svc.generate_ipose_mesh(sex)
        body_mesh = trimesh.Trimesh(body_verts, body_faces, process=False)

        # Compute binding data
        binding = compute_binding(garment_mesh.vertices, body_mesh)

        # Save binding + garment faces
        binding_path = self._cache / f"{garment_id}.npz"
        np.savez_compressed(
            binding_path,
            triangle_ids=binding["triangle_ids"],
            bary_coords=binding["bary_coords"],
            offsets=binding["offsets"],
            garment_faces=garment_mesh.faces.astype(np.int32),
            reference_sex=np.array([sex.value], dtype=object),
        )

        logger.info(
            "Binding computed for %s: %d vertices bound",
            garment_id,
            len(binding["triangle_ids"]),
        )
        return GarmentBindingResponse(
            garment_id=garment_id,
            num_bindings=len(binding["triangle_ids"]),
            status="bound",
        )

    # ------------------------------------------------------------------
    # Deform
    # ------------------------------------------------------------------
    def deform_garment(self, req: GarmentDeformRequest) -> GarmentDeformResponse:
        binding_path = self._cache / f"{req.garment_id}.npz"
        if not binding_path.exists():
            raise ValueError(
                f"No binding found for garment '{req.garment_id}'. "
                "Call the /bind endpoint first."
            )

        data = np.load(binding_path, allow_pickle=True)
        triangle_ids = data["triangle_ids"]
        bary_coords = data["bary_coords"]
        offsets = data["offsets"]
        garment_faces = data["garment_faces"]

        # Generate the fixed body from measurements
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
        body_verts, body_faces, _ = smpl_svc.generate_body_arrays(measurements)

        # Reconstruct garment on the deformed body with size control
        garment_verts = reconstruct_from_binding(
            body_verts,
            body_faces,
            triangle_ids,
            bary_coords,
            offsets,
            size_factor=req.size_factor,
        )

        return GarmentDeformResponse(
            vertices=garment_verts.reshape(-1).tolist(),
            faces=garment_faces.reshape(-1).tolist(),
            garment_id=req.garment_id,
            size_factor=req.size_factor,
        )

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------
    def list_garments(self) -> GarmentCatalogResponse:
        entries = []
        for entry in self._catalog.values():
            gid = entry["garment_id"]
            has_binding = (self._cache / f"{gid}.npz").exists()
            entries.append(
                GarmentCatalogEntry(
                    garment_id=gid,
                    filename=entry["filename"],
                    category=GarmentCategory(entry["category"]),
                    has_binding=has_binding,
                )
            )
        return GarmentCatalogResponse(garments=entries)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_glb(path: Path) -> trimesh.Trimesh:
        """Load a GLB file and return a single Trimesh."""
        loaded = trimesh.load(str(path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(
                [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        return loaded


@lru_cache(maxsize=1)
def get_garment_binding_service() -> GarmentBindingService:
    return GarmentBindingService()
