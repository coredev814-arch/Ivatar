from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple
import logging

import numpy as np
import torch

from ..core.config import settings

# Shoulder joint indices in body_pose (0-indexed, root excluded)
_BODY_POSE_L_SHOULDER = 15  # SMPL joint 16
_BODY_POSE_R_SHOULDER = 16  # SMPL joint 17
_BODY_POSE_L_HIP = 0       # SMPL joint 1
_BODY_POSE_R_HIP = 1       # SMPL joint 2
from ..models.schemas import BodyMeasurements, MeshResponse, Sex, Vector3
from ..mesh_ops.betas import measurements_to_betas
from ..mesh_ops.torso import apply_torso_sculpting
from ..mesh_ops.limbs import scale_limb_chain, scale_limb_segment, scale_bicep_cross_section
from ..mesh_ops.smoothing import smooth_arm_transitions

try:
    import smplx
except ImportError as exc:
    raise RuntimeError(
        "The 'smplx' package is required. Install with `pip install smplx`"
    ) from exc

logger = logging.getLogger(__name__)

# SMPL joint indices
SHOULDER_L, SHOULDER_R = 16, 17
ELBOW_L, ELBOW_R = 18, 19
WRIST_L, WRIST_R = 20, 21
HAND_L, HAND_R = 22, 23
HIP_L, HIP_R = 1, 2
KNEE_L, KNEE_R = 4, 5
ANKLE_L, ANKLE_R = 7, 8
FOOT_L, FOOT_R = 10, 11

# Joint names exposed in the API response
JOINT_INDICES = {
    "thigh_left": 1,
    "thigh_right": 2,
    "lower_leg_left": 4,
    "lower_leg_right": 5,
    "upper_arm_left": 16,
    "upper_arm_right": 17,
    "lower_arm_left": 18,
    "lower_arm_right": 19,
}


class SMPLService:
    """Service for loading SMPL models and generating meshes."""

    def __init__(self) -> None:
        self._cfg = settings.smpl
        try:
            self.male_model, self.faces = self._load_model(
                self._cfg.male_model_path, "MALE"
            )
            self.female_model, _ = self._load_model(
                self._cfg.female_model_path, "FEMALE"
            )
            logger.info("SMPL models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SMPL models: {e}")
            raise RuntimeError(f"SMPL model loading failed: {e}") from e

    @staticmethod
    def _load_model(
        path, gender: str
    ) -> Tuple[smplx.body_models.SMPLLayer, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"SMPL model not found at {path}")
        model = smplx.create(
            model_path=str(path),
            gender=gender,
            num_betas=settings.smpl.num_shape_params,
        )
        faces = model.faces.astype(np.int32)
        return model, faces

    def _select_model(self, sex: Sex) -> smplx.body_models.SMPLLayer:
        return self.male_model if sex == Sex.male else self.female_model

    def measurements_to_betas(self, m: BodyMeasurements) -> np.ndarray:
        """Public wrapper used by garment_service."""
        return measurements_to_betas(m, settings.smpl.num_shape_params)

    # ------------------------------------------------------------------
    # I-pose mesh (for garment binding)
    # ------------------------------------------------------------------
    def generate_ipose_mesh(
        self, sex: Sex,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate an SMPL body in I-pose with default proportions.

        Returns (vertices, faces, joints) as numpy arrays.
        """
        model = self._select_model(sex)
        betas = torch.zeros(1, self._cfg.num_shape_params, dtype=torch.float32)

        # Build body_pose: 23 joints × 3 = 69 values
        body_pose = torch.zeros(1, 69, dtype=torch.float32)
        angle = settings.garment.ipose_shoulder_angle
        body_pose[0, _BODY_POSE_L_SHOULDER * 3 + 2] = angle
        body_pose[0, _BODY_POSE_R_SHOULDER * 3 + 2] = -angle
        body_pose[0, _BODY_POSE_L_HIP * 3 + 2] = -0.05
        body_pose[0, _BODY_POSE_R_HIP * 3 + 2] = 0.05

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        with torch.no_grad():
            output = model(
                betas=betas.to(device),
                body_pose=body_pose.to(device),
                return_verts=True,
            )

        vertices = output.vertices[0].cpu().numpy().astype(np.float32)
        joints = output.joints[0].cpu().numpy().astype(np.float32)
        return vertices, self.faces.copy(), joints

    # ------------------------------------------------------------------
    # Body arrays (sculpted T-pose, for garment deformation at runtime)
    # ------------------------------------------------------------------
    def generate_body_arrays(
        self, measurements: BodyMeasurements,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the fully-sculpted T-pose body as raw arrays.

        Returns (vertices, faces, joints).
        """
        model = self._select_model(measurements.sex)
        betas_np = self.measurements_to_betas(measurements)
        betas = torch.from_numpy(betas_np).float().unsqueeze(0)

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        betas = betas.to(device)

        with torch.no_grad():
            smpl_output = model(betas=betas, return_verts=True)

        vertices = smpl_output.vertices[0].cpu().numpy().astype(np.float32)
        joints = smpl_output.joints[0].cpu().numpy().astype(np.float32)

        # Height scaling
        current_height = float(vertices[:, 1].max() - vertices[:, 1].min())
        target_height = measurements.height_cm / 100.0
        if current_height > 0:
            scale = target_height / current_height
            scale_vec = np.array(
                [1.0 + (scale - 1.0) * 0.3, scale, 1.0 + (scale - 1.0) * 0.3],
                dtype=np.float32,
            )
            vertices *= scale_vec
            joints *= scale_vec

        # Torso sculpting
        apply_torso_sculpting(vertices, joints, measurements)

        # Skinning weights for limb ops
        try:
            lbs_weights = model.lbs_weights.detach().cpu().numpy()
            dominant_joint = np.argmax(lbs_weights, axis=1)
        except AttributeError:
            lbs_weights = None
            dominant_joint = None

        # Arm scaling
        arm_target = max(measurements.arm_length_cm, 1.0) / 100.0
        disp_l = scale_limb_chain(
            vertices, joints, dominant_joint,
            SHOULDER_L, ELBOW_L, WRIST_L, arm_target,
            downstream_joint_ids=[HAND_L],
        )
        disp_r = scale_limb_chain(
            vertices, joints, dominant_joint,
            SHOULDER_R, ELBOW_R, WRIST_R, arm_target,
            downstream_joint_ids=[HAND_R],
        )

        # Bicep
        if dominant_joint is not None:
            scale_bicep_cross_section(
                vertices, joints, dominant_joint,
                measurements.bicep_cm,
                shoulder_pairs=[(SHOULDER_L, ELBOW_L), (SHOULDER_R, ELBOW_R)],
                forearm_pairs=[(ELBOW_L, WRIST_L), (ELBOW_R, WRIST_R)],
            )

        # Arm smoothing
        if lbs_weights is not None:
            smooth_arm_transitions(
                vertices, joints, self.faces, lbs_weights,
                disp_l, disp_r,
                shoulder_l=SHOULDER_L, shoulder_r=SHOULDER_R,
                elbow_l=ELBOW_L, elbow_r=ELBOW_R,
                wrist_l=WRIST_L, wrist_r=WRIST_R,
            )

        # Leg scaling
        leg_target_cm = max(measurements.leg_length_cm, 1.0)
        thigh_target = (leg_target_cm * 0.5) / 100.0
        scale_limb_segment(
            vertices, joints, dominant_joint,
            HIP_L, KNEE_L, thigh_target,
            downstream_joint_ids=[ANKLE_L, FOOT_L],
        )
        scale_limb_segment(
            vertices, joints, dominant_joint,
            HIP_R, KNEE_R, thigh_target,
            downstream_joint_ids=[ANKLE_R, FOOT_R],
        )

        return vertices, self.faces.copy(), joints

    # ------------------------------------------------------------------
    # Main mesh generation
    # ------------------------------------------------------------------
    def generate_mesh(self, measurements: BodyMeasurements) -> MeshResponse:
        model = self._select_model(measurements.sex)
        betas_np = self.measurements_to_betas(measurements)
        logger.info(
            "generate_mesh sex=%s betas[:4]=%s",
            measurements.sex.value,
            betas_np[:4].tolist(),
        )

        betas = torch.from_numpy(betas_np).float().unsqueeze(0)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        betas = betas.to(device)

        with torch.no_grad():
            smpl_output = model(betas=betas, return_verts=True)

        vertices = smpl_output.vertices[0].cpu().numpy().astype(np.float32)
        joints = smpl_output.joints[0].cpu().numpy().astype(np.float32)

        # --- Global height scaling ---
        current_height = float(vertices[:, 1].max() - vertices[:, 1].min())
        target_height = measurements.height_cm / 100.0
        if current_height > 0:
            scale = target_height / current_height
            scale_vec = np.array(
                [1.0 + (scale - 1.0) * 0.3, scale, 1.0 + (scale - 1.0) * 0.3],
                dtype=np.float32,
            )
            vertices *= scale_vec
            joints *= scale_vec

        logger.info(
            "generate_mesh scaled: current=%.3f target=%.3f scale=%.3f",
            current_height, target_height,
            scale if current_height > 0 else 1.0,
        )

        # --- Torso sculpting ---
        apply_torso_sculpting(vertices, joints, measurements)

        # --- Skinning weights ---
        try:
            lbs_weights = model.lbs_weights.detach().cpu().numpy()
            dominant_joint = np.argmax(lbs_weights, axis=1)
        except AttributeError:
            lbs_weights = None
            dominant_joint = None

        # --- Arm scaling ---
        arm_target = max(measurements.arm_length_cm, 1.0) / 100.0

        disp_l = scale_limb_chain(
            vertices, joints, dominant_joint,
            SHOULDER_L, ELBOW_L, WRIST_L, arm_target,
            downstream_joint_ids=[HAND_L],
        )
        disp_r = scale_limb_chain(
            vertices, joints, dominant_joint,
            SHOULDER_R, ELBOW_R, WRIST_R, arm_target,
            downstream_joint_ids=[HAND_R],
        )

        # --- Bicep cross-section ---
        if dominant_joint is not None:
            scale_bicep_cross_section(
                vertices, joints, dominant_joint,
                measurements.bicep_cm,
                shoulder_pairs=[(SHOULDER_L, ELBOW_L), (SHOULDER_R, ELBOW_R)],
                forearm_pairs=[(ELBOW_L, WRIST_L), (ELBOW_R, WRIST_R)],
            )

        # --- Arm transition smoothing ---
        if lbs_weights is not None:
            smooth_arm_transitions(
                vertices, joints, self.faces, lbs_weights,
                disp_l, disp_r,
                shoulder_l=SHOULDER_L, shoulder_r=SHOULDER_R,
                elbow_l=ELBOW_L, elbow_r=ELBOW_R,
                wrist_l=WRIST_L, wrist_r=WRIST_R,
            )

        # --- Leg scaling ---
        leg_target_cm = max(measurements.leg_length_cm, 1.0)
        thigh_target = (leg_target_cm * 0.5) / 100.0

        scale_limb_segment(
            vertices, joints, dominant_joint,
            HIP_L, KNEE_L, thigh_target,
            downstream_joint_ids=[ANKLE_L, FOOT_L],
        )
        scale_limb_segment(
            vertices, joints, dominant_joint,
            HIP_R, KNEE_R, thigh_target,
            downstream_joint_ids=[ANKLE_R, FOOT_R],
        )

        # --- Build response ---
        vertices_flat = vertices.reshape(-1).tolist()
        faces_flat = self.faces.reshape(-1).tolist()

        joints_payload: Dict[str, Vector3] = {}
        for name, idx in JOINT_INDICES.items():
            x, y, z = joints[idx].tolist()
            joints_payload[name] = Vector3(x=x, y=y, z=z)

        return MeshResponse(
            vertices=vertices_flat, faces=faces_flat, joints=joints_payload
        )


@lru_cache(maxsize=1)
def get_smpl_service() -> SMPLService:
    """Provide singleton SMPL service instance."""
    return SMPLService()
