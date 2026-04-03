"""Localised radial scaling of torso vertices for bust / waist / hip control."""

from __future__ import annotations

import numpy as np

from ..models.schemas import BodyMeasurements, Sex


def apply_torso_sculpting(
    vertices: np.ndarray,
    joints: np.ndarray,
    measurements: BodyMeasurements,
) -> None:
    """Radially scale torso vertices in three horizontal bands so that
    *bust*, *waist* and *hip* sliders each control the expected body region.

    Modifies *vertices* in place.

    Spine joint reference (SMPL):
        0  Pelvis
        1  L_Hip      2  R_Hip
        3  Spine1
        6  Spine2
        9  Spine3
       12  Neck
    """
    # --- Typical values (cm) ------------------------------------------------
    if measurements.sex == Sex.male:
        typical_bust, typical_waist, typical_hip = 100.0, 85.0, 104.0
    else:
        typical_bust, typical_waist, typical_hip = 92.0, 75.0, 100.0

    bust_ratio = float(np.clip(measurements.bust_cm / typical_bust, 0.7, 1.6))
    waist_ratio = float(np.clip(measurements.waist_cm / typical_waist, 0.6, 1.6))
    hip_ratio = float(np.clip(measurements.hip_cm / typical_hip, 0.7, 1.6))

    # --- Anchor heights (Y) from SMPL joints --------------------------------
    PELVIS, SPINE1, SPINE2, SPINE3, NECK = 0, 3, 6, 9, 12
    L_HIP, R_HIP = 1, 2

    hip_y = 0.5 * (joints[L_HIP, 1] + joints[R_HIP, 1])
    bust_centre_y = 0.5 * (joints[SPINE2, 1] + joints[SPINE3, 1])
    waist_centre_y = 0.5 * (joints[SPINE1, 1] + joints[SPINE2, 1])
    hip_centre_y = hip_y

    torso_len = max(float(joints[NECK, 1] - joints[PELVIS, 1]), 0.01)
    bust_sigma = torso_len * 0.18
    waist_sigma = torso_len * 0.16
    hip_sigma = torso_len * 0.20

    # --- Torso centreline ----------------------------------------------------
    spine_x = 0.5 * (joints[SPINE1, 0] + joints[SPINE2, 0])
    spine_z = 0.5 * (joints[SPINE1, 2] + joints[SPINE2, 2])

    vy = vertices[:, 1]

    def _band_weight(centre_y: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((vy - centre_y) / sigma) ** 2)

    w_bust = _band_weight(bust_centre_y, bust_sigma)
    w_waist = _band_weight(waist_centre_y, waist_sigma)
    w_hip = _band_weight(hip_centre_y, hip_sigma)

    total_w = w_bust + w_waist + w_hip + 1e-8
    blended_ratio = (
        w_bust * bust_ratio + w_waist * waist_ratio + w_hip * hip_ratio
    ) / total_w

    confidence = np.clip(total_w / (total_w.max() + 1e-8), 0.0, 1.0)
    eff_scale = 1.0 + confidence * (blended_ratio - 1.0)

    dx = vertices[:, 0] - spine_x
    dz = vertices[:, 2] - spine_z

    if measurements.sex == Sex.female:
        anterior_mask = (dz > 0).astype(np.float32)
        bust_extra_z_front = bust_ratio - blended_ratio
        bust_extra_z_back = (1.0 + (bust_ratio - 1.0) * 0.3) - blended_ratio
        bust_extra_x = (1.0 + (bust_ratio - 1.0) * 0.5) - blended_ratio
        bust_influence = w_bust / total_w

        eff_x = eff_scale + bust_influence * bust_extra_x * 0.5
        eff_z = eff_scale + bust_influence * (
            anterior_mask * bust_extra_z_front * 0.5
            + (1.0 - anterior_mask) * bust_extra_z_back * 0.3
        )
        vertices[:, 0] = spine_x + dx * eff_x
        vertices[:, 2] = spine_z + dz * eff_z
    else:
        vertices[:, 0] = spine_x + dx * eff_scale
        vertices[:, 2] = spine_z + dz * eff_scale
