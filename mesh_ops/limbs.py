"""Limb chain and segment scaling operations."""

from __future__ import annotations

import numpy as np


def scale_limb_chain(
    vertices: np.ndarray,
    joints: np.ndarray,
    dominant_joint: np.ndarray,
    root_idx: int,
    mid_idx: int,
    end_idx: int,
    target_total_length_m: float,
    downstream_joint_ids: list[int] | None = None,
) -> np.ndarray:
    """Scale a two-segment limb chain (e.g. shoulder->elbow->wrist).

    Modifies *vertices* and *joints* in place.

    Returns the per-vertex displacement array (N, 3) which is non-zero
    only for vertices that were moved.  The caller can diffuse this into
    neighbouring transition vertices.
    """
    disp = np.zeros_like(vertices)
    if dominant_joint is None or target_total_length_m <= 0:
        return disp

    seg1 = joints[mid_idx] - joints[root_idx]
    seg2 = joints[end_idx] - joints[mid_idx]
    len1 = float(np.linalg.norm(seg1))
    len2 = float(np.linalg.norm(seg2))
    current_len = len1 + len2
    if current_len <= 0:
        return disp

    scale_limb = target_total_length_m / current_len
    ortho_scale = float(np.clip(1.0 / max(scale_limb, 0.01) ** 0.2, 0.85, 1.15))

    # --- Segment 1: root -> mid (upper arm) ---
    dir1 = seg1 / len1 if len1 > 0 else seg1
    old_mid = joints[mid_idx].copy()
    joints[mid_idx] = joints[root_idx] + dir1 * (len1 * scale_limb)

    root_mask = dominant_joint == root_idx
    if np.any(root_mask):
        old_pos = vertices[root_mask].copy()
        rel = vertices[root_mask] - joints[root_idx]
        proj_scalar = np.dot(rel, dir1)
        proj = proj_scalar[:, None] * dir1
        ortho = rel - proj

        target_pos = joints[root_idx] + proj * scale_limb + ortho * ortho_scale

        t = np.clip(proj_scalar / (len1 + 1e-8), 0.0, 1.0)
        ramp = np.clip((t - 0.0) / 0.4, 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
        ramp = ramp[:, None]

        vertices[root_mask] = old_pos + (target_pos - old_pos) * ramp
        disp[root_mask] = vertices[root_mask] - old_pos

    # --- Segment 2: mid -> end (forearm) ---
    dir2 = seg2 / len2 if len2 > 0 else seg2
    old_end = joints[end_idx].copy()
    joints[end_idx] = joints[mid_idx] + dir2 * (len2 * scale_limb)
    end_offset = joints[end_idx] - old_end

    mid_mask = dominant_joint == mid_idx
    if np.any(mid_mask):
        old_pos = vertices[mid_mask].copy()
        rel = vertices[mid_mask] - old_mid
        proj_len = np.dot(rel, dir2)[:, None]
        proj = proj_len * dir2
        ortho = rel - proj
        vertices[mid_mask] = joints[mid_idx] + proj * scale_limb + ortho * ortho_scale
        disp[mid_mask] = vertices[mid_mask] - old_pos

    # --- Wrist: rigidly translate ---
    end_mask = dominant_joint == end_idx
    if np.any(end_mask):
        vertices[end_mask] += end_offset
        disp[end_mask] = end_offset

    # --- Downstream (hands/fingers): rigidly translate ---
    if downstream_joint_ids:
        for jid in downstream_joint_ids:
            joints[jid] += end_offset
        ds_mask = np.isin(dominant_joint, downstream_joint_ids)
        vertices[ds_mask] += end_offset
        disp[ds_mask] = end_offset

    return disp


def scale_limb_segment(
    vertices: np.ndarray,
    joints: np.ndarray,
    dominant_joint: np.ndarray,
    root_idx: int,
    end_idx: int,
    target_length_m: float,
    downstream_joint_ids: list[int] | None = None,
) -> None:
    """Scale a single segment (root->end) to target length.

    Modifies *vertices* and *joints* in place.
    """
    if dominant_joint is None or target_length_m <= 0:
        return
    seg = joints[end_idx] - joints[root_idx]
    current_len = float(np.linalg.norm(seg))
    if current_len <= 0:
        return
    scale_seg = target_length_m / current_len
    dir_unit = seg / current_len

    old_end = joints[end_idx].copy()
    joints[end_idx] = joints[root_idx] + dir_unit * target_length_m
    offset = joints[end_idx] - old_end

    ortho_scale = float(np.clip(1.0 / max(scale_seg, 0.01) ** 0.2, 0.85, 1.15))

    root_mask = dominant_joint == root_idx
    rel = vertices[root_mask] - joints[root_idx]
    proj_len = np.dot(rel, dir_unit)[:, None]
    proj = proj_len * dir_unit
    ortho = rel - proj
    vertices[root_mask] = joints[root_idx] + proj * scale_seg + ortho * ortho_scale

    translate_ids = [end_idx]
    if downstream_joint_ids:
        translate_ids.extend(downstream_joint_ids)
        for jid in downstream_joint_ids:
            joints[jid] += offset
    ds_mask = np.isin(dominant_joint, translate_ids)
    vertices[ds_mask] += offset


def scale_bicep_cross_section(
    vertices: np.ndarray,
    joints: np.ndarray,
    dominant_joint: np.ndarray,
    bicep_cm: float,
    shoulder_pairs: list[tuple[int, int]],
    forearm_pairs: list[tuple[int, int]],
) -> None:
    """Scale the cross-section of upper/lower arm segments for bicep size.

    *shoulder_pairs* — (shoulder_joint, elbow_joint) tuples; these get
    a tapered ramp so the deltoid cap is not distorted.

    *forearm_pairs* — (elbow_joint, wrist_joint) tuples; uniform scaling.

    Modifies *vertices* in place.
    """
    typical_bicep = 30.0
    muscle_factor = float(np.clip(bicep_cm / typical_bicep, 0.7, 1.8))

    def _scale_segment(jid: int, next_jid: int, taper: bool) -> None:
        mask = dominant_joint == jid
        if not np.any(mask):
            return

        seg = joints[next_jid] - joints[jid]
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 0:
            return
        direction = seg / seg_len

        rel = vertices[mask] - joints[jid]
        proj_len = np.dot(rel, direction)
        proj = proj_len[:, None] * direction
        ortho = rel - proj

        if taper:
            t = np.clip(proj_len / (seg_len + 1e-8), 0.0, 1.0)
            ramp = np.clip((t - 0.05) / 0.35, 0.0, 1.0)
            ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            per_vert_factor = 1.0 + (muscle_factor - 1.0) * ramp
            vertices[mask] = joints[jid] + proj + ortho * per_vert_factor[:, None]
        else:
            vertices[mask] = joints[jid] + proj + ortho * muscle_factor

    for jid, next_jid in shoulder_pairs:
        _scale_segment(jid, next_jid, taper=True)
    for jid, next_jid in forearm_pairs:
        _scale_segment(jid, next_jid, taper=False)
