"""Mesh smoothing utilities: adjacency, Laplacian smoothing, displacement diffusion."""

from __future__ import annotations

import numpy as np


def shoulder_influence(
    lbs_weights: np.ndarray,
    shoulder_idx: int,
    threshold: float = 0.01,
) -> np.ndarray:
    """Return the raw SMPL skinning weight for the shoulder joint,
    zeroed below *threshold*.

    Returns (N,) float32 in [0, 1].
    """
    w = lbs_weights[:, shoulder_idx].copy().astype(np.float32)
    w[w < threshold] = 0.0
    return w


def build_adjacency(faces: np.ndarray, n_verts: int):
    """Build a row-normalised sparse adjacency matrix from a face array.

    Multiplying the result by vertex positions gives the neighbourhood
    average (one Laplacian-smooth step).
    """
    from scipy import sparse

    edges = np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]],
    ], axis=0)
    rows, cols = edges[:, 0], edges[:, 1]
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
    adj.setdiag(0)
    adj = (adj > 0).astype(np.float32)
    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0
    inv_deg = sparse.diags(1.0 / degree)
    return inv_deg @ adj


def laplacian_smooth_region(
    vertices: np.ndarray,
    adj_norm,
    mask: np.ndarray,
    iterations: int = 3,
    lam: float = 0.5,
) -> None:
    """In-place Laplacian smooth restricted to *mask* vertices.

    Vertices where ``mask`` is False act as fixed anchors.
    """
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return
    for _ in range(iterations):
        avg = adj_norm.dot(vertices)
        delta = avg[indices] - vertices[indices]
        vertices[indices] += lam * delta


def diffuse_displacement(
    vertices: np.ndarray,
    adj_norm,
    displacement: np.ndarray,
    band_mask: np.ndarray,
    iterations: int = 4,
    lam: float = 0.6,
) -> None:
    """Diffuse arm-scaling displacement into the transition band
    so that the seam is bridged smoothly.

    Modifies *vertices* in place for band vertices.
    """
    disp = displacement.copy()
    band_idx = np.where(band_mask)[0]
    if len(band_idx) == 0:
        return
    for _ in range(iterations):
        avg_disp = adj_norm.dot(disp)
        disp[band_idx] = disp[band_idx] + lam * (
            avg_disp[band_idx] - disp[band_idx]
        )
    vertices[band_idx] += disp[band_idx]


def smooth_arm_transitions(
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    lbs_weights: np.ndarray,
    disp_l: np.ndarray,
    disp_r: np.ndarray,
    *,
    shoulder_l: int = 16,
    shoulder_r: int = 17,
    elbow_l: int = 18,
    elbow_r: int = 19,
    wrist_l: int = 20,
    wrist_r: int = 21,
) -> None:
    """Apply all arm-related transition smoothing in one call.

    Handles: shoulder displacement diffusion, shoulder/deltoid Laplacian
    smoothing, elbow transition smoothing, and wrist transition smoothing.

    Modifies *vertices* in place.
    """
    n_verts = vertices.shape[0]
    adj_norm = build_adjacency(faces, n_verts)

    sw_l = shoulder_influence(lbs_weights, shoulder_l)
    sw_r = shoulder_influence(lbs_weights, shoulder_r)

    # --- Shoulder transition: diffuse arm displacement ---
    for sw, disp in [(sw_l, disp_l), (sw_r, disp_r)]:
        arm_moved = np.linalg.norm(disp, axis=1) > 1e-8
        band = (sw > 0.02) & (~arm_moved)
        if np.any(band):
            diffuse_displacement(vertices, adj_norm, disp, band, iterations=6, lam=0.5)

    # --- Laplacian smooth: shoulder transition ---
    for sw in (sw_l, sw_r):
        transition = sw > 0.01
        if np.any(transition):
            laplacian_smooth_region(vertices, adj_norm, transition, iterations=6, lam=0.5)

    # --- Laplacian smooth: shoulder cap / deltoid / armpit ---
    for jid, next_jid in [(shoulder_l, elbow_l), (shoulder_r, elbow_r)]:
        joint_pos = joints[jid]
        dist = np.linalg.norm(vertices - joint_pos, axis=1)
        arm_len = float(np.linalg.norm(joints[next_jid] - joints[jid]))
        radius = arm_len * 0.55
        cap_region = dist < radius
        if np.any(cap_region):
            laplacian_smooth_region(vertices, adj_norm, cap_region, iterations=6, lam=0.5)

    # --- Laplacian smooth: elbow transition ---
    for s_jid, e_jid in [(shoulder_l, elbow_l), (shoulder_r, elbow_r)]:
        w_upper = lbs_weights[:, s_jid]
        w_lower = lbs_weights[:, e_jid]
        elbow_transition = (w_upper > 0.05) & (w_lower > 0.05)
        if np.any(elbow_transition):
            laplacian_smooth_region(vertices, adj_norm, elbow_transition, iterations=4, lam=0.5)

    # --- Laplacian smooth: wrist transition ---
    for e_jid, w_jid in [(elbow_l, wrist_l), (elbow_r, wrist_r)]:
        w_forearm = lbs_weights[:, e_jid]
        w_hand = lbs_weights[:, w_jid]
        wrist_transition = (w_forearm > 0.05) & (w_hand > 0.05)
        if np.any(wrist_transition):
            laplacian_smooth_region(vertices, adj_norm, wrist_transition, iterations=4, lam=0.5)
