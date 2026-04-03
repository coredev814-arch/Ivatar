"""Surface-binding utilities: bind garment vertices to a body mesh
and reconstruct them on a deformed body with size control."""

from __future__ import annotations

import numpy as np
import trimesh


def compute_binding(
    garment_vertices: np.ndarray,
    body_mesh: trimesh.Trimesh,
) -> dict:
    """Bind each garment vertex to the nearest point on the body surface.

    Returns a dict with:
        triangle_ids  (N,)   – index of the closest body triangle
        bary_coords   (N, 3) – barycentric coordinates on that triangle
        offsets        (N, 3) – offset vector from surface point to garment vertex
    """
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
        body_mesh, garment_vertices
    )

    # Compute barycentric coordinates for each closest point
    triangles = body_mesh.vertices[body_mesh.faces[triangle_ids]]  # (N, 3, 3)
    bary_coords = _barycentric_coords_batch(closest_points, triangles)

    # Offset from body surface to garment vertex
    offsets = garment_vertices - closest_points

    return {
        "triangle_ids": triangle_ids.astype(np.int32),
        "bary_coords": bary_coords.astype(np.float32),
        "offsets": offsets.astype(np.float32),
    }


def reconstruct_from_binding(
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    triangle_ids: np.ndarray,
    bary_coords: np.ndarray,
    offsets: np.ndarray,
    size_factor: float = 1.0,
) -> np.ndarray:
    """Reconstruct garment vertices on a deformed body.

    The body is fixed; only ``size_factor`` scales the offset to change
    garment fit (tighter / looser).
    """
    # Get the three vertices of each bound triangle on the deformed body
    triangles = body_vertices[body_faces[triangle_ids]]  # (N, 3, 3)

    # Interpolate surface anchor points using barycentric coordinates
    anchors = (
        bary_coords[:, 0, None] * triangles[:, 0]
        + bary_coords[:, 1, None] * triangles[:, 1]
        + bary_coords[:, 2, None] * triangles[:, 2]
    )

    # Compute per-triangle normals on the deformed body for
    # normal-aligned size scaling
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = normals / norms

    # Decompose offset into normal and tangential components
    normal_component = np.sum(offsets * normals, axis=1, keepdims=True) * normals
    tangential_component = offsets - normal_component

    # Scale only the normal component by size_factor
    scaled_offsets = normal_component * size_factor + tangential_component

    return (anchors + scaled_offsets).astype(np.float32)


def _barycentric_coords_batch(
    points: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Compute barycentric coordinates for points inside triangles.

    Parameters
    ----------
    points    : (N, 3) query points (assumed to lie on the triangle planes)
    triangles : (N, 3, 3) triangle vertex positions

    Returns
    -------
    bary : (N, 3) barycentric coordinates (u, v, w) with u + v + w ≈ 1
    """
    v0 = triangles[:, 1] - triangles[:, 0]
    v1 = triangles[:, 2] - triangles[:, 0]
    v2 = points - triangles[:, 0]

    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.stack([u, v, w], axis=1).astype(np.float32)
