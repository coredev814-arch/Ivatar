"""Lightweight replacements for kaolin v0.1 APIs used by SMPLicit.

Replaces:
  - kaolin.rep.TriangleMesh.from_tensors(vertices, faces)
  - kaolin.conversions.trianglemesh_to_sdf(mesh)
"""

import torch
import numpy as np


class SimpleMesh:
    """Minimal triangle mesh container, replacing kaolin.rep.TriangleMesh.

    Only stores vertices and faces as torch tensors, which is all
    SMPLicit uses from the kaolin TriangleMesh.
    """

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor):
        self.vertices = vertices
        self.faces = faces

    @staticmethod
    def from_tensors(vertices: torch.Tensor, faces: torch.Tensor) -> "SimpleMesh":
        return SimpleMesh(vertices, faces)


def mesh_to_sdf(mesh: SimpleMesh):
    """Return a callable that computes unsigned distance from query points
    to the mesh surface.

    Replaces kaolin.conversions.trianglemesh_to_sdf.

    The returned function signature: sdf(points: Tensor) -> Tensor
    where points is (N, 3) and output is (N,) unsigned distances.

    Uses point-to-triangle distance computed on the GPU/CPU depending
    on where the mesh lives.
    """
    verts = mesh.vertices  # (V, 3)
    faces = mesh.faces     # (F, 3)

    # Pre-extract triangle vertices
    v0 = verts[faces[:, 0]]  # (F, 3)
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    def _unsigned_distance(points: torch.Tensor) -> torch.Tensor:
        """Compute unsigned distance from each point to the closest triangle.

        points: (N, 3)
        returns: (N,) unsigned distances
        """
        device = points.device

        # Process in batches to avoid OOM with large point sets
        batch_size = 2048
        n_points = points.shape[0]
        all_dists = []

        for i in range(0, n_points, batch_size):
            pts = points[i:i + batch_size]  # (B, 3)
            # Compute distance from each point to each triangle
            # Using point-to-triangle projection
            dists = _point_to_triangles_distance(pts, v0.to(device), v1.to(device), v2.to(device))
            all_dists.append(dists)

        return torch.cat(all_dists, dim=0)

    return _unsigned_distance


def _point_to_triangles_distance(
    points: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> torch.Tensor:
    """Compute minimum unsigned distance from each point to the triangle soup.

    points: (N, 3)
    v0, v1, v2: (F, 3) triangle vertices
    returns: (N,) minimum distance per point
    """
    # For each point, find the closest triangle via sampling approach:
    # Compute distance to triangle centroids first, then refine with
    # exact point-to-triangle distance for the closest candidates.
    centroids = (v0 + v1 + v2) / 3.0  # (F, 3)

    # Distance from each point to each centroid
    # points: (N, 1, 3), centroids: (1, F, 3)
    diff = points.unsqueeze(1) - centroids.unsqueeze(0)  # (N, F, 3)
    centroid_dists = torch.sum(diff ** 2, dim=2)  # (N, F)

    # Take top-k closest triangles for exact computation
    k = min(64, v0.shape[0])
    _, topk_idx = torch.topk(centroid_dists, k, dim=1, largest=False)  # (N, k)

    # Gather closest triangle vertices
    t_v0 = v0[topk_idx]  # (N, k, 3)
    t_v1 = v1[topk_idx]
    t_v2 = v2[topk_idx]

    # Exact point-to-triangle distance for top-k candidates
    pts_exp = points.unsqueeze(1).expand(-1, k, -1)  # (N, k, 3)
    exact_dists = _exact_point_triangle_distance(pts_exp, t_v0, t_v1, t_v2)  # (N, k)

    # Return minimum distance per point
    min_dists, _ = exact_dists.min(dim=1)  # (N,)
    return min_dists


def _exact_point_triangle_distance(
    p: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> torch.Tensor:
    """Exact unsigned distance from points to triangles.

    All inputs have shape (..., 3). Output has shape (...).
    Based on the geometric approach by Ericson (Real-Time Collision Detection).
    """
    edge0 = v1 - v0
    edge1 = v2 - v0
    v0_to_p = p - v0

    a = (edge0 * edge0).sum(dim=-1)
    b = (edge0 * edge1).sum(dim=-1)
    c = (edge1 * edge1).sum(dim=-1)
    d = (edge0 * v0_to_p).sum(dim=-1)
    e = (edge1 * v0_to_p).sum(dim=-1)

    det = a * c - b * b
    det = torch.clamp(det, min=1e-12)

    s = b * e - c * d
    t = b * d - a * e

    # Clamp barycentric coordinates to triangle
    s = torch.clamp(s / det, 0.0, 1.0)
    t = torch.clamp(t / det, 0.0, 1.0)

    # Ensure s + t <= 1
    st_sum = s + t
    mask = st_sum > 1.0
    s = torch.where(mask, s / st_sum, s)
    t = torch.where(mask, t / st_sum, t)

    closest = v0 + s.unsqueeze(-1) * edge0 + t.unsqueeze(-1) * edge1
    diff = p - closest
    dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)

    return dist
