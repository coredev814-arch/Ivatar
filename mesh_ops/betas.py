"""Mapping from human-readable body measurements to SMPL shape parameters."""

from __future__ import annotations

import numpy as np

from ..models.schemas import BodyMeasurements, Sex


def measurements_to_betas(
    m: BodyMeasurements,
    num_shape_params: int,
) -> np.ndarray:
    """Map human-readable measurements to SMPL shape parameters.

    Only *global* body proportions are encoded here (height, weight,
    overall frame size).  Localised sculpting of bust, waist, hip,
    and bicep is handled post-hoc by separate mesh operations, so
    those channels are zeroed out to avoid double-counting.
    """
    features = np.array(
        [m.height_cm, m.weight_kg],
        dtype=np.float32,
    )

    typical = np.array([165.0, 65.0], dtype=np.float32)
    sex_bias = 1.0 if m.sex == Sex.male else -1.0

    normalized = (features - typical) / typical
    scale = np.array([4.0, -6.0], dtype=np.float32)
    normalized *= scale

    betas = np.zeros((num_shape_params,), dtype=np.float32)
    betas[: min(len(normalized), num_shape_params)] = normalized[:num_shape_params]
    betas[0] += sex_bias

    np.clip(betas, -3.5, 3.5, out=betas)
    return betas
