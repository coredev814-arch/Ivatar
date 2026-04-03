from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field


class Sex(str, Enum):
    male = "male"
    female = "female"


class BodyMeasurements(BaseModel):
    sex: Sex = Field(..., description="Biological sex for selecting SMPL model variant")
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    bust_cm: float = Field(..., gt=0)
    waist_cm: float = Field(..., gt=0)
    hip_cm: float = Field(..., gt=0)
    bicep_cm: float = Field(..., gt=0)
    arm_length_cm: float = Field(..., gt=0)
    leg_length_cm: float = Field(..., gt=0)


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class MeshResponse(BaseModel):
    vertices: List[float] = Field(
        ..., description="Flattened [x0,y0,z0,x1,y1,z1,...] vertex positions"
    )
    faces: List[int] = Field(
        ..., description="Flattened [i0,i1,i2,i3,i4,i5,...] triangle indices"
    )
    joints: Dict[str, Vector3] = Field(
        ...,
        description=(
            "Named joint positions used for posing and visualization. "
            "Keys: thigh_left, thigh_right, lower_leg_left, lower_leg_right, "
            "upper_arm_left, upper_arm_right, lower_arm_left, lower_arm_right."
        ),
    )


# ── Garment schemas ──────────────────────────────────────────────────────


class GarmentInfo(BaseModel):
    """Summary of a garment in the catalog."""
    id: str = Field(..., description="Unique garment identifier")
    source_image: str = Field(..., description="Source image the garment was fitted from")
    cloth_name: str = Field(..., description="Garment type: tshirt, coat, pants, skirt, shoe, hair")
    model_id: int = Field(..., description="SMPLicit model index: 0=upper, 1=pants, 2=skirts, 3=hair, 4=shoes")


class GarmentListResponse(BaseModel):
    """Response for listing all available garments."""
    garments: List[GarmentInfo]


class GarmentMeshRequest(BaseModel):
    """Request to generate a garment mesh on a specific body with size control."""
    garment_id: str = Field(..., description="ID of the garment from the catalog")
    sex: Sex = Field(..., description="Biological sex for SMPL model variant")
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    bust_cm: float = Field(..., gt=0)
    waist_cm: float = Field(..., gt=0)
    hip_cm: float = Field(..., gt=0)
    bicep_cm: float = Field(..., gt=0)
    arm_length_cm: float = Field(..., gt=0)
    leg_length_cm: float = Field(..., gt=0)
    size_offset: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Garment size adjustment. 0=as-fitted, negative=tighter, positive=looser",
    )


class GarmentMeshResponse(BaseModel):
    """Response containing the garment mesh."""
    vertices: List[float] = Field(
        ..., description="Flattened [x0,y0,z0,x1,y1,z1,...] garment vertex positions"
    )
    faces: List[int] = Field(
        ..., description="Flattened [i0,i1,i2,i3,i4,i5,...] triangle indices"
    )
    garment_id: str
    cloth_name: str
    size_offset: float

