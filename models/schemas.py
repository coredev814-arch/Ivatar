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


# ── Garment binding schemas ──────────────────────────────────────────────


class GarmentCategory(str, Enum):
    shirt = "shirt"
    pants = "pants"


class GarmentUploadResponse(BaseModel):
    garment_id: str
    filename: str
    category: GarmentCategory
    vertex_count: int
    face_count: int


class GarmentBindingRequest(BaseModel):
    sex: Sex = Field(..., description="Sex for SMPL model used in binding")


class GarmentBindingResponse(BaseModel):
    garment_id: str
    num_bindings: int
    status: str


class GarmentDeformRequest(BaseModel):
    garment_id: str = Field(..., description="ID of the garment to deform")
    sex: Sex = Field(..., description="Biological sex for SMPL model variant")
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    bust_cm: float = Field(..., gt=0)
    waist_cm: float = Field(..., gt=0)
    hip_cm: float = Field(..., gt=0)
    bicep_cm: float = Field(..., gt=0)
    arm_length_cm: float = Field(..., gt=0)
    leg_length_cm: float = Field(..., gt=0)
    size_factor: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="Garment size multiplier. 1.0=as-designed, <1=tighter, >1=looser",
    )


class GarmentDeformResponse(BaseModel):
    vertices: List[float] = Field(
        ..., description="Flattened [x0,y0,z0,...] garment vertex positions"
    )
    faces: List[int] = Field(
        ..., description="Flattened [i0,i1,i2,...] triangle indices"
    )
    garment_id: str
    size_factor: float


class GarmentCatalogEntry(BaseModel):
    garment_id: str
    filename: str
    category: GarmentCategory
    has_binding: bool


class GarmentCatalogResponse(BaseModel):
    garments: List[GarmentCatalogEntry]

