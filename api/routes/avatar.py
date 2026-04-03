from fastapi import APIRouter, Depends

from ...models.schemas import BodyMeasurements, MeshResponse
from ...services.smpl_service import SMPLService, get_smpl_service

router = APIRouter()


@router.post("/mesh", response_model=MeshResponse)
def generate_avatar_mesh(
    measurements: BodyMeasurements,
    smpl: SMPLService = Depends(get_smpl_service),
) -> MeshResponse:
    """Generate an SMPL mesh and joint positions from body measurements."""
    return smpl.generate_mesh(measurements)

