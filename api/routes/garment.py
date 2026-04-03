from fastapi import APIRouter, Depends, HTTPException

from ...models.schemas import (
    GarmentListResponse,
    GarmentMeshRequest,
    GarmentMeshResponse,
)
from ...services.garment_service import GarmentService, get_garment_service

router = APIRouter()


@router.get("/", response_model=GarmentListResponse)
def list_garments(
    svc: GarmentService = Depends(get_garment_service),
) -> GarmentListResponse:
    """List all available garments from the catalog."""
    return svc.list_garments()


@router.post("/mesh", response_model=GarmentMeshResponse)
def generate_garment_mesh(
    req: GarmentMeshRequest,
    svc: GarmentService = Depends(get_garment_service),
) -> GarmentMeshResponse:
    """Generate a garment mesh on a body defined by measurements, with size control."""
    try:
        return svc.generate_garment_mesh(req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/reload")
def reload_catalog(
    svc: GarmentService = Depends(get_garment_service),
) -> dict:
    """Reload the garment catalog from disk (after running fit_SMPLicit.py)."""
    count = svc.reload_catalog()
    return {"reloaded": count}
