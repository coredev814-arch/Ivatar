from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ...models.schemas import (
    GarmentBindingRequest,
    GarmentBindingResponse,
    GarmentCatalogResponse,
    GarmentCategory,
    GarmentDeformRequest,
    GarmentDeformResponse,
    GarmentUploadResponse,
)
from ...services.garment_binding_service import (
    GarmentBindingService,
    get_garment_binding_service,
)

router = APIRouter()


@router.get("/", response_model=GarmentCatalogResponse)
def list_garments(
    svc: GarmentBindingService = Depends(get_garment_binding_service),
) -> GarmentCatalogResponse:
    """List all uploaded garments and their binding status."""
    return svc.list_garments()


@router.post("/upload", response_model=GarmentUploadResponse)
async def upload_garment(
    file: UploadFile = File(...),
    category: GarmentCategory = Form(...),
    svc: GarmentBindingService = Depends(get_garment_binding_service),
) -> GarmentUploadResponse:
    """Upload a garment GLB file."""
    if not file.filename or not file.filename.lower().endswith((".glb", ".gltf")):
        raise HTTPException(status_code=400, detail="Only .glb/.gltf files are accepted")

    contents = await file.read()
    try:
        return svc.upload_garment(contents, file.filename, category)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/{garment_id}/bind", response_model=GarmentBindingResponse)
def bind_garment(
    garment_id: str,
    req: GarmentBindingRequest,
    svc: GarmentBindingService = Depends(get_garment_binding_service),
) -> GarmentBindingResponse:
    """Compute surface binding for a garment against the I-pose SMPL body."""
    try:
        return svc.compute_binding(garment_id, req.sex)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/deform", response_model=GarmentDeformResponse)
def deform_garment(
    req: GarmentDeformRequest,
    svc: GarmentBindingService = Depends(get_garment_binding_service),
) -> GarmentDeformResponse:
    """Deform a bound garment onto a fixed body with size control."""
    try:
        return svc.deform_garment(req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
