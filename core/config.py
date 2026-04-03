from pathlib import Path

from pydantic import BaseModel

# Project root: the Ivatar package directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SMPLConfig(BaseModel):
    """Configuration for SMPL model locations and defaults."""

    male_model_path: Path = PROJECT_ROOT / "lib" / "SMPLicit" / "utils" / "SMPL_MALE.pkl"
    female_model_path: Path = PROJECT_ROOT / "lib" / "SMPLicit" / "utils" / "SMPL_FEMALE.pkl"
    male_regressor_path: Path = PROJECT_ROOT / "lib" / "SMPLicit" / "utils" / "regressor_male.pt"
    female_regressor_path: Path = PROJECT_ROOT / "lib" / "SMPLicit" / "utils" / "regressor_female.pt"
    num_shape_params: int = 10


class GarmentBindingConfig(BaseModel):
    """Configuration for garment binding system."""

    garment_storage_path: Path = PROJECT_ROOT / "garments"
    binding_cache_path: Path = PROJECT_ROOT / "garment_bindings"
    max_upload_size_mb: int = 50
    # I-pose shoulder rotation (radians). Tune to match your garment GLBs.
    ipose_shoulder_angle: float = 1.13  # ~65 degrees


class AppConfig(BaseModel):
    smpl: SMPLConfig = SMPLConfig()
    garment: GarmentBindingConfig = GarmentBindingConfig()


settings = AppConfig()
