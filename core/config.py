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


class SMPLicitConfig(BaseModel):
    """Configuration for SMPLicit garment model locations."""

    garment_catalog_path: Path = PROJECT_ROOT / "lib" / "fit_SMPLicit" / "garment_catalog.json"
    checkpoints_path: Path = PROJECT_ROOT / "lib" / "checkpoints"
    clusters_path: Path = PROJECT_ROOT / "lib" / "clusters"


class AppConfig(BaseModel):
    smpl: SMPLConfig = SMPLConfig()
    smplicit: SMPLicitConfig = SMPLicitConfig()


settings = AppConfig()
