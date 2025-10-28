from pydantic_settings import BaseSettings
from functools import lru_cache
import torch


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    APP_NAME: str = "SPD-MVP"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./data/predictions.db"
    
    # Model settings - STAR Regression Model
    REGRESSION_MODEL_PATH: str = "models/regression/fd001"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    FAILURE_THRESHOLD: int = 30  # RUL threshold for failure warning
    
    # Legacy model path (deprecated)
    MODEL_PATH: str = "app/models/ml_models/model.pkl"
    
    # Validation settings
    DRIFT_THRESHOLD: float = 0.2
    PSI_THRESHOLD: float = 0.2
    OUTLIER_SENSITIVITY: str = "medium"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

