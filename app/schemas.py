from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    hour: int
    banner_pos: int
    site_id: str
    site_domain: str
    site_category: str
    app_id: str
    app_domain: str
    app_category: str
    device_id: str
    device_type: int
    device_conn_type: int
    C1: Optional[int] = None

class PredictionResponse(BaseModel):
    predicted_ctr: float
    is_weekend: bool
