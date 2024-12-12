from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SolarPowerData(BaseModel):
    date_time: str
    plant_id: int 
    source_key_power: str
    source_key_sensor: str
    dc_power: Optional[float]
    ac_power: Optional[float]
    daily_yield: Optional[float]
    total_yield: Optional[float]
    ambient_temperature: Optional[float]
    module_temperature: Optional[float]
    irradiation: Optional[float]
    
    