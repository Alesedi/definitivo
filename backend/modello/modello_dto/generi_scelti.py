from pydantic import BaseModel
from typing import List

class GeneriSceltiDTO(BaseModel):
    generi: List[str]
