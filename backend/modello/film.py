from pydantic import BaseModel
from typing import Optional
from enum_genere import EnumGenere

class FilmBase(BaseModel):
    id: Optional[int] = None
    titolo: str
    genere: EnumGenere
    media_voti: Optional[float] = None  # voto medio del film

    class Config:
        orm_mode = True
