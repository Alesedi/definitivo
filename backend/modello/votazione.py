from pydantic import BaseModel, Field
from typing import Optional

class VotazioneBase(BaseModel):
    id: Optional[int] = None
    utente_id: int     # id dell'utente che vota
    film_id: int       # id del film votato
    valutazione: int = Field(ge=1, le=5, description="voto da 1 a 5 stelle")
