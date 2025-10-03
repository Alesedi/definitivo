from pydantic import BaseModel
from typing import List
from modello.enum_genere import EnumGenere

# DTO generico (risposta API)
class UtenteDTO(BaseModel):
    id: str
    username: str
    email: str



