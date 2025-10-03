from pydantic import BaseModel, EmailStr
from typing import List, Optional
from votazione import VotazioneBase
from enum_genere import EnumGenere

class UtenteBase(BaseModel):
    id: Optional[int] = None
    username: str
    email: str
    password: str
    votazioni: List[VotazioneBase] = []  # lista dei voti dati dall'utente
    generi_preferiti = List[EnumGenere] = []

    class Config:
        orm_mode = True
