from pydantic import BaseModel, Field
from typing import Optional

class UtenteDTOLogin(BaseModel):
    """
    DTO per il login dell'utente.
    L'utente può usare username o email per autenticarsi.
    """
    username: Optional[str] = None
    email: Optional[str] = None
    password: str = Field(min_length=6, max_length=72, description="lunghezza minima 6 caratteri")

    class Config:
        from_attributes = True
