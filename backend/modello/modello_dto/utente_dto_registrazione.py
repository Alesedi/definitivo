from pydantic import BaseModel, Field, validator

class UtenteDTORegistrazione(BaseModel):
    username: str
    email: str = Field(description="Email dell'utente")
    password: str = Field(min_length=6, max_length=72)
    conferma_password: str = Field(min_length=6, max_length=72)

    @validator("conferma_password")
    def passwords_match(cls, v, values, **kwargs):
        if "password" in values and v != values["password"]:
            raise ValueError("Le password non coincidono")
        return v

    class Config:
        from_attributes = True
