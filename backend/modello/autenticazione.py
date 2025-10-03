from pydantic import BaseModel, EmailStr, constr
from typing import Optional

class LoginInput(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: str

class RegisterInput(BaseModel):
    username: str
    email: EmailStr
    password: constr(min_length=6)
    conferma_password: constr(min_length=6)
