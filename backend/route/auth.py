from fastapi import APIRouter
from modello.modello_dto.utente_dto import UtenteDTO
from modello.modello_dto.utente_dto_registrazione import UtenteDTORegistrazione
from modello.modello_dto.utente_dto_login import UtenteDTOLogin
from service.service_utente import ServiceUtente

router = APIRouter(tags=["auth"])

@router.post("/register", response_model=UtenteDTO)
def register(user_dto: UtenteDTORegistrazione):
    print('chiamato')
    return ServiceUtente.registra_utente(user_dto)

@router.post("/login")
def login(login_dto: UtenteDTOLogin):
    return ServiceUtente.login_utente(login_dto)
