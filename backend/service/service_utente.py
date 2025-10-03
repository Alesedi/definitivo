from modelli_ODM.utente_odm import Utente
from modello.modello_dto.utente_dto_registrazione import UtenteDTORegistrazione
from modello.modello_dto.utente_dto_login import UtenteDTOLogin
from modello.modello_dto.utente_dto import UtenteDTO
from service.service_auth import hash_password, verify_password, genera_token
from fastapi import HTTPException, status

class ServiceUtente:
    """
    Service layer per gestire tutta la logica dell'utente:
    registrazione, login, scelta generi, votazioni.
    """

    @staticmethod
    def registra_utente(user_dto: UtenteDTORegistrazione) -> UtenteDTO:
        # Controllo duplicati
        if Utente.objects(username=user_dto.username).first() or Utente.objects(email=user_dto.email).first():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Username o email già registrati")

        # Hash della password
        hashed_pw = hash_password(user_dto.password)

        # Creazione utente nel DB
        user = Utente(username=user_dto.username,
                      email=user_dto.email,
                      password=hashed_pw)
        user.save()

        return UtenteDTO(id=str(user.id), username=user.username, email=user.email)

    @staticmethod
    def login_utente(login_dto: UtenteDTOLogin) -> dict:
        # Ricerca utente per username o email
        user = Utente.objects(username=login_dto.username).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Credenziali non valide")

        # Verifica password
        if not verify_password(login_dto.password, user.password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Credenziali non valide")

        # Generazione JWT
        token = genera_token(user.email, ruolo="utente")  # qui puoi aggiungere ruoli futuri

        # Verifica se è il primo login (generi preferiti vuoti)
        is_first_login = len(user.generi_preferiti) == 0

        return {
            "access_token": token, 
            "utente": UtenteDTO(id=str(user.id), username=user.username, email=user.email),
            "first_login": is_first_login
        }
