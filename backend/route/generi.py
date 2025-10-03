from fastapi import APIRouter, HTTPException
from modelli_ODM.utente_odm import Utente
from modello.modello_dto.generi_scelti import GeneriSceltiDTO
from modello.enum_genere import EnumGenere

router = APIRouter(tags=["generi"])

@router.get("/")
def get_genres():
    """Ritorna tutti i generi disponibili"""
    return [{"id": g.value, "name": g.name.replace('_', ' ').title()} for g in EnumGenere]

@router.post("/scelta/{id_utente}")
def scelta_generi(id_utente: str, dto: GeneriSceltiDTO):
    utente = Utente.objects(id=id_utente).first()
    if not utente:
        raise HTTPException(status_code=404, detail="Utente non trovato")

    if len(dto.generi) < 3:
        raise HTTPException(status_code=400, detail="Seleziona almeno 3 generi")

    # Verifica validità generi
    valid_genres = [g.value for g in EnumGenere]
    for g in dto.generi:
        if g not in valid_genres:
            raise HTTPException(status_code=400, detail=f"Genere non valido: {g}")

    utente.generi_preferiti = dto.generi
    utente.save()

    return {"messaggio": "Generi preferiti aggiornati correttamente", "generi": dto.generi}
