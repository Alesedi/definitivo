from fastapi import APIRouter, HTTPException
from modelli_ODM.voto_odm import Votazione
from modelli_ODM.utente_odm import Utente

router = APIRouter(tags=["user-management"])

@router.delete("/user/{user_id}")
async def delete_user_safely(user_id: str):
    """Cancella un utente e tutte le sue votazioni in modo sicuro"""
    try:
        # Trova l'utente
        user = Utente.objects(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utente non trovato")
        
        # Conta le votazioni dell'utente
        user_votes = Votazione.objects(utente=user)
        votes_count = user_votes.count()
        
        # Cancella prima tutte le votazioni dell'utente
        user_votes.delete()
        
        # Poi cancella l'utente
        user.delete()
        
        return {
            "message": "Utente cancellato con successo",
            "deleted_user": user.username,
            "deleted_votes": votes_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella cancellazione: {str(e)}")