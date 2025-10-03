from fastapi import APIRouter, HTTPException
from modelli_ODM.voto_odm import Votazione
from modelli_ODM.utente_odm import Utente
from modelli_ODM.film_odm import Film

router = APIRouter(tags=["admin"])

@router.delete("/delete-user/{user_id}")
async def delete_user_with_auto_cleanup(user_id: str):
    """Cancella un utente (le votazioni vengono cancellate automaticamente dal signal)"""
    try:
        # Trova l'utente
        user = Utente.objects(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utente non trovato")
        
        # Conta le votazioni prima della cancellazione
        votes_count = Votazione.objects(utente=user).count()
        username = user.username
        
        # Cancella l'utente (il signal cancellerà automaticamente le votazioni)
        user.delete()
        
        return {
            "message": "Utente cancellato con successo (votazioni auto-cancellate)",
            "deleted_user": username,
            "auto_deleted_votes": votes_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella cancellazione: {str(e)}")

@router.get("/cleanup-orphaned-votes")
async def cleanup_orphaned_votes():
    """Pulisce le votazioni orfane (che puntano a utenti inesistenti)"""
    try:
        cleaned_votes = 0
        total_votes = Votazione.objects.count()
        
        # Trova e rimuovi votazioni orfane
        orphaned_votes = []
        
        for vote in Votazione.objects.all():
            try:
                # Tenta di accedere all'utente
                user = vote.utente
                if user is None:
                    orphaned_votes.append(vote.id)
            except Exception:
                # Se fallisce il dereference, è una votazione orfana
                orphaned_votes.append(vote.id)
        
        # Cancella le votazioni orfane
        if orphaned_votes:
            Votazione.objects(id__in=orphaned_votes).delete()
            cleaned_votes = len(orphaned_votes)
        
        remaining_votes = Votazione.objects.count()
        
        return {
            "message": "Cleanup completato",
            "total_votes_before": total_votes,
            "orphaned_votes_removed": cleaned_votes,
            "remaining_votes": remaining_votes,
            "users_count": Utente.objects.count(),
            "movies_count": Film.objects.count()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante cleanup: {str(e)}")

@router.get("/database-stats")
async def get_database_stats():
    """Mostra statistiche del database"""
    try:
        users = list(Utente.objects.all())
        user_stats = []
        
        for user in users:
            user_votes = Votazione.objects(utente=user).count()
            user_stats.append({
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email,
                "votes_count": user_votes
            })
        
        return {
            "total_users": len(users),
            "total_movies": Film.objects.count(),
            "total_votes": Votazione.objects.count(),
            "user_details": user_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero stats: {str(e)}")