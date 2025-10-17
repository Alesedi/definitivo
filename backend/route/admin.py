from fastapi import APIRouter, HTTPException
from modelli_ODM.voto_odm import Votazione
from modelli_ODM.utente_odm import Utente
from modelli_ODM.film_odm import Film
from service.service_ml import ml_service
from typing import List, Optional
from pydantic import BaseModel
import pandas as pd
from datetime import datetime

class KFactorOptimizationRequest(BaseModel):
    k_range: Optional[List[int]] = None

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
        raise HTTPException(status_code=500, detail=f"Errore nel recupero statistiche: {str(e)}")

# =====================================================
# ENDPOINTS PER TRACCIAMENTO FATTORE K - SVD
# =====================================================

@router.get("/ml/k-factor-analysis")
async def get_k_factor_analysis():
    """
    Analizza il fattore k utilizzato nella SVD
    
    Restituisce:
    - Analisi dettagliata del numero di componenti
    - Varianza spiegata per componente
    - Raccomandazioni per ottimizzazione
    """
    try:
        analysis = ml_service.analyze_k_factor()
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'analisi del fattore k: {str(e)}")

@router.post("/ml/optimize-k-factor")
async def optimize_k_factor(request: KFactorOptimizationRequest):
    """
    Ottimizza il fattore k testando diversi valori
    
    Args:
        request: Oggetto con k_range (lista opzionale di valori k da testare)
                 Se non specificata, usa range automatico
                 
    Restituisce:
    - Risultati dell'ottimizzazione
    - Valore k ottimale raccomandato
    - Performance comparison
    """
    try:
        k_range = request.k_range
        if k_range and (min(k_range) < 1 or max(k_range) > 100):
            raise HTTPException(status_code=400, detail="k_range deve essere tra 1 e 100")
            
        optimization = ml_service.optimize_k_factor(k_range)
        return optimization
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'ottimizzazione k: {str(e)}")

@router.get("/ml/k-factor-report")
async def get_k_factor_report():
    """
    Genera un report completo sul fattore k
    
    Restituisce:
    - Status attuale del modello
    - Analisi del fattore k
    - Storico delle ottimizzazioni
    - Raccomandazioni personalizzate
    """
    try:
        report = ml_service.get_k_factor_report()
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella generazione del report k: {str(e)}")

@router.get("/ml/model-status-extended")
async def get_model_status_extended():
    """
    Status esteso del modello ML con dettagli sul fattore k
    
    Restituisce:
    - Informazioni complete sul modello
    - Dettagli sul fattore k utilizzato
    - Efficienza dei componenti
    - Disponibilità ottimizzazioni
    """
    try:
        status = ml_service.get_model_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero status modello: {str(e)}")

@router.post("/ml/update-k-components/{new_k}")
async def update_k_components(new_k: int):
    """
    Aggiorna il numero di componenti k e riaddestra il modello
    
    Args:
        new_k: Nuovo numero di componenti SVD da utilizzare
        
    Restituisce:
    - Risultati del nuovo training
    - Confronto con il modello precedente
    """
    try:
        if new_k < 1 or new_k > 100:
            raise HTTPException(status_code=400, detail="new_k deve essere tra 1 e 100")
        
        # Salva stato precedente
        old_k = ml_service.actual_k_used
        old_variance = ml_service.explained_variance
        
        # Aggiorna parametri
        ml_service.n_components = new_k
        
        # Riaddestra modello
        new_stats = ml_service.train_model()
        
        # Confronto
        comparison = {
            "update_successful": True,
            "old_k": old_k,
            "new_k": ml_service.actual_k_used,
            "old_variance": float(old_variance) if old_variance else 0,
            "new_variance": float(ml_service.explained_variance),
            "improvement": float(ml_service.explained_variance) > float(old_variance) if old_variance else True,
            "training_stats": new_stats
        }
        
        return comparison
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'aggiornamento k: {str(e)}")

@router.get("/ml/live-monitor")
async def get_live_monitor():
    """
    Endpoint per monitoraggio live del sistema ML
    
    Restituisce informazioni in tempo reale su:
    - Status del modello
    - Ultima attività
    - Performance attuali
    - Logs recenti
    """
    try:
        # Status base
        status = ml_service.get_model_status()
        
        # Informazioni sul dataset
        dataset_info = {
            "total_users": Utente.objects.count(),
            "total_movies": Film.objects.count(), 
            "total_ratings": Votazione.objects.count()
        }
        
        # Informazioni aggiuntive per monitoring
        monitor_data = {
            "timestamp": datetime.now().isoformat(),
            "model_status": status,
            "dataset_info": dataset_info,
            "current_performance": {
                "k_used": ml_service.actual_k_used if ml_service.actual_k_used > 0 else None,
                "explained_variance": float(ml_service.explained_variance) if ml_service.explained_variance else 0,
                "k_efficiency": float(status.get('k_efficiency', 0)),
                "components_breakdown": ml_service.variance_per_component[:10] if hasattr(ml_service, 'variance_per_component') else []
            },
            "optimization_history": ml_service.k_performance_log if hasattr(ml_service, 'k_performance_log') else {},
            "recommendations": []
        }
        
        # Genera raccomandazioni real-time
        if ml_service.is_trained:
            if status.get('explained_variance', 0) < 0.5:
                monitor_data["recommendations"].append({
                    "level": "warning",
                    "message": "Bassa varianza spiegata - considera aumentare k"
                })
            
            if status.get('k_efficiency', 0) < 0.02:
                monitor_data["recommendations"].append({
                    "level": "optimization",
                    "message": "Bassa efficienza k - raccomando ottimizzazione"
                })
        
        return monitor_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel monitoraggio live: {str(e)}")