from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from service.service_ml import ml_service
from modelli_ODM.utente_odm import Utente
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommendations"])

class TrainingResponse(BaseModel):
    message: str
    stats: Dict[str, Any]

class RecommendationResponse(BaseModel):
    title: str
    predicted_rating: float
    tmdb_id: Optional[int]
    genres: List[str]
    poster_url: Optional[str]
    tmdb_rating: Optional[float]
    cluster: int

class UserHistoryResponse(BaseModel):
    title: str
    rating: int
    genres: List[str]
    poster_url: Optional[str]
    tmdb_id: Optional[int]
    tmdb_rating: Optional[float]

class EvaluationResponse(BaseModel):
    rmse: Optional[float]
    mae: Optional[float]
    test_samples: Optional[int]
    train_samples: Optional[int]
    explained_variance: Optional[float]
    model_status: str
    error: Optional[str] = None

class ClusteringDataResponse(BaseModel):
    points: List[Dict[str, Any]]
    centroids: List[Dict[str, Any]]
    n_clusters: int
    error: Optional[str] = None

@router.post("/train", response_model=TrainingResponse)
async def train_recommendation_model(background_tasks: BackgroundTasks):
    """Addestra il modello di raccomandazione in background"""
    try:
        def train_model_task():
            try:
                stats = ml_service.train_model()
                logger.info(f"Model training completed: {stats}")
            except Exception as e:
                logger.error(f"Background training failed: {e}")
        
        # Avvia training in background
        background_tasks.add_task(train_model_task)
        
        return TrainingResponse(
            message="Training del modello avviato in background",
            stats={"status": "training_started"}
        )
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nell'avvio del training: {str(e)}")

@router.get("/train-sync")
async def train_recommendation_model_sync():
    """Addestra il modello di raccomandazione in modo sincrono (per testing)"""
    try:
        stats = ml_service.train_model()
        return TrainingResponse(
            message="Modello addestrato con successo",
            stats=stats
        )
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel training del modello: {str(e)}")

@router.get("/user/{user_id}", response_model=List[RecommendationResponse])
async def get_user_recommendations(user_id: str, top_n: int = 10):
    """Ottieni raccomandazioni personalizzate per un utente"""
    try:
        # Verifica che l'utente esista
        user = Utente.objects(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utente non trovato")
        
        # Genera raccomandazioni
        recommendations = ml_service.get_user_recommendations(user_id, top_n)
        
        return [RecommendationResponse(**rec) for rec in recommendations]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel generare raccomandazioni: {str(e)}")

@router.get("/user/{user_id}/history", response_model=List[UserHistoryResponse])
async def get_user_history(user_id: str):
    """Ottieni lo storico dei voti dell'utente"""
    try:
        # Verifica che l'utente esista
        user = Utente.objects(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utente non trovato")
        
        # Recupera storico
        history = ml_service.get_user_history(user_id)
        
        return [UserHistoryResponse(**item) for item in history]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user history for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel recuperare storico utente: {str(e)}")

@router.get("/clustering", response_model=ClusteringDataResponse)
async def get_clustering_data():
    """Ottieni dati per visualizzazione clustering dei film"""
    try:
        clustering_data = ml_service.get_clustering_data()
        
        if "error" in clustering_data:
            return ClusteringDataResponse(
                points=[],
                centroids=[],
                n_clusters=0,
                error=clustering_data["error"]
            )
        
        return ClusteringDataResponse(**clustering_data)
        
    except Exception as e:
        logger.error(f"Error getting clustering data: {e}")
        return ClusteringDataResponse(
            points=[],
            centroids=[],
            n_clusters=0,
            error=str(e)
        )

@router.get("/evaluation", response_model=EvaluationResponse)
async def get_model_evaluation():
    """Ottieni metriche di valutazione del modello"""
    try:
        evaluation = ml_service.evaluate_model()
        
        if "error" in evaluation:
            return EvaluationResponse(
                rmse=None,
                mae=None,
                test_samples=None,
                train_samples=None,
                explained_variance=None,
                model_status="error",
                error=evaluation["error"]
            )
        
        return EvaluationResponse(**evaluation)
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return EvaluationResponse(
            rmse=None,
            mae=None,
            test_samples=None,
            train_samples=None,
            explained_variance=None,
            model_status="error",
            error=str(e)
        )

@router.get("/status")
async def get_model_status():
    """Ottieni stato del modello di raccomandazione"""
    try:
        return ml_service.get_model_status()
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return {"error": str(e)}

@router.get("/popular", response_model=List[RecommendationResponse])
async def get_popular_recommendations(top_n: int = 10):
    """Ottieni raccomandazioni basate sulla popolarità (per utenti senza voti)"""
    try:
        # Usa il metodo interno per raccomandazioni popolari
        recommendations = ml_service._get_popular_recommendations(top_n)
        
        return [RecommendationResponse(**rec) for rec in recommendations]
        
    except Exception as e:
        logger.error(f"Error getting popular recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel generare raccomandazioni popolari: {str(e)}")

@router.get("/dashboard/{user_id}", response_model=List[RecommendationResponse])
async def get_dashboard_recommendations(user_id: str):
    """Raccomandazioni specifiche per dashboard utente (non random)"""
    try:
        # Verifica che l'utente esista
        user = Utente.objects(id=user_id).first()
        if not user:
            # Se utente non esiste, usa raccomandazioni popolari
            logger.warning(f"Dashboard: Utente {user_id} non trovato, uso raccomandazioni popolari")
            recommendations = ml_service._get_popular_recommendations(top_n=6)
        else:
            # Usa raccomandazioni personalizzate
            logger.info(f"Dashboard: Genero raccomandazioni personalizzate per utente {user_id}")
            recommendations = ml_service.get_user_recommendations(user_id, top_n=6)
        
        return [RecommendationResponse(**rec) for rec in recommendations]
        
    except Exception as e:
        logger.error(f"Error getting dashboard recommendations for user {user_id}: {e}")
        # Fallback a raccomandazioni popolari
        recommendations = ml_service._get_popular_recommendations(top_n=6)
        return [RecommendationResponse(**rec) for rec in recommendations]