from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from service.service_ml import ml_service, get_ml_service_for_source, ml_service_tmdb, ml_service_omdb
from service.train_events import publish_event, event_generator
from fastapi.responses import StreamingResponse
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
async def train_recommendation_model(background_tasks: BackgroundTasks, source: Optional[str] = 'tmdb'):
    """Addestra il modello di raccomandazione in background"""
    try:
        def train_model_task():
            try:
                svc = get_ml_service_for_source(source)
                stats = svc.train_model()
                # Publish event async (schedule)
                try:
                    import asyncio
                    asyncio.get_event_loop().create_task(publish_event(source or 'tmdb', {"type": "training_completed", "stats": stats}))
                except Exception:
                    logger.debug("Non è stato possibile pubblicare evento training_completed")
                logger.info(f"Model training completed ({source}): {stats}")
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
async def train_recommendation_model_sync(source: Optional[str] = 'tmdb'):
    """Addestra il modello di raccomandazione in modo sincrono (per testing)"""
    try:
        svc = get_ml_service_for_source(source)
        stats = svc.train_model()
        # publish event
        try:
            import asyncio
            asyncio.get_event_loop().create_task(publish_event(source or 'tmdb', {"type": "training_completed", "stats": stats}))
        except Exception:
            logger.debug("Non è stato possibile pubblicare evento training_completed (sync)")
        return TrainingResponse(
            message="Modello addestrato con successo",
            stats=stats
        )
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel training del modello: {str(e)}")

@router.get("/user/{user_id}", response_model=List[RecommendationResponse])
async def get_user_recommendations(user_id: str, top_n: int = 10, source: Optional[str] = 'tmdb'):
    """Ottieni raccomandazioni personalizzate per un utente"""
    try:
        # Verifica che l'utente esista
        user = Utente.objects(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utente non trovato")
        # Genera raccomandazioni dalla sorgente selezionata
        svc = get_ml_service_for_source(source)
        try:
            recommendations = svc.get_user_recommendations(user_id, top_n)
            return [RecommendationResponse(**rec) for rec in recommendations]
        except ValueError as ve:
            # Se il modello non è addestrato, forniamo raccomandazioni popolari come fallback
            if 'Model not trained' in str(ve):
                logger.warning(f"Model not trained for source={source}. Returning service-specific popular recommendations as fallback.")
                # Usa la lista "popolare" costruita dall'istanza del servizio (se disponibile)
                if hasattr(svc, 'get_service_popular_recommendations'):
                    popular = svc.get_service_popular_recommendations(top_n)
                else:
                    popular = svc._get_popular_recommendations(top_n)
                return [RecommendationResponse(**rec) for rec in popular]
            # Altrimenti rilancia
            raise
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel generare raccomandazioni: {str(e)}")

@router.get("/user/{user_id}/history", response_model=List[UserHistoryResponse])
async def get_user_history(user_id: str, source: Optional[str] = 'tmdb'):
    """Ottieni lo storico dei voti dell'utente"""
    try:
        # Verifica che l'utente esista
        user = Utente.objects(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utente non trovato")
        
        # Recupera storico dalla sorgente richiesta (per coerenza con le raccomandazioni)
        svc = get_ml_service_for_source(source)
        history = svc.get_user_history(user_id)
        
        return [UserHistoryResponse(**item) for item in history]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user history for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel recuperare storico utente: {str(e)}")

@router.get("/clustering", response_model=ClusteringDataResponse)
async def get_clustering_data(source: Optional[str] = 'tmdb'):
    """Ottieni dati per visualizzazione clustering dei film per una sorgente specifica (tmdb|omdb)"""
    try:
        svc = get_ml_service_for_source(source)
        clustering_data = svc.get_clustering_data()

        if "error" in clustering_data:
            return ClusteringDataResponse(
                points=[],
                centroids=[],
                n_clusters=0,
                error=clustering_data["error"]
            )

        return ClusteringDataResponse(**clustering_data)

    except Exception as e:
        logger.error(f"Error getting clustering data for source={source}: {e}")
        return ClusteringDataResponse(
            points=[],
            centroids=[],
            n_clusters=0,
            error=str(e)
        )

@router.get("/evaluation", response_model=EvaluationResponse)
async def get_model_evaluation(source: Optional[str] = 'tmdb'):
    """Ottieni metriche di valutazione del modello"""
    try:
        svc = get_ml_service_for_source(source)
        evaluation = svc.evaluate_model()
        
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
async def get_model_status(source: Optional[str] = 'tmdb'):
    """Ottieni stato del modello di raccomandazione"""
    try:
        svc = get_ml_service_for_source(source)
        return svc.get_model_status()
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return {"error": str(e)}


    @router.get('/train-events')
    async def stream_train_events(source: Optional[str] = 'tmdb'):
        """SSE endpoint per eventi di training (training_completed, logs, etc.)"""
        async def event_stream():
            async for chunk in event_generator(source or 'tmdb'):
                yield chunk

        return StreamingResponse(event_stream(), media_type='text/event-stream')

@router.get("/popular", response_model=List[RecommendationResponse])
async def get_popular_recommendations(top_n: int = 10, source: Optional[str] = 'tmdb'):
    """Ottieni raccomandazioni basate sulla popolarità (per utenti senza voti) per una specifica sorgente"""
    try:
        svc = get_ml_service_for_source(source)
        # Preferisci la lista "service-specific" se esiste
        if hasattr(svc, 'get_service_popular_recommendations'):
            recommendations = svc.get_service_popular_recommendations(top_n)
        else:
            recommendations = svc._get_popular_recommendations(top_n)

        return [RecommendationResponse(**rec) for rec in recommendations]

    except Exception as e:
        logger.error(f"Error getting popular recommendations for source={source}: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel generare raccomandazioni popolari: {str(e)}")


@router.get("/train-both")
async def train_both_models():
    """Utility dev endpoint: addestra sincronicamente entrambi i modelli (tmdb e omdb) e restituisce i risultati"""
    try:
        stats_tmdb = ml_service_tmdb.train_model()
    except Exception as e:
        logger.error(f"Error training TMDB model: {e}")
        stats_tmdb = {"error": str(e)}

    try:
        stats_omdb = ml_service_omdb.train_model()
    except Exception as e:
        logger.error(f"Error training OMDb model: {e}")
        stats_omdb = {"error": str(e)}

    return {
        "tmdb": stats_tmdb,
        "omdb": stats_omdb
    }


@router.get("/diagnostics")
async def get_model_diagnostics(source: Optional[str] = 'tmdb'):
    """Endpoint diagnostico: restituisce una "fingerprint" dello stato interno del modello per la sorgente richiesta.

    Utile per confrontare istanze TMDB vs OMDb (dimensione dataset, k usato, explained variance, popular list, ecc.).
    """
    try:
        svc = get_ml_service_for_source(source)

        def safe_len(obj):
            try:
                return len(obj)
            except Exception:
                return None

        # Basic attributes
        info = {
            'instance_name': getattr(svc, 'instance_name', source),
            'is_trained': bool(getattr(svc, 'is_trained', False)),
            'training_source': getattr(svc, 'training_source', None),
            'last_trained_at': getattr(svc, 'last_trained_at', None),
            'use_tmdb_training': getattr(svc, 'use_tmdb_training', False),
            'use_omdb_training': getattr(svc, 'use_omdb_training', False),
        }

        # Training counts and k/variance
        info.update({
            'training_ratings_count': int(getattr(svc, '_training_ratings_count', 0) or 0),
            'actual_k_used': int(getattr(svc, 'actual_k_used', 0) or 0),
            'explained_variance': float(getattr(svc, 'explained_variance', 0.0) or 0.0),
        })

        # Encoders sizes
        try:
            user_enc_size = safe_len(getattr(svc, 'user_encoder').classes_) if getattr(svc, 'user_encoder', None) is not None else None
        except Exception:
            user_enc_size = None
        try:
            movie_enc_size = safe_len(getattr(svc, 'movie_encoder').classes_) if getattr(svc, 'movie_encoder', None) is not None else None
        except Exception:
            movie_enc_size = None

        info['user_encoder_size'] = user_enc_size
        info['movie_encoder_size'] = movie_enc_size

        # Popular list
        try:
            popular = getattr(svc, '_service_popular_recommendations', None)
            info['service_popular_count'] = safe_len(popular) if popular is not None else 0
            info['service_popular_sample'] = popular[:8] if popular else []
        except Exception:
            info['service_popular_count'] = None
            info['service_popular_sample'] = []

        # Movie factors shape and basic stats
        try:
            mf = getattr(svc, 'movie_factors', None)
            if mf is not None:
                import numpy as _np
                mf_arr = _np.asarray(mf)
                info['movie_factors_shape'] = list(mf_arr.shape)
                info['movie_factors_mean'] = float(_np.mean(mf_arr))
                info['movie_factors_std'] = float(_np.std(mf_arr))
            else:
                info['movie_factors_shape'] = None
                info['movie_factors_mean'] = None
                info['movie_factors_std'] = None
        except Exception:
            info['movie_factors_shape'] = None
            info['movie_factors_mean'] = None
            info['movie_factors_std'] = None

        # Additional metadata
        info['train_movie_genres_count'] = safe_len(getattr(svc, 'train_movie_genres', {})) if getattr(svc, 'train_movie_genres', None) is not None else 0
        info['k_performance_log_keys'] = list(getattr(svc, 'k_performance_log', {}).keys()) if getattr(svc, 'k_performance_log', None) is not None else []

        # Cache info
        try:
            cache_dir = getattr(svc, 'cache_dir', None)
            if cache_dir and os.path.exists(cache_dir):
                cached_files = len([f for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))])
            else:
                cached_files = 0
            info['cache_dir'] = cache_dir
            info['cache_files_count'] = int(cached_files)
        except Exception:
            info['cache_dir'] = getattr(svc, 'cache_dir', None)
            info['cache_files_count'] = None

        return info

    except Exception as e:
        logger.error(f"Error generating diagnostics for source={source}: {e}")
        raise HTTPException(status_code=500, detail=f"Errore diagnostica: {str(e)}")

@router.get("/dashboard/{user_id}", response_model=List[RecommendationResponse])
async def get_dashboard_recommendations(user_id: str, source: Optional[str] = 'tmdb'):
    """Raccomandazioni specifiche per dashboard utente (non random)"""
    try:
        # Verifica che l'utente esista
        user = Utente.objects(id=user_id).first()
        if not user:
            # Se utente non esiste, usa raccomandazioni popolari per la sorgente richiesta
            logger.warning(f"Dashboard: Utente {user_id} non trovato, uso raccomandazioni popolari (source={source})")
            svc = get_ml_service_for_source(source)
            if hasattr(svc, 'get_service_popular_recommendations'):
                recommendations = svc.get_service_popular_recommendations(top_n=6)
            else:
                recommendations = svc._get_popular_recommendations(top_n=6)
        else:
            # Usa raccomandazioni personalizzate
            logger.info(f"Dashboard: Genero raccomandazioni personalizzate per utente {user_id} (source={source})")
            svc = get_ml_service_for_source(source)
            recommendations = svc.get_user_recommendations(user_id, top_n=6)
        
        return [RecommendationResponse(**rec) for rec in recommendations]
        
    except Exception as e:
        logger.error(f"Error getting dashboard recommendations for user {user_id}: {e}")
        # Fallback a raccomandazioni popolari
        recommendations = ml_service._get_popular_recommendations(top_n=6)
        return [RecommendationResponse(**rec) for rec in recommendations]