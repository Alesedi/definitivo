from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from modelli_ODM.voto_odm import Votazione
from modelli_ODM.utente_odm import Utente
from modelli_ODM.film_odm import Film
from service.service_ml import ml_service
from typing import List, Optional, Generator
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import json
import asyncio
import time

class KFactorOptimizationRequest(BaseModel):
    k_range: Optional[List[int]] = None

class TMDBConfigRequest(BaseModel):
    api_key: str
    use_tmdb_training: bool = True
    use_tmdb_testing: bool = False

class KOptimizationRequest(BaseModel):
    auto_optimize_k_svd: bool = True
    auto_optimize_k_cluster: bool = True
    k_svd_range_start: int = 10
    k_svd_range_end: int = 100
    k_svd_range_step: int = 10
    k_cluster_range_start: int = 2
    k_cluster_range_end: int = 15

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

@router.post("/configure-tmdb")
async def configure_tmdb(config: TMDBConfigRequest):
    """Configura TMDB per training ibrido"""
    try:
        # Aggiorna configurazione ML service
        ml_service.tmdb_api_key = config.api_key
        ml_service.use_tmdb_training = config.use_tmdb_training
        ml_service.use_tmdb_testing = config.use_tmdb_testing
        
        return {
            "status": "success",
            "message": "Configurazione TMDB aggiornata",
            "config": {
                "tmdb_enabled": bool(config.api_key),
                "use_tmdb_training": config.use_tmdb_training,
                "use_tmdb_testing": config.use_tmdb_testing
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore configurazione TMDB: {str(e)}")

@router.get("/tmdb-status")
async def get_tmdb_status():
    """Ottiene stato configurazione TMDB"""
    try:
        return {
            "tmdb_configured": bool(ml_service.tmdb_api_key),
            "use_tmdb_training": ml_service.use_tmdb_training,
            "use_tmdb_testing": ml_service.use_tmdb_testing,
            "training_source": getattr(ml_service, 'training_source', 'not_trained'),
            "cache_dir": ml_service.cache_dir
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore stato TMDB: {str(e)}")

@router.delete("/clear-tmdb-cache")
async def clear_tmdb_cache():
    """Pulisce cache TMDB per rigenerare dati"""
    try:
        import os
        import shutil
        
        if os.path.exists(ml_service.cache_dir):
            shutil.rmtree(ml_service.cache_dir)
            os.makedirs(ml_service.cache_dir, exist_ok=True)
            
        return {
            "status": "success", 
            "message": "Cache TMDB pulita"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore pulizia cache: {str(e)}")

@router.post("/configure-k-optimization")
async def configure_k_optimization(config: KOptimizationRequest):
    """Configura ottimizzazione automatica K-values"""
    try:
        # Aggiorna configurazione ML service
        ml_service.auto_optimize_k_svd = config.auto_optimize_k_svd
        ml_service.auto_optimize_k_cluster = config.auto_optimize_k_cluster
        
        # Aggiorna range di ottimizzazione
        ml_service.k_svd_range = range(
            config.k_svd_range_start, 
            config.k_svd_range_end + 1, 
            config.k_svd_range_step
        )
        ml_service.k_cluster_range = range(
            config.k_cluster_range_start, 
            config.k_cluster_range_end + 1
        )
        
        return {
            "status": "success",
            "message": "Configurazione ottimizzazione K aggiornata",
            "config": {
                "auto_optimize_k_svd": config.auto_optimize_k_svd,
                "auto_optimize_k_cluster": config.auto_optimize_k_cluster,
                "k_svd_range": f"{config.k_svd_range_start}-{config.k_svd_range_end} (step {config.k_svd_range_step})",
                "k_cluster_range": f"{config.k_cluster_range_start}-{config.k_cluster_range_end}"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore configurazione K: {str(e)}")

@router.get("/k-optimization-status")  
async def get_k_optimization_status():
    """Ottiene stato configurazione ottimizzazione K"""
    try:
        return {
            "auto_optimize_k_svd": ml_service.auto_optimize_k_svd,
            "auto_optimize_k_cluster": ml_service.auto_optimize_k_cluster,
            "current_k_svd": ml_service.n_components,
            "current_k_cluster": ml_service.n_clusters,
            "optimization_history": getattr(ml_service, 'k_performance_log', {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore stato K: {str(e)}")

@router.get("/stream-k-optimization")
async def stream_k_optimization():
    """Streaming real-time dell'ottimizzazione K-values"""
    
    def generate_optimization_stream() -> Generator[str, None, None]:
        try:
            # Preparazione dati
            yield f"data: {json.dumps({'type': 'status', 'message': 'Preparazione dati...', 'progress': 0})}\n\n"
            
            df = ml_service.prepare_data()
            if len(df) < 10:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Dati insufficienti per ottimizzazione'})}\n\n"
                return
                
            # Crea matrice
            yield f"data: {json.dumps({'type': 'status', 'message': 'Creazione matrice ratings...', 'progress': 10})}\n\n"
            
            from sklearn.preprocessing import LabelEncoder
            from scipy.sparse import csr_matrix
            
            user_encoder = LabelEncoder()
            movie_encoder = LabelEncoder()
            df['user_idx'] = user_encoder.fit_transform(df['userId'])
            df['movie_idx'] = movie_encoder.fit_transform(df['title'])
            
            ratings_matrix = csr_matrix(
                (df['rating'], (df['user_idx'], df['movie_idx'])),
                shape=(df['user_idx'].nunique(), df['movie_idx'].nunique())
            )
            
            yield f"data: {json.dumps({'type': 'matrix_info', 'shape': ratings_matrix.shape, 'density': ratings_matrix.nnz / (ratings_matrix.shape[0] * ratings_matrix.shape[1])})}\n\n"
            
            # Ottimizzazione K-SVD
            if ml_service.auto_optimize_k_svd:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'k_svd', 'message': 'Inizio ottimizzazione K-SVD...', 'progress': 20})}\n\n"
                
                best_k_svd = None
                best_score_svd = -1
                total_k_svd = len(list(ml_service.k_svd_range))
                
                for i, k in enumerate(ml_service.k_svd_range):
                    try:
                        # Limite sicurezza
                        max_k = min(ratings_matrix.shape) - 1
                        if k >= max_k:
                            continue
                        
                        # Test SVD
                        from scipy.sparse.linalg import svds
                        import numpy as np
                        
                        U, sigma, Vt = svds(ratings_matrix, k=k)
                        
                        # Calcola metriche
                        total_variance = np.sum(sigma ** 2)
                        explained_variance = total_variance / (ratings_matrix.nnz if hasattr(ratings_matrix, 'nnz') else ratings_matrix.size)
                        efficiency = explained_variance / k
                        composite_score = explained_variance * 0.7 + efficiency * 0.3
                        
                        # Determina se è il migliore
                        is_best = composite_score > best_score_svd
                        if is_best:
                            best_score_svd = composite_score
                            best_k_svd = k
                        
                        # Stream risultato
                        progress = 20 + (i / total_k_svd * 40)  # 20-60%
                        result = {
                            'type': 'k_svd_result',
                            'k': k,
                            'explained_variance': float(explained_variance),
                            'efficiency': float(efficiency),
                            'composite_score': float(composite_score),
                            'is_best': is_best,
                            'progress': int(progress)
                        }
                        yield f"data: {json.dumps(result)}\n\n"
                        time.sleep(0.1)  # Simula calcolo
                        
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'warning', 'message': f'Errore K-SVD {k}: {str(e)}'})}\n\n"
                        continue
                
                if best_k_svd:
                    ml_service.n_components = best_k_svd
                    ml_service.current_k_factor = best_k_svd
                    yield f"data: {json.dumps({'type': 'k_svd_optimal', 'optimal_k': best_k_svd, 'score': float(best_score_svd)})}\n\n"
            
            # Ottimizzazione K-Cluster
            if ml_service.auto_optimize_k_cluster:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'k_cluster', 'message': 'Inizio ottimizzazione K-Cluster...', 'progress': 60})}\n\n"
                
                # Prepara dati per clustering
                try:
                    k_for_clustering = min(20, min(ratings_matrix.shape) - 1)
                    U_cluster, sigma_cluster, Vt_cluster = svds(ratings_matrix, k=k_for_clustering)
                    
                    if U_cluster.shape[1] >= 2:
                        cluster_data = U_cluster[:, :2]
                    else:
                        cluster_data = np.column_stack([U_cluster[:, 0], np.random.normal(0, 0.1, U_cluster.shape[0])])
                        
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Errore preparazione clustering: {str(e)}'})}\n\n"
                    return
                
                best_k_cluster = None
                best_score_cluster = -1
                total_k_cluster = len(list(ml_service.k_cluster_range))
                
                for i, k in enumerate(ml_service.k_cluster_range):
                    if k >= len(cluster_data):
                        continue
                        
                    try:
                        from sklearn.cluster import KMeans
                        from sklearn.metrics import silhouette_score
                        
                        # Test clustering
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(cluster_data)
                        
                        # Calcola metriche
                        silhouette = silhouette_score(cluster_data, labels)
                        unique, counts = np.unique(labels, return_counts=True)
                        balance = 1.0 - np.std(counts) / np.mean(counts) if len(counts) > 1 else 0
                        interpretability = 1.0 / k
                        composite_score = silhouette * 0.6 + balance * 0.3 + interpretability * 0.1
                        
                        # Determina se è il migliore
                        is_best = composite_score > best_score_cluster
                        if is_best:
                            best_score_cluster = composite_score
                            best_k_cluster = k
                        
                        # Stream risultato
                        progress = 60 + (i / total_k_cluster * 30)  # 60-90%
                        result = {
                            'type': 'k_cluster_result',
                            'k': k,
                            'silhouette_score': float(silhouette),
                            'balance': float(balance),
                            'interpretability': float(interpretability),
                            'composite_score': float(composite_score),
                            'is_best': is_best,
                            'progress': int(progress)
                        }
                        yield f"data: {json.dumps(result)}\n\n"
                        time.sleep(0.1)  # Simula calcolo
                        
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'warning', 'message': f'Errore K-Cluster {k}: {str(e)}'})}\n\n"
                        continue
                
                if best_k_cluster:
                    ml_service.n_clusters = best_k_cluster
                    yield f"data: {json.dumps({'type': 'k_cluster_optimal', 'optimal_k': best_k_cluster, 'score': float(best_score_cluster)})}\n\n"
            
            # Completamento
            yield f"data: {json.dumps({'type': 'completed', 'message': 'Ottimizzazione completata!', 'progress': 100, 'final_k_svd': ml_service.n_components, 'final_k_cluster': ml_service.n_clusters})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Errore generale: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        generate_optimization_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )