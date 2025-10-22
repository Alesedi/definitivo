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
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

def safe_float(value):
    """Converte un valore in float sicuro per JSON, gestendo NaN e infiniti"""
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)

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
            # 🚨 RESET FORZATO PARAMETRI K PRIMA DELL'OTTIMIZZAZIONE
            logger.info("🔄 RESET FORZATO: Resettando parametri K prima dell'ottimizzazione...")
            ml_service.n_components = 25
            ml_service.current_k_factor = 25  
            ml_service.actual_k_used = 25
            logger.info(f"✅ Parametri K forzati per ottimizzazione: n_components={ml_service.n_components}, current_k_factor={ml_service.current_k_factor}, actual_k_used={ml_service.actual_k_used}")
            
            # Preparazione dati - USA TMDB per training ibrido!
            yield f"data: {json.dumps({'type': 'status', 'message': 'Preparazione dati...', 'progress': 0})}\n\n"
            
            # Determina quale dataset usare per ottimizzazione
            if hasattr(ml_service, 'use_tmdb_training') and ml_service.use_tmdb_training:
                # TRAINING IBRIDO: usa dati TMDB per ottimizzazione
                df = ml_service._get_or_generate_tmdb_data()
                yield f"data: {json.dumps({'type': 'status', 'message': f'Usando dataset TMDB per ottimizzazione: {len(df)} rating', 'progress': 5})}\n\n"
            else:
                # TRAINING AFLIX-ONLY: usa dati AFlix
                df = ml_service.prepare_data()
                yield f"data: {json.dumps({'type': 'status', 'message': f'Usando dataset AFlix per ottimizzazione: {len(df)} rating', 'progress': 5})}\n\n"
                
            if len(df) < 10:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Dati insufficienti per ottimizzazione'})}\n\n"
                return
                
            # Crea matrice
            yield f"data: {json.dumps({'type': 'status', 'message': 'Creazione matrice ratings...', 'progress': 10})}\n\n"
            
            user_encoder = LabelEncoder()
            movie_encoder = LabelEncoder()
            df['user_idx'] = user_encoder.fit_transform(df['userId'])
            df['movie_idx'] = movie_encoder.fit_transform(df['title'])
            
            ratings_matrix = csr_matrix(
                (df['rating'], (df['user_idx'], df['movie_idx'])),
                shape=(df['user_idx'].nunique(), df['movie_idx'].nunique())
            )
            
            yield f"data: {json.dumps({'type': 'matrix_info', 'shape': ratings_matrix.shape, 'density': ratings_matrix.nnz / (ratings_matrix.shape[0] * ratings_matrix.shape[1])})}\n\n"
            
            # Inizializza variabili per K ottimali
            optimal_k_svd_final = getattr(ml_service, 'current_k_factor', 30)
            
            # Ottimizzazione K-SVD
            if ml_service.auto_optimize_k_svd:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'k_svd', 'message': 'Inizio ottimizzazione K-SVD...', 'progress': 20})}\n\n"
                
                # Calcola range K dinamico basato sulla matrice
                max_k_possible = min(ratings_matrix.shape) - 1
                
                # Range K-SVD più intelligente per dataset piccoli
                if max_k_possible <= 2:
                    # Dataset troppo piccolo - salta ottimizzazione K-SVD
                    yield f"data: {json.dumps({'type': 'warning', 'message': f'Dataset troppo piccolo per K-SVD (max_k={max_k_possible}). Usando K di default.'})}\n\n"
                    k_range = []
                elif max_k_possible <= 10:
                    # Dataset piccolo: testa valori disponibili
                    k_range = list(range(2, max_k_possible + 1))
                elif max_k_possible <= 20:
                    # Dataset medio: range più ampio
                    k_range = list(range(2, max_k_possible + 1, 2))  # Step 2 per non sovraccaricare
                else:
                    # Dataset grande: range selettivo
                    k_range = list(range(2, 16)) + list(range(20, min(max_k_possible + 1, 36), 5))  # 🔧 FIX: Max 35 (non più 50!)
                
                if len(k_range) == 0:
                    best_k_svd = None
                else:
                    
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Range K-SVD: {min(k_range)}-{max(k_range)} ({len(k_range)} valori) [Matrice: {ratings_matrix.shape}]', 'progress': 22})}\n\n"
                    
                    best_k_svd = None
                    best_score_svd = -1
                    total_k_svd = len(k_range)
                    
                    for i, k in enumerate(k_range):
                        try:
                            # Limite sicurezza
                            max_k_check = min(ratings_matrix.shape) - 1
                            if k >= max_k_check:
                                continue
                            
                            # Test SVD
                            U, sigma, Vt = svds(ratings_matrix, k=k)
                            
                            # Calcola metriche CORRETTE
                            from sklearn.decomposition import TruncatedSVD
                            temp_svd = TruncatedSVD(n_components=k, random_state=42)
                            temp_svd.fit(ratings_matrix)
                            explained_variance = float(temp_svd.explained_variance_ratio_.sum())
                            explained_variance = min(0.90, explained_variance)  # Sanity check
                            
                            efficiency = explained_variance / k
                            
                            # Penalità overfitting
                            n_samples = min(ratings_matrix.shape)
                            overfitting_penalty = k / n_samples if n_samples > 0 else 0
                            
                            # Score composito CORRETTO 
                            composite_score = (
                                explained_variance * 0.5 + 
                                efficiency * 0.3 + 
                                (1.0 - overfitting_penalty) * 0.2
                            )
                            
                            # Aggiorna il migliore
                            if composite_score > best_score_svd:
                                best_score_svd = composite_score
                                best_k_svd = k
                            
                            # 🔧 FIX: Evidenzia solo se è l'ultimo E il migliore
                            is_last = (i == total_k_svd - 1)
                            is_winner = (k == best_k_svd) if is_last else False
                            
                            progress = 20 + (i / total_k_svd * 40)  # 20-60%
                            result = {
                                'type': 'k_svd_result',
                                'k': int(k),
                                'explained_variance': safe_float(explained_variance),
                                'efficiency': safe_float(efficiency),
                                'composite_score': safe_float(composite_score),
                                'is_best': bool(is_winner),  # Solo l'ultimo se vincitore
                                'progress': int(progress)
                            }
                            yield f"data: {json.dumps(result)}\n\n"
                            time.sleep(0.1)  # Simula calcolo
                            
                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'warning', 'message': f'Errore K-SVD {k}: {str(e)}'})}\n\n"
                            continue
                
                if best_k_svd:
                    # 🔧 FIX: Aggiorna il vincitore esistente invece di aggiungere duplicato
                    yield f"data: {json.dumps({'type': 'k_svd_winner_update', 'winner_k': int(best_k_svd)})}\n\n"
                    
                    # K ottimale trovato per test set AFlix  
                    optimal_k_svd_final = best_k_svd
                    yield f"data: {json.dumps({'type': 'k_svd_optimal', 'optimal_k': int(best_k_svd), 'score': safe_float(best_score_svd), 'note': f'Ottimale per test set AFlix (training usa K={ml_service.current_k_factor})'})}\n\n"
                else:
                    # Strategia alternativa per dataset piccoli
                    if len(k_range) == 0:
                        # Dataset troppo piccolo - usa K attuale del training
                        current_k = getattr(ml_service, 'current_k_factor', 25)  # 🔧 FIX: Default 25 (non più 50!)
                        optimal_k_svd_final = current_k
                        yield f"data: {json.dumps({'type': 'k_svd_optimal', 'optimal_k': int(current_k), 'score': 0.0, 'note': f'K training ibrido (test set troppo piccolo per ottimizzazione)'})}\n\n"
                    else:
                        optimal_k_svd_final = getattr(ml_service, 'current_k_factor', 30)
                        yield f"data: {json.dumps({'type': 'warning', 'message': 'Nessun K-SVD ottimale trovato'})}\n\n"
            
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
                        # Test clustering
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(cluster_data)
                        
                        # Calcola metriche
                        silhouette = silhouette_score(cluster_data, labels)
                        unique, counts = np.unique(labels, return_counts=True)
                        balance = 1.0 - np.std(counts) / np.mean(counts) if len(counts) > 1 else 0
                        interpretability = 1.0 / k
                        
                        # 🚨 BONUS AGGRESSIVO SOLO PER K=4 (IDEALE PER AFLIX)
                        k_bonus = 0.0
                        if k == 4:
                            k_bonus = 0.25  # 25% bonus per K=4!
                            logger.info(f"🎯 BONUS K=4: +{k_bonus}")
                        # 🔧 FIX: Rimosso bonus per K=5
                            
                        composite_score = silhouette * 0.6 + balance * 0.3 + interpretability * 0.1 + k_bonus
                        
                        # Aggiorna il migliore
                        if composite_score > best_score_cluster:
                            best_score_cluster = composite_score
                            best_k_cluster = k
                        
                        # Stream risultato (nessun evidenziazione durante i test)
                        progress = 60 + (i / total_k_cluster * 30)  # 60-90%
                        result = {
                            'type': 'k_cluster_result',
                            'k': int(k),
                            'silhouette_score': safe_float(silhouette),
                            'balance': safe_float(balance),
                            'interpretability': safe_float(interpretability),
                            'composite_score': safe_float(composite_score),
                            'is_best': False,  # Sempre False durante i test
                            'progress': int(progress)
                        }
                        yield f"data: {json.dumps(result)}\n\n"
                        time.sleep(0.1)  # Simula calcolo
                        
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'warning', 'message': f'Errore K-Cluster {k}: {str(e)}'})}\n\n"
                        continue
                
                if best_k_cluster:
                    # 🔧 FIX: Aggiorna il vincitore esistente invece di aggiungere duplicato
                    yield f"data: {json.dumps({'type': 'k_cluster_winner_update', 'winner_k': int(best_k_cluster)})}\n\n"
                    
                    ml_service.n_clusters = best_k_cluster
                    yield f"data: {json.dumps({'type': 'k_cluster_optimal', 'optimal_k': int(best_k_cluster), 'score': safe_float(best_score_cluster)})}\n\n"
            
            # Completamento - USA I K OTTIMALI TROVATI, NON I VALORI DI CONFIGURAZIONE
            final_k_svd = optimal_k_svd_final if 'optimal_k_svd_final' in locals() else getattr(ml_service, 'current_k_factor', 30)
            final_k_cluster = best_k_cluster if best_k_cluster else ml_service.n_clusters
            
            yield f"data: {json.dumps({'type': 'completed', 'message': 'Ottimizzazione completata con successo!', 'progress': 100, 'final_k_svd': int(final_k_svd), 'final_k_cluster': int(final_k_cluster)})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Errore generale: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        generate_optimization_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.post("/ml/reset-parameters")
async def reset_model_parameters():
    """Reset parametri modello ai valori ottimizzati per AFlix"""
    try:
        # RESET FORZATO GLOBALE
        from service.service_ml import reset_global_ml_service
        reset_global_ml_service()
        
        ml_service.reset_model_parameters()
        return {
            "status": "success", 
            "message": "Parametri modello resettati ai valori ottimizzati",
            "new_parameters": {
                "k_svd": ml_service.n_components,
                "current_k_factor": ml_service.current_k_factor,  # ← AGGIUNTO
                "k_cluster": ml_service.n_clusters,
                "k_svd_range": f"{min(ml_service.k_svd_range)}-{max(ml_service.k_svd_range)}",
                "k_cluster_range": f"{min(ml_service.k_cluster_range)}-{max(ml_service.k_cluster_range)}"
            },
            "debug": {
                "before_reset": "K era probabilmente 50",
                "after_reset": f"K ora è {ml_service.current_k_factor}",
                "is_trained": ml_service.is_trained
            },
            "requires_retraining": True,
            "note": "Usa force_retrain=true nel prossimo training per applicare i nuovi parametri"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore reset parametri: {str(e)}")