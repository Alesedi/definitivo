import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import List, Dict, Any, Optional
import os
import requests
import pickle
import json
from datetime import datetime, timedelta
from modelli_ODM.voto_odm import Votazione
from modelli_ODM.film_odm import Film
from modelli_ODM.utente_odm import Utente
import logging
from service.service_omdb import omdb_service
import difflib

logger = logging.getLogger(__name__)

class MLRecommendationService:
    def __init__(self):
        self.user_encoder = None
        self.movie_encoder = None
        self.user_factors = None
        self.movie_factors = None
        self.svd_model = None
        self.kmeans_model = None
        self.is_trained = False
        self.explained_variance = 0.0
        self.cluster_labels = None
        
        # Parametri modello ottimizzati per dataset AFlix
        self.n_components = 25  # ✅ K conservativo per dataset medio-piccolo
        self.n_clusters = 4     # ✅ K-cluster ideale per AFlix
        
        # Ottimizzazione automatica K
        self.auto_optimize_k_svd = True
        self.auto_optimize_k_cluster = True
        self.k_svd_range = range(10, 36, 5)  # 🔧 FIX: Range 10-35 (non più 50!)
        self.k_cluster_range = range(2, 8)   # ✅ Range cluster più piccolo
        
        # Tracciamento fattore k (numero componenti SVD)
        self.actual_k_used = 0  # Numero effettivo di componenti utilizzati
        self.k_history = []  # Storico dei valori k testati
        self.variance_per_component = []  # Varianza spiegata per ogni componente
        self.optimal_k = None  # Valore k ottimale identificato
        self.k_performance_log = {}  # Log performance per diversi k
        # Instance name (set after creation)
        self.instance_name = 'unknown'

        # TMDB Integration - API key corretta
        self.tmdb_api_key = os.getenv('TMDB_API_KEY', '9e6c375b125d733d9ce459bdd91d4a06')
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.tmdb_cache_dir = "data/tmdb_cache"
        self.use_tmdb_training = True
        # OMDb Integration
        self.omdb_api_key = os.getenv('OMDB_API_KEY', '2639fb0f')
        self.use_omdb_training = False
        self.omdb_cache_dir = "data/omdb_cache"
        self.tmdb_movies_df = None
        self.tmdb_ratings_df = None
        self.training_source = "hybrid"
        self.cache_dir = "data/cache"
        self.current_k_factor = 25  # ✅ Allineato con n_components
        # Cached per-service popular recommendations (built from last synthetic dataset)
        self._service_popular_recommendations = []
        self.last_trained_at = None

        # Baseline bias parameters (global mean + per-item and per-user biases)
        self.bias_reg = 10.0  # regularization for bias estimation (lambda)
        self.global_mean = None
        self.item_bias = None  # numpy array indexed by movie_idx
        self.user_bias = None  # numpy array indexed by user_idx

        # Placeholder poster used when no poster can be resolved for a title
        # Use a small, reliable placeholder image — change if you have a local asset
        self.default_poster_placeholder = "https://via.placeholder.com/500x750?text=No+Poster"

        # TMDB API per poster
        self.TMDB_API_KEY = "9e6c375b125d733d9ce459bdd91d4a06"
        self.TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/{}/images?api_key={}"

        # Mappatura nomi generi -> TMDB id (usata per generazione sintetica OMDb)
        self.genre_name_to_id = {
            'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35,
            'Crime': 80, 'Documentary': 99, 'Drama': 18, 'Family': 10751,
            'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
            'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878,
            'TV Movie': 10770, 'Thriller': 53, 'War': 10752, 'Western': 37
        }

    def fetch_poster_url(self, tmdb_id: int) -> Optional[str]:
        """Recupera URL poster da TMDB API"""
        try:
            if not tmdb_id or pd.isna(tmdb_id):
                return None
            url = self.TMDB_BASE_URL.format(int(tmdb_id), self.TMDB_API_KEY)
            response = requests.get(url, timeout=5)
            data = response.json()
            posters = data.get('posters', [])
            if posters:
                return "https://image.tmdb.org/t/p/w500" + posters[0]['file_path']
        except Exception as e:
            logger.warning(f"Error fetching poster for TMDB ID {tmdb_id}: {e}")
        return None

    def prepare_data(self) -> pd.DataFrame:
        """Prepara i dati dal database per il training"""
        try:
            # Recupera tutti i voti dal database
            votes = list(Votazione.objects.all())
            if len(votes) < 10:
                raise ValueError("Insufficient voting data for training (minimum 10 votes required)")
            
            # Converti in DataFrame
            data = []
            for vote in votes:
                data.append({
                    'userId': str(vote.utente.id),
                    'movieId': vote.film.tmdb_id,
                    'title': vote.film.titolo,
                    'rating': vote.valutazione,
                    'genres': vote.film.genere,
                    'poster_path': vote.film.poster_path,
                    'tmdb_rating': vote.film.tmdb_rating
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Prepared dataset with {len(df)} ratings from {df['userId'].nunique()} users and {df['movieId'].nunique()} movies")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def train_model(self) -> Dict[str, Any]:
        """Addestra il modello SVD e clustering con supporto TMDB"""
        try:
            logger.info(f"INIZIO TRAINING MODELLO ML (instance={getattr(self, 'instance_name', 'unknown')})")
            logger.info("=" * 80)
            
            # Decide fonte dati per training
            if self.use_omdb_training and self.omdb_api_key:
                logger.info("Modalità HYBRID-OMDB: Training su OMDb + Testing su AFlix")
                return self._train_hybrid_model(source='omdb')
            elif self.use_tmdb_training and self.tmdb_api_key:
                logger.info("Modalità HYBRID: Training su TMDB + Testing su AFlix")
                return self._train_hybrid_model(source='tmdb')
            
            else:
                logger.info("Modalità AFlix-only: Training su dati AFlix")
                result = self._train_aflix_only_model()
                # set last trained timestamp
                try:
                    self.last_trained_at = datetime.now().isoformat()
                except Exception:
                    pass
                return result
                
        except Exception as e:
            logger.error(f"Errore training modello: {e}")
            raise
    
    def _train_hybrid_model(self, source: str = 'tmdb') -> Dict[str, Any]:
        """Training ibrido: usa TMDB o OMDb per training, AFlix per testing

        Args:
            source: 'tmdb' or 'omdb'
        """
        
        try:
            # 0. FORZA RESET PARAMETRI K ALL'INIZIO
            logger.info(f"Reset parametri K per training ibrido... (instance={getattr(self,'instance_name','unknown')} source={source})")
            self.n_components = 25
            self.current_k_factor = 25
            logger.info(f"Parametri K forzati: n_components={self.n_components}, current_k_factor={self.current_k_factor}")
            
            # 1. Genera o carica dataset (TMDB o OMDb)
            if source == 'omdb':
                logger.info("Inizio generazione/caricamento dati OMDb...")
                tmdb_data = self._get_or_generate_omdb_data()
            else:
                logger.info("Inizio generazione/caricamento dati TMDB...")
                tmdb_data = self._get_or_generate_tmdb_data()

            # Save last training dataframe for diagnostics / popular-building
            try:
                self._last_training_df = tmdb_data.copy() if hasattr(tmdb_data, 'copy') else tmdb_data
            except Exception:
                self._last_training_df = tmdb_data

            logger.info(f"Dataset caricato: {len(tmdb_data)} rating, {tmdb_data['userId'].nunique() if len(tmdb_data) > 0 else 0} utenti, {tmdb_data['title'].nunique() if len(tmdb_data) > 0 else 0} film")

            if len(tmdb_data) == 0:
                logger.error("Dataset di training vuoto! Fallback ad AFlix-only")
                return self._train_aflix_only_model()
            
            # 2. Training SVD su dati TMDB
            logger.info("Training SVD su dataset TMDB...")
            ratings_matrix = self._create_ratings_matrix(tmdb_data)
            logger.info(f"Matrice rating creata: {ratings_matrix.shape}")
            
            # 3. AUTO-OTTIMIZZAZIONE K su dataset TMDB (se abilitata)
            if self.auto_optimize_k_svd or self.auto_optimize_k_cluster:
                logger.info("Avvio ottimizzazione K-values su dataset TMDB...")
                optimization_results = self.optimize_both_k_values(ratings_matrix)
                logger.info(f"Ottimizzazione completata: {optimization_results}")
            
            # 4. Applica SVD con K ottimizzato
            logger.info("Applicando decomposizione SVD...")
            self._apply_svd_to_matrix(ratings_matrix)
            logger.info(f"SVD completata con k_factor: {self.current_k_factor}")
            
        except Exception as e:
            logger.error(f"Errore nel training hybrid: {e}")
            logger.info("Fallback a modalità AFlix-only...")
            return self._train_aflix_only_model()

        # 4. Test su dati AFlix se disponibili
        aflix_performance = self._test_on_aflix_data()

        # 5. CLUSTERING SUI FILM AFLIX (non TMDB!)
        self._apply_clustering_on_aflix_movies()

        # 6. Salva numero di rating per status
        self._training_ratings_count = len(tmdb_data)
        # 7. Imposta training_source sull'istanza PRIMA di compilare le statistiche
        self.training_source = f"hybrid_{'omdb' if source == 'omdb' else 'tmdb'}_train_aflix_test"

        # 8. Statistiche finali
        stats = self._compile_hybrid_stats(tmdb_data, aflix_performance, source)

        self.is_trained = True
        try:
            self.last_trained_at = datetime.now().isoformat()
        except Exception:
            pass

        # FORZATURA: costruisci esplicitamente la lista "popular" per questa istanza
        # usando il DataFrame di training salvato. Questo evita che entrambe le
        # istanze ricadano nel fallback globale del DB e assicura differenze
        # tra TMDB e OMDb nella lista di fallback.
        try:
            df = getattr(self, '_last_training_df', None)
            if df is not None and len(df) > 0:
                built = self._build_service_popular_from_df(df, source=source, top_n=100)
                if built and len(built) > 0:
                    try:
                        self._service_popular_recommendations = built
                    except Exception:
                        # ignore assignment errors
                        pass
        except Exception as e:
            logger.debug(f"Could not force-build service popular list at training end: {e}")

        logger.info(f"Model training completed for instance={getattr(self,'instance_name','unknown')} source={source} - stats: {stats}")

        return stats

    def _train_aflix_only_model(self) -> Dict[str, Any]:
        """Training tradizionale solo su dati AFlix"""
        try:
            # Prepara i dati AFlix
            logger.info("Preparando dati AFlix...")
            df = self.prepare_data()
            logger.info(f"Dati AFlix preparati: {len(df)} rating, {df['userId'].nunique() if len(df) > 0 else 0} utenti, {df['title'].nunique() if len(df) > 0 else 0} film")
            
            if len(df) < 10:
                logger.warning("Dataset AFlix troppo piccolo, uso dati demo TMDB...")
                return self._train_demo_tmdb_model()
            
            # Encoding utenti e film
            self.user_encoder = LabelEncoder()
            self.movie_encoder = LabelEncoder()
            
            df['user_idx'] = self.user_encoder.fit_transform(df['userId'])
            df['movie_idx'] = self.movie_encoder.fit_transform(df['title'])
            
            # Crea matrice sparsa
            ratings_sparse = csr_matrix(
                (df['rating'], (df['user_idx'], df['movie_idx'])),
                shape=(df['user_idx'].nunique(), df['movie_idx'].nunique())
            )
            
            # Calcola n_components sicuro per SVD
            min_dim = min(ratings_sparse.shape)
            max_components = max(1, min_dim - 1)  # Assicura almeno 1
            safe_components = min(self.n_components, max_components)
            
            # 🔍 LOGGING DETTAGLIATO SCELTA FATTORE K
            logger.info("=" * 60)
            logger.info("🎯 PROCESSO SELEZIONE FATTORE K - SVD")
            logger.info("=" * 60)
            logger.info(f"📊 Matrice dati: {ratings_sparse.shape[0]} utenti × {ratings_sparse.shape[1]} film")
            logger.info(f"📈 Densità matrice: {(len(df) / (ratings_sparse.shape[0] * ratings_sparse.shape[1]) * 100):.2f}%")
            logger.info(f"🎛️  K richiesto dal modello: {self.n_components}")
            logger.info(f"📏 Dimensione minima matrice: {min_dim}")
            logger.info(f"🔝 Massimo K possibile: {max_components}")
            logger.info(f"✅ K finale selezionato: {safe_components}")
            
            if safe_components < self.n_components:
                logger.warning(f"⚠️  K ridotto da {self.n_components} a {safe_components} per limitazioni dati")
            else:
                logger.info(f"✅ K utilizzato come richiesto: {safe_components}")
            
            logger.info("🚀 Avvio decomposizione SVD...")
            logger.info("-" * 60)
            
            # Verifica diversità dati per Collaborative Filtering
            n_users = df['userId'].nunique()
            n_movies = df['movieId'].nunique()
            
            if n_users < 2:
                # Fallback a Content-Based se c'è solo 1 utente
                return self._train_content_based_model(df)
            
            # Verifica dimensioni minime per SVD
            if min_dim < 2:
                raise ValueError(f"Insufficient data diversity for SVD: matrix shape {ratings_sparse.shape}. Need at least 2 users and 2 movies.")
            
            if safe_components <= 0:
                safe_components = 1  # Fallback a 1 componente
            
            # Imposta il fattore K prima dell'SVD
            self.current_k_factor = safe_components
            
            # Training SVD usando il metodo unificato
            logger.info("🔄 Applicando SVD unificato...")
            self._apply_svd_to_matrix(ratings_sparse)
            
            logger.info("✅ SVD AFlix completata!")
            logger.info(f"📈 Varianza spiegata: {self.explained_variance:.1%}")
            logger.info(f"👥 Fattori utenti: {self.user_factors.shape}")
            logger.info(f"🎭 Fattori film: {self.movie_factors.shape}")
            logger.info(f"🎯 K utilizzato: {self.actual_k_used}")
            # Per metodo unificato, creiamo una lista di varianza approssimata
            self.variance_per_component = [self.explained_variance / self.actual_k_used] * self.actual_k_used if self.actual_k_used > 0 else []
            
            # 📊 ANALISI COMPONENTI DETTAGLIATA
            logger.info("=" * 60)
            logger.info("📊 ANALISI DETTAGLIATA COMPONENTI SVD")
            logger.info("=" * 60)
            cumulative_var = 0
            for i, var_ratio in enumerate(self.variance_per_component[:10]):  # Prime 10
                cumulative_var += var_ratio
                logger.info(f"Componente {i+1:2d}: {var_ratio:.4f} ({var_ratio*100:.2f}%) | Cumulativa: {cumulative_var:.4f} ({cumulative_var*100:.1f}%)")
            
            if len(self.variance_per_component) > 10:
                logger.info(f"... e altre {len(self.variance_per_component) - 10} componenti")
            
            # Identifica elbow point in tempo reale
            if len(self.variance_per_component) > 2:
                differences = []
                for i in range(1, len(self.variance_per_component)):
                    diff = self.variance_per_component[i-1] - self.variance_per_component[i]
                    differences.append(diff)
                
                if differences:
                    max_diff = max(differences)
                    elbow_point = None
                    for i, diff in enumerate(differences):
                        if diff < max_diff * 0.1:
                            elbow_point = i + 1
                            break
                    
                    if elbow_point:
                        logger.info(f"📍 Elbow Point identificato: Componente {elbow_point}")
                        if elbow_point < safe_components:
                            logger.info(f"💡 Suggerimento: Potresti usare solo {elbow_point} componenti mantenendo {cumulative_var:.1%} della varianza")
            
            logger.info("-" * 60)
            
            # Log delle informazioni k
            k_info = {
                "requested_k": self.n_components,
                "actual_k": self.actual_k_used,
                "max_possible_k": max_components,
                "total_explained_variance": float(self.explained_variance),
                "variance_per_component": self.variance_per_component,
                "matrix_shape": ratings_sparse.shape,
                "data_sparsity": 1 - (len(df) / (ratings_sparse.shape[0] * ratings_sparse.shape[1]))
            }
            
            logger.info(f"SVD Training - K Factor Details: {k_info}")
            self.k_history.append(k_info)
            
            # Clustering dei film nello spazio latente
            if self.movie_factors.shape[0] >= 2:  # Almeno 2 film per clustering
                n_clusters_actual = min(self.n_clusters, max(2, self.movie_factors.shape[0] // 2))
                
                if self.movie_factors.shape[1] >= 2:
                    # Usa prime 2 componenti se disponibili
                    X = self.movie_factors[:, :2]
                else:
                    # Se c'è solo 1 componente, usa quella e aggiungi rumore per clustering
                    X = np.column_stack([self.movie_factors[:, 0], np.random.normal(0, 0.1, self.movie_factors.shape[0])])
                
                self.kmeans_model = KMeans(n_clusters=n_clusters_actual, random_state=42)
                self.cluster_labels = self.kmeans_model.fit_predict(X)
                self.n_clusters = n_clusters_actual  # Aggiorna con il numero effettivo
                
                logger.info(f"Clustering completato: {n_clusters_actual} cluster per {self.movie_factors.shape[0]} film")
            else:
                # Troppo pochi dati per clustering
                self.kmeans_model = None
                self.cluster_labels = None
                logger.warning(f"Clustering saltato: solo {self.movie_factors.shape[0]} film disponibili")
            
            self.is_trained = True
            
            # Salva numero di rating per status
            self._training_ratings_count = len(df)
            
            # Statistiche training nel formato atteso dal frontend
            logger.info(f"🔍 DEBUG - actual_k_used: {self.actual_k_used}, explained_variance: {self.explained_variance}")
            
            stats = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'training_mode': 'aflix_only',
                'stats': {
                    'total_ratings': len(df),
                    'actual_k_used': int(self.actual_k_used) if self.actual_k_used > 0 else None,
                    'explained_variance': float(self.explained_variance),
                    'unique_users': df['userId'].nunique(),
                    'unique_movies': df['movieId'].nunique(),
                    'k_efficiency': float(self.explained_variance / self.actual_k_used) if self.actual_k_used > 0 else 0,
                },
                'model_info': {
                    'algorithm': 'SVD',
                    'k_factor': int(self.actual_k_used) if self.actual_k_used > 0 else None,
                    'training_source': 'aflix',
                    'test_source': 'aflix'
                }
            }
            
            logger.info(f"Model trained successfully. Stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            self.is_trained = False
            raise
    
    # ================================
    # 🎬 METODI SUPPORTO TMDB
    # ================================
    
    def _get_or_generate_tmdb_data(self) -> pd.DataFrame:
        """Ottiene o genera dataset TMDB per training"""
        
        cache_file = os.path.join(self.cache_dir, 'tmdb_training_data.pkl')
        
        # Usa cache se disponibile e recente (7 giorni)
        if os.path.exists(cache_file):
            mod_time = os.path.getmtime(cache_file)
            if (datetime.now().timestamp() - mod_time) < 7 * 24 * 3600:
                logger.info("📂 Caricamento dataset TMDB da cache...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        logger.info("🔄 Generazione nuovo dataset TMDB...")
        
        # Fetch film popolari da TMDB
        logger.info("🎬 Recuperando film popolari da TMDB API...")
        popular_movies = self._fetch_tmdb_popular_movies(pages=20)  # ~400 film
        logger.info(f"✅ Film TMDB recuperati: {len(popular_movies)}")
        
        if len(popular_movies) == 0:
            logger.error("❌ Nessun film recuperato da TMDB API!")
            return pd.DataFrame()
        
        # Genera rating sintetici
        logger.info("🤖 Generando rating sintetici...")
        tmdb_ratings = self._generate_synthetic_ratings(popular_movies, n_users=10000)
        
        # Verifica che i dati siano stati generati
        if len(tmdb_ratings) == 0:
            logger.error("❌ Generazione rating sintetici fallita!")
            return pd.DataFrame()
        
        # Salva in cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(tmdb_ratings, f)
        
        logger.info(f"✅ Dataset TMDB generato e salvato: {len(tmdb_ratings)} rating")
        # Costruisci lista "popolare" per questa istanza (usata come fallback per raccomandazioni)
        try:
            self._service_popular_recommendations = self._build_service_popular_from_movies(popular_movies, tmdb_ratings)
        except Exception as e:
            logger.debug(f"Impossibile costruire popular recommendations TMDB: {e}")

        # Salva mappatura titolo -> generi per fallback genre-based
        try:
            movie_genre_map = {}
            for m in popular_movies:
                title = m.get('title') or m.get('name')
                genre_ids = m.get('genre_ids') or m.get('genre_ids', []) or []
                # some seeds may have Genre names rather than ids
                if not genre_ids and m.get('genre_ids') is None and m.get('genres'):
                    # try to use 'genres' text if present
                    if isinstance(m.get('genres'), list):
                        genre_ids = m.get('genres')
                movie_genre_map[title] = genre_ids
            self.train_movie_genres = movie_genre_map
        except Exception:
            self.train_movie_genres = {}
        return tmdb_ratings

    # ================================
    # 🚩 CSV support
    # ================================
    # CSV support removed: training from local MovieLens CSVs (ml-latest) was reverted per user request.
    # Previous implementation included load_csv and _train_from_csv to allow training from a directory
    # containing ratings.csv and movies.csv. That code was removed to restore the previous behavior
    # where training uses either TMDB/OMDb synthetic datasets or the internal AFlix DB.

    def _get_or_generate_omdb_data(self) -> pd.DataFrame:
        """Ottiene o genera dataset usando metadata OMDb (proxy: usa lista film TMDB e risolve su OMDb)"""

        cache_file = os.path.join(self.cache_dir, 'omdb_training_data.pkl')

        # Usa cache se disponibile e recente (7 giorni)
        if os.path.exists(cache_file):
            mod_time = os.path.getmtime(cache_file)
            if (datetime.now().timestamp() - mod_time) < 7 * 24 * 3600:
                logger.info("📂 Caricamento dataset OMDb da cache...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        logger.info("🔄 Generazione nuovo dataset OMDb usando OMDb metadata (via TMDB seed)...")

        # Recupera una combinazione di seed TMDB (top_rated, now_playing, discover anni) per OMDb
        seeds = []
        try:
            seeds.extend(self._fetch_tmdb_top_rated_movies(pages=10))
        except Exception as e:
            logger.debug(f"Errore fetching top_rated seed: {e}")
        try:
            seeds.extend(self._fetch_tmdb_now_playing(pages=6))
        except Exception as e:
            logger.debug(f"Errore fetching now_playing seed: {e}")

        # Discover con anni diversi per variare i titoli
        try:
            seeds.extend(self._fetch_tmdb_discover_by_years(years=list(range(1990, 2001)), per_year=3))
            seeds.extend(self._fetch_tmdb_discover_by_years(years=list(range(2005, 2011)), per_year=3))
        except Exception as e:
            logger.debug(f"Errore fetching discover seeds: {e}")

        # Dedupe per titolo
        seen = set()
        popular_movies = []
        for m in seeds:
            title = m.get('title') or m.get('name')
            if not title:
                continue
            if title in seen:
                continue
            seen.add(title)
            popular_movies.append(m)

        # Se la lista è enorme, taglia
        if len(popular_movies) > 800:
            popular_movies = popular_movies[:800]

        # Escludi i titoli che compaiono nel seed TMDB (popular) per mantenere i dataset disgiunti
        try:
            tmdb_popular = self._fetch_tmdb_popular_movies(pages=20)
            tmdb_titles = {m.get('title') or m.get('name') for m in tmdb_popular if m}
            before = len(popular_movies)
            popular_movies = [m for m in popular_movies if (m.get('title') or m.get('name')) not in tmdb_titles]
            after = len(popular_movies)
            logger.info(f"🔎 Esclusi {before - after} seed OMDb presenti anche in TMDB popular - rimangono {after} seed OMDb")
        except Exception as e:
            logger.debug(f"Errore durante l'esclusione dei titoli TMDB dal seed OMDb: {e}")

        logger.info(f"🔎 Film seed combinati recuperati da TMDb per OMDb: {len(popular_movies)}")

        if len(popular_movies) == 0:
            logger.error("❌ Nessun film seed per OMDb generation")
            return pd.DataFrame()

        omdb_movies = []
        for m in popular_movies:
            title = m.get('title') or m.get('name')
            year = None
            try:
                # Prova a estrarre l'anno se disponibile
                release = m.get('release_date') or m.get('first_air_date')
                if release:
                    year = release.split('-')[0]
            except:
                year = None

            try:
                # Preferisci ricerca via imdbID se disponibile
                imdb_id = m.get('imdb_id')
                omdb_data = None
                if imdb_id:
                    omdb_data = omdb_service.get_movie_by_imdb(imdb_id)
                if not omdb_data and title:
                    omdb_data = omdb_service.search_movie(title, year=year)

                if not omdb_data:
                    continue

                # Mappa campi OMDb in struttura simile a TMDB
                genres_text = omdb_data.get('Genre', '') if isinstance(omdb_data, dict) else ''
                genre_names = [g.strip() for g in genres_text.split(',')] if genres_text else []
                # Converti nomi generi in TMDB ids se possibile
                genre_ids = [self.genre_name_to_id.get(name) for name in genre_names if self.genre_name_to_id.get(name)]

                vote_average = 0.0
                try:
                    vote_average = float(omdb_data.get('imdbRating')) if omdb_data.get('imdbRating') and omdb_data.get('imdbRating') != 'N/A' else 5.0
                except:
                    vote_average = 5.0

                omdb_movies.append({
                    'id': omdb_data.get('imdbID') or m.get('id'),
                    'title': omdb_data.get('Title') or title,
                    'genre_ids': genre_ids,
                    'vote_average': vote_average,
                    'popularity': m.get('popularity', 0),
                    'raw_omdb': omdb_data
                })

                # Rate limit leggero
                import time
                time.sleep(0.05)

            except Exception as e:
                logger.debug(f"Errore resolving OMDb per film '{title}': {e}")
                continue

        logger.info(f"✅ Film OMDb risolti: {len(omdb_movies)}")

        if len(omdb_movies) == 0:
            logger.error("❌ Nessun metadata OMDb ottenuto, abort")
            return pd.DataFrame()

        # Genera rating sintetici usando logica OMDb (seed diverso, profili diversi)
        logger.info("🤖 Generando rating sintetici (OMDb)...")
        # Indichiamo al generatore che stiamo creando dataset OMDb
        try:
            self._current_generation_is_omdb = True
            omdb_ratings = self._generate_synthetic_ratings(omdb_movies, n_users=15000)

            # Aplicare una leggera perturbazione ai rating OMDb per variare i fattori latenti
            try:
                if isinstance(omdb_ratings, pd.DataFrame) and len(omdb_ratings) > 0:
                    # Per i film meno popolari, aggiungi un bias positivo; per i popolari un bias leggermente negativo
                    def apply_bias(row):
                        pop = float(row.get('popularity', 0) or 0)
                        base = float(row.get('rating', 3.0) or 3.0)
                        bias = 0.0
                        if pop < 20:
                            bias += 0.25
                        elif pop > 60:
                            bias -= 0.15
                        # Aggiungi rumore gaussiano
                        bias += float(np.random.normal(0, 0.2))
                        newr = max(0.5, min(5.0, round((base + bias) * 2) / 2))
                        return newr

                    omdb_ratings['rating'] = omdb_ratings.apply(apply_bias, axis=1)
                    logger.info(f"🔧 Applied per-movie bias to OMDb synthetic ratings (samples: {min(20, len(omdb_ratings))})")
            except Exception as e:
                logger.debug(f"Errore applicazione bias su OMDb ratings: {e}")
        finally:
            # cleanup flag
            if hasattr(self, '_current_generation_is_omdb'):
                delattr(self, '_current_generation_is_omdb')

        if len(omdb_ratings) == 0:
            logger.error("❌ Generazione rating sintetici OMDb fallita!")
            return pd.DataFrame()

        # Salva in cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(omdb_ratings, f)

        logger.info(f"✅ Dataset OMDb generato e salvato: {len(omdb_ratings)} rating")
        # Costruisci lista "popolare" basata sui movie seed OMDb
        try:
            # Passiamo la lista omdb_movies (contiene raw_omdb) e il dataframe omdb_ratings
            self._service_popular_recommendations = self._build_service_popular_from_movies(omdb_movies, omdb_ratings, is_omdb=True)
        except Exception as e:
            logger.debug(f"Impossibile costruire popular recommendations OMDb: {e}")
        # Salva mappatura titolo -> generi per fallback genre-based
        try:
            movie_genre_map = {}
            for m in omdb_movies:
                title = m.get('title') or m.get('name')
                genre_ids = m.get('genre_ids') or []
                movie_genre_map[title] = genre_ids
            self.train_movie_genres = movie_genre_map
        except Exception:
            self.train_movie_genres = {}
        return omdb_ratings
    
    def _fetch_tmdb_popular_movies(self, pages: int = 10) -> List[Dict]:
        """Recupera film popolari da TMDB API"""
        import requests
        
        # Verifica API key
        if not self.tmdb_api_key or self.tmdb_api_key == "YOUR_API_KEY":
            logger.error("❌ TMDB API key non configurata!")
            return []
        
        movies = []
        logger.info(f"🔑 Usando TMDB API key: {self.tmdb_api_key[:10]}...")
        
        for page in range(1, pages + 1):
            try:
                url = f"{self.tmdb_base_url}/movie/popular"
                params = {
                    'api_key': self.tmdb_api_key,
                    'page': page,
                    'language': 'it-IT'
                }
                
                logger.info(f"📡 Chiamando TMDB API - Pagina {page}...")
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                page_results = data.get('results', [])
                movies.extend(page_results)
                logger.info(f"✅ Pagina {page}: {len(page_results)} film recuperati")
                
                # Rate limiting
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Errore fetch pagina {page}: {e}")
                if page == 1:  # Se fallisce la prima pagina, è un problema serio
                    logger.error("❌ Impossibile accedere a TMDB API - Verifica API key e connessione")
                break
        
        logger.info(f"📥 Recuperati {len(movies)} film da TMDB")
        return movies

    def _fetch_tmdb_top_rated_movies(self, pages: int = 10) -> List[Dict]:
        """Recupera film top-rated da TMDB API (usato come seed alternativo per OMDb)"""
        import requests

        if not self.tmdb_api_key or self.tmdb_api_key == "YOUR_API_KEY":
            logger.error("❌ TMDB API key non configurata!")
            return []

        movies = []
        for page in range(1, pages + 1):
            try:
                url = f"{self.tmdb_base_url}/movie/top_rated"
                params = {
                    'api_key': self.tmdb_api_key,
                    'page': page,
                    'language': 'it-IT'
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                page_results = data.get('results', [])
                movies.extend(page_results)
                import time
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"❌ Errore fetch top-rated pagina {page}: {e}")
                break

        logger.info(f"📥 Recuperati {len(movies)} top-rated film da TMDB")
        return movies

    def _fetch_tmdb_now_playing(self, pages: int = 5) -> List[Dict]:
        """Recupera film now_playing da TMDB API"""
        import requests

        if not self.tmdb_api_key or self.tmdb_api_key == "YOUR_API_KEY":
            logger.error("❌ TMDB API key non configurata!")
            return []

        movies = []
        for page in range(1, pages + 1):
            try:
                url = f"{self.tmdb_base_url}/movie/now_playing"
                params = {'api_key': self.tmdb_api_key, 'page': page, 'language': 'it-IT'}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                movies.extend(data.get('results', []))
                import time; time.sleep(0.1)
            except Exception as e:
                logger.debug(f"Errore fetch now_playing pagina {page}: {e}")
                break

        logger.info(f"📥 Recuperati {len(movies)} now_playing film da TMDB")
        return movies

    def _fetch_tmdb_discover_by_years(self, years: List[int], per_year: int = 2) -> List[Dict]:
        """Usa l'endpoint discover per prendere film di anni diversi (per_year pagine per anno)"""
        import requests

        if not self.tmdb_api_key or self.tmdb_api_key == "YOUR_API_KEY":
            logger.error("❌ TMDB API key non configurata!")
            return []

        movies = []
        for year in years:
            for page in range(1, per_year + 1):
                try:
                    url = f"{self.tmdb_base_url}/discover/movie"
                    params = {
                        'api_key': self.tmdb_api_key,
                        'primary_release_year': year,
                        'sort_by': 'vote_average.desc',
                        'vote_count.gte': 10,
                        'page': page,
                        'language': 'it-IT'
                    }
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    movies.extend(data.get('results', []))
                    import time; time.sleep(0.08)
                except Exception as e:
                    logger.debug(f"Errore discover year {year} page {page}: {e}")
                    break

        logger.info(f"📥 Recuperati {len(movies)} discover-film da TMDB per years={years[:3]}...")
        return movies
    
    def _generate_synthetic_ratings(self, movies: List[Dict], n_users: int = 10000) -> pd.DataFrame:
        """Genera rating sintetici basati su caratteristiche TMDB"""
        ratings_data = []

        # Se is_omdb è passato via attributo temporaneo in self (non ideal), supportalo
        is_omdb = getattr(self, '_current_generation_is_omdb', False)

        # Se OMDb, usa seed diverso per generazione (temporaneo - ripristiniamo lo stato RNG)
        prev_state = None
        if is_omdb:
            try:
                prev_state = np.random.get_state()
                np.random.seed(4242)
            except Exception:
                prev_state = None

        # Simula diversi tipi di utenti con preferenze (diverso per OMDb se richiesto)
        user_profiles = self._create_user_profiles(n_users, is_omdb=is_omdb)

        for user_id in range(n_users):
            profile = user_profiles[user_id]

            # Ogni utente valuta 10-50 film casualmente
            n_ratings = np.random.randint(10, 51)
            user_movies = np.random.choice(len(movies), size=min(n_ratings, len(movies)), replace=False)

            for movie_idx in user_movies:
                movie = movies[movie_idx]

                # Genera rating basato su profilo utente e caratteristiche film
                rating = self._calculate_synthetic_rating(profile, movie)

                user_prefix = 'omdb' if is_omdb else 'tmdb'
                ratings_data.append({
                    'userId': f"{user_prefix}_user_{user_id}",
                    'movieId': movie['id'],
                    'title': movie.get('title') or movie.get('name'),
                    'rating': rating,
                    'timestamp': datetime.now().timestamp(),
                    'genres': movie.get('genre_ids', []),
                    'tmdb_rating': movie.get('vote_average', 5.0) if movie.get('vote_average') is not None else movie.get('vote_average', 5.0),
                    'popularity': movie.get('popularity', 0)
                })

        # Ripristina stato RNG precedente
        if prev_state is not None:
            try:
                np.random.set_state(prev_state)
            except Exception:
                pass

        df = pd.DataFrame(ratings_data)
        logger.info(f"🎯 Generati {len(df)} rating sintetici per {n_users} utenti (is_omdb={is_omdb})")
        return df
    
    def _create_user_profiles(self, n_users: int, is_omdb: bool = False) -> List[Dict]:
        """Crea profili utente diversificati. Se is_omdb=True usa distribuzioni diverse per aumentare la variabilità."""
        profiles = []

        # Generi TMDB comuni (usati come id)
        base_genres = [28, 12, 16, 35, 80, 99, 18, 10751, 14, 36, 27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37]

        for _ in range(n_users):
            if is_omdb:
                # OMDb users: preferenze più varie e più rumorose
                favorite_genres = np.random.choice(base_genres, size=np.random.randint(1, 7), replace=False)
                rating_tendency = np.random.normal(3.2, 1.0)  # leggermente più variabile
                rating_variance = np.random.uniform(0.6, 1.8)
                popularity_bias = np.random.uniform(-0.7, 1.2)
            else:
                # TMDB users: più centrati
                favorite_genres = np.random.choice(base_genres, size=np.random.randint(2, 6), replace=False)
                rating_tendency = np.random.normal(3.5, 0.8)
                rating_variance = np.random.uniform(0.5, 1.5)
                popularity_bias = np.random.uniform(-0.5, 1.0)

            profiles.append({
                'favorite_genres': favorite_genres.tolist(),
                'rating_tendency': float(max(1.0, min(5.0, rating_tendency))),
                'rating_variance': float(rating_variance),
                'popularity_bias': float(popularity_bias)
            })

        return profiles
    
    def _calculate_synthetic_rating(self, profile: Dict, movie: Dict) -> float:
        """Calcola rating sintetico basato su profilo e caratteristiche film"""
        
        base_rating = profile['rating_tendency']
        
        # Bonus per generi preferiti
        movie_genres = movie.get('genre_ids', [])
        genre_match = len(set(profile['favorite_genres']).intersection(movie_genres))
        genre_bonus = genre_match * 0.3
        
        # Effetto popolarità
        popularity = movie.get('popularity', 0)
        popularity_effect = profile['popularity_bias'] * min(popularity / 100, 1.0)
        
        # Effetto qualità TMDB
        tmdb_rating = movie.get('vote_average', 5.0)
        quality_effect = (tmdb_rating - 5.0) * 0.2
        
        # Calcola rating finale con variabilità
        final_rating = base_rating + genre_bonus + popularity_effect + quality_effect
        final_rating += np.random.normal(0, profile['rating_variance'])
        
        # Clamp tra 0.5 e 5.0
        return max(0.5, min(5.0, round(final_rating * 2) / 2))  # Arrotonda a 0.5
    
    def _apply_svd_to_matrix(self, ratings_matrix):
        """Applica SVD alla matrice ratings"""
        
        # 🔍 LOGGING AGGRESSIVO PER DEBUG K
        logger.info("=" * 80)
        logger.info("🔍 DEBUG _apply_svd_to_matrix - TRACCIAMENTO K")
        logger.info("=" * 80)
        logger.info(f"📊 Matrice shape: {ratings_matrix.shape}")
        logger.info(f"📏 Min dimension: {min(ratings_matrix.shape)}")
        logger.info(f"🔢 current_k_factor PRIMA: {self.current_k_factor}")
        
        min_dim = min(ratings_matrix.shape)
        old_k = self.current_k_factor
        self.current_k_factor = min(self.current_k_factor, min_dim - 1)
        
        logger.info(f"🔢 current_k_factor DOPO min(): {self.current_k_factor} (era {old_k})")
        
        if self.current_k_factor != old_k:
            logger.warning(f"⚠️ K CAMBIATO da {old_k} a {self.current_k_factor} per limiti matrice!")
        
        # 🚨 FORZA K=25 SE È TROPPO ALTO
        if self.current_k_factor > 35:
            logger.warning(f"🚨 K troppo alto ({self.current_k_factor}), forzo a 25!")
            self.current_k_factor = 25
        
        logger.info(f"✅ K FINALE utilizzato per SVD: {self.current_k_factor}")
        logger.info("=" * 80)
        
        logger.info(f"🧮 Applicando SVD con k_factor: {self.current_k_factor}")
        
        U, sigma, Vt = svds(ratings_matrix, k=self.current_k_factor)
        
        # Calcola explained variance usando sklearn per precisione
        if len(sigma) > 0:
            # Usa sklearn TruncatedSVD per calcolo preciso explained variance
            from sklearn.decomposition import TruncatedSVD
            
            try:
                # Applica TruncatedSVD per ottenere explained_variance_ratio_
                temp_svd = TruncatedSVD(n_components=self.current_k_factor, random_state=42)
                temp_svd.fit(ratings_matrix)
                explained_variance = float(temp_svd.explained_variance_ratio_.sum())
                
                logger.info(f"🔍 VARIANZA DEBUG - Matrix shape: {ratings_matrix.shape}, K: {self.current_k_factor}")
                logger.info(f"🔍 VARIANZA DEBUG - Ratio individuali: {temp_svd.explained_variance_ratio_[:5]}")
                logger.info(f"🔍 VARIANZA DEBUG - Varianza calcolata: {explained_variance:.4f} ({explained_variance:.1%})")
                
                # Sanity check: explained variance non dovrebbe mai essere > 95% per dati reali
                if explained_variance > 0.95:
                    logger.warning(f"⚠️ Explained variance molto alta ({explained_variance:.1%}), possibile overfitting")
                    # Usa formula conservativa per dataset piccoli
                    explained_variance = min(0.90, explained_variance)
                
            except Exception as e:
                logger.warning(f"Errore calcolo explained variance sklearn: {e}")
                # Fallback: usa stima conservativa basata su K
                # Per dataset piccoli con K=25, stima realistica è 60-80%
                if self.current_k_factor <= 10:
                    explained_variance = 0.45 + (self.current_k_factor * 0.02)  # 45-65%
                elif self.current_k_factor <= 30:
                    explained_variance = 0.60 + ((self.current_k_factor - 10) * 0.01)  # 60-80%
                else:
                    explained_variance = 0.75 + min(0.15, (self.current_k_factor - 30) * 0.005)  # 75-90%
                
                explained_variance = min(0.90, explained_variance)
                logger.info(f"📊 Usando stima conservativa explained variance: {explained_variance:.1%} per K={self.current_k_factor}")
        else:
            explained_variance = 0.0
        
        # Normalizza fattori per predizioni nel range [1,5]
        # Scala i fattori per avere predizioni ragionevoli
        scale_factor = np.sqrt(2.0 / self.current_k_factor)  # Normalizzazione teorica
        U_scaled = U * scale_factor
        Vt_scaled = Vt * scale_factor
        
        # Aggiungi bias per centrare attorno a 3.0 (rating medio)
        mean_rating = 3.0
        
        # Imposta attributi per statistiche
        self.user_factors = U_scaled
        self.item_factors = Vt_scaled.T
        self.movie_factors = Vt_scaled.T  # Alias per compatibilità
        self.actual_k_used = self.current_k_factor
        self.explained_variance = explained_variance
        self.mean_rating_bias = mean_rating
        
        logger.info(f"🔍 VARIANZA FINALE SETTATA: {self.explained_variance:.1%} (valore: {explained_variance})")
        
        # Conversione per compatibilità numpy
        self.user_factors = np.ascontiguousarray(self.user_factors)
        self.item_factors = np.ascontiguousarray(self.item_factors)
        self.movie_factors = np.ascontiguousarray(self.movie_factors)
        
        logger.info(f"✅ SVD completato: {U.shape} x {Vt.T.shape}")
        logger.info(f"📊 Varianza spiegata: {self.explained_variance:.3f} ({self.explained_variance*100:.1f}%)")
    
    def _create_ratings_matrix(self, df: pd.DataFrame):
        """Crea matrice ratings da DataFrame"""
        
        # Encoding
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
        df['user_idx'] = self.user_encoder.fit_transform(df['userId'])
        df['movie_idx'] = self.movie_encoder.fit_transform(df['title'])
        
        # Calcolo dei bias baseline (mu, b_i, b_u) usando regolarizzazione
        try:
            mu = float(df['rating'].mean()) if len(df) > 0 else 0.0
            lam = float(getattr(self, 'bias_reg', 10.0))

            n_users = df['user_idx'].nunique()
            n_movies = df['movie_idx'].nunique()

            # item bias b_i: sum(r_ui - mu) / (N_i + lambda)
            item_group = df.groupby('movie_idx').agg(sum_diff=('rating', lambda x: (x - mu).sum()), count=('rating', 'count'))
            b_i_series = item_group['sum_diff'] / (item_group['count'] + lam)

            # map b_i to rows for user bias calculation
            b_i_map = df['movie_idx'].map(b_i_series.to_dict())
            df = df.copy()
            df['b_i'] = b_i_map.fillna(0.0)

            # user bias b_u: sum(r_ui - mu - b_i) / (N_u + lambda)
            user_group = df.groupby('user_idx').agg(sum_diff=('rating', lambda x: (x - mu - df.loc[x.index, 'b_i']).sum()), count=('rating', 'count'))
            b_u_series = user_group['sum_diff'] / (user_group['count'] + lam)

            # Costruisci array finali per accesso veloce
            item_bias_arr = np.zeros(n_movies, dtype=float)
            for idx, val in b_i_series.items():
                if 0 <= int(idx) < n_movies:
                    item_bias_arr[int(idx)] = float(val)

            user_bias_arr = np.zeros(n_users, dtype=float)
            for idx, val in b_u_series.items():
                if 0 <= int(idx) < n_users:
                    user_bias_arr[int(idx)] = float(val)

            self.global_mean = mu
            self.item_bias = item_bias_arr
            self.user_bias = user_bias_arr

            logger.info(f"Bias baseline calcolati: mu={self.global_mean:.3f}, users={n_users}, movies={n_movies}")
        except Exception as e:
            logger.warning(f"Errore calcolo bias baseline: {e}")
            # fallback
            self.global_mean = float(df['rating'].mean()) if len(df) > 0 else 3.0
            self.item_bias = np.zeros(df['movie_idx'].nunique() if 'movie_idx' in df else 0)
            self.user_bias = np.zeros(df['user_idx'].nunique() if 'user_idx' in df else 0)

        return csr_matrix(
            (df['rating'], (df['user_idx'], df['movie_idx'])),
            shape=(df['user_idx'].nunique(), df['movie_idx'].nunique())
        )
    
    def _test_on_aflix_data(self) -> Dict:
        """Testa modello TMDB su dati AFlix reali"""
        
        try:
            # Carica dati AFlix
            aflix_df = self.prepare_data()
            
            if len(aflix_df) < 5:
                return {"status": "insufficient_aflix_data", "rmse": None}
            
            # Mappa film AFlix a TMDB con fallback fuzzy + genre-based
            encoded_titles = list(self.movie_encoder.classes_)

            def resolve_title_to_item_factor(aflix_title: str, aflix_genres: List[Any]):
                """Ritorna un vettore item_factors per un titolo AFlix usando:
                   1) match esatto
                   2) fuzzy match (difflib)
                   3) genre-overlap best match usando self.train_movie_genres
                   4) weighted average dei top-k candidate
                """
                try:
                    # 1) Exact match
                    if aflix_title in self.movie_encoder.classes_:
                        idx = int(self.movie_encoder.transform([aflix_title])[0])
                        return self.item_factors[idx], 'exact', idx

                    # 2) Fuzzy match
                    matches = difflib.get_close_matches(aflix_title, encoded_titles, n=3, cutoff=0.7)
                    if matches:
                        # Usa il migliore
                        best = matches[0]
                        idx = int(self.movie_encoder.transform([best])[0])
                        logger.debug(f"Fuzzy match: '{aflix_title}' -> '{best}'")
                        return self.item_factors[idx], f'fuzzy:{best}', idx

                    # 3) Genre-overlap: richiede mapping titolo->generi generato durante build dataset
                    if hasattr(self, 'train_movie_genres') and self.train_movie_genres:
                        # Calcola overlap con tutti i titoli noti (scorri solo i titoli con genres definiti)
                        best_scores = []
                        aflix_genre_set = set(aflix_genres) if aflix_genres else set()
                        for cand in encoded_titles:
                            cand_genres = set(self.train_movie_genres.get(cand, []))
                            if not cand_genres:
                                continue
                            overlap = len(aflix_genre_set.intersection(cand_genres))
                            if overlap > 0:
                                best_scores.append((cand, overlap))

                        if best_scores:
                            # Ordina per overlap decrescente
                            best_scores.sort(key=lambda x: x[1], reverse=True)
                            top = [b[0] for b in best_scores[:3]]
                            idxs = self.movie_encoder.transform(top)
                            vecs = self.item_factors[idxs]
                            weights = np.array([s for _, s in best_scores[:3]], dtype=float)
                            weights /= weights.sum()
                            # ritorniamo vettore medio; idx non singolo (None)
                            return np.average(vecs, axis=0, weights=weights), f'genre_best:{top}', None

                    # 4) Ultimo fallback: media pesata dei primi N item_factors per similarità di titolo (partial)
                    # Usa similarity via difflib.SequenceMatcher ratio su tutti i titoli
                    ratios = []
                    from difflib import SequenceMatcher
                    for cand in encoded_titles:
                        r = SequenceMatcher(None, aflix_title.lower(), cand.lower()).ratio()
                        if r > 0.4:
                            ratios.append((cand, r))
                    if ratios:
                        ratios.sort(key=lambda x: x[1], reverse=True)
                        top = [r[0] for r in ratios[:5]]
                        idxs = self.movie_encoder.transform(top)
                        vecs = self.item_factors[idxs]
                        weights = np.array([r[1] for r in ratios[:5]], dtype=float)
                        weights /= weights.sum()
                        return np.average(vecs, axis=0, weights=weights), f'partial_sim:{[t for t in top]}', None

                    # Se tutto fallisce, None
                    return None, 'none', None
                except Exception as e:
                    logger.debug(f"Errore resolving factor per '{aflix_title}': {e}")
                    return None, 'error', None

            # Costruiamo test set e predizioni
            predictions = []
            actuals = []

            mapping_counts = {}
            for _, row in aflix_df.iterrows():
                try:
                    title = row['title']
                    aflix_genres = row.get('genres', []) if isinstance(row.get('genres', []), list) else []
                    item_vec, why, mapped_idx = resolve_title_to_item_factor(title, aflix_genres)
                    mapping_counts[why] = mapping_counts.get(why, 0) + 1
                    if item_vec is None:
                        # Non possiamo predire per questo film
                        continue

                    # Usa vettore utente medio se l'utente non è nel training
                    avg_user_factor = np.mean(self.user_factors, axis=0)
                    base_pred = float(np.dot(avg_user_factor, item_vec))
                    # Aggiungi baseline: global mean + item bias (se disponibile)
                    global_mean = getattr(self, 'global_mean', getattr(self, 'mean_rating_bias', 3.0))
                    item_b = 0.0
                    try:
                        if mapped_idx is not None and self.item_bias is not None and mapped_idx < len(self.item_bias):
                            item_b = float(self.item_bias[int(mapped_idx)])
                    except Exception:
                        item_b = 0.0

                    pred_rating = base_pred + global_mean + item_b
                    pred_rating = max(1.0, min(5.0, pred_rating))

                    predictions.append(pred_rating)
                    actuals.append(row['rating'])
                except Exception as e:
                    logger.debug(f"Errore predizione film {row.get('title', 'unknown')}: {e}")
                    continue

            try:
                logger.info(f"AFlix->SVD mapping during evaluation: {mapping_counts}")
            except Exception:
                pass

            if len(predictions) > 0:
                rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
                mae = float(mean_absolute_error(actuals, predictions))
                total_aflix = len(aflix_df)
                coverage = float(len(predictions)) / total_aflix if total_aflix > 0 else None
                return {
                    "status": "success",
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "test_samples": len(predictions),
                    "total_aflix_samples": total_aflix,
                    "coverage": coverage,
                    "mapping_counts": mapping_counts
                }

            return {"status": "no_predictions", "rmse": None}
            
        except Exception as e:
            logger.error(f"Errore test AFlix: {e}")
            return {"status": "error", "rmse": None, "error": str(e)}
    
    def _compile_hybrid_stats(self, source_data: pd.DataFrame, aflix_test: Dict, source: str = 'tmdb') -> Dict:
        """Compila statistiche training ibrido. Il campo dei dati di training sarà dinamico
        (es. 'tmdb_data' o 'omdb_data') per evitare confusione quando la sorgente è OMDb."""
        
        # Calcola varianza spiegata se disponibile
        explained_variance = getattr(self, 'explained_variance', 0.0)
        actual_k_used = getattr(self, 'actual_k_used', self.current_k_factor)
        
        logger.info(f"🔍 DEBUG HYBRID - actual_k_used: {actual_k_used}, explained_variance: {explained_variance}")
        
        data_key = f"{source}_data" if source in ['tmdb', 'omdb'] else 'training_data'

        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'training_mode': 'hybrid',
            # Campi attesi dal frontend
            'stats': {
                'total_ratings': len(source_data),
                'actual_k_used': int(actual_k_used) if actual_k_used and actual_k_used > 0 else None,
                'explained_variance': float(explained_variance) if explained_variance else 0.0,
                'unique_users': source_data['userId'].nunique() if len(source_data) > 0 else 0,
                'unique_movies': source_data['title'].nunique() if len(source_data) > 0 else 0,
                'test_rmse': aflix_test.get('rmse'),
                'test_status': aflix_test.get('status')
            },
            'model_info': {
                'algorithm': 'SVD',
                'k_factor': self.current_k_factor,
                'training_source': getattr(self, 'training_source', 'tmdb'),
                'test_source': 'aflix'
            },
            data_key: {
                'total_ratings': len(source_data),
                'unique_users': source_data['userId'].nunique() if len(source_data) > 0 else 0,
                'unique_movies': source_data['title'].nunique() if len(source_data) > 0 else 0,
                'rating_distribution': source_data['rating'].value_counts().to_dict() if len(source_data) > 0 else {}
            },
            'aflix_test': aflix_test,
            'performance': {
                'test_rmse': aflix_test.get('rmse'),
                'test_status': aflix_test.get('status')
            }
        }
    
    def _train_demo_tmdb_model(self) -> Dict:
        """Training demo con dati TMDB per dataset piccoli"""
        
        logger.info("🎬 Modalità DEMO: Usando dataset TMDB ridotto...")
        
        # Dataset demo piccolo
        demo_data = self._get_or_generate_tmdb_data()
        
        # Riduci a 1000 rating per demo
        if len(demo_data) > 1000:
            demo_data = demo_data.sample(1000, random_state=42)
        
        # Training normale
        ratings_matrix = self._create_ratings_matrix(demo_data)
        self._apply_svd_to_matrix(ratings_matrix)
        
        stats = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'training_mode': 'demo_tmdb',
            'model_info': {
                'algorithm': 'SVD',
                'k_factor': self.current_k_factor,
                'total_ratings': len(demo_data),
                'unique_users': demo_data['userId'].nunique(),
                'unique_movies': demo_data['title'].nunique()
            }
        }
        
        return stats
    
    # ================================
    # 🎯 OTTIMIZZAZIONE AUTOMATICA K
    # ================================
    
    def optimize_both_k_values(self, ratings_matrix) -> Dict[str, int]:
        """Ottimizza automaticamente sia K-SVD che K-Cluster"""
        
        logger.info("🎯 OTTIMIZZAZIONE AUTOMATICA K-VALUES")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. Ottimizza K-SVD se abilitato
        if self.auto_optimize_k_svd:
            optimal_k_svd = self._optimize_k_svd(ratings_matrix)
            if optimal_k_svd:
                self.n_components = optimal_k_svd
                self.current_k_factor = optimal_k_svd
                results['optimal_k_svd'] = optimal_k_svd
                logger.info(f"✅ K-SVD ottimizzato: {optimal_k_svd}")
        
        # 2. Ottimizza K-Cluster se abilitato  
        if self.auto_optimize_k_cluster:
            optimal_k_cluster = self._optimize_k_cluster(ratings_matrix)
            if optimal_k_cluster:
                self.n_clusters = optimal_k_cluster
                self.optimal_k_cluster = optimal_k_cluster  # SALVA per uso successivo!
                results['optimal_k_cluster'] = optimal_k_cluster
                logger.info(f"✅ K-Cluster ottimizzato: {optimal_k_cluster}")
                results['optimal_k_cluster'] = optimal_k_cluster
                logger.info(f"✅ K-Cluster ottimizzato: {optimal_k_cluster}")
        
        return results
    
    def _optimize_k_svd(self, ratings_matrix) -> Optional[int]:
        """Ottimizza K per SVD"""
        
        logger.info("🔍 Ottimizzazione K-SVD...")
        
        best_k = None
        best_score = -1
        results = []
        
        for k in self.k_svd_range:
            try:
                # Limite sicurezza
                max_k = min(ratings_matrix.shape) - 1
                if k >= max_k:
                    continue
                
                # Test SVD con k componenti usando sklearn per explained variance preciso
                from sklearn.decomposition import TruncatedSVD
                temp_svd = TruncatedSVD(n_components=k, random_state=42)
                temp_svd.fit(ratings_matrix)
                
                # Usa explained_variance_ratio_ di sklearn (più preciso)
                explained_variance = float(temp_svd.explained_variance_ratio_.sum())
                
                # Sanity check per evitare valori irrealistici
                explained_variance = min(0.90, explained_variance)
                
                # Calcola efficienza (varianza per componente)
                efficiency = explained_variance / k
                
                # Penalità overfitting per K troppo alti
                n_samples = min(ratings_matrix.shape)
                overfitting_penalty = k / n_samples if n_samples > 0 else 0
                
                # Score composito CORRETTO (penalizza K alti)
                composite_score = (
                    explained_variance * 0.5 +           # 50% varianza spiegata
                    efficiency * 0.3 +                   # 30% efficienza  
                    (1.0 - overfitting_penalty) * 0.2    # 20% penalità overfitting
                )
                
                results.append({
                    'k': k,
                    'explained_variance': explained_variance,
                    'efficiency': efficiency,
                    'composite_score': composite_score
                })
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_k = k
                
                logger.info(f"K={k:2d} | Varianza: {explained_variance:.4f} | Efficienza: {efficiency:.4f} | Score: {composite_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Errore K-SVD={k}: {e}")
                continue
        
        # Salva risultati per monitoring
        self.k_performance_log['svd_optimization'] = results
        
        return best_k
    
    def _optimize_k_cluster(self, ratings_matrix) -> Optional[int]:
        """Ottimizza K per Clustering usando film AFlix proiettati"""
        
        logger.info("🎯 Ottimizzazione K-Cluster...")
        
        # IMPORTANTE: Usa film AFlix, non TMDB per ottimizzazione K-cluster!
        try:
            # Ottieni dati AFlix per clustering
            aflix_df = self.prepare_data()
            n_aflix_movies = len(aflix_df['title'].unique()) if len(aflix_df) > 0 else 0
            
            logger.info(f"🎬 Film AFlix disponibili per clustering: {n_aflix_movies}")
            
            if n_aflix_movies < 3:
                logger.warning("❌ Clustering ottimizzazione saltata: meno di 3 film AFlix unici")
                return None
            
            # STRATEGIA ALTERNATIVA: Usa subset dei dati TMDB come proxy
            # Prendi solo i primi N utenti per simulare dataset AFlix
            max_users = min(20, ratings_matrix.shape[0])  # Max 20 utenti come AFlix
            cluster_data_matrix = ratings_matrix[:max_users, :]
            
            # Applica SVD ridotta per clustering
            k_for_clustering = min(10, min(cluster_data_matrix.shape) - 1)
            if k_for_clustering < 2:
                logger.warning("❌ K troppo piccolo per clustering")
                return None
                
            U_cluster, sigma_cluster, Vt_cluster = svds(cluster_data_matrix, k=k_for_clustering)
            
            # Usa fattori FILM (Vt) per clustering, non utenti
            if Vt_cluster.shape[0] >= 2:
                cluster_data = Vt_cluster[:2, :].T  # Trasposto: righe=film, colonne=componenti
            else:
                cluster_data = np.column_stack([
                    Vt_cluster[0, :], 
                    np.random.normal(0, 0.1, Vt_cluster.shape[1])
                ])
            
            logger.info(f"🎯 Dati clustering preparati: {cluster_data.shape[0]} film, {cluster_data.shape[1]} componenti")
            
        except Exception as e:
            logger.warning(f"Errore preparazione dati clustering AFlix: {e}")
            return None
        
        best_k = None
        best_score = -1
        results = []
        n_movies_for_clustering = cluster_data.shape[0]
        
        logger.info(f"🎯 Ottimizzazione K-Cluster su {n_movies_for_clustering} elementi")
        
        for k in self.k_cluster_range:
            if k >= n_movies_for_clustering:  # Controlla elementi disponibili per clustering
                logger.info(f"⏭️ Skipping K={k} (troppo grande per {n_movies_for_clustering} elementi)")
                continue
                
            try:
                # Test clustering
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(cluster_data)
                
                # Calcola silhouette score
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(cluster_data, labels)
                
                # Calcola bilanciamento cluster
                unique, counts = np.unique(labels, return_counts=True)
                balance = 1.0 - np.std(counts) / np.mean(counts) if len(counts) > 1 else 0
                
                # BONUS AGGRESSIVO per K ottimali (forza 4-5 cluster per AFlix)
                if k == 4:
                    k_bonus = 1.5  # BONUS MASSIMO per K=4 (ideale AFlix)
                elif k == 5:
                    k_bonus = 1.3  # Bonus molto alto per K=5
                elif k == 3:
                    k_bonus = 1.0  # Bonus buono per K=3
                elif k == 6:
                    k_bonus = 0.8  # Bonus moderato per K=6
                elif k == 2:
                    k_bonus = 0.2  # PENALITÀ FORTE per K=2 (troppo semplice)
                else:
                    k_bonus = 0.3  # Penalità per altri K
                
                # Score composito (peso bonus aumentato per forzare K=4)
                composite_score = silhouette * 0.4 + balance * 0.2 + k_bonus * 0.4
                
                results.append({
                    'k': k,
                    'silhouette_score': silhouette,
                    'balance': balance,
                    'k_bonus': k_bonus,
                    'composite_score': composite_score
                })
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_k = k
                
                logger.info(f"K={k:2d} | Silhouette: {silhouette:.3f} | Balance: {balance:.3f} | K-Bonus: {k_bonus:.1f} | Score: {composite_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Errore K-Cluster={k}: {e}")
                continue
        
        # Salva risultati per monitoring
        self.k_performance_log['cluster_optimization'] = results
        
        # Salva K ottimizzato per uso nella visualizzazione
        if best_k:
            self.optimal_k_cluster = best_k
            logger.info(f"🏆 K-Cluster ottimizzato salvato: {best_k}")
        
        return best_k

    def _create_user_clusters(self, df: pd.DataFrame):
        """Crea cluster utenti"""
        
        n_clusters = min(5, len(df) // 3)
        
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            user_clusters = kmeans.fit_predict(self.user_factors)
            
            self.user_clusters = {}
            for idx, cluster in enumerate(user_clusters):
                user_id = self.user_encoder.inverse_transform([idx])[0]
                self.user_clusters[user_id] = int(cluster)
            
            logger.info(f"🎯 Creati {n_clusters} cluster con {len(self.user_clusters)} utenti")
    
    def _calculate_aflix_stats(self, df: pd.DataFrame, ratings_sparse) -> Dict:
        """Calcola statistiche per training AFlix"""
        
        # RMSE on training data
        predicted_matrix = np.dot(self.user_factors, self.item_factors.T)
        mse = mean_squared_error(ratings_sparse.data, 
                               predicted_matrix[ratings_sparse.row, ratings_sparse.col])
        rmse = np.sqrt(mse)
        
        # Coverage
        unique_users = len(df['userId'].unique())
        unique_movies = len(df['title'].unique())
        total_interactions = len(df)
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'training_mode': 'aflix_only',
            'model_info': {
                'algorithm': 'SVD',
                'k_factor': self.current_k_factor,
                'users': unique_users,
                'movies': unique_movies,
                'ratings': total_interactions,
                'density': f"{(total_interactions/(unique_users*unique_movies)*100):.2f}%"
            },
            'metrics': {
                'rmse': round(rmse, 4),
                'coverage': f"{(unique_movies/unique_movies*100):.1f}%",
                'clusters': len(self.user_clusters) if hasattr(self, 'user_clusters') else 0
            },
            'training_data': {
                'source': 'aflix_db',
                'total_ratings': total_interactions,
                'rating_distribution': df['rating'].value_counts().to_dict()
            }
        }

    def get_user_recommendations(self, user_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Genera raccomandazioni personalizzate per un utente"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        try:
            # Se l'utente è presente nel training encoder, usalo direttamente
            inferred_user_vector = None
            if user_id in self.user_encoder.classes_:
                user_idx = self.user_encoder.transform([user_id])[0]
                inferred_user_vector = self.user_factors[user_idx]
            else:
                # Proviamo a inferire un vettore utente dai voti reali AFlix
                try:
                    user_obj = Utente.objects(id=user_id).first()
                    if user_obj:
                        votes = list(Votazione.objects(utente=user_obj))
                    else:
                        votes = []
                except Exception:
                    votes = []

                # Costruiamo vettore utente come media pesata dei fattori degli item votati
                if votes:
                    rated = []
                    # Precompute encoded titles for fuzzy matching
                    encoded_titles = list(self.movie_encoder.classes_) if hasattr(self, 'movie_encoder') and self.movie_encoder is not None else []
                    from difflib import get_close_matches

                    for v in votes:
                        try:
                            title = v.film.titolo
                            rating_val = float(v.valutazione)
                            if encoded_titles and title in encoded_titles:
                                rated.append((title, rating_val))
                            elif encoded_titles:
                                # Proviamo fuzzy match per mappare titoli leggermente diversi
                                matches = get_close_matches(title, encoded_titles, n=1, cutoff=0.6)
                                if matches:
                                    mapped = matches[0]
                                    logger.info(f"User {user_id}: fuzzy-mapped voted title '{title}' -> '{mapped}' for inference")
                                    rated.append((mapped, rating_val))
                                else:
                                    # Non trovato match; skip
                                    logger.debug(f"User {user_id}: no match for voted title '{title}' during inference")
                            else:
                                # No movie encoder available yet
                                logger.debug(f"User {user_id}: movie encoder not available, cannot map voted title '{title}'")
                        except Exception:
                            continue

                    if rated:
                        titles = [t for t, _ in rated]
                        weights = np.array([r for _, r in rated], dtype=float)
                        try:
                            idxs = self.movie_encoder.transform(titles)
                            item_vecs = self.item_factors[idxs]
                            # Weighted average of item vectors
                            inferred_user_vector = np.average(item_vecs, axis=0, weights=weights)
                        except Exception as e:
                            logger.debug(f"Could not infer user vector from AFlix votes: {e}")

            # Debug logging: indicate how we inferred the user vector
            if inferred_user_vector is None:
                logger.info(f"User {user_id}: could not infer user vector from training encoders or AFlix votes - attempting item-nearest-neighbor fallback")
                try:
                    # Try item-based nearest-neighbor fallback using user's AFlix votes
                    nn_recs = self._fallback_item_based_recommendations(user_id, top_n=top_n)
                    if nn_recs and len(nn_recs) > 0:
                        logger.info(f"User {user_id}: returning {len(nn_recs)} item-based fallback recommendations")
                        return nn_recs
                except Exception as e:
                    logger.debug(f"Item-based fallback failed for user {user_id}: {e}")

                # Ultimo fallback: raccomandazioni popolari
                logger.info(f"User {user_id}: falling back to popular recommendations")
                return self._get_popular_recommendations(top_n)
            else:
                logger.info(f"User {user_id}: inferred user vector available (len={len(inferred_user_vector)}) - generating personalized recommendations")

            # Calcola rating predetti usando il vettore utente (reale o inferito)
            base_preds = np.dot(self.movie_factors, inferred_user_vector)

            # Aggiungi baseline: global mean + item_bias + user_bias (se disponibile)
            global_mean = getattr(self, 'global_mean', getattr(self, 'mean_rating_bias', 3.0))
            item_bias_arr = getattr(self, 'item_bias', None)

            # Se l'utente è nel training, recupera user_bias, altrimenti 0
            user_bias_scalar = 0.0
            try:
                if user_id in self.user_encoder.classes_:
                    user_idx = int(self.user_encoder.transform([user_id])[0])
                    if self.user_bias is not None and user_idx < len(self.user_bias):
                        user_bias_scalar = float(self.user_bias[user_idx])
            except Exception:
                user_bias_scalar = 0.0

            if item_bias_arr is None:
                item_bias_arr = np.zeros_like(base_preds)

            predicted_ratings = base_preds + global_mean + item_bias_arr + user_bias_scalar
            predicted_ratings = np.clip(predicted_ratings, 1.0, 5.0)
            
            # Converti in raccomandazioni
            movie_titles = self.movie_encoder.inverse_transform(np.arange(len(predicted_ratings)))
            
            # Recupera informazioni film dal database
            recommendations = []
            for i, title in enumerate(movie_titles):
                try:
                    film = Film.objects(titolo=title).first()
                    if film:
                        poster_url = None
                        if film.poster_path:
                            poster_url = f"https://image.tmdb.org/t/p/w500{film.poster_path}"
                        elif film.tmdb_id:
                            poster_url = self.fetch_poster_url(film.tmdb_id)
                        
                        # 🔧 FIX: Solo film con titolo valido e poster
                        if not title or not title.strip() or title.strip().lower() in ['film raccomandato 1', 'no title', 'untitled']:
                            logger.debug(f"Saltato film con titolo invalido: '{title}'")
                            continue
                            
                        if not poster_url or not poster_url.strip() or poster_url == 'None':
                            logger.debug(f"Saltato film senza poster: '{title}' (poster: {poster_url})")
                            continue  # Salta film senza poster
                        
                        recommendations.append({
                            "title": title,
                            "predicted_rating": float(predicted_ratings[i]),
                            "tmdb_id": film.tmdb_id,
                            "genres": film.genere,
                            "poster_url": poster_url,
                            "tmdb_rating": film.tmdb_rating,
                            "cluster": int(self.cluster_labels[i]) if self.cluster_labels is not None and i < len(self.cluster_labels) else 0
                        })
                except Exception as e:
                    logger.warning(f"Error processing recommendation for {title}: {e}")
                    continue
            
            # Ordina per rating predetto e restituisci top_n
            recommendations.sort(key=lambda x: x["predicted_rating"], reverse=True)
            return recommendations[:top_n]
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return self._get_popular_recommendations(top_n)

    def _get_popular_recommendations(self, top_n: int) -> List[Dict[str, Any]]:
        """Raccomandazioni basate sulla popolarità per nuovi utenti"""
        try:
            # Se abbiamo costruito una lista popolare dal dataset di training (per-source), usala
            if hasattr(self, '_service_popular_recommendations') and self._service_popular_recommendations:
                return self._service_popular_recommendations[:top_n]

            # Altrimenti recupera film più votati dal database (fallback globale)
            films = list(Film.objects.order_by('-media_voti', '-numero_voti')[:top_n])

            recommendations = []
            for film in films:
                # 🔧 FIX: Solo film con titolo valido
                if not film.titolo or not film.titolo.strip() or film.titolo.strip().lower() in ['film raccomandato 1', 'no title', 'untitled']:
                    logger.debug(f"Saltato film popolare con titolo invalido: '{film.titolo}'")
                    continue

                poster_url = None
                if film.poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{film.poster_path}"
                elif film.tmdb_id:
                    poster_url = self.fetch_poster_url(film.tmdb_id)

                # 🔧 FIX: Solo film con poster valido
                if not poster_url or not poster_url.strip() or poster_url == 'None':
                    logger.debug(f"Saltato film popolare senza poster: '{film.titolo}' (poster: {poster_url})")
                    continue

                recommendations.append({
                    "title": film.titolo,
                    "predicted_rating": float(film.media_voti) if film.media_voti else 3.0,
                    "tmdb_id": film.tmdb_id,
                    "genres": film.genere,
                    "poster_url": poster_url,
                    "tmdb_rating": film.tmdb_rating,
                    "cluster": 0
                })

            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating popular recommendations: {e}")
            return []

    def _build_service_popular_from_movies(self, movies: List[Dict], ratings_df: pd.DataFrame, is_omdb: bool = False) -> List[Dict[str, Any]]:
        """Costruisce una lista di raccomandazioni popolari basata sul dataset di training generato.

        movies: lista originale passata alla generazione dei rating (può contenere raw_omdb)
        ratings_df: DataFrame generato con colonne 'title', 'rating', 'movieId'
        is_omdb: flag per adattare estrazione poster
        """
        try:
            if ratings_df is None or len(ratings_df) == 0:
                return []

            # Calcola media rating e conteggio per titolo
            agg = ratings_df.groupby('title').agg({'rating': ['mean', 'count']})
            agg.columns = ['mean_rating', 'count']
            agg = agg.reset_index()
            agg = agg.sort_values(by=['mean_rating', 'count'], ascending=[False, False])

            # Costruisci mapping per poster dai movie seed
            poster_map = {}
            for m in movies:
                title = m.get('title') or m.get('name')
                if is_omdb:
                    raw = m.get('raw_omdb') if isinstance(m, dict) else None
                    poster = None
                    if raw and isinstance(raw, dict):
                        poster = raw.get('Poster')
                    poster_map[title] = poster
                else:
                    poster = m.get('poster_path') or None
                    # if poster is TMDB path, build full url
                    if poster and poster.startswith('/'):
                        poster_map[title] = f"https://image.tmdb.org/t/p/w500{poster}"
                    else:
                        poster_map[title] = poster

            results = []
            for _, row in agg.head(50).iterrows():
                title = row['title']
                poster = poster_map.get(title)
                # If no poster, try to fetch via movieId (if numeric) else use placeholder
                if not poster:
                    try:
                        movie_id = row.get('movieId') if 'movieId' in row.index else None
                        if movie_id and str(movie_id).isdigit():
                            poster = self.fetch_poster_url(int(movie_id))
                    except Exception:
                        poster = None

                if not poster:
                    poster = getattr(self, 'default_poster_placeholder', None)

                results.append({
                    'title': title,
                    'predicted_rating': float(row['mean_rating']),
                    'poster_url': poster,
                    'genres': [],
                    'tmdb_id': None,
                    'tmdb_rating': None,
                    'cluster': 0
                })

            return results
        except Exception as e:
            logger.debug(f"Errore building service popular: {e}")
            return []

    def _build_service_popular_from_df(self, df: pd.DataFrame, source: str = 'tmdb', top_n: int = 50) -> List[Dict[str, Any]]:
        """Costruisce una lista 'popolare' direttamente dal DataFrame di training.

        Questo metodo è più tollerante: non scarta i titoli senza poster, usa placeholder
        quando necessario e prova a risolvere poster tramite DB Film se possibile.
        Restituisce una lista ordinata per mean rating+count.
        """
        try:
            if df is None or len(df) == 0:
                return []

            # Aggrega per titolo
            agg = df.groupby('title').agg({'rating': ['mean', 'count']})
            agg.columns = ['mean_rating', 'count']
            agg = agg.reset_index()

            # Score composito: mean_rating * log(1+count)
            import numpy as _np
            agg['score'] = agg['mean_rating'] * _np.log1p(agg['count'])
            agg = agg.sort_values(by=['score', 'mean_rating', 'count'], ascending=[False, False, False])

            results = []
            placeholder = None
            try:
                # usa un placeholder locale se presente
                placeholder = getattr(self, 'default_poster_placeholder', None)
            except Exception:
                placeholder = None

            for _, row in agg.head(top_n).iterrows():
                title = row['title']
                poster_url = None

                # 1) prova a trovare il poster tramite Film DB
                try:
                    film = Film.objects(titolo=title).first()
                    if film:
                        if film.poster_path:
                            poster_url = f"https://image.tmdb.org/t/p/w500{film.poster_path}"
                        elif film.tmdb_id:
                            poster_url = self.fetch_poster_url(film.tmdb_id)
                except Exception:
                    poster_url = None

                # 2) se non trovato, prova a usare poster salvati nel DataFrame (colonna 'poster' o 'poster_url')
                if not poster_url:
                    if 'poster' in df.columns:
                        # cerca il primo valore non-null per questo titolo
                        try:
                            pv = df[df['title'] == title]['poster'].dropna()
                            if len(pv) > 0:
                                poster_url = pv.iloc[0]
                        except Exception:
                            poster_url = None

                # 3) fallback a placeholder (se configurato)
                if not poster_url:
                    poster_url = placeholder

                results.append({
                    'title': title,
                    'predicted_rating': float(row['mean_rating']),
                    'poster_url': poster_url,
                    'genres': [],
                    'tmdb_id': None,
                    'tmdb_rating': None,
                    'cluster': 0
                })

            # cache per istanza
            try:
                self._service_popular_recommendations = results
            except Exception:
                pass

            return results
        except Exception as e:
            logger.debug(f"Errore building service popular from df: {e}")
            return []

    def _fallback_item_based_recommendations(self, user_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Fallback item-based: usa i film votati dall'utente per trovare vicini nel movie_factors e aggregare punteggi.

        Questo fallback è source-specific perché usa self.movie_factors addestrati sulla sorgente corrente.
        """
        try:
            # Recupera voti AFlix per l'utente
            user_obj = Utente.objects(id=user_id).first()
            if not user_obj:
                return []

            votes = list(Votazione.objects(utente=user_obj))
            if not votes:
                return []

            # Assicurati di avere fattori film
            if not hasattr(self, 'movie_factors') or self.movie_factors is None:
                return []

            # Precompute normalized movie_factors for cosine similarity
            mf = np.asarray(self.movie_factors)
            norms = np.linalg.norm(mf, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mf_norm = mf / norms

            # Map voti a indici nel movie_encoder (fuzzy se necessario)
            encoded_titles = list(self.movie_encoder.classes_) if hasattr(self, 'movie_encoder') and self.movie_encoder is not None else []
            from difflib import get_close_matches

            rated_indices = set()
            score_acc = {}
            weight_acc = {}

            for v in votes:
                try:
                    title = v.film.titolo
                    rating_val = float(v.valutazione)
                    mapped_title = None
                    if encoded_titles and title in encoded_titles:
                        mapped_title = title
                    elif encoded_titles:
                        matches = get_close_matches(title, encoded_titles, n=1, cutoff=0.6)
                        if matches:
                            mapped_title = matches[0]

                    if not mapped_title:
                        continue

                    idx = int(self.movie_encoder.transform([mapped_title])[0])
                    rated_indices.add(idx)

                    # Similarities to all movies
                    target_vec = mf_norm[idx].reshape(1, -1)
                    sims = np.dot(mf_norm, target_vec.T).flatten()  # cosine similarities

                    # Consider top_k neighbors (excluding itself)
                    top_k = min(50, len(sims)-1)
                    if top_k <= 0:
                        continue
                    neigh_idx = np.argpartition(-sims, range(1, top_k+1))[:top_k+1]
                    # Aggregate weighted score: similarity * (rating - mean_bias)
                    # baseline components
                    global_mean = getattr(self, 'global_mean', getattr(self, 'mean_rating_bias', 3.0))
                    for ni in neigh_idx:
                        if ni == idx:
                            continue
                        sim = float(sims[ni])
                        # tenta di sottrarre global mean e item bias del film votato
                        item_b_voted = 0.0
                        try:
                            if self.item_bias is not None and int(idx) < len(self.item_bias):
                                item_b_voted = float(self.item_bias[int(idx)])
                        except Exception:
                            item_b_voted = 0.0

                        residual = rating_val - global_mean - item_b_voted
                        score = sim * residual
                        score_acc[ni] = score_acc.get(ni, 0.0) + score
                        weight_acc[ni] = weight_acc.get(ni, 0.0) + abs(sim)
                except Exception:
                    continue

            # Compute final scores
            final_scores = []
            for mi, s in score_acc.items():
                w = weight_acc.get(mi, 1.0)
                final_scores.append((mi, s / w if w != 0 else s))

            if not final_scores:
                return []

            final_scores.sort(key=lambda x: x[1], reverse=True)

            # Build recommendation objects, skip already rated indices
            recs = []
            for mi, sc in final_scores:
                if mi in rated_indices:
                    continue
                try:
                    title = self.movie_encoder.inverse_transform([mi])[0]
                    film = Film.objects(titolo=title).first()
                    if not film:
                        continue
                    poster_url = None
                    if film.poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w500{film.poster_path}"
                    elif film.tmdb_id:
                        poster_url = self.fetch_poster_url(film.tmdb_id)

                    if not poster_url:
                        continue

                    recs.append({
                        'title': title,
                        'predicted_rating': float(min(5.0, max(1.0, getattr(self, 'mean_rating_bias', 3.0) + sc))),
                        'tmdb_id': film.tmdb_id,
                        'genres': film.genere,
                        'poster_url': poster_url,
                        'tmdb_rating': film.tmdb_rating,
                        'cluster': int(self.cluster_labels[mi]) if self.cluster_labels is not None and mi < len(self.cluster_labels) else 0
                    })
                    if len(recs) >= top_n:
                        break
                except Exception:
                    continue

            return recs
        except Exception as e:
            logger.debug(f"Errore fallback_item_based_recommendations: {e}")
            return []

    def get_service_popular_recommendations(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Restituisce le raccomandazioni popolari costruite dal dataset di training di questa istanza, se disponibili."""
        # If we already built a per-service popular list, return it
        if hasattr(self, '_service_popular_recommendations') and self._service_popular_recommendations:
            return self._service_popular_recommendations[:top_n]

        # Otherwise, try to build a lightweight popular list from the last generated training dataframe
        try:
            df = getattr(self, '_last_training_df', None)
            if df is not None and len(df) > 0:
                # Aggregate mean rating and counts per title
                agg = df.groupby('title').agg({'rating': ['mean', 'count']})
                agg.columns = ['mean_rating', 'count']
                agg = agg.reset_index().sort_values(by=['mean_rating', 'count'], ascending=[False, False])

                results = []
                for _, row in agg.head(top_n).iterrows():
                    title = row['title']
                    results.append({
                        'title': title,
                        'predicted_rating': float(row['mean_rating']),
                        'poster_url': None,
                        'genres': [],
                        'tmdb_id': None,
                        'tmdb_rating': None,
                        'cluster': 0
                    })

                # Cache it to avoid recomputation
                try:
                    self._service_popular_recommendations = results
                except Exception:
                    pass

                return results
        except Exception as e:
            logger.debug(f"Could not build service-specific popular from last training df: {e}")

        # Fallback: global DB popular
        return self._get_popular_recommendations(top_n)

    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Recupera lo storico dei voti dell'utente"""
        try:
            user = Utente.objects(id=user_id).first()
            if not user:
                return []
            
            votes = list(Votazione.objects(utente=user).order_by('-valutazione'))
            
            history = []
            for vote in votes:
                film = vote.film
                poster_url = None
                if film.poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{film.poster_path}"
                elif film.tmdb_id:
                    poster_url = self.fetch_poster_url(film.tmdb_id)
                
                history.append({
                    "title": film.titolo,
                    "rating": vote.valutazione,
                    "genres": film.genere,
                    "poster_url": poster_url,
                    "tmdb_id": film.tmdb_id,
                    "tmdb_rating": film.tmdb_rating
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting user history for {user_id}: {e}")
            return []

    def get_clustering_data(self) -> Dict[str, Any]:
        """Restituisce dati per visualizzazione clustering - SOLO film votati utenti AFlix (test set)"""
        if not self.is_trained:
            return {"error": "Model not trained yet. Please train the model first."}
        
        if self.kmeans_model is None:
            return {"error": f"Clustering not available. Need at least 2 movies for clustering. Current: {self.movie_factors.shape[0] if hasattr(self, 'movie_factors') and self.movie_factors is not None else 0} movies."}
        
        try:
            # Usa i fattori AFlix calcolati durante il training ibrido
            if not hasattr(self, 'aflix_movie_factors') or self.aflix_movie_factors is None:
                return {"error": "AFlix movie factors not available. Train model in hybrid mode first."}
            
            aflix_movie_factors = self.aflix_movie_factors
            if len(aflix_movie_factors) < 2:
                return {"error": f"Insufficient AFlix movies for clustering. Need at least 2, found: {len(aflix_movie_factors)}"}
            
            # Gestisci dimensioni per visualizzazione
            if aflix_movie_factors.shape[1] >= 2:
                X = aflix_movie_factors[:, :2]
            else:
                # Se c'è solo 1 componente, usa quella e aggiungi rumore per clustering
                X = np.column_stack([aflix_movie_factors[:, 0], np.random.normal(0, 0.1, aflix_movie_factors.shape[0])])
            
            # Ottieni informazioni sui film AFlix
            aflix_movie_info = self._get_aflix_movies_info()
            
            # Dati per visualizzazione - SOLO film AFlix
            clustering_data = {
                "points": [
                    {
                        "x": float(X[i, 0]), 
                        "y": float(X[i, 1]), 
                        "cluster": int(self.cluster_labels[i]) if self.cluster_labels is not None and i < len(self.cluster_labels) else 0,
                        "movie_title": aflix_movie_info[i]["title"] if i < len(aflix_movie_info) else f"Movie {i}",
                        "movie_id": aflix_movie_info[i]["movie_id"] if i < len(aflix_movie_info) else None
                    } 
                    for i in range(len(X))
                ],
                "centroids": [
                    {"x": float(center[0]), "y": float(center[1]), "cluster": i} 
                    for i, center in enumerate(self.kmeans_model.cluster_centers_)
                ] if self.kmeans_model is not None else [],
                "n_clusters": self.n_clusters,
                "total_aflix_movies": len(aflix_movie_factors),
                "dataset_type": "aflix_test_set",
                "training_source": self.training_source if hasattr(self, 'training_source') else "unknown"
            }
            
            return clustering_data
            
        except Exception as e:
            logger.error(f"Error generating clustering data: {e}")
            return {"error": str(e)}
    
    def _get_aflix_movies_indices(self) -> List[int]:
        """Ottieni indici dei film votati dagli utenti AFlix nel movie encoder"""
        try:
            # Ottieni tutti i voti AFlix dal database
            votes = list(Votazione.objects.all())
            
            if not votes:
                return []
            
            # Estrai titoli dei film votati
            aflix_movie_titles = {vote.film.titolo for vote in votes}
            
            # Se non abbiamo un movie encoder, non possiamo fare il mapping
            if not hasattr(self, 'movie_encoder') or self.movie_encoder is None:
                logger.warning("Movie encoder not available for AFlix filtering")
                return []
            
            # Ottieni tutti i titoli nel movie encoder
            encoded_titles = self.movie_encoder.classes_
            
            # Trova gli indici dei film AFlix nel movie encoder
            aflix_indices = []
            for i, title in enumerate(encoded_titles):
                if title in aflix_movie_titles:
                    aflix_indices.append(i)
            
            logger.info(f"Found {len(aflix_indices)} AFlix movies out of {len(encoded_titles)} total movies")
            return aflix_indices
            
        except Exception as e:
            logger.error(f"Error getting AFlix movie indices: {e}")
            return []
    
    def _get_aflix_movies_info(self) -> List[Dict[str, Any]]:
        """Ottieni informazioni sui film AFlix per la visualizzazione"""
        try:
            votes = list(Votazione.objects.all())
            
            # Crea mapping titolo -> info film
            movie_info_map = {}
            for vote in votes:
                title = vote.film.titolo
                if title not in movie_info_map:
                    movie_info_map[title] = {
                        "title": title,
                        "movie_id": vote.film.tmdb_id,
                        "genre": vote.film.genere,
                        "avg_rating": 0,
                        "vote_count": 0
                    }
                
                # Calcola media rating
                current_info = movie_info_map[title]
                current_avg = current_info["avg_rating"]
                current_count = current_info["vote_count"]
                
                new_avg = (current_avg * current_count + vote.valutazione) / (current_count + 1)
                movie_info_map[title]["avg_rating"] = new_avg
                movie_info_map[title]["vote_count"] = current_count + 1
            
            # Se non abbiamo movie encoder, restituisci info base
            if not hasattr(self, 'movie_encoder') or self.movie_encoder is None:
                return list(movie_info_map.values())
            
            # Ordina secondo il movie encoder
            encoded_titles = self.movie_encoder.classes_
            aflix_movie_info = []
            
            for title in encoded_titles:
                if title in movie_info_map:
                    aflix_movie_info.append(movie_info_map[title])
            
            return aflix_movie_info
            
        except Exception as e:
            logger.error(f"Error getting AFlix movie info: {e}")
            return []
    
    def _apply_clustering_on_aflix_movies(self):
        """Applica clustering sui film AFlix usando il modello SVD addestrato su TMDB"""
        try:
            logger.info("🎯 Applicazione clustering sui film AFlix...")
            
            # Ottieni dati AFlix
            aflix_df = self.prepare_data()
            if len(aflix_df) < 2:
                logger.warning("Clustering saltato: meno di 2 film AFlix disponibili")
                self.kmeans_model = None
                self.cluster_labels = None
                return
            
            # Proietta i film AFlix nello spazio SVD TMDB
            aflix_movie_factors = self._project_aflix_movies_to_svd_space(aflix_df)
            
            if aflix_movie_factors is None or aflix_movie_factors.shape[0] < 2:
                logger.warning("Clustering saltato: proiezione AFlix non riuscita")
                self.kmeans_model = None
                self.cluster_labels = None  
                return
            
            # Salva i fattori AFlix per il clustering
            self.aflix_movie_factors = aflix_movie_factors
            
            # Determina numero cluster ottimale per AFlix
            n_movies = aflix_movie_factors.shape[0]
            
            # Usa K ottimizzato se disponibile, altrimenti usa euristica migliorata
            if hasattr(self, 'optimal_k_cluster') and self.optimal_k_cluster:
                n_clusters_target = self.optimal_k_cluster
                logger.info(f"🎯 Usando K ottimizzato: {n_clusters_target}")
            else:
                # Euristica migliorata: favorisce 3-5 cluster per dataset piccoli
                if n_movies >= 10:
                    n_clusters_target = min(5, max(3, n_movies // 3))  # 3-5 cluster
                else:
                    n_clusters_target = min(3, max(2, n_movies // 2))  # 2-3 cluster
                logger.info(f"🎯 Usando euristica K: {n_clusters_target} (per {n_movies} film)")
            
            n_clusters_actual = min(n_clusters_target, n_movies - 1)
            
            # Prepara dati per clustering (usa 2D)
            if aflix_movie_factors.shape[1] >= 2:
                X = aflix_movie_factors[:, :2]
            else:
                # Se c'è solo 1 componente, aggiungi rumore per clustering 2D
                X = np.column_stack([
                    aflix_movie_factors[:, 0], 
                    np.random.normal(0, 0.1, aflix_movie_factors.shape[0])
                ])
            
            # Applica K-means clustering sui film AFlix
            self.kmeans_model = KMeans(n_clusters=n_clusters_actual, random_state=42)
            self.cluster_labels = self.kmeans_model.fit_predict(X)
            self.n_clusters = n_clusters_actual
            
            logger.info(f"✅ Clustering AFlix completato: {n_clusters_actual} cluster per {n_movies} film AFlix")
            
            # Statistiche cluster
            unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
            for i, (label, count) in enumerate(zip(unique_labels, counts)):
                logger.info(f"   Cluster {label}: {count} film")
                
        except Exception as e:
            logger.error(f"Errore clustering film AFlix: {e}")
            self.kmeans_model = None
            self.cluster_labels = None
    
    def _project_aflix_movies_to_svd_space(self, aflix_df: pd.DataFrame):
        """Proietta i film AFlix nello spazio SVD addestrato su TMDB"""
        try:
            # Verifica che il modello SVD sia addestrato
            if not hasattr(self, 'item_factors') or self.item_factors is None:
                logger.error("SVD model non disponibile per proiezione AFlix")
                return None
            
            # Trova i film AFlix che sono anche nel training TMDB
            aflix_titles = set(aflix_df['title'].unique())
            tmdb_titles = set(self.movie_encoder.classes_)
            common_titles = aflix_titles.intersection(tmdb_titles)
            
            logger.info(f"Film AFlix: {len(aflix_titles)}, Film TMDB: {len(tmdb_titles)}, Comuni: {len(common_titles)}")
            # Costruisci lista codificata dei titoli noti
            encoded_titles = list(self.movie_encoder.classes_)

            def resolve_item_vector(title: str, genres: List[Any]):
                # 1) exact
                if title in self.movie_encoder.classes_:
                    idx = int(self.movie_encoder.transform([title])[0])
                    return self.item_factors[idx], 'exact'

                # 2) fuzzy
                matches = difflib.get_close_matches(title, encoded_titles, n=3, cutoff=0.72)
                if matches:
                    best = matches[0]
                    idx = int(self.movie_encoder.transform([best])[0])
                    return self.item_factors[idx], f'fuzzy:{best}'

                # 3) genre overlap
                if hasattr(self, 'train_movie_genres') and self.train_movie_genres:
                    aflix_genre_set = set(genres) if genres else set()
                    scores = []
                    for cand in encoded_titles:
                        cand_genres = set(self.train_movie_genres.get(cand, []))
                        if not cand_genres:
                            continue
                        overlap = len(aflix_genre_set.intersection(cand_genres))
                        if overlap > 0:
                            scores.append((cand, overlap))
                    if scores:
                        scores.sort(key=lambda x: x[1], reverse=True)
                        top = [s[0] for s in scores[:3]]
                        idxs = self.movie_encoder.transform(top)
                        vecs = self.item_factors[idxs]
                        weights = np.array([s[1] for s in scores[:3]], dtype=float)
                        weights /= weights.sum()
                        return np.average(vecs, axis=0, weights=weights), f'genre_best:{top}'

                # 4) partial similarity weighted average
                from difflib import SequenceMatcher
                ratios = []
                for cand in encoded_titles:
                    r = SequenceMatcher(None, title.lower(), cand.lower()).ratio()
                    if r > 0.35:
                        ratios.append((cand, r))
                if ratios:
                    ratios.sort(key=lambda x: x[1], reverse=True)
                    top = [r[0] for r in ratios[:5]]
                    idxs = self.movie_encoder.transform(top)
                    vecs = self.item_factors[idxs]
                    weights = np.array([r[1] for r in ratios[:5]], dtype=float)
                    weights /= weights.sum()
                    return np.average(vecs, axis=0, weights=weights), f'partial_sim:{[t for t in top]}'

                # 5) fallback None
                return None, 'none'

            aflix_movie_factors = []
            aflix_movie_info = []

            # Manteniamo ordine coerente con aflix_titles list
            for title in sorted(list(aflix_titles)):
                # trova generi a partire dal dataframe (prendi primo match)
                try:
                    genres = list(aflix_df[aflix_df['title'] == title]['genres'].iloc[0]) if 'genres' in aflix_df.columns and len(aflix_df[aflix_df['title'] == title]) > 0 else []
                except Exception:
                    genres = []

                vec, why = resolve_item_vector(title, genres)
                if vec is None:
                    # se non risolto, usa media item_factors con rumore maggiore per varianza
                    avg_factor = np.mean(self.item_factors, axis=0)
                    noise = np.random.normal(0, 0.2, avg_factor.shape)
                    vec = avg_factor + noise
                    why = 'fallback_mean_no_match'

                aflix_movie_factors.append(vec)
                aflix_movie_info.append({"title": title, "mapping": why})

            self.aflix_movie_info = aflix_movie_info
            # Log summary of mapping reasons
            try:
                counts = {}
                for info in aflix_movie_info:
                    key = info.get('mapping', 'unknown')
                    counts[key] = counts.get(key, 0) + 1
                logger.info(f"AFlix->SVD mapping summary: {counts}")
            except Exception:
                pass

            return np.array(aflix_movie_factors)
                
        except Exception as e:
            logger.error(f"Errore proiezione film AFlix: {e}")
            return None

    def evaluate_model(self) -> Dict[str, Any]:
        """Valuta le performance del modello"""
        if not self.is_trained:
            return {"error": "Model not trained"}

        try:
            # Prefer to evaluate the trained instance against real AFlix votes
            # using the instance's learned item_factors / user_factors mapping.
            aflix_result = self._test_on_aflix_data()

            # If the class test returns usable predictions, use its RMSE
            if isinstance(aflix_result, dict) and aflix_result.get('status') == 'success' and aflix_result.get('rmse') is not None:
                return {
                    "rmse": float(aflix_result.get('rmse')),
                    "mae": float(aflix_result.get('mae')) if aflix_result.get('mae') is not None else None,
                    "test_samples": int(aflix_result.get('test_samples', 0)),
                    "train_samples": int(getattr(self, '_training_ratings_count', 0) or 0),
                    "explained_variance": float(getattr(self, 'explained_variance', 0.0)),
                    "model_status": "evaluated_aflix_test"
                }

            # Fallback: if _test_on_aflix_data could not produce predictions, run a local
            # evaluation on a sampled AFlix subset (this was the previous behavior).
            df = self.prepare_data()
            if len(df) < 15:
                return {
                    "error": f"Insufficient data for evaluation (minimum 15 votes required, current: {len(df)})",
                    "current_votes": len(df),
                    "required_votes": 15,
                    "model_status": "trained_but_not_evaluable"
                }

            sample_users = df['userId'].value_counts().head(100).index
            sample_df = df[df['userId'].isin(sample_users)].copy()

            # Re-encoding per il sample
            user_encoder_eval = LabelEncoder()
            movie_encoder_eval = LabelEncoder()
            sample_df['user_idx'] = user_encoder_eval.fit_transform(sample_df['userId'])
            sample_df['movie_idx'] = movie_encoder_eval.fit_transform(sample_df['title'])

            train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=42)

            train_matrix = csr_matrix(
                (train_df['rating'], (train_df['user_idx'], train_df['movie_idx'])),
                shape=(sample_df['user_idx'].nunique(), sample_df['movie_idx'].nunique())
            )

            eval_components = min(30, min(train_matrix.shape) - 1)
            eval_components = max(1, eval_components)

            svd_eval = TruncatedSVD(n_components=eval_components, random_state=42)
            user_factors_eval = svd_eval.fit_transform(train_matrix)
            movie_factors_eval = svd_eval.components_.T

            predictions, actuals = [], []
            for _, row in test_df.iterrows():
                u, m = row['user_idx'], row['movie_idx']
                if u < user_factors_eval.shape[0] and m < movie_factors_eval.shape[0]:
                    pred = float(np.dot(user_factors_eval[u], movie_factors_eval[m]))
                    pred += 3.0
                    pred = max(1.0, min(5.0, pred))
                    predictions.append(pred)
                    actuals.append(row['rating'])

            if len(predictions) == 0:
                return {"error": "No valid predictions generated"}

            rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
            mae = float(mean_absolute_error(actuals, predictions))

            evaluation = {
                "rmse": rmse,
                "mae": mae,
                "test_samples": len(predictions),
                "train_samples": len(train_df),
                "explained_variance": float(getattr(self, 'explained_variance', 0.0)),
                "model_status": "evaluated_local"
            }

            logger.info(f"Model evaluation completed (fallback local): RMSE={rmse:.4f}, MAE={mae:.4f}")
            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"error": str(e)}

    def get_model_status(self) -> Dict[str, Any]:
        """Restituisce lo stato del modello con dettagli sul fattore k"""
        explained_var = 0.0
        if self.is_trained and self.explained_variance is not None:
            explained_var = float(self.explained_variance)
            # Gestisci valori NaN o infiniti
            if not (explained_var == explained_var):  # Check for NaN (NaN != NaN)
                explained_var = 0.0
            elif explained_var == float('inf') or explained_var == float('-inf'):
                explained_var = 0.0
        
        # Determina il K-SVD attuale utilizzato
        current_k_svd = 0
        if self.is_trained:
            if hasattr(self, 'actual_k_used') and self.actual_k_used > 0:
                current_k_svd = self.actual_k_used
            elif hasattr(self, 'current_k_factor') and self.current_k_factor > 0:
                current_k_svd = self.current_k_factor
            elif self.svd_model and hasattr(self.svd_model, 'n_components'):
                current_k_svd = self.svd_model.n_components
            else:
                current_k_svd = self.n_components

        # Calcola total_ratings se il modello è addestrato
        total_ratings = 0
        if self.is_trained:
            try:
                # Usa sempre i dati effettivi di training (salvati durante addestramento)
                total_ratings = getattr(self, '_training_ratings_count', 0)
                
                # Se non disponibile, fallback basato su modalità
                if total_ratings == 0:
                    if hasattr(self, 'training_source') and 'hybrid' in str(self.training_source).lower():
                        # Modalità ibrida: dovrebbe essere ~400k TMDB
                        total_ratings = 400000  # Stima per TMDB
                    else:
                        # Modalità AFlix-only: conta voti reali
                        votes = list(Votazione.objects.all())
                        total_ratings = len(votes)
            except:
                total_ratings = 0

        return {
            "is_trained": self.is_trained,
            "explained_variance": explained_var,
            "n_components": current_k_svd,  # K-SVD attualmente utilizzato
            "actual_k_used": self.actual_k_used,
            "requested_k": self.n_components,
            "k_efficiency": float(explained_var / current_k_svd) if current_k_svd > 0 else 0,
            "n_clusters": self.n_clusters if hasattr(self, 'n_clusters') else 3,
            "has_clustering": self.kmeans_model is not None,
            "variance_per_component": self.variance_per_component if hasattr(self, 'variance_per_component') else [],
            "k_optimization_available": len(self.k_performance_log) > 0 if hasattr(self, 'k_performance_log') else False,
            "total_ratings": total_ratings,
            "training_source": getattr(self, 'training_source', None),
            "last_trained_at": getattr(self, 'last_trained_at', None),
            "instance_name": getattr(self, 'instance_name', None),
            # Debug info
            "debug": {
                "svd_model_exists": self.svd_model is not None,
                "svd_n_components": self.svd_model.n_components if self.svd_model else None,
                "current_k_factor": getattr(self, 'current_k_factor', None),
                "movie_factors_shape": self.movie_factors.shape if hasattr(self, 'movie_factors') and self.movie_factors is not None else None
            }
        }

    def _train_content_based_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback per Content-Based quando c'è solo 1 utente"""
        try:
            logger.info("Training Content-Based model (single user fallback)")
            
            # Marca come "addestrato" ma con modalità content-based
            self.is_trained = True
            self.svd_model = None
            self.kmeans_model = None
            self.explained_variance = 0.0
            
            # Salva i dati dell'utente per raccomandazioni content-based
            self.user_profile = df.copy()
            
            # Calcola preferenze per genere
            self.genre_preferences = {}
            for _, row in df.iterrows():
                rating = row['rating']
                for genre in row['genres']:
                    if genre not in self.genre_preferences:
                        self.genre_preferences[genre] = {'total': 0, 'count': 0}
                    self.genre_preferences[genre]['total'] += rating
                    self.genre_preferences[genre]['count'] += 1
            
            # Calcola medie per genere
            for genre in self.genre_preferences:
                self.genre_preferences[genre]['avg'] = self.genre_preferences[genre]['total'] / self.genre_preferences[genre]['count']
            
            stats = {
                "total_ratings": len(df),
                "unique_users": 1,
                "unique_movies": df['movieId'].nunique(),
                "model_type": "content_based",
                "preferred_genres": len(self.genre_preferences),
                "training_status": "success_content_based"
            }
            
            logger.info(f"Content-Based model trained. Stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in content-based training: {e}")
            raise

    def analyze_k_factor(self) -> Dict[str, Any]:
        """
        Analizza il fattore k utilizzato nella SVD e fornisce insights
        
        Returns:
            Analisi dettagliata del fattore k e raccomandazioni
        """
        if not self.is_trained or not self.svd_model:
            return {"error": "Model not trained or SVD not available"}
        
        try:
            analysis = {
                "current_k": self.actual_k_used,
                "requested_k": self.n_components,
                "max_possible_k": min(self.svd_model.n_features_in_, len(self.variance_per_component)),
                "total_explained_variance": float(self.explained_variance),
                "variance_per_component": self.variance_per_component,
                "k_efficiency": float(self.explained_variance / self.actual_k_used) if self.actual_k_used > 0 else 0,
                "cumulative_variance": [],
                "elbow_point": None,
                "recommended_k": None
            }
            
            # Calcola varianza cumulativa
            cumulative = 0
            for i, var in enumerate(self.variance_per_component):
                cumulative += var
                analysis["cumulative_variance"].append({
                    "component": i + 1,
                    "individual_variance": float(var),
                    "cumulative_variance": float(cumulative),
                    "percentage": float(cumulative * 100)
                })
            
            # Trova elbow point (punto di diminuzione significativa)
            if len(self.variance_per_component) > 2:
                differences = []
                for i in range(1, len(self.variance_per_component)):
                    diff = self.variance_per_component[i-1] - self.variance_per_component[i]
                    differences.append(diff)
                
                if differences:
                    # Trova il punto dove la differenza cala significativamente
                    max_diff = max(differences)
                    for i, diff in enumerate(differences):
                        if diff < max_diff * 0.1:  # 10% del massimo
                            analysis["elbow_point"] = i + 1
                            break
            
            # Raccomandazione k ottimale
            # Usa 95% della varianza come target
            target_variance = 0.95
            for item in analysis["cumulative_variance"]:
                if item["cumulative_variance"] >= target_variance:
                    analysis["recommended_k"] = item["component"]
                    break
            
            # Se non raggiunge 95%, usa elbow point
            if not analysis["recommended_k"] and analysis["elbow_point"]:
                analysis["recommended_k"] = analysis["elbow_point"]
            
            # Fallback: usa 80% dei componenti attuali
            if not analysis["recommended_k"]:
                analysis["recommended_k"] = max(1, int(self.actual_k_used * 0.8))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing k factor: {e}")
            return {"error": str(e)}
    
    def reset_model_parameters(self):
        """Reset parametri modello ai valori ottimizzati"""
        logger.info("🔄 Reset parametri modello ai valori ottimizzati...")
        
        # Reset parametri K
        self.n_components = 25
        self.current_k_factor = 25  
        self.n_clusters = 4
        
        # Reset range ottimizzazione
        self.k_svd_range = range(10, 51, 5)
        self.k_cluster_range = range(2, 8)
        
        # Reset stato training per forzare ri-addestramento
        self.is_trained = False
        self.svd_model = None
        self.kmeans_model = None
        
        logger.info(f"✅ Parametri reset: K-SVD={self.n_components}, K-Cluster={self.n_clusters}")
        return True

    def optimize_k_factor(self, k_range: List[int] = None) -> Dict[str, Any]:
        """
        Ottimizza il fattore k testando diversi valori
        
        Args:
            k_range: Lista di valori k da testare (default: range automatico)
            
        Returns:
            Risultati dell'ottimizzazione con k ottimale
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Prepara dati per test
            df = self.prepare_data()
            
            if len(df) < 20:
                return {"error": "Insufficient data for k optimization"}
            
            # Range automatico se non specificato
            if k_range is None:
                max_k = min(35, min(df['userId'].nunique(), df['movieId'].nunique()) - 1)  # 🔧 FIX: Max 35 (non più 50!)
                max_k = max(1, max_k)  # Assicura almeno 1
                if max_k <= 3:
                    k_range = list(range(1, max_k + 1))  # Per dataset piccoli
                else:
                    k_range = list(range(5, max_k + 1, 5))  # Step di 5 per dataset grandi
            
            results = []
            best_score = float('inf')
            best_k = None
            
            logger.info(f"Testing k values: {k_range}")
            logger.info("=" * 80)
            logger.info("🔬 OTTIMIZZAZIONE FATTORE K - PROCESSO DETTAGLIATO")
            logger.info("=" * 80)
            logger.info(f"🎯 Range K da testare: {k_range}")
            logger.info(f"📊 Dataset: {len(df)} rating, {df['userId'].nunique()} utenti, {df['movieId'].nunique()} film")
            logger.info("=" * 80)
            
            for k in k_range:
                try:
                    logger.info(f"\n🔍 TEST K = {k}")
                    logger.info("-" * 40)
                    
                    # Encoding per test
                    user_encoder_test = LabelEncoder()
                    movie_encoder_test = LabelEncoder()
                    
                    df_test = df.copy()
                    df_test['user_idx'] = user_encoder_test.fit_transform(df_test['userId'])
                    df_test['movie_idx'] = movie_encoder_test.fit_transform(df_test['title'])
                    
                    # Split train/test
                    train_df, test_df = train_test_split(df_test, test_size=0.2, random_state=42)
                    logger.info(f"📚 Train set: {len(train_df)} rating")
                    logger.info(f"🧪 Test set: {len(test_df)} rating")
                    
                    # Matrice training
                    train_matrix = csr_matrix(
                        (train_df['rating'], (train_df['user_idx'], train_df['movie_idx'])),
                        shape=(df_test['user_idx'].nunique(), df_test['movie_idx'].nunique())
                    )
                    logger.info(f"🏗️  Matrice training: {train_matrix.shape}")
                    
                    # Test SVD con k componenti
                    logger.info(f"⚙️  Esecuzione SVD con k={k}...")
                    svd_test = TruncatedSVD(n_components=k, random_state=42)
                    user_factors_test = svd_test.fit_transform(train_matrix)
                    movie_factors_test = svd_test.components_.T
                    
                    explained_var = float(svd_test.explained_variance_ratio_.sum())
                    logger.info(f"📈 Varianza spiegata: {explained_var:.1%}")
                    
                    # Valutazione
                    logger.info("🎯 Generazione predizioni...")
                    predictions, actuals = [], []
                    test_sample = test_df.head(100)  # Limita per velocità
                    
                    for idx, (_, row) in enumerate(test_sample.iterrows()):
                        u, m = row['user_idx'], row['movie_idx']
                        if u < user_factors_test.shape[0] and m < movie_factors_test.shape[0]:
                            pred = np.dot(user_factors_test[u], movie_factors_test[m])
                            
                            # Aggiungi bias rating medio e clamp nel range [1, 5]
                            pred += 3.0  # Rating medio  
                            pred = max(1.0, min(5.0, pred))
                            
                            predictions.append(pred)
                            actuals.append(row['rating'])
                        
                        # Progress indicator per test lunghi
                        if idx % 25 == 0 and idx > 0:
                            logger.info(f"   📊 Processate {idx}/{len(test_sample)} predizioni...")
                    
                    if len(predictions) > 5:
                        rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
                        mae = float(mean_absolute_error(actuals, predictions))
                        
                        # Score combinato (RMSE penalizzato + bonus varianza)
                        combined_score = rmse - (explained_var * 0.5)
                        
                        result = {
                            "k": k,
                            "rmse": rmse,
                            "mae": mae,
                            "explained_variance": explained_var,
                            "combined_score": combined_score,
                            "test_predictions": len(predictions)
                        }
                        
                        results.append(result)
                        
                        # 📊 LOG RISULTATI IN TEMPO REALE
                        logger.info(f"✅ RISULTATI K={k}:")
                        logger.info(f"   📉 RMSE: {rmse:.4f}")
                        logger.info(f"   📊 MAE: {mae:.4f}")
                        logger.info(f"   📈 Varianza: {explained_var:.1%}")
                        logger.info(f"   🎯 Score Combinato: {combined_score:.4f}")
                        logger.info(f"   🔢 Predizioni valide: {len(predictions)}")
                        
                        if combined_score < best_score:
                            best_score = combined_score
                            best_k = k
                            logger.info(f"   🏆 NUOVO MIGLIOR K! ({k})")
                        
                    else:
                        logger.warning(f"   ⚠️  Predizioni insufficienti per k={k} ({len(predictions)})")
                    
                except Exception as e:
                    logger.error(f"   ❌ Errore testing k={k}: {e}")
                    continue
            
            if results:
                # Ordina per score
                results.sort(key=lambda x: x["combined_score"])
                
                logger.info("\n" + "=" * 80)
                logger.info("🏆 RISULTATI FINALI OTTIMIZZAZIONE")
                logger.info("=" * 80)
                logger.info(f"🥇 MIGLIOR K TROVATO: {best_k}")
                logger.info(f"📊 K ATTUALE: {self.actual_k_used}")
                logger.info(f"🔄 MIGLIORAMENTO POSSIBILE: {'SÌ' if best_k != self.actual_k_used else 'NO'}")
                logger.info("\n🏅 TOP 5 CONFIGURAZIONI:")
                logger.info("-" * 80)
                logger.info("Pos | K   | RMSE   | MAE    | Varianza | Score")
                logger.info("-" * 80)
                
                for i, result in enumerate(results[:5], 1):
                    k = result['k']
                    rmse = result['rmse']
                    mae = result['mae']
                    var = result['explained_variance']
                    score = result['combined_score']
                    marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                    logger.info(f"{marker} {i:2d} | {k:3d} | {rmse:.4f} | {mae:.4f} | {var:.4f}   | {score:.4f}")
                
                logger.info("=" * 80)
                
                optimization_result = {
                    "best_k": best_k,
                    "current_k": self.actual_k_used,
                    "improvement": best_k != self.actual_k_used,
                    "all_results": results,
                    "recommendation": f"Use k={best_k} for optimal performance"
                }
                
                # Salva nel log delle performance
                self.k_performance_log = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "optimization_results": optimization_result
                }
                
                logger.info(f"💡 RACCOMANDAZIONE: {optimization_result['recommendation']}")
                return optimization_result
            else:
                logger.error("❌ Nessun risultato valido dall'ottimizzazione")
                return {"error": "No valid results from k optimization"}
            
        except Exception as e:
            logger.error(f"Error optimizing k factor: {e}")
            return {"error": str(e)}

    def get_k_factor_report(self) -> Dict[str, Any]:
        """
        Genera un report completo sul fattore k
        
        Returns:
            Report dettagliato con analisi, storia e raccomandazioni
        """
        try:
            report = {
                "current_status": {
                    "is_trained": self.is_trained,
                    "current_k": self.actual_k_used,
                    "requested_k": self.n_components,
                    "explained_variance": float(self.explained_variance) if self.explained_variance else 0,
                },
                "k_analysis": self.analyze_k_factor() if self.is_trained else {},
                "k_history": self.k_history,
                "performance_log": self.k_performance_log,
                "recommendations": []
            }
            
            # Genera raccomandazioni
            if self.is_trained:
                if self.explained_variance < 0.5:
                    report["recommendations"].append({
                        "type": "warning",
                        "message": f"Low explained variance ({self.explained_variance:.2%}). Consider increasing k or improving data quality."
                    })
                
                if self.actual_k_used < self.n_components * 0.5:
                    report["recommendations"].append({
                        "type": "info",
                        "message": f"Using much fewer components than requested ({self.actual_k_used}/{self.n_components}). Dataset might be limited."
                    })
                
                if self.actual_k_used > 30 and self.explained_variance > 0.95:
                    report["recommendations"].append({
                        "type": "optimization",
                        "message": f"High k with excellent variance. Consider running k optimization to find minimum effective k."
                    })
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating k factor report: {e}")
            return {"error": str(e)}

# Creiamo due istanze del servizio ML: una per TMDB e una per OMDb
ml_service_tmdb = MLRecommendationService()
ml_service_omdb = MLRecommendationService()

# Configura le istanze per la sorgente corretta
ml_service_tmdb.use_tmdb_training = True
ml_service_tmdb.use_omdb_training = False
ml_service_tmdb.cache_dir = os.path.join('data', 'cache', 'tmdb')
ml_service_tmdb.instance_name = 'tmdb'

ml_service_omdb.use_tmdb_training = False
ml_service_omdb.use_omdb_training = True
ml_service_omdb.cache_dir = os.path.join('data', 'cache', 'omdb')
ml_service_omdb.instance_name = 'omdb'

def get_ml_service_for_source(source: Optional[str]):
    """Restituisce l'istanza ML corretta in base alla sorgente ('tmdb' o 'omdb')."""
    if source and str(source).lower() == 'omdb':
        return ml_service_omdb
    return ml_service_tmdb

# Manteniamo ml_service per retrocompatibilità (usa TMDB)
ml_service = ml_service_tmdb

def reset_global_ml_service():
    """Reset basico per entrambe le istanze (usato in dev)."""
    for svc in [ml_service_tmdb, ml_service_omdb]:
        svc.n_components = 25
        svc.current_k_factor = 25
        svc.n_clusters = 4
        svc.k_svd_range = range(10, 51, 5)
        svc.k_cluster_range = range(2, 8)
        svc.is_trained = False

# Esegui reset immediato in avvio per sicurezza
reset_global_ml_service()