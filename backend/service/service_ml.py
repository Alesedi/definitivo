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
        
        # Parametri modello
        self.n_components = 50
        self.n_clusters = 3
        
        # Ottimizzazione automatica K
        self.auto_optimize_k_svd = True
        self.auto_optimize_k_cluster = True
        self.k_svd_range = range(10, 101, 10)
        self.k_cluster_range = range(2, 16)
        
        # Tracciamento fattore k (numero componenti SVD)
        self.actual_k_used = 0  # Numero effettivo di componenti utilizzati
        self.k_history = []  # Storico dei valori k testati
        self.variance_per_component = []  # Varianza spiegata per ogni componente
        self.optimal_k = None  # Valore k ottimale identificato
        self.k_performance_log = {}  # Log performance per diversi k
        
        # TMDB Integration
        self.tmdb_api_key = os.getenv('TMDB_API_KEY', '8265bd1679663a7ea12ac168da84d2e8')
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.tmdb_cache_dir = "data/tmdb_cache"
        self.use_tmdb_training = True
        self.tmdb_movies_df = None
        self.tmdb_ratings_df = None
        self.training_source = "hybrid"
        self.cache_dir = "data/cache"
        self.current_k_factor = 50
        
        # TMDB API per poster
        self.TMDB_API_KEY = "9e6c375b125d733d9ce459bdd91d4a06"
        self.TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/{}/images?api_key={}"

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
            logger.info("🚀 INIZIO TRAINING MODELLO ML")
            logger.info("=" * 80)
            
            # Decide fonte dati per training
            if self.use_tmdb_training and self.tmdb_api_key:
                logger.info("🎬 Modalità HYBRID: Training su TMDB + Testing su AFlix")
                return self._train_hybrid_model()
            else:
                logger.info("🏠 Modalità AFlix-only: Training su dati AFlix")
                return self._train_aflix_only_model()
                
        except Exception as e:
            logger.error(f"❌ Errore training modello: {e}")
            raise
    
    def _train_hybrid_model(self) -> Dict[str, Any]:
        """Training ibrido: TMDB per training, AFlix per testing"""
        
        # 1. Genera o carica dataset TMDB
        tmdb_data = self._get_or_generate_tmdb_data()
        
        # 2. Training SVD su dati TMDB
        logger.info("🧠 Training SVD su dataset TMDB...")
        ratings_matrix = self._create_ratings_matrix(tmdb_data)
        
        # 3. Applica SVD
        self._apply_svd_to_matrix(ratings_matrix)
        
        # 4. Test su dati AFlix se disponibili
        aflix_performance = self._test_on_aflix_data()
        
        # 5. Statistiche finali
        stats = self._compile_hybrid_stats(tmdb_data, aflix_performance)
        
        self.is_trained = True
        self.training_source = "hybrid"
        
        return stats
    
    def _train_aflix_only_model(self) -> Dict[str, Any]:
        """Training tradizionale solo su dati AFlix"""
        try:
            # Prepara i dati AFlix
            df = self.prepare_data()
            
            if len(df) < 10:
                logger.warning("⚠️ Dataset AFlix tropico piccolo, uso dati demo TMDB...")
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
        
        # Training SVD
        logger.info("🔄 Esecuzione TruncatedSVD...")
        self.svd_model = TruncatedSVD(n_components=safe_components, random_state=42)
        logger.info("📊 Calcolo fattori latenti utenti...")
        self.user_factors = self.svd_model.fit_transform(ratings_sparse)
        logger.info("🎬 Calcolo fattori latenti film...")
        self.movie_factors = self.svd_model.components_.T
        self.explained_variance = self.svd_model.explained_variance_ratio_.sum()
            
        logger.info("✅ SVD completata!")
        logger.info(f"📈 Varianza totale spiegata: {self.explained_variance:.1%}")
        logger.info(f"👥 Fattori utenti: {self.user_factors.shape}")
        logger.info(f"🎭 Fattori film: {self.movie_factors.shape}")
        
        # Tracciamento dettagliato del fattore k
        self.actual_k_used = safe_components
        self.variance_per_component = self.svd_model.explained_variance_ratio_.tolist()
        
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
        if self.movie_factors.shape[0] > self.n_clusters and self.movie_factors.shape[1] >= 2:
            # Usa prime 2 componenti se disponibili
            X = self.movie_factors[:, :2]
            self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.cluster_labels = self.kmeans_model.fit_predict(X)
        elif self.movie_factors.shape[0] > self.n_clusters and self.movie_factors.shape[1] == 1:
            # Se c'è solo 1 componente, usa quella e aggiungi rumore per clustering
            X = np.column_stack([self.movie_factors[:, 0], np.random.normal(0, 0.1, self.movie_factors.shape[0])])
            self.kmeans_model = KMeans(n_clusters=min(self.n_clusters, self.movie_factors.shape[0]), random_state=42)
            self.cluster_labels = self.kmeans_model.fit_predict(X)
        else:
            # Troppi pochi dati per clustering significativo
            self.kmeans_model = None
            self.cluster_labels = None
        
        self.is_trained = True
        
        # Statistiche training
        stats = {
            "total_ratings": len(df),
            "unique_users": df['userId'].nunique(),
            "unique_movies": df['movieId'].nunique(),
            "explained_variance": float(self.explained_variance),
            "n_components": self.svd_model.n_components,
            "actual_k_used": self.actual_k_used,
            "k_efficiency": float(self.explained_variance / self.actual_k_used) if self.actual_k_used > 0 else 0,
            "training_status": "success"
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
        popular_movies = self._fetch_tmdb_popular_movies(pages=20)  # ~400 film
        
        # Genera rating sintetici
        tmdb_ratings = self._generate_synthetic_ratings(popular_movies, n_users=10000)
        
        # Salva in cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(tmdb_ratings, f)
        
        logger.info(f"✅ Dataset TMDB generato: {len(tmdb_ratings)} rating")
        return tmdb_ratings
    
    def _fetch_tmdb_popular_movies(self, pages: int = 10) -> List[Dict]:
        """Recupera film popolari da TMDB API"""
        import requests
        
        movies = []
        
        for page in range(1, pages + 1):
            try:
                url = f"{self.tmdb_base_url}/movie/popular"
                params = {
                    'api_key': self.tmdb_api_key,
                    'page': page,
                    'language': 'it-IT'
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data.get('results', []))
                
                # Rate limiting
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Errore fetch pagina {page}: {e}")
                break
        
        logger.info(f"📥 Recuperati {len(movies)} film da TMDB")
        return movies
    
    def _generate_synthetic_ratings(self, movies: List[Dict], n_users: int = 10000) -> pd.DataFrame:
        """Genera rating sintetici basati su caratteristiche TMDB"""
        
        ratings_data = []
        
        # Simula diversi tipi di utenti con preferenze
        user_profiles = self._create_user_profiles(n_users)
        
        for user_id in range(n_users):
            profile = user_profiles[user_id]
            
            # Ogni utente valuta 10-50 film casualmente
            n_ratings = np.random.randint(10, 51)
            user_movies = np.random.choice(len(movies), size=min(n_ratings, len(movies)), replace=False)
            
            for movie_idx in user_movies:
                movie = movies[movie_idx]
                
                # Genera rating basato su profilo utente e caratteristiche film
                rating = self._calculate_synthetic_rating(profile, movie)
                
                ratings_data.append({
                    'userId': f"tmdb_user_{user_id}",
                    'movieId': movie['id'],
                    'title': movie['title'],
                    'rating': rating,
                    'timestamp': datetime.now().timestamp(),
                    'genres': movie.get('genre_ids', []),
                    'tmdb_rating': movie.get('vote_average', 5.0),
                    'popularity': movie.get('popularity', 0)
                })
        
        df = pd.DataFrame(ratings_data)
        logger.info(f"🎯 Generati {len(df)} rating sintetici per {n_users} utenti")
        return df
    
    def _create_user_profiles(self, n_users: int) -> List[Dict]:
        """Crea profili utente diversificati"""
        profiles = []
        
        # Generi TMDB comuni
        genres = [28, 12, 16, 35, 80, 99, 18, 10751, 14, 36, 27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37]
        
        for _ in range(n_users):
            # Preferenze casuali ma realistiche
            favorite_genres = np.random.choice(genres, size=np.random.randint(2, 6), replace=False)
            rating_tendency = np.random.normal(3.5, 0.8)  # Tendenza rating
            rating_variance = np.random.uniform(0.5, 1.5)  # Variabilità rating
            
            profiles.append({
                'favorite_genres': favorite_genres.tolist(),
                'rating_tendency': max(1.0, min(5.0, rating_tendency)),
                'rating_variance': rating_variance,
                'popularity_bias': np.random.uniform(-0.5, 1.0)  # Bias verso film popolari
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
        
        min_dim = min(ratings_matrix.shape)
        self.current_k_factor = min(self.current_k_factor, min_dim - 1)
        
        logger.info(f"🧮 Applicando SVD con k_factor: {self.current_k_factor}")
        
        U, sigma, Vt = svds(ratings_matrix, k=self.current_k_factor)
        self.user_factors = U
        self.item_factors = Vt.T
        
        # Conversione per compatibilità numpy
        self.user_factors = np.ascontiguousarray(self.user_factors)
        self.item_factors = np.ascontiguousarray(self.item_factors)
        
        logger.info(f"✅ SVD completato: {U.shape} x {Vt.T.shape}")
    
    def _create_ratings_matrix(self, df: pd.DataFrame):
        """Crea matrice ratings da DataFrame"""
        
        # Encoding
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
        df['user_idx'] = self.user_encoder.fit_transform(df['userId'])
        df['movie_idx'] = self.movie_encoder.fit_transform(df['title'])
        
        # Matrice sparsa
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
            
            # Mappa film AFlix a TMDB (semplificato)
            # In produzione, useresti fuzzy matching o mapping ID
            common_movies = []
            for title in aflix_df['title'].unique():
                if title in self.movie_encoder.classes_:
                    common_movies.append(title)
            
            if len(common_movies) < 3:
                return {"status": "no_common_movies", "rmse": None}
            
            # Test su film in comune
            test_data = aflix_df[aflix_df['title'].isin(common_movies)]
            
            # Predizioni
            predictions = []
            actuals = []
            
            for _, row in test_data.iterrows():
                try:
                    movie_idx = self.movie_encoder.transform([row['title']])[0]
                    # Usa utente generico (media)
                    pred_rating = np.mean(self.item_factors[movie_idx])
                    predictions.append(pred_rating)
                    actuals.append(row['rating'])
                except:
                    continue
            
            if len(predictions) > 0:
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                return {
                    "status": "success",
                    "rmse": float(rmse),
                    "test_samples": len(predictions),
                    "common_movies": len(common_movies)
                }
            
            return {"status": "no_predictions", "rmse": None}
            
        except Exception as e:
            logger.error(f"Errore test AFlix: {e}")
            return {"status": "error", "rmse": None, "error": str(e)}
    
    def _compile_hybrid_stats(self, tmdb_data: pd.DataFrame, aflix_test: Dict) -> Dict:
        """Compila statistiche training ibrido"""
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'training_mode': 'hybrid',
            'model_info': {
                'algorithm': 'SVD',
                'k_factor': self.current_k_factor,
                'training_source': 'tmdb',
                'test_source': 'aflix'
            },
            'tmdb_data': {
                'total_ratings': len(tmdb_data),
                'unique_users': tmdb_data['userId'].nunique(),
                'unique_movies': tmdb_data['title'].nunique(),
                'rating_distribution': tmdb_data['rating'].value_counts().to_dict()
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
                
                # Test SVD con k componenti
                U, sigma, Vt = svds(ratings_matrix, k=k)
                
                # Calcola varianza spiegata approssimata
                total_variance = np.sum(sigma ** 2)
                explained_variance = total_variance / (ratings_matrix.nnz if hasattr(ratings_matrix, 'nnz') else ratings_matrix.size)
                
                # Calcola efficienza (varianza per componente)
                efficiency = explained_variance / k
                
                # Score composito (personalizzabile)
                composite_score = explained_variance * 0.7 + efficiency * 0.3
                
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
        """Ottimizza K per Clustering"""
        
        logger.info("🎯 Ottimizzazione K-Cluster...")
        
        # Usa SVD per ridurre dimensionalità per clustering
        try:
            k_for_clustering = min(20, min(ratings_matrix.shape) - 1)
            U, sigma, Vt = svds(ratings_matrix, k=k_for_clustering)
            
            # Usa prime 2 componenti se possibile
            if U.shape[1] >= 2:
                cluster_data = U[:, :2]
            else:
                # Aggiungi rumore per clustering 1D
                cluster_data = np.column_stack([U[:, 0], np.random.normal(0, 0.1, U.shape[0])])
            
        except Exception as e:
            logger.warning(f"Errore preparazione dati clustering: {e}")
            return None
        
        best_k = None
        best_score = -1
        results = []
        
        for k in self.k_cluster_range:
            if k >= len(cluster_data):
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
                
                # Interpretabilità (preferisci meno cluster)
                interpretability = 1.0 / k
                
                # Score composito
                composite_score = silhouette * 0.6 + balance * 0.3 + interpretability * 0.1
                
                results.append({
                    'k': k,
                    'silhouette_score': silhouette,
                    'balance': balance,
                    'interpretability': interpretability,
                    'composite_score': composite_score
                })
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_k = k
                
                logger.info(f"K={k:2d} | Silhouette: {silhouette:.3f} | Balance: {balance:.3f} | Score: {composite_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Errore K-Cluster={k}: {e}")
                continue
        
        # Salva risultati
        self.k_performance_log['cluster_optimization'] = results
        
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
            # Verifica che l'utente esista
            if user_id not in self.user_encoder.classes_:
                return self._get_popular_recommendations(top_n)
            
            user_idx = self.user_encoder.transform([user_id])[0]
            
            # Calcola rating predetti
            predicted_ratings = np.dot(self.movie_factors, self.user_factors[user_idx])
            predicted_ratings = np.clip(predicted_ratings, 0.5, 5.0)
            
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
            # Recupera film più votati dal database
            films = list(Film.objects.order_by('-media_voti', '-numero_voti')[:top_n])
            
            recommendations = []
            for film in films:
                poster_url = None
                if film.poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{film.poster_path}"
                elif film.tmdb_id:
                    poster_url = self.fetch_poster_url(film.tmdb_id)
                
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
        """Restituisce dati per visualizzazione clustering"""
        if not self.is_trained or self.kmeans_model is None:
            return {"error": "Clustering model not available. Need more diverse data for clustering."}
        
        try:
            # Gestisci sia il caso di 1 che 2+ componenti
            if self.movie_factors.shape[1] >= 2:
                X = self.movie_factors[:, :2]
            else:
                # Se c'è solo 1 componente, ricostruisci la matrice usata per clustering
                X = np.column_stack([self.movie_factors[:, 0], np.random.normal(0, 0.1, self.movie_factors.shape[0])])
            
            # Dati per visualizzazione
            clustering_data = {
                "points": [{"x": float(X[i, 0]), "y": float(X[i, 1]), "cluster": int(self.cluster_labels[i])} 
                          for i in range(len(X))],
                "centroids": [{"x": float(center[0]), "y": float(center[1]), "cluster": i} 
                             for i, center in enumerate(self.kmeans_model.cluster_centers_)],
                "n_clusters": self.n_clusters
            }
            
            return clustering_data
            
        except Exception as e:
            logger.error(f"Error generating clustering data: {e}")
            return {"error": str(e)}

    def evaluate_model(self) -> Dict[str, Any]:
        """Valuta le performance del modello"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Prepara dati per valutazione
            df = self.prepare_data()
            
            if len(df) < 15:  # Abbassato da 20 a 15
                return {
                    "error": f"Insufficient data for evaluation (minimum 15 votes required, current: {len(df)})",
                    "current_votes": len(df),
                    "required_votes": 15,
                    "model_status": "trained_but_not_evaluable"
                }
            
            
            # Campiona utenti per valutazione più veloce
            sample_users = df['userId'].value_counts().head(100).index
            sample_df = df[df['userId'].isin(sample_users)].copy()
            
            # Re-encoding per il sample
            user_encoder_eval = LabelEncoder()
            movie_encoder_eval = LabelEncoder()
            sample_df['user_idx'] = user_encoder_eval.fit_transform(sample_df['userId'])
            sample_df['movie_idx'] = movie_encoder_eval.fit_transform(sample_df['title'])
            
            # Split train/test
            train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=42)
            
            # Training matrix
            train_matrix = csr_matrix(
                (train_df['rating'], (train_df['user_idx'], train_df['movie_idx'])),
                shape=(sample_df['user_idx'].nunique(), sample_df['movie_idx'].nunique())
            )
            
            # SVD per valutazione
            eval_components = min(30, min(train_matrix.shape) - 1)
            eval_components = max(1, eval_components)  # Assicura almeno 1 componente
            
            svd_eval = TruncatedSVD(n_components=eval_components, random_state=42)
            user_factors_eval = svd_eval.fit_transform(train_matrix)
            movie_factors_eval = svd_eval.components_.T
            
            # Predizioni
            predictions, actuals = [], []
            for _, row in test_df.iterrows():
                u, m = row['user_idx'], row['movie_idx']
                if u < user_factors_eval.shape[0] and m < movie_factors_eval.shape[0]:
                    pred = np.dot(user_factors_eval[u], movie_factors_eval[m])
                    predictions.append(pred)
                    actuals.append(row['rating'])
            
            if len(predictions) == 0:
                return {"error": "No valid predictions generated"}
            
            # Metriche
            rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
            mae = float(mean_absolute_error(actuals, predictions))
            
            evaluation = {
                "rmse": rmse,
                "mae": mae,
                "test_samples": len(predictions),
                "train_samples": len(train_df),
                "explained_variance": float(self.explained_variance),
                "model_status": "evaluated"
            }
            
            logger.info(f"Model evaluation completed: RMSE={rmse:.4f}, MAE={mae:.4f}")
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
        
        return {
            "is_trained": self.is_trained,
            "explained_variance": explained_var,
            "n_components": self.svd_model.n_components if self.svd_model else 0,
            "actual_k_used": self.actual_k_used,
            "requested_k": self.n_components,
            "k_efficiency": float(explained_var / self.actual_k_used) if self.actual_k_used > 0 else 0,
            "n_clusters": self.n_clusters,
            "has_clustering": self.kmeans_model is not None,
            "variance_per_component": self.variance_per_component if hasattr(self, 'variance_per_component') else [],
            "k_optimization_available": len(self.k_performance_log) > 0 if hasattr(self, 'k_performance_log') else False
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
                max_k = min(50, min(df['userId'].nunique(), df['movieId'].nunique()) - 1)
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

# Istanza globale del servizio
ml_service = MLRecommendationService()