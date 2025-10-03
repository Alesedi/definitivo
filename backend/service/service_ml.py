import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from typing import List, Dict, Any, Optional
import os
import requests
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
        
        # Content-based fallback
        self.user_profile = None
        self.genre_preferences = None
        
        # Configurazionipy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from typing import List, Dict, Any, Optional
import os
import requests
from modelli_ODM.utente_odm import Utente
from modelli_ODM.film_odm import Film
from modelli_ODM.voto_odm import Votazione
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
        """Addestra il modello SVD e clustering"""
        try:
            # Prepara i dati
            df = self.prepare_data()
            
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
            self.svd_model = TruncatedSVD(n_components=safe_components, random_state=42)
            self.user_factors = self.svd_model.fit_transform(ratings_sparse)
            self.movie_factors = self.svd_model.components_.T
            self.explained_variance = self.svd_model.explained_variance_ratio_.sum()
            
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
                "training_status": "success"
            }
            
            logger.info(f"Model trained successfully. Stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            self.is_trained = False
            raise

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
        """Restituisce lo stato del modello"""
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
            "n_clusters": self.n_clusters,
            "has_clustering": self.kmeans_model is not None
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

# Istanza globale del servizio
ml_service = MLRecommendationService()