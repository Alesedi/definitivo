import pandas as pd
import os
import requests
from typing import List, Dict, Any
from pathlib import Path
import time

class CSVDataService:
    """Servizio per leggere i dati dai file CSV di TMDB"""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.movies_df = None
        self.ratings_df = None
        self.links_df = None
        self.tags_df = None
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")
        self.tmdb_base_url = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
        self.poster_cache = {
            # Film popolari con poster garantiti
            1: "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",     # Toy Story (1995)
            2: "/vzmL6fP7aPKNKPRTFnZmiUfciyV.jpg",     # Jumanji (1995)
            3: "/1tcFSoKWyeUmJrrpvMsHwezEdi8.jpg",     # Grumpier Old Men (1995)
            260: "/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",   # Star Wars (1977)
            296: "/sM33SANp9z6rXW8Itn7NnG1GOEs.jpg",   # Pulp Fiction (1994)
            318: "/iVZ3JAcAjmguGPnRNfWFOtLHOuY.jpg",   # Shawshank Redemption (1994)
            356: "/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg",   # Forrest Gump (1994)
            480: "/kqjL17yufvn9OVLyXYpvtyrFfak.jpg",   # Jurassic Park (1993)
            593: "/3AQdBSwdLlbAEL5odJRnBfWdOLJ.jpg",   # Silence of the Lambs (1991)
            858: "/ow3wq89wM8qd5X7hWKxiRfsFf9C.jpg",   # Godfather (1972)
            # Aggiungi altri film per test
            4: "/bvCkDhff6FOmBFE2bHQQhO9OG3h.jpg",     # Waiting to Exhale (1995)
            5: "/bVtqOq6bH3E3c9nInFGdIJMOdcb.jpg",     # Father of the Bride Part II (1995)
            6: "/3L0S0wJ6gvD1JCGJfYOJ1iqKZIM.jpg",     # Heat (1995)
            7: "/wWr1aq7Hzv6Hx7v3gV7WyLnFt1U.jpg",     # Sabrina (1995)
            8: "/4zqjWEdhEGhWRtEjHv3R1F4Nw2V.jpg",     # Tom and Huck (1995)
        }  # Cache dinamica per i poster
        self._load_data()
    
    def _load_data(self):
        """Carica tutti i CSV in memoria"""
        try:
            self.movies_df = pd.read_csv(self.data_dir / "movies.csv")
            self.ratings_df = pd.read_csv(self.data_dir / "ratings.csv")
            try:
                self.links_df = pd.read_csv(self.data_dir / "links.csv")
            except:
                self.links_df = None  # Funziona senza links
            try:
                self.tags_df = pd.read_csv(self.data_dir / "tags.csv")
            except:
                self.tags_df = None
            print("✅ Dati CSV caricati con successo")
        except Exception as e:
            print(f"❌ Errore nel caricamento CSV: {e}")
    
    def _get_poster_from_tmdb(self, tmdb_id: int) -> str:
        """Recupera il poster da TMDB API con cache"""
        if not self.tmdb_api_key:
            return None
            
        # Controlla cache
        if tmdb_id in self.poster_cache:
            return self.poster_cache[tmdb_id]
        
        try:
            url = f"{self.tmdb_base_url}/movie/{tmdb_id}"
            response = requests.get(url, params={"api_key": self.tmdb_api_key}, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get('poster_path')
                # Salva in cache
                self.poster_cache[tmdb_id] = poster_path
                return poster_path
            else:
                # Cache anche i fallimenti per evitare ripetute chiamate
                self.poster_cache[tmdb_id] = None
                return None
                
        except Exception as e:
            print(f"Errore TMDB per film {tmdb_id}: {e}")
            self.poster_cache[tmdb_id] = None
            return None
    
    def get_movies_by_genre(self, genres: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """Recupera film filtrati per genere (versione sincrona per ora)"""
        if self.movies_df is None:
            return []
        
        # Converti generi per matching
        genre_mapping = {
            "Action": "Action", "Adventure": "Adventure", "Animation": "Animation",
            "Comedy": "Comedy", "Crime": "Crime", "Drama": "Drama",
            "Fantasy": "Fantasy", "Horror": "Horror", "Romance": "Romance",
            "Science Fiction": "Sci-Fi", "Thriller": "Thriller", "War": "War"
        }
        
        # Filtra per generi
        filtered_movies = self.movies_df.copy()
        for genre in genres:
            mapped_genre = genre_mapping.get(genre, genre)
            filtered_movies = filtered_movies[
                filtered_movies['genres'].str.contains(mapped_genre, na=False, case=False)
            ]
        
        # Prendi sample casuale e limita
        if len(filtered_movies) > limit:
            filtered_movies = filtered_movies.sample(n=limit)
        
        # Converti in formato API (senza poster per ora)
        movies = []
        for _, row in filtered_movies.iterrows():
            tmdb_id = row['movieId']
            if self.links_df is not None:
                tmdb_row = self.links_df[self.links_df['movieId'] == row['movieId']]
                if len(tmdb_row) > 0:
                    tmdb_id = tmdb_row['tmdbId'].iloc[0]
            
            # Ottieni poster da TMDB API reale
            poster_path = self._get_poster_from_tmdb(int(tmdb_id))
            
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == row['movieId']]
            avg_rating = movie_ratings['rating'].mean() if len(movie_ratings) > 0 else 3.5

            movies.append({
                "id": int(tmdb_id),
                "title": row['title'],
                "overview": f"Un film del genere {row['genres'].replace('|', ', ')}. " +
                           "Scopri questa avventura cinematografica selezionata per te!",
                "poster_path": poster_path,  # Poster reale da TMDB API
                "release_date": "2023-01-01",
                "vote_average": round(float(avg_rating * 2), 1),
                "genre_ids": self._parse_genre_ids(row['genres']),
                "genres": row['genres']
            })
        
        return movies
    
    def _parse_genre_ids(self, genres_str: str) -> List[int]:
        """Converte stringa generi in ID numerici"""
        genre_id_map = {
            "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
            "Crime": 80, "Drama": 18, "Fantasy": 14, "Horror": 27,
            "Romance": 10749, "Sci-Fi": 878, "Thriller": 53, "War": 10752
        }
        
        if pd.isna(genres_str):
            return []
        
        genre_names = genres_str.split('|')
        return [genre_id_map.get(g, 99) for g in genre_names if g in genre_id_map]
    
    def get_random_movies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Recupera film casuali per il rating iniziale"""
        if self.movies_df is None:
            return []
        
        # Prendi film casuali
        random_movies = self.movies_df.sample(n=min(limit, len(self.movies_df)))
        
        movies = []
        for _, row in random_movies.iterrows():
            tmdb_id = row['movieId']
            if self.links_df is not None:
                tmdb_row = self.links_df[self.links_df['movieId'] == row['movieId']]
                if len(tmdb_row) > 0:
                    tmdb_id = tmdb_row['tmdbId'].iloc[0]
            
            # Ottieni poster da TMDB API reale  
            poster_path = self._get_poster_from_tmdb(int(tmdb_id))
            
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == row['movieId']]
            avg_rating = movie_ratings['rating'].mean() if len(movie_ratings) > 0 else 3.5
            
            movies.append({
                "id": int(tmdb_id),
                "title": row['title'],
                "overview": f"Un {row['genres'].replace('|', '/')} che potrebbe piacerti. " +
                           "Votalo per aiutarci a capire i tuoi gusti!",
                "poster_path": poster_path,  # Poster reale da TMDB API
                "release_date": "2023-01-01",
                "vote_average": round(float(avg_rating * 2), 1),
                "genre_ids": self._parse_genre_ids(row['genres']),
                "genres": row['genres']
            })
        
        return movies

# Instanza globale
csv_service = CSVDataService()