import httpx
import asyncio
from typing import List, Dict, Any
from database.connessione import TMDB_API_KEY, TMDB_BASE_URL
from modello.enum_genere import EnumGenere


class ServiceTMDB:
    """Service per interagire con TMDB API"""
    
    def __init__(self):
        self.api_key = TMDB_API_KEY
        self.base_url = TMDB_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json"
        }
    
    async def get_genres_from_tmdb(self) -> List[Dict[str, Any]]:
        """Recupera tutti i generi da TMDB API"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/genre/movie/list",
                headers=self.headers,
                params={"language": "it-IT"}
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("genres", [])
            else:
                raise Exception(f"Errore TMDB API: {response.status_code}")
    
    async def get_movies_by_genres(self, genre_ids: List[int], page: int = 1) -> List[Dict[str, Any]]:
        """Recupera film basati sui generi selezionati"""
        genre_string = ",".join(map(str, genre_ids))
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/discover/movie",
                headers=self.headers,
                params={
                    "with_genres": genre_string,
                    "language": "it-IT",
                    "sort_by": "popularity.desc",
                    "page": page,
                    "vote_count.gte": 100  # Solo film con almeno 100 voti
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                raise Exception(f"Errore TMDB API: {response.status_code}")
    
    async def get_50_movies_for_rating(self, user_genre_preferences: List[str]) -> List[Dict[str, Any]]:
        """Recupera 50 film popolari basati sui generi preferiti dell'utente"""
        # Mappa i generi enum ai ID TMDB (questi ID andrebbero recuperati dinamicamente)
        genre_mapping = {
            "Action": 28,
            "Adventure": 12,
            "Animation": 16,
            "Comedy": 35,
            "Crime": 80,
            "Documentary": 99,
            "Drama": 18,
            "Family": 10751,
            "Fantasy": 14,
            "History": 36,
            "Horror": 27,
            "Music": 10402,
            "Mystery": 9648,
            "Romance": 10749,
            "Science Fiction": 878,
            "TV Movie": 10770,
            "Thriller": 53,
            "War": 10752,
            "Western": 37
        }
        
        # Converti generi preferiti in ID TMDB
        genre_ids = [genre_mapping.get(genre) for genre in user_genre_preferences if genre_mapping.get(genre)]
        
        all_movies = []
        page = 1
        
        # Recupera film da più pagine fino ad avere almeno 50 film
        while len(all_movies) < 50 and page <= 3:
            movies = await self.get_movies_by_genres(genre_ids, page)
            all_movies.extend(movies)
            page += 1
        
        # Restituisci i primi 50 film
        return all_movies[:50]


# Istanza globale del servizio
tmdb_service = ServiceTMDB()