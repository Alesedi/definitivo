from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from service.service_tmdb import tmdb_service
try:
    # service_csv may have been removed/archived; import if present
    from service.service_csv import csv_service  # type: ignore
except Exception:
    csv_service = None
from modelli_ODM.utente_odm import Utente
from modello.modello_dto.generi_scelti import GeneriSceltiDTO
from modello.enum_genere import EnumGenere
from service.service_auth import verifica_token

router = APIRouter(tags=["onboarding"])

@router.get("/genres")
async def get_available_genres() -> Dict[str, Any]:
    """Recupera tutti i generi disponibili da TMDB per la selezione iniziale"""
    try:
        # Mock temporaneo se TMDB non è configurato
        mock_genres = [
            {"id": 28, "name": "Azione"},
            {"id": 12, "name": "Avventura"},
            {"id": 16, "name": "Animazione"},
            {"id": 35, "name": "Commedia"},
            {"id": 80, "name": "Crime"},
            {"id": 18, "name": "Drammatico"},
            {"id": 14, "name": "Fantasy"},
            {"id": 27, "name": "Horror"},
            {"id": 10749, "name": "Romance"},
            {"id": 878, "name": "Fantascienza"},
            {"id": 53, "name": "Thriller"},
            {"id": 10752, "name": "Guerra"}
        ]
        
        try:
            tmdb_genres = await tmdb_service.get_genres_from_tmdb()
            return {
                "message": "Generi recuperati con successo",
                "genres": tmdb_genres
            }
        except:
            return {
                "message": "Generi mock caricati",
                "genres": mock_genres
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero generi: {str(e)}")

@router.post("/select-genres/{user_id}")
async def select_preferred_genres(user_id: str, genres_dto: GeneriSceltiDTO):
    """Permette all'utente di selezionare i suoi generi preferiti (minimo 3)"""
    print(f"DEBUG: Ricevuti generi: {genres_dto.generi}")
    
    # Trova l'utente
    user = Utente.objects(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utente non trovato")
    
    # Verifica che abbia selezionato almeno 3 generi
    if len(genres_dto.generi) < 3:
        raise HTTPException(status_code=400, detail="Seleziona almeno 3 generi")
    
    # Mappa ID generi TMDB e nomi italiani ai valori enum
    genre_mapping = {
        # ID TMDB
        28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
        80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
        14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
        9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
        10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
        # Nomi italiani
        "Azione": "Action", "Avventura": "Adventure", "Animazione": "Animation",
        "Commedia": "Comedy", "Crime": "Crime", "Drammatico": "Drama",
        "Fantasy": "Fantasy", "Horror": "Horror", "Romance": "Romance",
        "Fantascienza": "Science Fiction", "Thriller": "Thriller", "Guerra": "War"
    }
    
    # Converte gli ID/nomi nei valori enum
    mapped_genres = []
    for genre_item in genres_dto.generi:
        if genre_item in genre_mapping:
            mapped_genres.append(genre_mapping[genre_item])
        elif isinstance(genre_item, str) and genre_item in [g.value for g in EnumGenere]:
            mapped_genres.append(genre_item)
        else:
            print(f"DEBUG: Genere non mappato: {genre_item} (tipo: {type(genre_item)})")
    
    print(f"DEBUG: Generi mappati: {mapped_genres}")
    
    if len(mapped_genres) < 3:
        raise HTTPException(status_code=400, detail=f"Generi non validi selezionati. Ricevuti: {genres_dto.generi}, Mappati: {mapped_genres}")
    
    # Aggiorna i generi preferiti
    user.generi_preferiti = mapped_genres
    user.save()
    
    return {
        "message": "Generi preferiti salvati con successo",
        "preferred_genres": mapped_genres
    }

@router.get("/movies-for-rating/{user_id}")
async def get_movies_for_initial_rating(user_id: str) -> Dict[str, Any]:
    """Recupera 50 film da valutare basati sui generi preferiti dell'utente"""
    # Trova l'utente
    user = Utente.objects(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utente non trovato")
    
    # Verifica che abbia già selezionato i generi
    if not user.generi_preferiti:
        raise HTTPException(status_code=400, detail="L'utente deve prima selezionare i generi preferiti")
    
    try:
        # Usa i dati CSV reali se disponibili, altrimenti fallback a TMDB
        if csv_service is not None:
            try:
                movies = csv_service.get_random_movies(50)
            except Exception:
                movies = None
        else:
            movies = None

        # TMDB fallback (async)
        if not movies:
            try:
                movies = await tmdb_service.get_50_movies_for_rating(user.generi_preferiti)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Errore nel recupero film da TMDB: {e}")

        return {
            "message": "Film per valutazione recuperati con successo",
            "movies": movies,
            "user_preferred_genres": user.generi_preferiti
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero film: {str(e)}")