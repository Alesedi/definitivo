from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from modelli_ODM.utente_odm import Utente
from modelli_ODM.film_odm import Film
from modelli_ODM.voto_odm import Votazione

router = APIRouter(tags=["ratings"])

# Mapping per convertire ID generi numerici in nomi
GENRE_ID_TO_NAME = {
    12: "Adventure", 16: "Animation", 18: "Drama", 27: "Horror", 28: "Action",
    35: "Comedy", 36: "History", 37: "Western", 53: "Thriller", 80: "Crime",
    99: "Documentary", 878: "Science Fiction", 9648: "Mystery", 10402: "Music",
    10749: "Romance", 10751: "Family", 10752: "War", 10770: "TV Movie", 14: "Fantasy"
}

def convert_genres_to_names(genres_list: List[str]) -> List[str]:
    """Converte una lista di generi (che possono essere ID o nomi) in nomi completi"""
    converted_genres = []
    
    print(f"DEBUG: convert_genres_to_names ricevuto: {genres_list}")
    
    for genre in genres_list:
        print(f"DEBUG: Processando genere: {genre}")
        # Se il genere è nel formato "GenreXX", estrai l'ID
        if genre.startswith("Genre"):
            try:
                genre_id = int(genre.replace("Genre", ""))
                print(f"DEBUG: ID estratto: {genre_id}")
                genre_name = GENRE_ID_TO_NAME.get(genre_id, "Drama")  # Default fallback
                print(f"DEBUG: Nome genere: {genre_name}")
                converted_genres.append(genre_name)
            except ValueError as e:
                print(f"DEBUG: Errore conversione {genre}: {e}")
                converted_genres.append("Drama")  # Fallback
        # Se è già un nome, mantienilo
        elif genre in ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
                      "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", 
                      "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"]:
            print(f"DEBUG: Genere già nome valido: {genre}")
            converted_genres.append(genre)
        else:
            print(f"DEBUG: Genere sconosciuto {genre}, usando Drama")
            converted_genres.append("Drama")  # Fallback per generi sconosciuti
    
    print(f"DEBUG: Generi convertiti finali: {converted_genres}")
    return converted_genres

class VoteRequest(BaseModel):
    tmdb_id: int
    rating: int  # da 1 a 5
    title: str
    genres: List[str] = []
    poster_path: Optional[str] = None
    release_date: Optional[str] = None
    tmdb_rating: Optional[float] = 0.0

class MultipleVotesRequest(BaseModel):
    votes: List[VoteRequest]

@router.post("/vote/{user_id}")
async def vote_movie(user_id: str, vote_request: VoteRequest):
    """Permette all'utente di votare un singolo film"""
    try:
        print(f"DEBUG: Ricevuto voto da utente {user_id}")
        print(f"DEBUG: Dati voto: {vote_request}")
        print(f"DEBUG: Film: {vote_request.title} - {vote_request.rating} stelle")
        print(f"DEBUG: Generi: {vote_request.genres}")
        print(f"DEBUG: Poster path: {vote_request.poster_path}")
        
        # Verifica che l'utente esista
        user = Utente.objects(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utente non trovato")
        
        # Verifica che il voto sia valido
        if vote_request.rating < 1 or vote_request.rating > 5:
            raise HTTPException(status_code=400, detail="Il voto deve essere tra 1 e 5")
        
        # Cerca o crea il film nel database
        film = Film.objects(tmdb_id=vote_request.tmdb_id).first()
        if not film:
            print(f"DEBUG: Creando nuovo film nel database...")
            print(f"DEBUG: Generi ricevuti prima conversione: {vote_request.genres}")
            # Converti i generi dal formato GenreXX ai nomi completi
            converted_genres = convert_genres_to_names(vote_request.genres)
            print(f"DEBUG: Generi convertiti: {converted_genres}")
            
            # Crea nuovo film con dati da TMDB
            film = Film(
                tmdb_id=vote_request.tmdb_id,
                titolo=vote_request.title,
                genere=converted_genres,
                poster_path=vote_request.poster_path,
                release_date=vote_request.release_date,
                tmdb_rating=vote_request.tmdb_rating
            )
            print(f"DEBUG: Film creato con generi: {film.genere}")
            print(f"DEBUG: Film creato, salvando...")
            try:
                film.save()
                print(f"DEBUG: Film salvato con successo!")
            except Exception as save_error:
                print(f"ERROR: Errore salvando film: {save_error}")
                print(f"ERROR: Generi che hanno causato errore: {film.genere}")
                raise
        else:
            print(f"DEBUG: Film già esistente nel database")
        
        # Verifica se l'utente ha già votato questo film
        existing_vote = Votazione.objects(utente=user, film=film).first()
        if existing_vote:
            # Aggiorna il voto esistente
            old_rating = existing_vote.valutazione
            existing_vote.valutazione = vote_request.rating
            existing_vote.save()
            
            # Aggiorna la media del film
            _update_film_rating(film, old_rating, vote_request.rating, is_update=True)
        else:
            # Crea nuovo voto
            new_vote = Votazione(
                utente=user,
                film=film,
                valutazione=vote_request.rating
            )
            new_vote.save()
            
            # Aggiorna la lista votazioni dell'utente
            user.votazioni.append(new_vote)
            user.save()
            print(f"DEBUG: Aggiunta votazione alla lista utente. Totale voti utente: {len(user.votazioni)}")
            
            # Aggiorna la media del film
            _update_film_rating(film, 0, vote_request.rating, is_update=False)
        
        return {
            "message": "Voto salvato con successo",
            "movie": vote_request.title,
            "rating": vote_request.rating
        }
        
    except Exception as e:
        print(f"ERROR: Errore durante il salvataggio del voto: {e}")
        print(f"ERROR: Tipo errore: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Errore nella validazione dei dati: {str(e)}")

@router.post("/vote-multiple/{user_id}")
async def vote_multiple_movies(user_id: str, votes_request: MultipleVotesRequest):
    """Permette all'utente di votare più film contemporaneamente"""
    # Verifica che l'utente esista
    user = Utente.objects(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utente non trovato")
    
    successful_votes = []
    failed_votes = []
    
    for vote in votes_request.votes:
        try:
            # Verifica che il voto sia valido
            if vote.rating < 1 or vote.rating > 5:
                failed_votes.append({"movie": vote.title, "error": "Voto deve essere tra 1 e 5"})
                continue
            
            # Cerca o crea il film
            film = Film.objects(tmdb_id=vote.tmdb_id).first()
            if not film:
                # Converti i generi dal formato GenreXX ai nomi completi
                converted_genres = convert_genres_to_names(vote.genres)
                
                film = Film(
                    tmdb_id=vote.tmdb_id,
                    titolo=vote.title,
                    genere=converted_genres,
                    poster_path=vote.poster_path,
                    release_date=vote.release_date,
                    tmdb_rating=vote.tmdb_rating
                )
                film.save()
            
            # Verifica se esiste già un voto
            existing_vote = Votazione.objects(utente=user, film=film).first()
            if existing_vote:
                old_rating = existing_vote.valutazione
                existing_vote.valutazione = vote.rating
                existing_vote.save()
                _update_film_rating(film, old_rating, vote.rating, is_update=True)
            else:
                new_vote = Votazione(
                    utente=user,
                    film=film,
                    valutazione=vote.rating
                )
                new_vote.save()
                
                # Aggiorna la lista votazioni dell'utente
                user.votazioni.append(new_vote)
                
                _update_film_rating(film, 0, vote.rating, is_update=False)
            
            successful_votes.append({"movie": vote.title, "rating": vote.rating})
            
        except Exception as e:
            failed_votes.append({"movie": vote.title, "error": str(e)})
    
    # Salva l'utente con tutte le nuove votazioni
    user.save()
    
    return {
        "message": f"Elaborati {len(votes_request.votes)} voti",
        "successful_votes": successful_votes,
        "failed_votes": failed_votes,
        "success_count": len(successful_votes),
        "failed_count": len(failed_votes)
    }

def _update_film_rating(film: Film, old_rating: int, new_rating: int, is_update: bool):
    """Funzione helper per aggiornare la media dei voti di un film"""
    if is_update:
        # Aggiornamento: rimuovi il vecchio voto e aggiungi il nuovo
        if film.numero_voti > 0:
            total = film.media_voti * film.numero_voti
            total = total - old_rating + new_rating
            film.media_voti = total / film.numero_voti
    else:
        # Nuovo voto: aggiungi al totale
        total = film.media_voti * film.numero_voti + new_rating
        film.numero_voti += 1
        film.media_voti = total / film.numero_voti
    
    film.save()

@router.get("/user-ratings/{user_id}")
async def get_user_ratings(user_id: str):
    """Recupera tutti i voti di un utente"""
    user = Utente.objects(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utente non trovato")
    
    votes = Votazione.objects(utente=user).select_related()
    
    user_ratings = []
    for vote in votes:
        user_ratings.append({
            "tmdb_id": vote.film.tmdb_id,
            "title": vote.film.titolo,
            "rating": vote.valutazione,
            "genres": vote.film.genere,
            "poster_path": vote.film.poster_path
        })
    
    return {
        "user_id": user_id,
        "total_ratings": len(user_ratings),
        "ratings": user_ratings
    }