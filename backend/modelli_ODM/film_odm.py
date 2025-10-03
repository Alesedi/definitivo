from mongoengine import Document, StringField, FloatField, ListField, IntField
from modello.enum_genere import EnumGenere

class Film(Document):
    # Dati da TMDB
    tmdb_id = IntField(required=True, unique=True)  # ID univoco da TMDB
    titolo = StringField(required=True, max_length=200)
    descrizione = StringField()
    poster_path = StringField()  # Path del poster da TMDB
    release_date = StringField()  # Data di rilascio
    
    # Lista di generi del film, corrispondenti ai valori di GenereEnum
    genere = ListField(StringField(choices=[g.value for g in EnumGenere]), default=[])

    # Rating da TMDB
    tmdb_rating = FloatField(min_value=0.0, max_value=10.0, default=0.0)  # rating da TMDB (1-10)
    tmdb_vote_count = IntField(default=0)  # numero di voti su TMDB
    
    # Rating interno della nostra app
    media_voti = FloatField(min_value=0.0, max_value=5.0, default=0.0)  # media voti dei nostri utenti (1-5)
    numero_voti = IntField(default=0)  # numero di voti dei nostri utenti

    meta = {
        "collection": "film",
        "indexes": ["tmdb_id"]
    }
