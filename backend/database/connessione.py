import os
from mongoengine import connect
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Costanti per il database e TMDB
MONGODB_URI = os.getenv("MONGODB_URI")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")

# Verifica le variabili essenziali
if not MONGODB_URI:
    raise ValueError("Devi impostare la variabile d'ambiente MONGODB_URI con la stringa di connessione a MongoDB Atlas")

if not TMDB_API_KEY:
    raise ValueError("Devi impostare la variabile d'ambiente TMDB_API_KEY con la tua chiave TMDB")

def connetti_mongodb():
    """Connette a MongoDB Atlas con alias 'default'."""
    connect(host=MONGODB_URI, alias="default")
    print("✅ Connessione a MongoDB Atlas riuscita!")

# Connessione al DB all'import
connetti_mongodb()

print("✅ TMDB API configurata correttamente")
