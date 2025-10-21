from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route import auth, generi, onboarding, ratings, recommendations, admin
from database.connessione import connetti_mongodb
from database.change_stream_monitor import change_monitor
import uvicorn
import atexit

app = FastAPI(title="AFlix Movie Recommendation App", debug=True)

# Configurazione CORS per permettere chiamate dal frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route di test
@app.get("/")
async def root():
    return {"message": "AFlix API - Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

# Connessione a MongoDB
connetti_mongodb()

# Avvia il Change Stream Monitor per auto-delete votazioni
@app.on_event("startup")
async def startup_event():
    """Avvia il monitoraggio dei change streams all'avvio del server"""
    try:
        change_monitor.start_monitoring()
    except Exception as e:
        print(f"⚠️ Impossibile avviare Change Stream Monitor: {e}")
        print("💡 Se usi MongoDB Atlas, usa i Triggers invece dei Change Streams")

# Ferma il monitor alla chiusura
@app.on_event("shutdown")
async def shutdown_event():
    """Ferma il monitoraggio alla chiusura del server"""
    change_monitor.stop_monitoring()

# Anche per chiusura improvvisa
atexit.register(change_monitor.stop_monitoring)

# Includi tutti i router con prefisso /api
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(generi.router, prefix="/api/generi", tags=["generi"])
app.include_router(onboarding.router, prefix="/api/onboarding", tags=["onboarding"])
app.include_router(ratings.router, prefix="/api/ratings", tags=["ratings"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=True)
