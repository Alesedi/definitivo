import asyncio
import threading
import pymongo
from modelli_ODM.voto_odm import Votazione
from database.connessione import connetti_mongodb
import logging
import os

logger = logging.getLogger(__name__)

class MongoChangeStreamMonitor:
    def __init__(self):
        self.client = None
        self.db = None
        self.change_stream = None
        self.monitor_thread = None
        self.running = False

    def start_monitoring(self):
        """Avvia il monitoraggio dei change streams"""
        if self.running:
            return
            
        try:
            # Connetti a MongoDB
            mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://...")
            self.client = pymongo.MongoClient(mongo_uri)
            self.db = self.client[os.getenv("MONGO_DB_NAME", "movie_rec")]
            
            # Configura change stream per la collection utenti
            pipeline = [
                {
                    "$match": {
                        "operationType": "delete",
                        "ns.coll": "utenti"
                    }
                }
            ]
            
            self.change_stream = self.db.utenti.watch(pipeline, full_document="updateLookup")
            self.running = True
            
            # Avvia il thread di monitoraggio
            self.monitor_thread = threading.Thread(target=self._monitor_changes, daemon=True)
            self.monitor_thread.start()
            
            logger.info("🔍 Change Stream Monitor avviato per auto-delete votazioni")
            
        except Exception as e:
            logger.error(f"❌ Errore avvio Change Stream Monitor: {e}")

    def _monitor_changes(self):
        """Monitora i cambiamenti e cancella le votazioni automaticamente"""
        try:
            for change in self.change_stream:
                if change["operationType"] == "delete":
                    user_id = change["documentKey"]["_id"]
                    self._auto_delete_votes(user_id)
                    
        except Exception as e:
            logger.error(f"❌ Errore nel monitoring Change Stream: {e}")
        finally:
            self.running = False

    def _auto_delete_votes(self, user_id):
        """Cancella automaticamente le votazioni dell'utente"""
        try:
            # Usa MongoEngine per cancellare le votazioni
            from bson import ObjectId
            
            # Conta le votazioni prima della cancellazione
            votes_count = self.db.votazioni.count_documents({"utente": user_id})
            
            # Cancella le votazioni
            result = self.db.votazioni.delete_many({"utente": user_id})
            
            logger.info(f"🗑️ AUTO-DELETE: Cancellate {result.deleted_count} votazioni per utente {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Errore auto-delete votazioni per utente {user_id}: {e}")

    def stop_monitoring(self):
        """Ferma il monitoraggio"""
        self.running = False
        if self.change_stream:
            self.change_stream.close()
        if self.client:
            self.client.close()
        logger.info("🛑 Change Stream Monitor fermato")

# Istanza globale del monitor
change_monitor = MongoChangeStreamMonitor()