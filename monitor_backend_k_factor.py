"""
MONITOR VISUALE IN TEMPO REALE - PROCESSO SCELTA FATTORE K
=========================================================

Questo script mostra visualmente cosa succede lato backend durante:
1. Training del modello con selezione automatica del fattore k
2. Ottimizzazione del fattore k
3. Aggiornamento del modello con nuovo k

Utilizza polling dell'endpoint live-monitor per visualizzare in tempo reale
lo stato del backend durante i processi di machine learning.
"""

import requests
import time
import os
import json
from datetime import datetime
from typing import Dict, Any
import threading
import queue

class BackendMonitor:
    """Monitor visuale del backend per tracciamento fattore k"""
    
    def __init__(self, base_url: str = "http://localhost:8005"):
        self.base_url = base_url
        self.monitoring = False
        self.log_queue = queue.Queue()
        
    def clear_screen(self):
        """Pulisce lo schermo"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Stampa intestazione del monitor"""
        print("=" * 80)
        print("🔍 MONITOR BACKEND AFLIX - TRACCIAMENTO FATTORE K")
        print("=" * 80)
        print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Backend URL: {self.base_url}")
        print("=" * 80)
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Recupera status live dal backend"""
        try:
            response = requests.get(f"{self.base_url}/admin/ml/live-monitor", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def display_model_status(self, data: Dict[str, Any]):
        """Visualizza status del modello"""
        if "error" in data:
            print(f"❌ Errore connessione: {data['error']}")
            return
        
        status = data.get("model_status", {})
        performance = data.get("current_performance", {})
        
        print("\n📊 STATUS MODELLO ML")
        print("-" * 50)
        print(f"🤖 Addestrato: {'✅ SÌ' if status.get('is_trained') else '❌ NO'}")
        print(f"🎯 K Utilizzato: {performance.get('k_used', 'N/A')}")
        print(f"📋 K Richiesto: {status.get('requested_k', 'N/A')}")
        print(f"📈 Varianza Spiegata: {performance.get('explained_variance', 0):.1%}")
        print(f"⚡ Efficienza K: {performance.get('k_efficiency', 0):.4f}")
        print(f"🔧 Clustering: {'✅' if status.get('has_clustering') else '❌'}")
        
        # Mostra breakdown componenti se disponibile
        components = performance.get('components_breakdown', [])
        if components:
            print(f"\n📊 TOP 5 COMPONENTI SVD:")
            print("-" * 30)
            for i, var in enumerate(components[:5], 1):
                print(f"  Comp {i}: {var:.4f} ({var*100:.2f}%)")
    
    def display_recommendations(self, data: Dict[str, Any]):
        """Visualizza raccomandazioni"""
        recommendations = data.get("recommendations", [])
        if recommendations:
            print(f"\n💡 RACCOMANDAZIONI:")
            print("-" * 30)
            for rec in recommendations:
                level = rec['level'].upper()
                icon = "⚠️" if level == "WARNING" else "🔧" if level == "OPTIMIZATION" else "ℹ️"
                print(f"  {icon} [{level}] {rec['message']}")
    
    def display_optimization_history(self, data: Dict[str, Any]):
        """Visualizza storico ottimizzazioni"""
        history = data.get("optimization_history", {})
        if history:
            results = history.get("optimization_results", {})
            if results:
                print(f"\n📈 ULTIMA OTTIMIZZAZIONE:")
                print("-" * 30)
                print(f"  🏆 K Ottimale: {results.get('best_k', 'N/A')}")
                print(f"  🔄 Miglioramento: {'SÌ' if results.get('improvement') else 'NO'}")
                timestamp = history.get("timestamp", "")
                if timestamp:
                    print(f"  🕐 Quando: {timestamp[:19].replace('T', ' ')}")
    
    def monitor_live(self, interval: int = 2):
        """Monitoraggio live con refresh automatico"""
        print("🚀 Avvio monitoraggio live...")
        print("Premi Ctrl+C per fermare\n")
        
        self.monitoring = True
        
        try:
            while self.monitoring:
                self.clear_screen()
                self.print_header()
                
                # Recupera dati
                data = self.get_backend_status()
                
                # Visualizza informazioni
                self.display_model_status(data)
                self.display_recommendations(data)
                self.display_optimization_history(data)
                
                print(f"\n🔄 Prossimo aggiornamento in {interval} secondi...")
                print("💡 Premi Ctrl+C per fermare il monitoraggio")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoraggio fermato dall'utente")
            self.monitoring = False
    
    def trigger_training_and_monitor(self):
        """Triggera training e monitora il processo"""
        print("🔄 Triggering training del modello...")
        
        try:
            # Avvia training sincrono per vedere i risultati immediati
            response = requests.get(f"{self.base_url}/recommendations/train-sync", timeout=30)
            
            if response.status_code == 200:
                print("✅ Training completato!")
                result = response.json()
                print(f"📊 K utilizzato: {result.get('n_components', 'N/A')}")
                print(f"📈 Varianza: {result.get('explained_variance', 0):.1%}")
            else:
                print(f"❌ Errore training: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    def trigger_optimization_and_monitor(self, k_range=None):
        """Triggera ottimizzazione k e monitora"""
        print("🔧 Triggering ottimizzazione fattore k...")
        
        if k_range is None:
            k_range = [5, 10, 15, 20, 25, 30, 35, 40]
        
        print(f"🎯 Testing range: {k_range}")
        
        try:
            payload = {"k_range": k_range}
            response = requests.post(f"{self.base_url}/admin/ml/optimize-k-factor", 
                                   json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Ottimizzazione completata!")
                print(f"🏆 K ottimale: {result.get('best_k', 'N/A')}")
                print(f"🔄 Miglioramento: {'SÌ' if result.get('improvement') else 'NO'}")
                
                # Mostra top 3 risultati
                top_results = result.get('all_results', [])[:3]
                if top_results:
                    print("\n🏅 TOP 3 RISULTATI:")
                    for i, res in enumerate(top_results, 1):
                        print(f"  {i}. K={res['k']}: RMSE={res['rmse']:.4f}, Var={res['explained_variance']:.1%}")
            else:
                print(f"❌ Errore ottimizzazione: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    def trigger_k_update_and_monitor(self, new_k: int):
        """Triggera aggiornamento k e monitora"""
        print(f"🔧 Aggiornamento a K={new_k}...")
        
        try:
            response = requests.post(f"{self.base_url}/admin/ml/update-k-components/{new_k}", 
                                   timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Aggiornamento completato!")
                print(f"📊 Vecchio K: {result.get('old_k', 'N/A')}")
                print(f"📊 Nuovo K: {result.get('new_k', 'N/A')}")
                print(f"📈 Miglioramento: {'SÌ' if result.get('improvement') else 'NO'}")
            else:
                print(f"❌ Errore aggiornamento: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Errore: {e}")

def show_process_visualization():
    """Mostra visualizzazione del processo"""
    print("=" * 80)
    print("🎬 VISUALIZZAZIONE PROCESSO SCELTA FATTORE K")
    print("=" * 80)
    print("""
📊 FASE 1: ANALISI DATI
├── Caricamento ratings dal database
├── Costruzione matrice utenti×film (sparsa)
├── Calcolo dimensioni: min(n_users, n_movies)
└── Determinazione K massimo possibile

🎯 FASE 2: SELEZIONE K INIZIALE
├── K richiesto: {configurato nel modello}
├── K massimo: min(dim_matrice) - 1
├── K sicuro: min(K_richiesto, K_massimo)
└── ⚠️  Warning se K ridotto per limitazioni

🔄 FASE 3: DECOMPOSIZIONE SVD
├── Esecuzione TruncatedSVD(n_components=K)
├── Calcolo fattori latenti utenti
├── Calcolo fattori latenti film
└── 📈 Calcolo varianza spiegata per componente

📊 FASE 4: ANALISI COMPONENTI
├── Varianza individuale per ogni componente
├── Varianza cumulativa
├── 📍 Identificazione Elbow Point
└── 💡 Suggerimenti per ottimizzazione

🔧 FASE 5: OTTIMIZZAZIONE (opzionale)
├── Test di diversi valori K
├── Train/Test split per ogni K
├── Calcolo RMSE, MAE, Varianza
├── Score combinato per ranking
└── 🏆 Identificazione K ottimale

✅ FASE 6: FINALIZZAZIONE
├── Clustering K-means nello spazio latente
├── Salvataggio configurazione ottimale
├── Log delle performance
└── 📋 Report finale
    """)
    print("=" * 80)

def interactive_menu():
    """Menu interattivo per il monitoring"""
    monitor = BackendMonitor()
    
    while True:
        print("\n" + "=" * 60)
        print("🎛️  MENU MONITOR BACKEND - FATTORE K")
        print("=" * 60)
        print("1. 📊 Status attuale modello")
        print("2. 🔍 Monitoraggio live (auto-refresh)")
        print("3. 🔄 Triggera training + monitor")
        print("4. 🔧 Triggera ottimizzazione K + monitor")
        print("5. ⚙️  Aggiorna K specifico + monitor")
        print("6. 🎬 Visualizza processo completo")
        print("7. 📋 Test connessione backend")
        print("0. 🚪 Esci")
        print("=" * 60)
        
        choice = input("Scegli opzione (0-7): ").strip()
        
        if choice == "0":
            print("👋 Arrivederci!")
            break
        elif choice == "1":
            print("\n📊 RECUPERO STATUS...")
            data = monitor.get_backend_status()
            monitor.display_model_status(data)
            monitor.display_recommendations(data)
        elif choice == "2":
            interval = input("Intervallo refresh (secondi, default 2): ").strip()
            try:
                interval = int(interval) if interval else 2
            except ValueError:
                interval = 2
            monitor.monitor_live(interval)
        elif choice == "3":
            monitor.trigger_training_and_monitor()
        elif choice == "4":
            k_input = input("Range K (es: 5,10,15) o ENTER per default: ").strip()
            k_range = None
            if k_input:
                try:
                    k_range = [int(x.strip()) for x in k_input.split(',')]
                except ValueError:
                    print("❌ Formato non valido, uso default")
            monitor.trigger_optimization_and_monitor(k_range)
        elif choice == "5":
            try:
                new_k = int(input("Nuovo valore K: "))
                monitor.trigger_k_update_and_monitor(new_k)
            except ValueError:
                print("❌ Valore K non valido")
        elif choice == "6":
            show_process_visualization()
        elif choice == "7":
            print("\n🔌 TEST CONNESSIONE...")
            data = monitor.get_backend_status()
            if "error" in data:
                print(f"❌ Connessione fallita: {data['error']}")
            else:
                print("✅ Connessione OK!")
                print(f"📊 Modello addestrato: {'SÌ' if data.get('model_status', {}).get('is_trained') else 'NO'}")
        else:
            print("❌ Opzione non valida")
        
        if choice not in ["0", "2"]:  # Non pausare per exit o live monitor
            input("\nPremi ENTER per continuare...")

if __name__ == "__main__":
    print("🎬 MONITOR VISUALE BACKEND AFLIX")
    print("=" * 60)
    print("Questo tool mostra in tempo reale cosa succede nel backend")
    print("durante i processi di selezione e ottimizzazione del fattore K.")
    print("=" * 60)
    
    # Verifica connessione iniziale
    monitor = BackendMonitor()
    data = monitor.get_backend_status()
    
    if "error" in data:
        print(f"⚠️  Attenzione: Backend non raggiungibile")
        print(f"❌ Errore: {data['error']}")
        print("💡 Assicurati che il backend FastAPI sia avviato su http://localhost:8000")
        print()
    else:
        print("✅ Backend connesso e funzionante!")
        print()
    
    interactive_menu()