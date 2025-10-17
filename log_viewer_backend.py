"""
LOG VIEWER IN TEMPO REALE - BACKEND AFLIX
==========================================

Questo script semplice permette di vedere i log dettagliati del backend
durante i processi di machine learning, specialmente per la scelta del fattore K.

Per usarlo:
1. Avvia il backend FastAPI
2. Esegui questo script
3. Triggera operazioni ML dal frontend o API
4. Osserva i log dettagliati in tempo reale
"""

import requests
import time
import json
from datetime import datetime

def show_live_logs():
    """Mostra esempio di log che vedrai nel backend"""
    print("=" * 80)
    print("📋 ESEMPIO LOG BACKEND - PROCESSO SCELTA FATTORE K")
    print("=" * 80)
    print("""
[2025-10-17 15:30:45] INFO: ============================================================
[2025-10-17 15:30:45] INFO: 🎯 PROCESSO SELEZIONE FATTORE K - SVD
[2025-10-17 15:30:45] INFO: ============================================================
[2025-10-17 15:30:45] INFO: 📊 Matrice dati: 25 utenti × 487 film
[2025-10-17 15:30:45] INFO: 📈 Densità matrice: 2.34%
[2025-10-17 15:30:45] INFO: 🎛️  K richiesto dal modello: 50
[2025-10-17 15:30:45] INFO: 📏 Dimensione minima matrice: 25
[2025-10-17 15:30:45] INFO: 🔝 Massimo K possibile: 24
[2025-10-17 15:30:45] INFO: ✅ K finale selezionato: 24
[2025-10-17 15:30:45] WARNING: ⚠️  K ridotto da 50 a 24 per limitazioni dati
[2025-10-17 15:30:45] INFO: 🚀 Avvio decomposizione SVD...
[2025-10-17 15:30:45] INFO: ------------------------------------------------------------
[2025-10-17 15:30:45] INFO: 🔄 Esecuzione TruncatedSVD...
[2025-10-17 15:30:45] INFO: 📊 Calcolo fattori latenti utenti...
[2025-10-17 15:30:45] INFO: 🎬 Calcolo fattori latenti film...
[2025-10-17 15:30:46] INFO: ✅ SVD completata!
[2025-10-17 15:30:46] INFO: 📈 Varianza totale spiegata: 89.2%
[2025-10-17 15:30:46] INFO: 👥 Fattori utenti: (25, 24)
[2025-10-17 15:30:46] INFO: 🎭 Fattori film: (487, 24)
[2025-10-17 15:30:46] INFO: ============================================================
[2025-10-17 15:30:46] INFO: 📊 ANALISI DETTAGLIATA COMPONENTI SVD
[2025-10-17 15:30:46] INFO: ============================================================
[2025-10-17 15:30:46] INFO: Componente  1: 0.2341 (23.41%) | Cumulativa: 0.2341 (23.4%)
[2025-10-17 15:30:46] INFO: Componente  2: 0.1876 (18.76%) | Cumulativa: 0.4217 (42.2%)
[2025-10-17 15:30:46] INFO: Componente  3: 0.1234 (12.34%) | Cumulativa: 0.5451 (54.5%)
[2025-10-17 15:30:46] INFO: Componente  4: 0.0987 (9.87%) | Cumulativa: 0.6438 (64.4%)
[2025-10-17 15:30:46] INFO: Componente  5: 0.0754 (7.54%) | Cumulativa: 0.7192 (71.9%)
[2025-10-17 15:30:46] INFO: Componente  6: 0.0621 (6.21%) | Cumulativa: 0.7813 (78.1%)
[2025-10-17 15:30:46] INFO: Componente  7: 0.0543 (5.43%) | Cumulativa: 0.8356 (83.6%)
[2025-10-17 15:30:46] INFO: Componente  8: 0.0398 (3.98%) | Cumulativa: 0.8754 (87.5%)
[2025-10-17 15:30:46] INFO: Componente  9: 0.0187 (1.87%) | Cumulativa: 0.8941 (89.4%)
[2025-10-17 15:30:46] INFO: Componente 10: 0.0156 (1.56%) | Cumulativa: 0.9097 (91.0%)
[2025-10-17 15:30:46] INFO: ... e altre 14 componenti
[2025-10-17 15:30:46] INFO: 📍 Elbow Point identificato: Componente 8
[2025-10-17 15:30:46] INFO: 💡 Suggerimento: Potresti usare solo 8 componenti mantenendo 87.5% della varianza
[2025-10-17 15:30:46] INFO: ------------------------------------------------------------
""")

def trigger_training_with_logs():
    """Triggera training e mostra che tipo di log aspettarsi"""
    print("\n🔄 TRIGGER TRAINING MODELLO...")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:8005/recommendations/train-sync", timeout=30)
        
        print("📋 Log che vedrai nel backend durante il training:")
        print("-" * 60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ TRAINING COMPLETATO!")
            print(f"📊 K utilizzato: {result.get('n_components', 'N/A')}")
            print(f"📈 Varianza spiegata: {result.get('explained_variance', 0):.1%}")
            print(f"🎯 Efficienza K: {result.get('k_efficiency', 0):.4f}")
        else:
            print(f"❌ Errore: HTTP {response.status_code}")
            print("💡 Assicurati che il backend sia avviato")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore connessione: {e}")
        print("💡 Backend non raggiungibile - mostro esempio log...")
        show_live_logs()

def trigger_optimization_with_logs():
    """Triggera ottimizzazione e mostra log attesi"""
    print("\n🔧 TRIGGER OTTIMIZZAZIONE FATTORE K...")
    print("=" * 60)
    
    k_range = [5, 10, 15, 20]
    print(f"🎯 Testing K: {k_range}")
    
    try:
        payload = {"k_range": k_range}
        response = requests.post("http://localhost:8005/admin/ml/optimize-k-factor", 
                               json=payload, timeout=60)
        
        print("\n📋 Log che vedrai durante l'ottimizzazione:")
        print("-" * 60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ OTTIMIZZAZIONE COMPLETATA!")
            print(f"🏆 K ottimale: {result.get('best_k', 'N/A')}")
            print(f"🔄 Miglioramento disponibile: {'SÌ' if result.get('improvement') else 'NO'}")
        else:
            print(f"❌ Errore: HTTP {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore connessione: {e}")
        print("💡 Mostro esempio log ottimizzazione...")
        show_optimization_logs()

def show_optimization_logs():
    """Mostra esempio log ottimizzazione"""
    print("""
[2025-10-17 15:35:12] INFO: ================================================================================
[2025-10-17 15:35:12] INFO: 🔬 OTTIMIZZAZIONE FATTORE K - PROCESSO DETTAGLIATO
[2025-10-17 15:35:12] INFO: ================================================================================
[2025-10-17 15:35:12] INFO: 🎯 Range K da testare: [5, 10, 15, 20]
[2025-10-17 15:35:12] INFO: 📊 Dataset: 285 rating, 25 utenti, 134 film
[2025-10-17 15:35:12] INFO: ================================================================================

[2025-10-17 15:35:12] INFO: 🔍 TEST K = 5
[2025-10-17 15:35:12] INFO: ----------------------------------------
[2025-10-17 15:35:12] INFO: 📚 Train set: 228 rating
[2025-10-17 15:35:12] INFO: 🧪 Test set: 57 rating
[2025-10-17 15:35:12] INFO: 🏗️  Matrice training: (25, 134)
[2025-10-17 15:35:12] INFO: ⚙️  Esecuzione SVD con k=5...
[2025-10-17 15:35:12] INFO: 📈 Varianza spiegata: 67.2%
[2025-10-17 15:35:12] INFO: 🎯 Generazione predizioni...
[2025-10-17 15:35:13] INFO: ✅ RISULTATI K=5:
[2025-10-17 15:35:13] INFO:    📉 RMSE: 0.8734
[2025-10-17 15:35:13] INFO:    📊 MAE: 0.6521
[2025-10-17 15:35:13] INFO:    📈 Varianza: 67.2%
[2025-10-17 15:35:13] INFO:    🎯 Score Combinato: 0.5374
[2025-10-17 15:35:13] INFO:    🔢 Predizioni valide: 45
[2025-10-17 15:35:13] INFO:    🏆 NUOVO MIGLIOR K! (5)

[2025-10-17 15:35:13] INFO: 🔍 TEST K = 10
[2025-10-17 15:35:13] INFO: ----------------------------------------
[2025-10-17 15:35:13] INFO: 📚 Train set: 228 rating
[2025-10-17 15:35:13] INFO: 🧪 Test set: 57 rating
[2025-10-17 15:35:13] INFO: 🏗️  Matrice training: (25, 134)
[2025-10-17 15:35:13] INFO: ⚙️  Esecuzione SVD con k=10...
[2025-10-17 15:35:13] INFO: 📈 Varianza spiegata: 78.9%
[2025-10-17 15:35:13] INFO: 🎯 Generazione predizioni...
[2025-10-17 15:35:14] INFO:    📊 Processate 25/57 predizioni...
[2025-10-17 15:35:14] INFO:    📊 Processate 50/57 predizioni...
[2025-10-17 15:35:14] INFO: ✅ RISULTATI K=10:
[2025-10-17 15:35:14] INFO:    📉 RMSE: 0.7891
[2025-10-17 15:35:14] INFO:    📊 MAE: 0.5987
[2025-10-17 15:35:14] INFO:    📈 Varianza: 78.9%
[2025-10-17 15:35:14] INFO:    🎯 Score Combinato: 0.3946
[2025-10-17 15:35:14] INFO:    🔢 Predizioni valide: 52
[2025-10-17 15:35:14] INFO:    🏆 NUOVO MIGLIOR K! (10)

[2025-10-17 15:35:15] INFO: ================================================================================
[2025-10-17 15:35:15] INFO: 🏆 RISULTATI FINALI OTTIMIZZAZIONE
[2025-10-17 15:35:15] INFO: ================================================================================
[2025-10-17 15:35:15] INFO: 🥇 MIGLIOR K TROVATO: 10
[2025-10-17 15:35:15] INFO: 📊 K ATTUALE: 24
[2025-10-17 15:35:15] INFO: 🔄 MIGLIORAMENTO POSSIBILE: SÌ

[2025-10-17 15:35:15] INFO: 🏅 TOP 5 CONFIGURAZIONI:
[2025-10-17 15:35:15] INFO: --------------------------------------------------------------------------------
[2025-10-17 15:35:15] INFO: Pos | K   | RMSE   | MAE    | Varianza | Score
[2025-10-17 15:35:15] INFO: --------------------------------------------------------------------------------
[2025-10-17 15:35:15] INFO: 🥇  1 |  10 | 0.7891 | 0.5987 | 0.7890   | 0.3946
[2025-10-17 15:35:15] INFO: 🥈  2 |  15 | 0.8234 | 0.6123 | 0.8456   | 0.4006
[2025-10-17 15:35:15] INFO: 🥉  3 |   5 | 0.8734 | 0.6521 | 0.6720   | 0.5374
[2025-10-17 15:35:15] INFO: ================================================================================
[2025-10-17 15:35:15] INFO: 💡 RACCOMANDAZIONE: Use k=10 for optimal performance
""")

def show_real_time_status():
    """Mostra status in tempo reale"""
    print("\n📊 RECUPERO STATUS TEMPO REALE...")
    
    try:
        response = requests.get("http://localhost:8005/admin/ml/live-monitor", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Status recuperato!")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Errore: HTTP {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore connessione: {e}")
        print("💡 Backend non raggiungibile")

def main_menu():
    """Menu principale"""
    while True:
        print("\n" + "=" * 70)
        print("📋 LOG VIEWER BACKEND - PROCESSI FATTORE K")
        print("=" * 70)
        print("1. 🔄 Triggera training + mostra log")
        print("2. 🔧 Triggera ottimizzazione + mostra log")
        print("3. 📊 Status tempo reale")
        print("4. 📋 Esempio log training")
        print("5. 🔬 Esempio log ottimizzazione")
        print("6. 💡 Come vedere i log nel backend")
        print("0. 🚪 Esci")
        print("=" * 70)
        
        choice = input("Scegli opzione (0-6): ").strip()
        
        if choice == "0":
            print("👋 Arrivederci!")
            break
        elif choice == "1":
            trigger_training_with_logs()
        elif choice == "2":
            trigger_optimization_with_logs()
        elif choice == "3":
            show_real_time_status()
        elif choice == "4":
            show_live_logs()
        elif choice == "5":
            show_optimization_logs()
        elif choice == "6":
            show_backend_log_instructions()
        else:
            print("❌ Opzione non valida")
        
        if choice != "0":
            input("\nPremi ENTER per continuare...")

def show_backend_log_instructions():
    """Mostra istruzioni per vedere i log nel backend"""
    print("\n" + "=" * 70)
    print("💡 COME VEDERE I LOG NEL BACKEND")
    print("=" * 70)
    print("""
🔧 METODO 1: Avvia backend in modalità verbose
   Nella cartella backend, esegui:
   uvicorn main:app --reload --log-level debug

📋 METODO 2: Controlla console VS Code
   Se avvii il backend da VS Code, i log appaiono nel terminale

🖥️  METODO 3: Usa questo script
   Questo script triggera le operazioni e mostra esempi dei log

📊 METODO 4: Monitor live
   Usa il file 'monitor_backend_k_factor.py' per monitoraggio real-time

🔍 COSA VEDRAI NEI LOG:
   ✅ Processo completo di selezione K
   📊 Analisi dettagliata componenti SVD
   🔧 Progress dell'ottimizzazione step-by-step
   🏆 Risultati finali con ranking
   ⚠️  Warning e raccomandazioni

💡 LIVELLI LOG:
   INFO: Operazioni normali
   WARNING: Situazioni che richiedono attenzione
   ERROR: Errori che interrompono il processo
   DEBUG: Dettagli tecnici approfonditi
""")

if __name__ == "__main__":
    print("📋 LOG VIEWER BACKEND AFLIX - FATTORE K")
    print("=" * 70)
    print("Questo tool ti aiuta a vedere cosa succede nel backend")
    print("durante i processi di machine learning, specialmente per")
    print("la selezione e ottimizzazione del fattore K nella SVD.")
    print("=" * 70)
    
    main_menu()