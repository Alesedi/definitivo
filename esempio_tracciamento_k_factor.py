"""
ESEMPIO PRATICO: TRACCIAMENTO E OTTIMIZZAZIONE DEL FATTORE K NELLA SVD
=====================================================================

Questo file mostra come utilizzare le funzionalità di tracciamento del fattore k
implementate nel sistema di raccomandazione AFlix.

Il fattore k rappresenta il numero di componenti (dimensioni latenti) utilizzate
nella decomposizione SVD e ha un impatto critico sulle performance del modello.
"""

import requests
import json
from typing import Dict, List, Any

# Base URL del tuo backend FastAPI
BASE_URL = "http://localhost:8005"

class KFactorMonitor:
    """Classe per monitorare e ottimizzare il fattore k nella SVD"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def get_current_k_status(self) -> Dict[str, Any]:
        """Recupera lo status attuale del fattore k"""
        try:
            response = requests.get(f"{self.base_url}/admin/ml/model-status-extended")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_k_factor(self) -> Dict[str, Any]:
        """Analizza il fattore k e riceve raccomandazioni"""
        try:
            response = requests.get(f"{self.base_url}/admin/ml/k-factor-analysis")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def optimize_k_factor(self, k_range: List[int] = None) -> Dict[str, Any]:
        """Ottimizza il fattore k testando diversi valori"""
        try:
            payload = {"k_range": k_range} if k_range else {}
            response = requests.post(f"{self.base_url}/admin/ml/optimize-k-factor", 
                                   json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_full_report(self) -> Dict[str, Any]:
        """Genera report completo sul fattore k"""
        try:
            response = requests.get(f"{self.base_url}/admin/ml/k-factor-report")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def update_k_components(self, new_k: int) -> Dict[str, Any]:
        """Aggiorna il numero di componenti k"""
        try:
            response = requests.post(f"{self.base_url}/admin/ml/update-k-components/{new_k}")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

def print_k_analysis(analysis: Dict[str, Any]):
    """Stampa l'analisi del fattore k in modo leggibile"""
    print("\n" + "="*60)
    print("ANALISI FATTORE K - SVD")
    print("="*60)
    
    if "error" in analysis:
        print(f"❌ Errore: {analysis['error']}")
        return
    
    print(f"📊 K Attuale: {analysis.get('current_k', 'N/A')}")
    print(f"📋 K Richiesto: {analysis.get('requested_k', 'N/A')}")
    print(f"📈 Varianza Totale Spiegata: {analysis.get('total_explained_variance', 0):.1%}")
    print(f"⚡ Efficienza K: {analysis.get('k_efficiency', 0):.4f}")
    
    if analysis.get('elbow_point'):
        print(f"📍 Elbow Point: {analysis['elbow_point']}")
    
    if analysis.get('recommended_k'):
        print(f"💡 K Raccomandato: {analysis['recommended_k']}")
    
    # Mostra varianza per componente
    if analysis.get('cumulative_variance'):
        print("\n📈 VARIANZA CUMULATIVA PER COMPONENTE:")
        print("-" * 50)
        for item in analysis['cumulative_variance'][:10]:  # Prime 10
            comp = item['component']
            individual = item['individual_variance']
            cumulative = item['cumulative_variance']
            percentage = item['percentage']
            print(f"Componente {comp:2d}: {individual:.4f} | Cumulativa: {cumulative:.4f} ({percentage:.1f}%)")

def print_optimization_results(results: Dict[str, Any]):
    """Stampa i risultati dell'ottimizzazione k"""
    print("\n" + "="*60)
    print("RISULTATI OTTIMIZZAZIONE FATTORE K")
    print("="*60)
    
    if "error" in results:
        print(f"❌ Errore: {results['error']}")
        return
    
    print(f"🎯 K Ottimale: {results.get('best_k', 'N/A')}")
    print(f"📊 K Attuale: {results.get('current_k', 'N/A')}")
    print(f"🔄 Miglioramento Disponibile: {'Sì' if results.get('improvement') else 'No'}")
    print(f"💬 Raccomandazione: {results.get('recommendation', 'N/A')}")
    
    # Mostra top 5 risultati
    if results.get('all_results'):
        print("\n🏆 TOP 5 CONFIGURAZIONI K:")
        print("-" * 70)
        print("K    | RMSE   | MAE    | Var.Spiega | Score Combinato")
        print("-" * 70)
        
        for result in results['all_results'][:5]:
            k = result['k']
            rmse = result['rmse']
            mae = result['mae']
            var = result['explained_variance']
            score = result['combined_score']
            print(f"{k:3d}  | {rmse:.4f} | {mae:.4f} | {var:.4f}      | {score:.4f}")

def example_workflow():
    """Esempio completo di workflow per tracciamento k"""
    print("🚀 AVVIO WORKFLOW TRACCIAMENTO FATTORE K")
    print("="*60)
    
    # Inizializza monitor
    monitor = KFactorMonitor()
    
    # 1. Controlla status attuale
    print("\n📊 1. CONTROLLO STATUS ATTUALE...")
    status = monitor.get_current_k_status()
    
    if not status.get("error"):
        print(f"✅ Modello addestrato: {status.get('is_trained')}")
        print(f"📊 K utilizzato: {status.get('actual_k_used')}")
        print(f"📈 Varianza spiegata: {status.get('explained_variance', 0):.1%}")
        print(f"⚡ Efficienza K: {status.get('k_efficiency', 0):.4f}")
    else:
        print(f"❌ Errore status: {status['error']}")
        return
    
    # 2. Analisi dettagliata
    print("\n🔍 2. ANALISI DETTAGLIATA DEL FATTORE K...")
    analysis = monitor.analyze_k_factor()
    print_k_analysis(analysis)
    
    # 3. Ottimizzazione (se necessaria)
    if analysis.get('recommended_k') and analysis['recommended_k'] != status.get('actual_k_used'):
        print("\n⚡ 3. OTTIMIZZAZIONE FATTORE K...")
        
        # Test range intorno al valore raccomandato
        recommended = analysis['recommended_k']
        test_range = list(range(max(1, recommended-10), recommended+11, 2))
        
        optimization = monitor.optimize_k_factor(test_range)
        print_optimization_results(optimization)
        
        # 4. Applicazione del k ottimale (opzionale)
        if optimization.get('best_k') and optimization.get('improvement'):
            apply = input(f"\n🔄 Applicare k ottimale ({optimization['best_k']})? (y/n): ")
            if apply.lower() == 'y':
                print(f"\n🔧 4. APPLICAZIONE K={optimization['best_k']}...")
                update_result = monitor.update_k_components(optimization['best_k'])
                
                if not update_result.get("error"):
                    print("✅ Aggiornamento completato!")
                    print(f"📊 Nuovo K: {update_result.get('new_k')}")
                    print(f"📈 Nuova Varianza: {update_result.get('new_variance', 0):.1%}")
                    print(f"🎯 Miglioramento: {'Sì' if update_result.get('improvement') else 'No'}")
                else:
                    print(f"❌ Errore aggiornamento: {update_result['error']}")
    else:
        print("\n✅ 3. K ATTUALE GIÀ OTTIMALE - Nessuna ottimizzazione necessaria")
    
    # 5. Report finale
    print("\n📋 5. REPORT FINALE...")
    final_report = monitor.get_full_report()
    
    if not final_report.get("error"):
        recommendations = final_report.get('recommendations', [])
        if recommendations:
            print("\n💡 RACCOMANDAZIONI FINALI:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. [{rec['type'].upper()}] {rec['message']}")
        else:
            print("✅ Nessuna raccomandazione aggiuntiva")
    
    print("\n🎉 WORKFLOW COMPLETATO!")

def monitoring_dashboard():
    """Dashboard di monitoraggio continuo"""
    print("📊 DASHBOARD MONITORAGGIO FATTORE K")
    print("="*60)
    
    monitor = KFactorMonitor()
    
    while True:
        print("\nOpzioni disponibili:")
        print("1. Status modello")
        print("2. Analisi fattore k")
        print("3. Ottimizzazione k")
        print("4. Report completo")
        print("5. Aggiorna k")
        print("0. Esci")
        
        choice = input("\nScegli opzione (0-5): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            status = monitor.get_current_k_status()
            print(json.dumps(status, indent=2))
        elif choice == "2":
            analysis = monitor.analyze_k_factor()
            print_k_analysis(analysis)
        elif choice == "3":
            k_input = input("Range k (es: 5,10,15,20) o ENTER per automatico: ").strip()
            k_range = None
            if k_input:
                try:
                    k_range = [int(x.strip()) for x in k_input.split(',')]
                except ValueError:
                    print("❌ Formato non valido. Uso range automatico.")
            
            optimization = monitor.optimize_k_factor(k_range)
            print_optimization_results(optimization)
        elif choice == "4":
            report = monitor.get_full_report()
            print(json.dumps(report, indent=2))
        elif choice == "5":
            try:
                new_k = int(input("Nuovo valore k: "))
                result = monitor.update_k_components(new_k)
                print(json.dumps(result, indent=2))
            except ValueError:
                print("❌ Valore k non valido")
        else:
            print("❌ Opzione non valida")

if __name__ == "__main__":
    print("🎬 SISTEMA TRACCIAMENTO FATTORE K - AFLIX")
    print("=" * 60)
    print("Questo script dimostra come tracciare e ottimizzare")
    print("il fattore k (numero componenti) nella SVD.")
    print("=" * 60)
    
    mode = input("\nModalità: (1) Workflow automatico (2) Dashboard interattiva: ").strip()
    
    if mode == "1":
        example_workflow()
    elif mode == "2":
        monitoring_dashboard()
    else:
        print("❌ Modalità non valida")
        example_workflow()  # Default