"""
Comparazione K-SVD vs K-Cluster - Ottimizzazione Performance
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class KOptimizer:
    def __init__(self, ratings_matrix):
        self.ratings_matrix = ratings_matrix
        self.svd_results = {}
        self.cluster_results = {}
    
    def optimize_k_svd(self, k_range=range(5, 101, 5)):
        """
        Ottimizza K per SVD basato su:
        - Varianza spiegata
        - RMSE predizioni
        - Tempo computazionale
        """
        print("🔍 OTTIMIZZAZIONE K-SVD")
        print("=" * 50)
        
        results = []
        
        for k in k_range:
            try:
                # Test SVD con k componenti
                svd = TruncatedSVD(n_components=k, random_state=42)
                user_factors = svd.fit_transform(self.ratings_matrix)
                movie_factors = svd.components_.T
                
                # Calcola metriche
                explained_variance = svd.explained_variance_ratio_.sum()
                
                # Ricostruzione errore (RMSE approssimato)
                reconstructed = np.dot(user_factors, movie_factors.T)
                mse = np.mean((self.ratings_matrix.toarray() - reconstructed) ** 2)
                rmse = np.sqrt(mse)
                
                # Efficienza (varianza per componente)
                efficiency = explained_variance / k
                
                result = {
                    'k_svd': k,
                    'explained_variance': explained_variance,
                    'rmse': rmse,
                    'efficiency': efficiency,
                    'overfitting_risk': 1.0 if k > self.ratings_matrix.shape[0] * 0.1 else 0.0
                }
                
                results.append(result)
                
                print(f"K={k:3d} | Varianza: {explained_variance:.3f} | RMSE: {rmse:.4f} | Efficienza: {efficiency:.4f}")
                
            except Exception as e:
                print(f"Errore K={k}: {e}")
                continue
        
        self.svd_results = pd.DataFrame(results)
        return self._find_optimal_k_svd()
    
    def optimize_k_cluster(self, k_range=range(2, 21)):
        """
        Ottimizza K per Clustering basato su:
        - Silhouette Score
        - Inertia (Within-cluster sum of squares)
        - Metodo Elbow
        """
        print("\n🎯 OTTIMIZZAZIONE K-CLUSTER")  
        print("=" * 50)
        
        # Usa prime 2 componenti SVD per clustering
        svd = TruncatedSVD(n_components=min(50, self.ratings_matrix.shape[1]-1), random_state=42)
        factors = svd.fit_transform(self.ratings_matrix)
        
        # Se abbiamo abbastanza dimensioni, usa prime 2
        if factors.shape[1] >= 2:
            cluster_data = factors[:, :2]
        else:
            # Aggiungi rumore per clustering 1D
            cluster_data = np.column_stack([factors[:, 0], np.random.normal(0, 0.1, factors.shape[0])])
        
        results = []
        
        for k in k_range:
            if k >= len(cluster_data):
                continue
                
            try:
                # Test clustering con k cluster
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(cluster_data)
                
                # Calcola metriche
                silhouette = silhouette_score(cluster_data, labels)
                inertia = kmeans.inertia_
                
                # Bilanciamento cluster (quanto sono equilibrati)
                unique, counts = np.unique(labels, return_counts=True)
                balance = 1.0 - np.std(counts) / np.mean(counts)  # 1 = perfetto balance
                
                result = {
                    'k_cluster': k,
                    'silhouette_score': silhouette,
                    'inertia': inertia,
                    'balance': balance,
                    'interpretability': 1.0 / k  # Meno cluster = più interpretabile
                }
                
                results.append(result)
                
                print(f"K={k:2d} | Silhouette: {silhouette:.3f} | Inertia: {inertia:.0f} | Balance: {balance:.3f}")
                
            except Exception as e:
                print(f"Errore K={k}: {e}")
                continue
        
        self.cluster_results = pd.DataFrame(results)
        return self._find_optimal_k_cluster()
    
    def _find_optimal_k_svd(self):
        """Trova K ottimale per SVD"""
        if self.svd_results.empty:
            return None
            
        df = self.svd_results.copy()
        
        # Normalizza metriche (0-1)
        df['explained_variance_norm'] = df['explained_variance'] / df['explained_variance'].max()
        df['rmse_norm'] = 1 - (df['rmse'] / df['rmse'].max())  # Inverti (meno RMSE = meglio)
        df['efficiency_norm'] = df['efficiency'] / df['efficiency'].max()
        
        # Score composito (pesi personalizzabili)
        df['composite_score'] = (
            df['explained_variance_norm'] * 0.4 +  # 40% varianza
            df['rmse_norm'] * 0.4 +                # 40% accuratezza  
            df['efficiency_norm'] * 0.2             # 20% efficienza
        )
        
        optimal = df.loc[df['composite_score'].idxmax()]
        
        print(f"\n✅ K-SVD OTTIMALE: {optimal['k_svd']}")
        print(f"   Varianza: {optimal['explained_variance']:.3f}")
        print(f"   RMSE: {optimal['rmse']:.4f}")
        print(f"   Score: {optimal['composite_score']:.3f}")
        
        return int(optimal['k_svd'])
    
    def _find_optimal_k_cluster(self):
        """Trova K ottimale per Clustering"""
        if self.cluster_results.empty:
            return None
            
        df = self.cluster_results.copy()
        
        # Normalizza metriche
        df['silhouette_norm'] = df['silhouette_score'] / df['silhouette_score'].max()
        df['balance_norm'] = df['balance'] / df['balance'].max()
        df['interpretability_norm'] = df['interpretability'] / df['interpretability'].max()
        
        # Score composito per clustering
        df['composite_score'] = (
            df['silhouette_norm'] * 0.5 +      # 50% qualità cluster
            df['balance_norm'] * 0.3 +         # 30% bilanciamento
            df['interpretability_norm'] * 0.2   # 20% interpretabilità
        )
        
        optimal = df.loc[df['composite_score'].idxmax()]
        
        print(f"\n✅ K-CLUSTER OTTIMALE: {optimal['k_cluster']}")
        print(f"   Silhouette: {optimal['silhouette_score']:.3f}")
        print(f"   Balance: {optimal['balance']:.3f}")
        print(f"   Score: {optimal['composite_score']:.3f}")
        
        return int(optimal['k_cluster'])
    
    def plot_optimization_results(self):
        """Visualizza risultati ottimizzazione"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # SVD: Varianza spiegata vs K
        if not self.svd_results.empty:
            ax1.plot(self.svd_results['k_svd'], self.svd_results['explained_variance'], 'b-o')
            ax1.set_title('K-SVD: Varianza Spiegata')
            ax1.set_xlabel('K (Componenti SVD)')
            ax1.set_ylabel('Varianza Spiegata')
            ax1.grid(True)
            
            # SVD: RMSE vs K
            ax2.plot(self.svd_results['k_svd'], self.svd_results['rmse'], 'r-o')
            ax2.set_title('K-SVD: Errore RMSE')
            ax2.set_xlabel('K (Componenti SVD)')
            ax2.set_ylabel('RMSE')
            ax2.grid(True)
        
        # Cluster: Silhouette Score vs K
        if not self.cluster_results.empty:
            ax3.plot(self.cluster_results['k_cluster'], self.cluster_results['silhouette_score'], 'g-o')
            ax3.set_title('K-Cluster: Silhouette Score')
            ax3.set_xlabel('K (Numero Cluster)')
            ax3.set_ylabel('Silhouette Score')
            ax3.grid(True)
            
            # Cluster: Inertia vs K (Elbow Method)
            ax4.plot(self.cluster_results['k_cluster'], self.cluster_results['inertia'], 'm-o')
            ax4.set_title('K-Cluster: Inertia (Elbow Method)')
            ax4.set_xlabel('K (Numero Cluster)')
            ax4.set_ylabel('Inertia')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('k_optimization_results.png', dpi=300, bbox_inches='tight')
        return fig

def compare_k_values_impact():
    """
    Confronta l'impatto dei diversi K sui risultati finali
    """
    print("\n" + "="*80)
    print("🔬 ANALISI IMPATTO K-VALUES SULLE PERFORMANCE")
    print("="*80)
    
    impacts = {
        'K-SVD': {
            'description': 'Controlla dimensionalità spazio latente',
            'effects': {
                'Troppo basso (K<10)': 'Underfitting - perde pattern importanti',
                'Ottimale (K=20-50)': 'Bilancia accuratezza e generalizzazione', 
                'Troppo alto (K>100)': 'Overfitting - memorizza rumore'
            },
            'optimization_goal': 'Massimizzare varianza spiegata minimizzando overfitting'
        },
        'K-Cluster': {
            'description': 'Controlla numero gruppi utenti/film',
            'effects': {
                'Troppo basso (K=2-3)': 'Gruppi troppo generici, poca granularità',
                'Ottimale (K=5-10)': 'Gruppi significativi e interpretabili',
                'Troppo alto (K>20)': 'Gruppi troppo specifici, poco utili'
            },
            'optimization_goal': 'Massimizzare coesione intra-cluster e separazione inter-cluster'
        }
    }
    
    for k_type, info in impacts.items():
        print(f"\n📊 {k_type}:")
        print(f"   {info['description']}")
        print(f"   Obiettivo: {info['optimization_goal']}")
        print("   Effetti:")
        for scenario, effect in info['effects'].items():
            print(f"     • {scenario}: {effect}")
    
    print(f"\n💡 RACCOMANDAZIONI:")
    print(f"   • K-SVD: Usa cross-validation per trovare sweet spot")
    print(f"   • K-Cluster: Usa metodo Elbow + Silhouette Score")
    print(f"   • Monitora entrambi: performance possono essere indipendenti")
    print(f"   • Dataset piccoli: K più bassi per evitare overfitting")
    print(f"   • Dataset grandi: K più alti per catturare complessità")

if __name__ == "__main__":
    compare_k_values_impact()