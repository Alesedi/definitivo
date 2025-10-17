import React, { useState, useEffect } from 'react';
import './MLMonitor.css';

const MLMonitor = () => {
  const [modelStatus, setModelStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [logs, setLogs] = useState([]);
  const [isAutoRefresh, setIsAutoRefresh] = useState(false);

  const API_BASE = 'http://localhost:8005';

  // Funzione per aggiungere log
  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-20), { // Mantieni solo gli ultimi 20 log
      id: Date.now(),
      timestamp,
      message,
      type
    }]);
  };

  // Recupera status del modello
  const fetchModelStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/admin/ml/live-monitor`);
      if (response.ok) {
        const data = await response.json();
        setModelStatus(data);
        return data;
      }
    } catch (error) {
      console.error('Error fetching model status:', error);
      const errorMessage = error.message || error.toString() || 'Errore di connessione';
      addLog(`Errore connessione backend: ${errorMessage}`, 'error');
    }
    return null;
  };

  // Training del modello
  const handleTraining = async () => {
    setIsLoading(true);
    addLog('🚀 Avvio training del modello...', 'info');
    
    try {
      const response = await fetch(`${API_BASE}/recommendations/train-sync`);
      if (response.ok) {
        const result = await response.json();
        addLog('✅ Training completato con successo!', 'success');
        addLog(`📊 Statistiche: ${result.stats?.total_ratings || 0} rating processati`, 'info');
        addLog(`🎯 K utilizzato: ${result.stats?.actual_k_used || 'N/A'}`, 'info');
        addLog(`📈 Varianza spiegata: ${(result.stats?.explained_variance * 100 || 0).toFixed(1)}%`, 'info');
        
        // Aggiorna status
        setTimeout(fetchModelStatus, 1000);
      } else {
        let errorMessage = 'Errore sconosciuto';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || `HTTP ${response.status}`;
        } catch (parseError) {
          errorMessage = `HTTP ${response.status} - ${response.statusText}`;
        }
        addLog(`❌ Errore training: ${errorMessage}`, 'error');
      }
    } catch (error) {
      const errorMessage = error.message || error.toString() || 'Errore di connessione';
      addLog(`❌ Errore training: ${errorMessage}`, 'error');
    }
    
    setIsLoading(false);
  };

  // Ottimizzazione fattore K
  const handleOptimization = async () => {
    setIsLoading(true);
    addLog('🔧 Avvio ottimizzazione fattore K...', 'info');
    
    // Prima controlliamo se il modello è addestrato
    const currentStatus = await fetchModelStatus();
    
    if (!currentStatus?.model_status?.is_trained) {
      addLog('❌ Il modello non è ancora addestrato!', 'error');
      addLog('💡 Esegui prima il training del modello', 'info');
      setIsLoading(false);
      return;
    }
    
    // Controlliamo se abbiamo un K attuale valido
    const currentK = currentStatus?.current_performance?.k_used;
    if (!currentK || currentK === 0) {
      addLog('❌ K attuale non valido - riaddestra il modello', 'error');
      setIsLoading(false);
      return;
    }
    
    // Calcoliamo range K basato sui dati attuali
    let k_range = [1]; // Default per dataset molto piccoli
    
    if (currentStatus?.dataset_info) {
      const users = currentStatus.dataset_info.total_users || 2;
      const movies = currentStatus.dataset_info.total_movies || 53;
      const max_k = Math.min(users, movies, 10); // Limita a 10 max
      
      if (max_k >= 3) {
        k_range = [1, 2, 3];
      } else if (max_k >= 2) {
        k_range = [1, 2];
      }
      
      addLog(`📊 Dataset: ${users} utenti, ${movies} film - K max: ${max_k}`, 'info');
    } else {
      // Fallback basato su K attuale
      const maxTest = Math.min(currentK + 2, 5);
      k_range = Array.from({length: maxTest}, (_, i) => i + 1);
    }
    
    addLog(`🎯 Testing K range: [${k_range.join(', ')}]`, 'info');
    
    try {
      const response = await fetch(`${API_BASE}/admin/ml/optimize-k-factor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          k_range: k_range
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        
        addLog('✅ Ottimizzazione completata!', 'success');
        addLog(`🏆 K ottimale trovato: ${result.best_k || 'N/A'}`, 'success');
        addLog(`🔄 Miglioramento disponibile: ${result.improvement ? 'SÌ' : 'NO'}`, 'info');
        
        // Mostra top risultati
        if (result.all_results && result.all_results.length > 0) {
          result.all_results.slice(0, 3).forEach((res, index) => {
            addLog(`${index + 1}° K=${res.k}: RMSE=${res.rmse?.toFixed(4)}, Var=${(res.explained_variance * 100)?.toFixed(1)}%`, 'info');
          });
        }
        
        setTimeout(fetchModelStatus, 1000);
      } else {
        let errorMessage = 'Errore sconosciuto';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || JSON.stringify(errorData);
        } catch (parseError) {
          errorMessage = `HTTP ${response.status} - ${response.statusText}`;
        }
        addLog(`❌ Errore ottimizzazione: ${errorMessage}`, 'error');
      }
    } catch (error) {
      const errorMessage = error.message || error.toString() || 'Errore di connessione';
      addLog(`❌ Errore ottimizzazione: ${errorMessage}`, 'error');
    }
    
    setIsLoading(false);
  };

  // Analisi fattore K
  const fetchKAnalysis = async () => {
    try {
      const response = await fetch(`${API_BASE}/admin/ml/k-factor-analysis`);
      if (response.ok) {
        const analysis = await response.json();
        
        addLog('📊 Analisi fattore K completata', 'info');
        addLog(`🎯 K attuale: ${analysis.current_k || 'N/A'}`, 'info');
        addLog(`📋 K raccomandato: ${analysis.recommended_k || 'N/A'}`, 'info');
        addLog(`📍 Elbow point: ${analysis.elbow_point || 'N/A'}`, 'info');
        
        return analysis;
      }
    } catch (error) {
      const errorMessage = error.message || error.toString() || 'Errore di connessione';
      addLog(`❌ Errore analisi K: ${errorMessage}`, 'error');
    }
    return null;
  };

  // Auto-refresh
  useEffect(() => {
    let interval;
    if (isAutoRefresh) {
      interval = setInterval(() => {
        fetchModelStatus();
      }, 3000); // Ogni 3 secondi
    }
    return () => clearInterval(interval);
  }, [isAutoRefresh]); // fetchModelStatus è definito nel componente, non serve nelle dipendenze

  // Caricamento iniziale
  useEffect(() => {
    const initializeMonitor = () => {
      fetchModelStatus();
      addLog('🎬 Monitor ML inizializzato', 'info');
    };
    
    initializeMonitor();
  }, []); // Array vuoto per eseguire solo al mount

  const getStatusColor = (status) => {
    if (!status) return 'gray';
    return status.model_status?.is_trained ? 'green' : 'orange';
  };

  const getEfficiencyColor = (efficiency) => {
    if (efficiency > 0.03) return '#27ae60';
    if (efficiency > 0.01) return '#f39c12';
    return '#e74c3c';
  };

  return (
    <div className="ml-monitor">
      <div className="ml-monitor-header">
        <h2>🧠 Monitor Machine Learning - Fattore K</h2>
        <div className={`connection-status ${modelStatus ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          {modelStatus ? 'Connesso' : 'Disconnesso'}
        </div>
      </div>

      <div className="ml-monitor-content">
        {/* Controlli */}
        <div className="ml-controls">
          <button 
            onClick={handleTraining} 
            disabled={isLoading}
            className="btn btn-primary"
          >
            {isLoading ? '⏳' : '🚀'} Training Modello
          </button>
          
          <button 
            onClick={handleOptimization} 
            disabled={isLoading}
            className="btn btn-warning"
          >
            {isLoading ? '⏳' : '🔧'} Ottimizza K
          </button>
          
          <button 
            onClick={fetchKAnalysis}
            className="btn btn-info"
          >
            📊 Analizza K
          </button>
          
          <button 
            onClick={() => setIsAutoRefresh(!isAutoRefresh)}
            className={`btn ${isAutoRefresh ? 'btn-success' : 'btn-secondary'}`}
          >
            {isAutoRefresh ? '⏹️ Ferma' : '▶️ Auto'} Refresh
          </button>
        </div>

        {/* Status Cards */}
        <div className="status-grid">
          <div className="status-card">
            <h3>📊 Status Modello</h3>
            <div className="status-item">
              <span>Addestrato:</span>
              <span className={`status-value ${getStatusColor(modelStatus)}`}>
                {modelStatus?.model_status?.is_trained ? '✅ SÌ' : '❌ NO'}
              </span>
            </div>
            <div className="status-item">
              <span>K Utilizzato:</span>
              <span className="status-value">
                {modelStatus?.current_performance?.k_used > 0 
                  ? modelStatus.current_performance.k_used 
                  : (modelStatus?.model_status?.is_trained ? 'In caricamento...' : 'Non addestrato')}
              </span>
            </div>
            <div className="status-item">
              <span>Varianza Spiegata:</span>
              <span className="status-value">
                {modelStatus?.current_performance?.explained_variance 
                  ? `${(modelStatus.current_performance.explained_variance * 100).toFixed(1)}%`
                  : 'N/A'}
              </span>
            </div>
          </div>

          <div className="status-card">
            <h3>📊 Dataset Info</h3>
            <div className="status-item">
              <span>Utenti:</span>
              <span className="status-value">
                {modelStatus?.dataset_info?.total_users || 'N/A'}
              </span>
            </div>
            <div className="status-item">
              <span>Film:</span>
              <span className="status-value">
                {modelStatus?.dataset_info?.total_movies || 'N/A'}
              </span>
            </div>
            <div className="status-item">
              <span>Rating Totali:</span>
              <span className="status-value">
                {modelStatus?.dataset_info?.total_ratings || 'N/A'}
              </span>
            </div>
          </div>

          <div className="status-card">
            <h3>⚡ Performance</h3>
            <div className="status-item">
              <span>Efficienza K:</span>
              <span 
                className="status-value"
                style={{ 
                  color: getEfficiencyColor(modelStatus?.current_performance?.k_efficiency || 0)
                }}
              >
                {modelStatus?.current_performance?.k_efficiency?.toFixed(4) || 'N/A'}
              </span>
            </div>
            <div className="status-item">
              <span>Clustering:</span>
              <span className="status-value">
                {modelStatus?.model_status?.has_clustering ? '✅ Attivo' : '❌ Disattivo'}
              </span>
            </div>
            
            {/* Progress Bar Varianza */}
            <div className="progress-container">
              <div className="progress-label">Varianza Spiegata</div>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ 
                    width: `${(modelStatus?.current_performance?.explained_variance || 0) * 100}%`
                  }}
                ></div>
              </div>
              <div className="progress-text">
                {modelStatus?.current_performance?.explained_variance 
                  ? `${(modelStatus.current_performance.explained_variance * 100).toFixed(1)}%`
                  : '0%'}
              </div>
            </div>
          </div>

          {/* Componenti SVD */}
          <div className="status-card">
            <h3>📈 Componenti SVD</h3>
            <div className="components-chart">
              {modelStatus?.current_performance?.components_breakdown?.slice(0, 5).map((variance, index) => (
                <div key={index} className="component-row">
                  <span className="component-label">Comp {index + 1}</span>
                  <div className="component-bar">
                    <div 
                      className="component-fill"
                      style={{ width: `${variance * 100}%` }}
                    ></div>
                  </div>
                  <span className="component-value">{(variance * 100).toFixed(1)}%</span>
                </div>
              )) || <p>Nessun componente disponibile</p>}
            </div>
          </div>
        </div>

        {/* Raccomandazioni */}
        {modelStatus?.recommendations?.length > 0 && (
          <div className="recommendations">
            <h3>💡 Raccomandazioni Sistema</h3>
            {modelStatus.recommendations.map((rec, index) => (
              <div key={index} className={`recommendation ${rec.level}`}>
                <span className="rec-icon">
                  {rec.level === 'warning' ? '⚠️' : rec.level === 'optimization' ? '🔧' : 'ℹ️'}
                </span>
                <span className="rec-message">{rec.message}</span>
              </div>
            ))}
          </div>
        )}

        {/* Log in tempo reale */}
        <div className="logs-container">
          <h3>📋 Log Processo ML</h3>
          <div className="logs-content">
            {logs.map(log => (
              <div key={log.id} className={`log-entry log-${log.type}`}>
                <span className="log-timestamp">[{log.timestamp}]</span>
                <span className="log-message">{log.message}</span>
              </div>
            ))}
            {logs.length === 0 && (
              <div className="log-entry log-info">
                <span className="log-message">In attesa di operazioni ML...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLMonitor;