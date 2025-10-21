import React, { useState, useEffect, useRef } from 'react';
import './KOptimizationMonitor.css';

const KOptimizationMonitor = () => {
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [currentPhase, setCurrentPhase] = useState(null);
  const [kSvdResults, setKSvdResults] = useState([]);
  const [kClusterResults, setKClusterResults] = useState([]);
  const [optimalResults, setOptimalResults] = useState({});
  const [matrixInfo, setMatrixInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isOptimizing, setIsOptimizing] = useState(false);
  
  const eventSourceRef = useRef(null);

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-49), { timestamp, message, type }]);
  };

  const startOptimization = async () => {
    if (isOptimizing) return;
    
    setIsOptimizing(true);
    setStatus('running');
    setProgress(0);
    setKSvdResults([]);
    setKClusterResults([]);
    setOptimalResults({});
    setLogs([]);
    
    addLog('🚀 Avvio ottimizzazione K-values...', 'info');
    
    // Crea EventSource per streaming
    eventSourceRef.current = new EventSource('/api/admin/stream-k-optimization');
    
    eventSourceRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleStreamData(data);
      } catch (error) {
        console.error('Errore parsing stream data:', error);
        addLog(`❌ Errore parsing: ${error.message}`, 'error');
      }
    };
    
    eventSourceRef.current.onerror = (error) => {
      console.error('Errore EventSource:', error);
      addLog('❌ Connessione streaming interrotta', 'error');
      setIsOptimizing(false);
      setStatus('error');
    };
  };

  const handleStreamData = (data) => {
    switch (data.type) {
      case 'status':
        addLog(`📊 ${data.message}`, 'info');
        if (data.progress) setProgress(data.progress);
        break;
        
      case 'matrix_info':
        setMatrixInfo(data);
        addLog(`📈 Matrice: ${data.shape[0]}×${data.shape[1]} (densità: ${(data.density * 100).toFixed(2)}%)`, 'info');
        break;
        
      case 'phase':
        setCurrentPhase(data.phase);
        addLog(`🎯 ${data.message}`, 'phase');
        if (data.progress) setProgress(data.progress);
        break;
        
      case 'k_svd_result':
        setKSvdResults(prev => [...prev, data]);
        if (data.is_best) {
          addLog(`✨ Nuovo K-SVD migliore: ${data.k} (score: ${data.composite_score.toFixed(4)})`, 'success');
        } else {
          addLog(`📊 K-SVD ${data.k}: score ${data.composite_score.toFixed(4)}`, 'info');
        }
        setProgress(data.progress);
        break;
        
      case 'k_cluster_result':
        setKClusterResults(prev => [...prev, data]);
        if (data.is_best) {
          addLog(`✨ Nuovo K-Cluster migliore: ${data.k} (score: ${data.composite_score.toFixed(4)})`, 'success');
        } else {
          addLog(`🎯 K-Cluster ${data.k}: score ${data.composite_score.toFixed(4)}`, 'info');
        }
        setProgress(data.progress);
        break;
        
      case 'k_svd_optimal':
        setOptimalResults(prev => ({...prev, k_svd: data.optimal_k, svd_score: data.score}));
        addLog(`🏆 K-SVD OTTIMALE: ${data.optimal_k}`, 'success');
        break;
        
      case 'k_cluster_optimal':
        setOptimalResults(prev => ({...prev, k_cluster: data.optimal_k, cluster_score: data.score}));
        addLog(`🏆 K-Cluster OTTIMALE: ${data.optimal_k}`, 'success');
        break;
        
      case 'completed':
        addLog(`🎉 ${data.message}`, 'success');
        addLog(`📊 Risultati finali: K-SVD=${data.final_k_svd}, K-Cluster=${data.final_k_cluster}`, 'success');
        setProgress(100);
        setStatus('completed');
        setIsOptimizing(false);
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }
        break;
        
      case 'error':
        addLog(`❌ ${data.message}`, 'error');
        setStatus('error');
        setIsOptimizing(false);
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }
        break;
        
      case 'warning':
        addLog(`⚠️ ${data.message}`, 'warning');
        break;
        
      default:
        console.log('Tipo messaggio sconosciuto:', data);
    }
  };

  const stopOptimization = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    setIsOptimizing(false);
    setStatus('stopped');
    addLog('⏹️ Ottimizzazione interrotta dall\'utente', 'warning');
  };

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const clearLogs = () => {
    setLogs([]);
  };

  return (
    <div className="k-optimization-monitor">
      <div className="monitor-header">
        <h2>🎯 Monitoraggio Ottimizzazione K-Values</h2>
        <div className="monitor-controls">
          <button 
            onClick={startOptimization}
            disabled={isOptimizing}
            className={`btn-start ${isOptimizing ? 'disabled' : ''}`}
          >
            {isOptimizing ? '🔄 In corso...' : '🚀 Avvia Ottimizzazione'}
          </button>
          
          {isOptimizing && (
            <button onClick={stopOptimization} className="btn-stop">
              ⏹️ Interrompi
            </button>
          )}
          
          <button onClick={clearLogs} className="btn-clear">
            🗑️ Pulisci Log
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="progress-section">
        <div className="progress-info">
          <span className="progress-label">Progresso: {progress}%</span>
          {currentPhase && (
            <span className="current-phase">
              Fase: {currentPhase === 'k_svd' ? '🧮 K-SVD' : '🎯 K-Cluster'}
            </span>
          )}
        </div>
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Matrix Info */}
      {matrixInfo && (
        <div className="matrix-info">
          <h3>📊 Informazioni Matrice</h3>
          <div className="info-grid">
            <div className="info-item">
              <span className="label">Dimensioni:</span>
              <span className="value">{matrixInfo.shape[0]} × {matrixInfo.shape[1]}</span>
            </div>
            <div className="info-item">
              <span className="label">Densità:</span>
              <span className="value">{(matrixInfo.density * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* Results Grid */}
      <div className="results-grid">
        
        {/* K-SVD Results */}
        <div className="results-panel">
          <h3>🧮 Risultati K-SVD</h3>
          {kSvdResults.length > 0 ? (
            <div className="results-chart">
              <div className="chart-header">
                <span>K</span>
                <span>Varianza</span>
                <span>Efficienza</span>
                <span>Score</span>
                <span>Migliore</span>
              </div>
              {kSvdResults.map((result, idx) => (
                <div 
                  key={idx} 
                  className={`chart-row ${result.is_best ? 'best' : ''}`}
                >
                  <span className="k-value">{result.k}</span>
                  <span className="metric">{result.explained_variance.toFixed(3)}</span>
                  <span className="metric">{result.efficiency.toFixed(3)}</span>
                  <span className="metric">{result.composite_score.toFixed(4)}</span>
                  <span className="best-indicator">{result.is_best ? '✨' : ''}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-data">Nessun risultato K-SVD</div>
          )}
        </div>

        {/* K-Cluster Results */}
        <div className="results-panel">
          <h3>🎯 Risultati K-Cluster</h3>
          {kClusterResults.length > 0 ? (
            <div className="results-chart">
              <div className="chart-header">
                <span>K</span>
                <span>Silhouette</span>
                <span>Balance</span>
                <span>Score</span>
                <span>Migliore</span>
              </div>
              {kClusterResults.map((result, idx) => (
                <div 
                  key={idx} 
                  className={`chart-row ${result.is_best ? 'best' : ''}`}
                >
                  <span className="k-value">{result.k}</span>
                  <span className="metric">{result.silhouette_score.toFixed(3)}</span>
                  <span className="metric">{result.balance.toFixed(3)}</span>
                  <span className="metric">{result.composite_score.toFixed(4)}</span>
                  <span className="best-indicator">{result.is_best ? '✨' : ''}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-data">Nessun risultato K-Cluster</div>
          )}
        </div>
      </div>

      {/* Optimal Results Summary */}
      {Object.keys(optimalResults).length > 0 && (
        <div className="optimal-results">
          <h3>🏆 Risultati Ottimali</h3>
          <div className="optimal-grid">
            {optimalResults.k_svd && (
              <div className="optimal-item">
                <div className="optimal-title">🧮 K-SVD Ottimale</div>
                <div className="optimal-value">{optimalResults.k_svd}</div>
                <div className="optimal-score">Score: {optimalResults.svd_score?.toFixed(4)}</div>
              </div>
            )}
            {optimalResults.k_cluster && (
              <div className="optimal-item">
                <div className="optimal-title">🎯 K-Cluster Ottimale</div>
                <div className="optimal-value">{optimalResults.k_cluster}</div>
                <div className="optimal-score">Score: {optimalResults.cluster_score?.toFixed(4)}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Live Log */}
      <div className="log-section">
        <div className="log-header">
          <h3>📝 Log Real-time</h3>
          <span className="log-count">{logs.length} messaggi</span>
        </div>
        <div className="log-container">
          {logs.map((log, idx) => (
            <div key={idx} className={`log-entry ${log.type}`}>
              <span className="log-timestamp">{log.timestamp}</span>
              <span className="log-message">{log.message}</span>
            </div>
          ))}
          {logs.length === 0 && (
            <div className="no-logs">Nessun log disponibile</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default KOptimizationMonitor;