import React, { useState, useEffect } from 'react';
import './TMDBConfig.css';

const TMDBConfig = () => {
  const [config, setConfig] = useState({
    api_key: '',
    use_tmdb_training: true,
    use_tmdb_testing: false
  });
  
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    fetchTMDBStatus();
  }, []);

  const fetchTMDBStatus = async () => {
    try {
      const response = await fetch('/api/admin/tmdb-status');
      const data = await response.json();
      setStatus(data);
      
      if (data.tmdb_configured) {
        setConfig(prev => ({
          ...prev,
          use_tmdb_training: data.use_tmdb_training,
          use_tmdb_testing: data.use_tmdb_testing
        }));
      }
    } catch (error) {
      console.error('Error fetching TMDB status:', error);
    }
  };

  const handleConfigSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    
    try {
      const response = await fetch('/api/admin/configure-tmdb', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      
      const result = await response.json();
      
      if (response.ok) {
        setMessage({ type: 'success', text: result.message });
        fetchTMDBStatus();
      } else {
        setMessage({ type: 'error', text: result.detail });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Errore di connessione' });
    } finally {
      setLoading(false);
    }
  };

  const handleClearCache = async () => {
    if (!window.confirm('Vuoi davvero pulire la cache TMDB?')) return;
    
    setLoading(true);
    
    try {
      const response = await fetch('/api/admin/clear-tmdb-cache', {
        method: 'DELETE'
      });
      
      const result = await response.json();
      
      if (response.ok) {
        setMessage({ type: 'success', text: result.message });
      } else {
        setMessage({ type: 'error', text: result.detail });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Errore di connessione' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tmdb-config">
      <div className="tmdb-header">
        <h2>🎬 Configurazione TMDB</h2>
        <p>Configura l'integrazione con The Movie Database per training avanzato</p>
      </div>

      {/* Status attuale */}
      {status && (
        <div className="tmdb-status">
          <h3>📊 Stato Attuale</h3>
          <div className="status-grid">
            <div className="status-item">
              <span className="label">TMDB Configurato:</span>
              <span className={`value ${status.tmdb_configured ? 'enabled' : 'disabled'}`}>
                {status.tmdb_configured ? '✅ Sì' : '❌ No'}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Training Source:</span>
              <span className="value">{status.training_source}</span>
            </div>
            <div className="status-item">
              <span className="label">TMDB Training:</span>
              <span className={`value ${status.use_tmdb_training ? 'enabled' : 'disabled'}`}>
                {status.use_tmdb_training ? '✅ Abilitato' : '❌ Disabilitato'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Configurazione */}
      <form onSubmit={handleConfigSubmit} className="tmdb-form">
        <h3>⚙️ Configurazione</h3>
        
        <div className="form-group">
          <label htmlFor="api_key">
            🔑 TMDB API Key
            <span className="required">*</span>
          </label>
          <input
            type="password"
            id="api_key"
            value={config.api_key}
            onChange={(e) => setConfig(prev => ({ ...prev, api_key: e.target.value }))}
            placeholder="Inserisci la tua TMDB API Key"
            required
          />
          <small>Ottieni la tua API key da <a href="https://www.themoviedb.org/settings/api" target="_blank" rel="noopener noreferrer">TMDB</a></small>
        </div>

        <div className="form-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={config.use_tmdb_training}
              onChange={(e) => setConfig(prev => ({ ...prev, use_tmdb_training: e.target.checked }))}
            />
            <span className="checkmark">🎯</span>
            Usa TMDB per Training
            <small>Addestra il modello ML su dati TMDB (milioni di utenti)</small>
          </label>
        </div>

        <div className="form-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={config.use_tmdb_testing}
              onChange={(e) => setConfig(prev => ({ ...prev, use_tmdb_testing: e.target.checked }))}
            />
            <span className="checkmark">🧪</span>
            Usa TMDB per Testing
            <small>Testa anche su dati TMDB (oltre agli utenti AFlix)</small>
          </label>
        </div>

        <div className="form-actions">
          <button 
            type="submit" 
            className="btn-primary"
            disabled={loading}
          >
            {loading ? '🔄 Configurando...' : '💾 Salva Configurazione'}
          </button>
          
          <button 
            type="button"
            className="btn-secondary"
            onClick={handleClearCache}
            disabled={loading || !status?.tmdb_configured}
          >
            🗑️ Pulisci Cache
          </button>
        </div>
      </form>

      {/* Messaggio */}
      {message && (
        <div className={`message ${message.type}`}>
          {message.type === 'success' ? '✅' : '❌'} {message.text}
        </div>
      )}

      {/* Info training ibrido */}
      <div className="training-info">
        <h3>🔬 Training Ibrido</h3>
        <div className="info-box">
          <h4>🎯 Come funziona:</h4>
          <ol>
            <li><strong>Training:</strong> Il modello viene addestrato su milioni di rating TMDB</li>
            <li><strong>Testing:</strong> Le predizioni vengono testate su utenti AFlix reali</li>
            <li><strong>Vantaggi:</strong> Modello robusto anche con pochi utenti AFlix</li>
          </ol>
          
          <h4>📈 Benefici:</h4>
          <ul>
            <li>Raccomandazioni immediate per nuovi utenti</li>
            <li>Migliore qualità predizioni</li>
            <li>Resistente al cold start problem</li>
            <li>Aggiornamento automatico dataset</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default TMDBConfig;