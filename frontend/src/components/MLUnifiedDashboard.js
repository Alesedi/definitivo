import React, { useEffect, useRef } from 'react';
import styled from 'styled-components';
import { theme } from '../styles/theme';
import { 
  FaBrain, FaChartLine, FaCogs, FaStar, FaPlay, FaHistory, 
  FaStop, FaTrash, FaSync, FaRocket 
} from 'react-icons/fa';
import { useML } from '../contexts/MLContext';
import ClusteringChart from './ClusteringChart';

const UnifiedContainer = styled.div`
  padding: ${theme.spacing.xl};
  background: white;
`;

const SectionGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${theme.spacing.xl};
  margin-bottom: ${theme.spacing.xl};
  
  @media (max-width: 1200px) {
    grid-template-columns: 1fr;
  }
`;

const FullWidthSection = styled.div`
  grid-column: 1 / -1;
  background: ${theme.colors.backgroundLight};
  border-radius: ${theme.borderRadius.lg};
  padding: ${theme.spacing.xl};
  box-shadow: ${theme.boxShadow.base};
  margin-bottom: ${theme.spacing.xl};
`;

const Section = styled.div`
  background: ${theme.colors.backgroundLight};
  border-radius: ${theme.borderRadius.lg};
  padding: ${theme.spacing.xl};
  box-shadow: ${theme.boxShadow.base};
`;

const SectionHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${theme.spacing.lg};
  border-bottom: 2px solid ${theme.colors.primaryLighter};
  padding-bottom: ${theme.spacing.md};
`;

const SectionTitle = styled.h2`
  color: ${theme.colors.primary};
  font-size: ${theme.fontSize.xl};
  margin: 0;
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
`;

const Button = styled.button`
  background: ${props => props.variant === 'danger' ? theme.colors.error : 
              props.variant === 'warning' ? theme.colors.warning : 
              props.variant === 'success' ? theme.colors.success : theme.colors.secondary};
  color: white;
  border: none;
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  border-radius: ${theme.borderRadius.md};
  font-size: ${theme.fontSize.sm};
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: ${theme.spacing.xs};
  transition: all 0.3s ease;
  
  &:hover {
    opacity: 0.8;
    transform: translateY(-1px);
  }
  
  &:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: ${theme.spacing.sm};
  flex-wrap: wrap;
`;

const StatusGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${theme.spacing.md};
  margin-bottom: ${theme.spacing.lg};
`;

const MetricCard = styled.div`
  background: white;
  padding: ${theme.spacing.lg};
  border-radius: ${theme.borderRadius.md};
  text-align: center;
  border: 1px solid ${theme.colors.primaryLighter};
`;

const MetricValue = styled.div`
  font-size: ${theme.fontSize['2xl']};
  font-weight: bold;
  color: ${theme.colors.primary};
  margin-bottom: ${theme.spacing.xs};
`;

const MetricLabel = styled.div`
  font-size: ${theme.fontSize.sm};
  color: #666;
`;

const StatusBadge = styled.span`
  padding: ${theme.spacing.xs} ${theme.spacing.sm};
  background: ${props => props.isReady ? theme.colors.successLighter || '#d4edda' : '#fef3c7'};
  color: ${props => props.isReady ? theme.colors.success : '#92400e'};
  border-radius: ${theme.borderRadius.sm};
  font-size: ${theme.fontSize.sm};
  font-weight: bold;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 20px;
  background: ${theme.colors.primaryLighter};
  border-radius: ${theme.borderRadius.sm};
  overflow: hidden;
  margin: ${theme.spacing.sm} 0;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: ${theme.colors.secondary};
  width: ${props => props.progress}%;
  transition: width 0.3s ease;
`;

const LogContainer = styled.div`
  background: #1a1a1a;
  color: #fff;
  padding: ${theme.spacing.md};
  border-radius: ${theme.borderRadius.md};
  max-height: 300px;
  overflow-y: auto;
  font-family: 'Courier New', monospace;
  font-size: ${theme.fontSize.sm};
`;

const LogEntry = styled.div`
  margin-bottom: ${theme.spacing.xs};
  color: ${props => 
    props.type === 'error' ? '#ff6b6b' :
    props.type === 'success' ? '#51cf66' :
    props.type === 'warning' ? '#ffd43b' :
    props.type === 'phase' ? '#74c0fc' : '#fff'
  };
`;

const MovieCard = styled.div`
  display: flex;
  gap: ${theme.spacing.md};
  padding: ${theme.spacing.md};
  border: 1px solid ${theme.colors.primaryLighter};
  border-radius: ${theme.borderRadius.md};
  margin-bottom: ${theme.spacing.md};
  background: white;
  transition: transform 0.2s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: ${theme.boxShadow.base};
  }
`;

const MoviePoster = styled.img`
  width: 60px;
  height: 90px;
  object-fit: cover;
  border-radius: ${theme.borderRadius.sm};
  background: ${theme.colors.primaryLighter};
`;

const MovieInfo = styled.div`
  flex: 1;
`;

const MovieTitle = styled.h4`
  color: ${theme.colors.primary};
  margin: 0 0 ${theme.spacing.xs} 0;
  font-size: ${theme.fontSize.sm};
`;

const MovieRating = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.xs};
  color: ${theme.colors.secondary};
  font-weight: bold;
  margin-bottom: ${theme.spacing.xs};
  font-size: ${theme.fontSize.sm};
`;

const ResultsTable = styled.div`
  background: white;
  border-radius: ${theme.borderRadius.md};
  overflow: hidden;
  margin-top: ${theme.spacing.md};
`;

const TableHeader = styled.div`
  display: grid;
  grid-template-columns: 1fr 2fr 2fr 2fr 1fr;
  background: ${theme.colors.primary};
  color: white;
  padding: ${theme.spacing.sm};
  font-weight: bold;
  font-size: ${theme.fontSize.sm};
`;

const TableRow = styled.div`
  display: grid;
  grid-template-columns: 1fr 2fr 2fr 2fr 1fr;
  padding: ${theme.spacing.sm};
  border-bottom: 1px solid ${theme.colors.primaryLighter};
  background: ${props => props.isBest ? theme.colors.successLighter || '#d4edda' : 'white'};
  font-size: ${theme.fontSize.sm};
  
  &:hover {
    background: ${theme.colors.primaryLighter};
  }
`;

const ErrorMessage = styled.div`
  background: #fee2e2;
  color: #dc2626;
  padding: ${theme.spacing.md};
  border-radius: ${theme.borderRadius.md};
  text-align: center;
  margin: ${theme.spacing.md} 0;
`;

const MLUnifiedDashboard = ({ user }) => {
  const {
    // State
    modelStatus, isTraining, recommendations, userHistory, evaluation, clustering,
    logs, isAutoRefresh, kOptimization, loading, error,
    
    // Actions
    addLog, clearLogs, setAutoRefresh,
    
    // API Functions
    fetchModelStatus, trainModel, fetchRecommendations, fetchUserHistory,
    fetchEvaluation, fetchClustering,
    
    // K-Optimization Functions
    startKOptimization, stopKOptimization, setKOptimizationProgress,
    setKOptimizationPhase, addKSvdResult, addKClusterResult, 
    setOptimalResults, setMatrixInfo
  } = useML();

  const eventSourceRef = useRef(null);
  const timeoutRef = useRef(null);

  // Funzione per immagini TMDB
  const getTMDBImageUrl = (posterPath) => {
    if (!posterPath) {
      const placeholderSVG = `data:image/svg+xml,${encodeURIComponent(`
        <svg width="60" height="90" xmlns="http://www.w3.org/2000/svg">
          <rect width="100%" height="100%" fill="#2a5298"/>
          <text x="50%" y="50%" text-anchor="middle" fill="white" font-size="8" font-family="Arial">
            No Image
          </text>
        </svg>
      `)}`;
      return placeholderSVG;
    }
    return posterPath.startsWith('http') ? posterPath : `https://image.tmdb.org/t/p/w200${posterPath}`;
  };

  // Auto-refresh effect
  useEffect(() => {
    let interval;
    if (isAutoRefresh) {
      interval = setInterval(() => {
        fetchModelStatus();
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [isAutoRefresh, fetchModelStatus]);

  // Caricamento iniziale
  useEffect(() => {
    fetchModelStatus();
    // Rimuoviamo il log duplicato - inizializzazione silenziosa
  }, [fetchModelStatus]);

  // Carica clustering automaticamente quando il modello è addestrato
  useEffect(() => {
    if (modelStatus?.is_trained && !clustering) {
      fetchClustering();
    }
  }, [modelStatus?.is_trained, clustering, fetchClustering]);

  // Gestione K-Optimization Streaming
  const handleKOptimization = async () => {
    if (kOptimization.isOptimizing) return;
    
    addLog('🚀 Avvio ottimizzazione K...', 'info');
    startKOptimization();
    
    // Crea EventSource per streaming
    eventSourceRef.current = new EventSource('/api/admin/stream-k-optimization');
    
    // Timeout di sicurezza (15 minuti)
    timeoutRef.current = setTimeout(() => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        addLog('⏱️ Timeout ottimizzazione K (15 minuti)', 'warning');
        stopKOptimization();
      }
    }, 15 * 60 * 1000);
    
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
      
      // Se la connessione è chiusa dal server (normale fine stream)
      if (eventSourceRef.current.readyState === EventSource.CLOSED) {
        addLog('📡 Streaming completato', 'info');
      } else {
        // Errore di connessione
        addLog('❌ Connessione streaming interrotta. Verifica che il backend sia attivo.', 'error');
        addLog('💡 Controlla: http://127.0.0.1:8005/docs per verificare lo stato del server', 'warning');
      }
      
      stopKOptimization();
    };
  };

  const handleStreamData = (data) => {
    switch (data.type) {
      case 'status':
        addLog(`📊 ${data.message}`, 'info');
        if (data.progress) setKOptimizationProgress(data.progress);
        break;
        
      case 'matrix_info':
        setMatrixInfo(data);
        addLog(`📈 Matrice: ${data.shape[0]}×${data.shape[1]} (densità: ${(data.density * 100).toFixed(2)}%)`, 'info');
        break;
        
      case 'phase':
        setKOptimizationPhase(data.phase);
        addLog(`🎯 ${data.message}`, 'phase');
        if (data.progress) setKOptimizationProgress(data.progress);
        break;
        
      case 'k_svd_result':
        addKSvdResult(data);
        if (data.is_best) {
          addLog(`✨ Nuovo K-SVD migliore: ${data.k} (score: ${data.composite_score.toFixed(4)})`, 'success');
        } else {
          addLog(`📊 K-SVD ${data.k}: score ${data.composite_score.toFixed(4)}`, 'info');
        }
        setKOptimizationProgress(data.progress);
        break;
        
      case 'k_cluster_result':
        addKClusterResult(data);
        if (data.is_best) {
          addLog(`✨ Nuovo K-Cluster migliore: ${data.k} (score: ${data.composite_score.toFixed(4)})`, 'success');
        } else {
          addLog(`🎯 K-Cluster ${data.k}: score ${data.composite_score.toFixed(4)}`, 'info');
        }
        setKOptimizationProgress(data.progress);
        break;
        
      case 'k_svd_optimal':
        console.log('K-SVD Optimal received:', data);
        setOptimalResults(prev => ({ ...prev, k_svd: data.optimal_k, svd_score: data.score }));
        if (data.note) {
          addLog(`🎯 K-SVD SUGGERITO: ${data.optimal_k} (${data.note})`, 'info');
        } else {
          addLog(`🏆 K-SVD OTTIMALE: ${data.optimal_k}`, 'success');
        }
        break;
        
      case 'k_cluster_optimal':
        setOptimalResults(prev => ({ ...prev, k_cluster: data.optimal_k, cluster_score: data.score }));
        addLog(`🏆 K-Cluster OTTIMALE: ${data.optimal_k}`, 'success');
        break;
        
      case 'completed':
        addLog(`🎉 ${data.message}`, 'success');
        addLog(`📊 Risultati finali: K-SVD=${data.final_k_svd}, K-Cluster=${data.final_k_cluster}`, 'success');
        setKOptimizationProgress(100);
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
        stopKOptimization(false); // Completamento naturale, non interruzione utente
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }
        break;
        
      case 'error':
        addLog(`❌ ${data.message}`, 'error');
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
        stopKOptimization();
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

  const handleStopKOptimization = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    stopKOptimization();
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return (
    <UnifiedContainer>
      {error && <ErrorMessage>{error}</ErrorMessage>}
      
      {/* Sezione Controlli Principali */}
      <FullWidthSection>
        <SectionHeader>
          <SectionTitle>
            <FaRocket />
            Controlli Principali ML
          </SectionTitle>
          <ButtonGroup>
            <Button 
              onClick={trainModel} 
              disabled={isTraining || loading}
              variant="success"
            >
              <FaRocket />
              {isTraining ? 'Training...' : 'Addestra Modello'}
            </Button>
            <Button 
              onClick={handleKOptimization} 
              disabled={kOptimization.isOptimizing}
              variant="warning"
            >
              <FaCogs />
              {kOptimization.isOptimizing ? 'Ottimizzando...' : 'Ottimizza K'}
            </Button>
            {kOptimization.isOptimizing && (
              <Button onClick={handleStopKOptimization} variant="danger">
                <FaStop />
                Ferma
              </Button>
            )}
            <Button 
              onClick={() => setAutoRefresh(!isAutoRefresh)}
              variant={isAutoRefresh ? 'success' : 'secondary'}
            >
              <FaSync />
              Auto-refresh {isAutoRefresh ? 'ON' : 'OFF'}
            </Button>
            <Button onClick={clearLogs} variant="secondary">
              <FaTrash />
              Pulisci Log
            </Button>
          </ButtonGroup>
        </SectionHeader>

        {/* Status Grid */}
        <StatusGrid>
          <MetricCard>
            <MetricValue>
              <StatusBadge isReady={modelStatus?.is_trained}>
                {modelStatus?.is_trained ? '✅ ATTIVO' : '❌ INATTIVO'}
              </StatusBadge>
            </MetricValue>
            <MetricLabel>Stato Modello</MetricLabel>
          </MetricCard>

          <MetricCard>
            <MetricValue>
              {modelStatus?.n_components || 'N/A'}
            </MetricValue>
            <MetricLabel>K-SVD Attuale</MetricLabel>
          </MetricCard>

          <MetricCard>
            <MetricValue>
              {modelStatus?.explained_variance 
                ? `${(modelStatus.explained_variance * 100).toFixed(1)}%`
                : 'N/A'}
            </MetricValue>
            <MetricLabel>Varianza Spiegata</MetricLabel>
          </MetricCard>

          <MetricCard>
            <MetricValue>
              {kOptimization.optimalResults.k_svd || 'N/A'}
            </MetricValue>
            <MetricLabel>K-SVD Ottimale</MetricLabel>
          </MetricCard>

          <MetricCard>
            <MetricValue>
              {kOptimization.optimalResults.k_cluster || 'N/A'}
            </MetricValue>
            <MetricLabel>K-Cluster Ottimale</MetricLabel>
          </MetricCard>
        </StatusGrid>

        {/* Progress Bar K-Optimization */}
        {kOptimization.isOptimizing && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Progresso K-Optimization: {kOptimization.progress}%</span>
              {kOptimization.currentPhase && (
                <span>Fase: {kOptimization.currentPhase === 'k_svd' ? '🧮 K-SVD' : '🎯 K-Cluster'}</span>
              )}
            </div>
            <ProgressBar>
              <ProgressFill progress={kOptimization.progress} />
            </ProgressBar>
          </div>
        )}
      </FullWidthSection>

      <SectionGrid>
        {/* Sezione Raccomandazioni */}
        <Section>
          <SectionHeader>
            <SectionTitle>
              <FaStar />
              Raccomandazioni
            </SectionTitle>
            <ButtonGroup>
              <Button 
                onClick={() => fetchRecommendations(user.id)}
                disabled={loading || !modelStatus?.is_trained}
              >
                <FaPlay />
                Genera
              </Button>
              <Button onClick={() => fetchUserHistory(user.id)} disabled={loading}>
                <FaHistory />
                Storico
              </Button>
            </ButtonGroup>
          </SectionHeader>
          
          {recommendations.length > 0 && (
            <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
              {recommendations.slice(0, 10).map((movie, index) => (
                <MovieCard key={index}>
                  <MoviePoster 
                    src={getTMDBImageUrl(movie.poster_url)} 
                    alt={movie.title}
                    onError={(e) => {
                      e.target.src = getTMDBImageUrl(null);
                    }}
                  />
                  <MovieInfo>
                    <MovieTitle>{movie.title}</MovieTitle>
                    <MovieRating>
                      <FaStar />
                      {movie.predicted_rating.toFixed(2)} (previsto)
                    </MovieRating>
                  </MovieInfo>
                </MovieCard>
              ))}
            </div>
          )}
        </Section>

        {/* Sezione Valutazione */}
        <Section>
          <SectionHeader>
            <SectionTitle>
              <FaChartLine />
              Performance
            </SectionTitle>
            <ButtonGroup>
              <Button 
                onClick={fetchEvaluation}
                disabled={loading || !modelStatus?.is_trained}
              >
                <FaChartLine />
                Valuta
              </Button>
              <Button 
                onClick={fetchClustering}
                disabled={loading || !modelStatus?.is_trained}
              >
                <FaBrain />
                Clustering
              </Button>
            </ButtonGroup>
          </SectionHeader>
          
          {evaluation && !evaluation.error && (
            <StatusGrid>
              <MetricCard>
                <MetricValue>{evaluation.rmse?.toFixed(4) || 'N/A'}</MetricValue>
                <MetricLabel>RMSE</MetricLabel>
              </MetricCard>
              <MetricCard>
                <MetricValue>{evaluation.mae?.toFixed(4) || 'N/A'}</MetricValue>
                <MetricLabel>MAE</MetricLabel>
              </MetricCard>
            </StatusGrid>
          )}

          {/* Grafico Clustering */}
          {clustering && !clustering.error ? (
            <div style={{ marginTop: '20px' }}>
              <div style={{ 
                padding: '10px', 
                backgroundColor: '#e3f2fd', 
                borderRadius: '6px', 
                marginBottom: '15px',
                fontSize: '14px',
                color: '#1565c0'
              }}>
                💡 <strong>Clustering Test Set:</strong> Mostra solo i film votati dagli utenti AFlix, 
                raggruppati nello spazio delle caratteristiche SVD per scoprire pattern di preferenze.
              </div>
              <div style={{ height: '450px' }}>
                <ClusteringChart data={clustering} />
              </div>
            </div>
          ) : (
            <div style={{ 
              marginTop: '20px', 
              padding: '20px', 
              textAlign: 'center', 
              color: '#666',
              backgroundColor: '#f8f9fa',
              borderRadius: '8px'
            }}>
              {clustering?.error ? (
                <div>❌ Errore nel caricamento clustering: {clustering.error}</div>
              ) : (
                <div>📊 Nessun dato di clustering disponibile. Clicca "Mostra Clustering" per generarlo.</div>
              )}
            </div>
          )}
        </Section>
      </SectionGrid>

      {/* Sezione Risultati K-Optimization */}
      {(kOptimization.kSvdResults.length > 0 || kOptimization.kClusterResults.length > 0) && (
        <FullWidthSection>
          <SectionHeader>
            <SectionTitle>
              <FaCogs />
              Risultati K-Optimization
            </SectionTitle>
          </SectionHeader>

          <SectionGrid>
            {/* K-SVD Results */}
            {kOptimization.kSvdResults.length > 0 && (
              <div>
                <h4>🧮 Risultati K-SVD</h4>
                <ResultsTable>
                  <TableHeader>
                    <span>K</span>
                    <span>Varianza</span>
                    <span>Efficienza</span>
                    <span>Score</span>
                    <span>Best</span>
                  </TableHeader>
                  {kOptimization.kSvdResults.map((result, idx) => (
                    <TableRow key={idx} isBest={result.is_best}>
                      <span>{result.k}</span>
                      <span>{result.explained_variance.toFixed(3)}</span>
                      <span>{result.efficiency.toFixed(3)}</span>
                      <span>{result.composite_score.toFixed(4)}</span>
                      <span>{result.is_best ? '✨' : ''}</span>
                    </TableRow>
                  ))}
                </ResultsTable>
              </div>
            )}

            {/* K-Cluster Results */}
            {kOptimization.kClusterResults.length > 0 && (
              <div>
                <h4>🎯 Risultati K-Cluster</h4>
                <ResultsTable>
                  <TableHeader>
                    <span>K</span>
                    <span>Silhouette</span>
                    <span>Balance</span>
                    <span>Score</span>
                    <span>Best</span>
                  </TableHeader>
                  {kOptimization.kClusterResults.map((result, idx) => (
                    <TableRow key={idx} isBest={result.is_best}>
                      <span>{result.k}</span>
                      <span>{result.silhouette_score.toFixed(3)}</span>
                      <span>{result.balance.toFixed(3)}</span>
                      <span>{result.composite_score.toFixed(4)}</span>
                      <span>{result.is_best ? '✨' : ''}</span>
                    </TableRow>
                  ))}
                </ResultsTable>
              </div>
            )}
          </SectionGrid>
        </FullWidthSection>
      )}

      {/* Sezione Log */}
      <FullWidthSection>
        <SectionHeader>
          <SectionTitle>
            <FaChartLine />
            Log Real-time ML ({logs.length} messaggi)
          </SectionTitle>
        </SectionHeader>
        
        <LogContainer>
          {logs.length > 0 ? (
            logs.map((log) => (
              <LogEntry key={log.id} type={log.type}>
                [{log.timestamp}] {log.message}
              </LogEntry>
            ))
          ) : (
            <LogEntry>In attesa di operazioni ML...</LogEntry>
          )}
        </LogContainer>
      </FullWidthSection>
    </UnifiedContainer>
  );
};

export default MLUnifiedDashboard;