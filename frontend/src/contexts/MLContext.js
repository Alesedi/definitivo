import React, { createContext, useContext, useReducer, useCallback } from 'react';
import api from '../services/api';

// Stato iniziale
const initialState = {
  // Stato del modello
  modelStatus: null,
  isTraining: false,
  
  // Raccomandazioni
  recommendations: [],
  userHistory: [],
  evaluation: null,
  clustering: null,
  
  // Monitor ML
  logs: [],
  isAutoRefresh: false,
  
  // K-Optimization
  kOptimization: {
    status: 'idle',
    progress: 0,
    currentPhase: null,
    kSvdResults: [],
    kClusterResults: [],
    optimalResults: {},
    matrixInfo: null,
    isOptimizing: false
  },
  
  // Stato generale
  loading: false,
  error: null
};

// Actions
const ML_ACTIONS = {
  // Model Status
  SET_MODEL_STATUS: 'SET_MODEL_STATUS',
  SET_TRAINING: 'SET_TRAINING',
  
  // Recommendations
  SET_RECOMMENDATIONS: 'SET_RECOMMENDATIONS',
  SET_USER_HISTORY: 'SET_USER_HISTORY',
  SET_EVALUATION: 'SET_EVALUATION',
  SET_CLUSTERING: 'SET_CLUSTERING',
  
  // Monitor ML
  ADD_LOG: 'ADD_LOG',
  CLEAR_LOGS: 'CLEAR_LOGS',
  SET_AUTO_REFRESH: 'SET_AUTO_REFRESH',
  
  // K-Optimization
  SET_K_OPTIMIZATION_STATUS: 'SET_K_OPTIMIZATION_STATUS',
  SET_K_OPTIMIZATION_PROGRESS: 'SET_K_OPTIMIZATION_PROGRESS',
  SET_K_OPTIMIZATION_PHASE: 'SET_K_OPTIMIZATION_PHASE',
  ADD_K_SVD_RESULT: 'ADD_K_SVD_RESULT',
  ADD_K_CLUSTER_RESULT: 'ADD_K_CLUSTER_RESULT',
  SET_OPTIMAL_RESULTS: 'SET_OPTIMAL_RESULTS',
  SET_MATRIX_INFO: 'SET_MATRIX_INFO',
  SET_K_OPTIMIZING: 'SET_K_OPTIMIZING',
  RESET_K_OPTIMIZATION: 'RESET_K_OPTIMIZATION',
  
  // General
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR'
};

// Reducer
const mlReducer = (state, action) => {
  switch (action.type) {
    case ML_ACTIONS.SET_MODEL_STATUS:
      return { ...state, modelStatus: action.payload };
      
    case ML_ACTIONS.SET_TRAINING:
      return { ...state, isTraining: action.payload };
      
    case ML_ACTIONS.SET_RECOMMENDATIONS:
      return { ...state, recommendations: action.payload };
      
    case ML_ACTIONS.SET_USER_HISTORY:
      return { ...state, userHistory: action.payload };
      
    case ML_ACTIONS.SET_EVALUATION:
      return { ...state, evaluation: action.payload };
      
    case ML_ACTIONS.SET_CLUSTERING:
      return { ...state, clustering: action.payload };
      
    case ML_ACTIONS.ADD_LOG:
      const newLog = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        message: action.payload.message,
        type: action.payload.type || 'info'
      };
      return {
        ...state,
        logs: [...state.logs.slice(-49), newLog] // Mantieni solo gli ultimi 50 log
      };
      
    case ML_ACTIONS.CLEAR_LOGS:
      return { ...state, logs: [] };
      
    case ML_ACTIONS.SET_AUTO_REFRESH:
      return { ...state, isAutoRefresh: action.payload };
      
    case ML_ACTIONS.SET_K_OPTIMIZATION_STATUS:
      return {
        ...state,
        kOptimization: { ...state.kOptimization, status: action.payload }
      };
      
    case ML_ACTIONS.SET_K_OPTIMIZATION_PROGRESS:
      return {
        ...state,
        kOptimization: { ...state.kOptimization, progress: action.payload }
      };
      
    case ML_ACTIONS.SET_K_OPTIMIZATION_PHASE:
      return {
        ...state,
        kOptimization: { ...state.kOptimization, currentPhase: action.payload }
      };
      
    case ML_ACTIONS.ADD_K_SVD_RESULT:
      return {
        ...state,
        kOptimization: {
          ...state.kOptimization,
          kSvdResults: [...state.kOptimization.kSvdResults, action.payload]
        }
      };
      
    case ML_ACTIONS.ADD_K_CLUSTER_RESULT:
      return {
        ...state,
        kOptimization: {
          ...state.kOptimization,
          kClusterResults: [...state.kOptimization.kClusterResults, action.payload]
        }
      };
      
    case ML_ACTIONS.SET_OPTIMAL_RESULTS:
      return {
        ...state,
        kOptimization: {
          ...state.kOptimization,
          optimalResults: { ...state.kOptimization.optimalResults, ...action.payload }
        }
      };
      
    case ML_ACTIONS.SET_MATRIX_INFO:
      return {
        ...state,
        kOptimization: { ...state.kOptimization, matrixInfo: action.payload }
      };
      
    case ML_ACTIONS.SET_K_OPTIMIZING:
      return {
        ...state,
        kOptimization: { ...state.kOptimization, isOptimizing: action.payload }
      };
      
    case ML_ACTIONS.RESET_K_OPTIMIZATION:
      return {
        ...state,
        kOptimization: {
          ...initialState.kOptimization,
          optimalResults: state.kOptimization.optimalResults, // Preserva i risultati precedenti
          isOptimizing: false
        }
      };
      
    case ML_ACTIONS.SET_LOADING:
      return { ...state, loading: action.payload };
      
    case ML_ACTIONS.SET_ERROR:
      return { ...state, error: action.payload };
      
    case ML_ACTIONS.CLEAR_ERROR:
      return { ...state, error: null };
      
    default:
      return state;
  }
};

// Context
const MLContext = createContext();

// Provider Component
export const MLProvider = ({ children }) => {
  const [state, dispatch] = useReducer(mlReducer, initialState);

  // Utility function per aggiungere log
  const addLog = useCallback((message, type = 'info') => {
    dispatch({
      type: ML_ACTIONS.ADD_LOG,
      payload: { message, type }
    });
  }, []);

  // API Functions
  const fetchModelStatus = useCallback(async () => {
    try {
      const response = await api.get('/api/recommendations/status');
      dispatch({ type: ML_ACTIONS.SET_MODEL_STATUS, payload: response.data });
      return response.data;
    } catch (error) {
      console.error('Error fetching model status:', error);
      dispatch({ 
        type: ML_ACTIONS.SET_ERROR, 
        payload: 'Errore nel recupero dello stato del modello' 
      });
      addLog(`❌ Errore connessione backend: ${error.message}`, 'error');
      return null;
    }
  }, [addLog]);

  const trainModel = useCallback(async () => {
    dispatch({ type: ML_ACTIONS.SET_TRAINING, payload: true });
    dispatch({ type: ML_ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ML_ACTIONS.CLEAR_ERROR });
    addLog('🚀 Avvio training del modello...', 'info');
    
    try {
      const response = await api.get('/api/recommendations/train-sync');
      const result = response.data;
      
      addLog('✅ Training completato con successo!', 'success');
      
      // Aggiorna status del modello che mostrerà le statistiche aggiornate
      setTimeout(() => {
        fetchModelStatus().then(status => {
          if (status) {
            addLog(`📊 Statistiche: ${status.total_ratings || 0} rating processati`, 'info');
            addLog(`🎯 K utilizzato: ${status.actual_k_used || 'N/A'}`, 'info');
            addLog(`📈 Varianza spiegata: ${(status.explained_variance * 100 || 0).toFixed(1)}%`, 'info');
          }
        });
      }, 500);
      
      return result;
    } catch (error) {
      console.error('Error training model:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Errore sconosciuto';
      dispatch({ 
        type: ML_ACTIONS.SET_ERROR, 
        payload: `Errore nel training del modello: ${errorMessage}` 
      });
      addLog(`❌ Errore training: ${errorMessage}`, 'error');
      throw error;
    } finally {
      dispatch({ type: ML_ACTIONS.SET_TRAINING, payload: false });
      dispatch({ type: ML_ACTIONS.SET_LOADING, payload: false });
    }
  }, [addLog, fetchModelStatus]);

  const fetchRecommendations = useCallback(async (userId, topN = 10) => {
    dispatch({ type: ML_ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ML_ACTIONS.CLEAR_ERROR });
    
    try {
      const response = await api.get(`/api/recommendations/user/${userId}?top_n=${topN}`);
      dispatch({ type: ML_ACTIONS.SET_RECOMMENDATIONS, payload: response.data });
      addLog(`✅ Caricate ${response.data.length} raccomandazioni`, 'success');
      return response.data;
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      const errorMessage = error.response?.data?.detail || error.message;
      dispatch({ 
        type: ML_ACTIONS.SET_ERROR, 
        payload: `Errore nel recupero delle raccomandazioni: ${errorMessage}` 
      });
      addLog(`❌ Errore raccomandazioni: ${errorMessage}`, 'error');
      throw error;
    } finally {
      dispatch({ type: ML_ACTIONS.SET_LOADING, payload: false });
    }
  }, [addLog]);

  const fetchUserHistory = useCallback(async (userId) => {
    dispatch({ type: ML_ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ML_ACTIONS.CLEAR_ERROR });
    
    try {
      const response = await api.get(`/api/recommendations/user/${userId}/history`);
      dispatch({ type: ML_ACTIONS.SET_USER_HISTORY, payload: response.data });
      addLog(`✅ Caricato storico: ${response.data.length} voti`, 'success');
      return response.data;
    } catch (error) {
      console.error('Error fetching user history:', error);
      const errorMessage = error.response?.data?.detail || error.message;
      dispatch({ 
        type: ML_ACTIONS.SET_ERROR, 
        payload: `Errore nel recupero dello storico: ${errorMessage}` 
      });
      addLog(`❌ Errore storico: ${errorMessage}`, 'error');
      throw error;
    } finally {
      dispatch({ type: ML_ACTIONS.SET_LOADING, payload: false });
    }
  }, [addLog]);

  const fetchEvaluation = useCallback(async () => {
    dispatch({ type: ML_ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ML_ACTIONS.CLEAR_ERROR });
    
    try {
      const response = await api.get('/api/recommendations/evaluation');
      dispatch({ type: ML_ACTIONS.SET_EVALUATION, payload: response.data });
      addLog('✅ Valutazione modello completata', 'success');
      return response.data;
    } catch (error) {
      console.error('Error fetching evaluation:', error);
      const errorMessage = error.response?.data?.detail || error.message;
      dispatch({ 
        type: ML_ACTIONS.SET_ERROR, 
        payload: `Errore nel recupero della valutazione: ${errorMessage}` 
      });
      addLog(`❌ Errore valutazione: ${errorMessage}`, 'error');
      throw error;
    } finally {
      dispatch({ type: ML_ACTIONS.SET_LOADING, payload: false });
    }
  }, [addLog]);

  const fetchClustering = useCallback(async () => {
    dispatch({ type: ML_ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ML_ACTIONS.CLEAR_ERROR });
    
    try {
      const response = await api.get('/api/recommendations/clustering');
      dispatch({ type: ML_ACTIONS.SET_CLUSTERING, payload: response.data });
      addLog('✅ Clustering caricato con successo', 'success');
      return response.data;
    } catch (error) {
      console.error('Error fetching clustering:', error);
      
      let errorMessage = 'Errore sconosciuto';
      if (error.code === 'NETWORK_ERROR' || !navigator.onLine) {
        errorMessage = 'Backend non raggiungibile. Verifica che il server sia attivo su http://127.0.0.1:8005';
      } else if (error.response?.status === 500) {
        errorMessage = 'Il modello ML non è ancora addestrato. Esegui prima il training.';
      } else {
        errorMessage = error.response?.data?.detail || error.message;
      }
      
      dispatch({ 
        type: ML_ACTIONS.SET_ERROR, 
        payload: `Errore nel recupero del clustering: ${errorMessage}` 
      });
      dispatch({ type: ML_ACTIONS.SET_CLUSTERING, payload: { error: errorMessage } });
      addLog(`❌ Errore clustering: ${errorMessage}`, 'error');
      throw error;
    } finally {
      dispatch({ type: ML_ACTIONS.SET_LOADING, payload: false });
    }
  }, [addLog]);

  // K-Optimization Functions
  const startKOptimization = useCallback(() => {
    dispatch({ type: ML_ACTIONS.RESET_K_OPTIMIZATION });
    dispatch({ type: ML_ACTIONS.SET_K_OPTIMIZING, payload: true });
    dispatch({ type: ML_ACTIONS.SET_K_OPTIMIZATION_STATUS, payload: 'running' });
    addLog('🚀 Avvio ottimizzazione K-values...', 'info');
  }, [addLog]);

  const stopKOptimization = useCallback((userTriggered = true) => {
    dispatch({ type: ML_ACTIONS.SET_K_OPTIMIZING, payload: false });
    dispatch({ type: ML_ACTIONS.SET_K_OPTIMIZATION_STATUS, payload: 'stopped' });
    if (userTriggered) {
      addLog('⏹️ Ottimizzazione interrotta dall\'utente', 'warning');
    }
  }, [addLog]);

  // Context value
  const value = {
    // State
    ...state,
    
    // Actions
    addLog,
    clearLogs: () => dispatch({ type: ML_ACTIONS.CLEAR_LOGS }),
    setAutoRefresh: (value) => dispatch({ type: ML_ACTIONS.SET_AUTO_REFRESH, payload: value }),
    setError: (error) => dispatch({ type: ML_ACTIONS.SET_ERROR, payload: error }),
    clearError: () => dispatch({ type: ML_ACTIONS.CLEAR_ERROR }),
    
    // API Functions
    fetchModelStatus,
    trainModel,
    fetchRecommendations,
    fetchUserHistory,
    fetchEvaluation,
    fetchClustering,
    
    // K-Optimization Functions
    startKOptimization,
    stopKOptimization,
    setKOptimizationProgress: (progress) => dispatch({ 
      type: ML_ACTIONS.SET_K_OPTIMIZATION_PROGRESS, 
      payload: progress 
    }),
    setKOptimizationPhase: (phase) => dispatch({ 
      type: ML_ACTIONS.SET_K_OPTIMIZATION_PHASE, 
      payload: phase 
    }),
    addKSvdResult: (result) => dispatch({ 
      type: ML_ACTIONS.ADD_K_SVD_RESULT, 
      payload: result 
    }),
    addKClusterResult: (result) => dispatch({ 
      type: ML_ACTIONS.ADD_K_CLUSTER_RESULT, 
      payload: result 
    }),
    setOptimalResults: (results) => dispatch({ 
      type: ML_ACTIONS.SET_OPTIMAL_RESULTS, 
      payload: results 
    }),
    setMatrixInfo: (info) => dispatch({ 
      type: ML_ACTIONS.SET_MATRIX_INFO, 
      payload: info 
    })
  };

  return (
    <MLContext.Provider value={value}>
      {children}
    </MLContext.Provider>
  );
};

// Hook per usare il context
export const useML = () => {
  const context = useContext(MLContext);
  if (!context) {
    throw new Error('useML must be used within an MLProvider');
  }
  return context;
};

export { ML_ACTIONS };