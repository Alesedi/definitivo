import React, { createContext, useContext, useReducer, useCallback } from 'react';
import api from '../services/api';
import { recommendationsAPI } from '../services/api';

// Stato iniziale
const initialState = {
  // Stato del modello
  modelStatus: null,
    modelStatusBySource: {
      tmdb: null,
      omdb: null
    },
  isTraining: false,
  
  // Raccomandazioni
  recommendations: [],
  userHistory: [],
  evaluation: null,
  clustering: null,
    // Per-sorgente (tmdb / omdb)
    evaluationBySource: {
      tmdb: null,
      omdb: null
    },
    clusteringBySource: {
      tmdb: null,
      omdb: null
    },
  
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
  // Stato per sorgente (tmdb / omdb) - permette split-view futuro
  kOptimizationBySource: {
    tmdb: {
      status: 'idle',
      progress: 0,
      currentPhase: null,
      kSvdResults: [],
      kClusterResults: [],
      optimalResults: {},
      matrixInfo: null,
      isOptimizing: false
    },
    omdb: {
      status: 'idle',
      progress: 0,
      currentPhase: null,
      kSvdResults: [],
      kClusterResults: [],
      optimalResults: {},
      matrixInfo: null,
      isOptimizing: false
    }
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
      // Supporta payload.source per memorizzare lo stato per sorgente
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const payloadCopy = { ...action.payload };
        delete payloadCopy.source;
        return {
          ...state,
          modelStatus: payloadCopy,
          modelStatusBySource: { ...state.modelStatusBySource, [src]: payloadCopy }
        };
      }
      return { ...state, modelStatus: action.payload };
      
    case ML_ACTIONS.SET_TRAINING:
      return { ...state, isTraining: action.payload };
      
    case ML_ACTIONS.SET_RECOMMENDATIONS:
      return { ...state, recommendations: action.payload };
      
    case ML_ACTIONS.SET_USER_HISTORY:
      return { ...state, userHistory: action.payload };
      
    case ML_ACTIONS.SET_EVALUATION:
      // Supporta payload.source per memorizzare evaluation per sorgente
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const payloadCopy = { ...action.payload };
        delete payloadCopy.source;
        return {
          ...state,
          evaluation: payloadCopy,
          evaluationBySource: { ...state.evaluationBySource, [src]: payloadCopy }
        };
      }
      return { ...state, evaluation: action.payload };
      
    case ML_ACTIONS.SET_CLUSTERING:
      // Supporta payload.source per memorizzare clustering per sorgente
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const payloadCopy = { ...action.payload };
        delete payloadCopy.source;
        return {
          ...state,
          clustering: payloadCopy,
          clusteringBySource: { ...state.clusteringBySource, [src]: payloadCopy }
        };
      }
      return { ...state, clustering: action.payload };
      
    case ML_ACTIONS.ADD_LOG:
      // Evita duplicati consecutivi (stesso messaggio + type)
      const lastLog = state.logs.length > 0 ? state.logs[state.logs.length - 1] : null;
      const incomingMessage = action.payload.message;
      const incomingType = action.payload.type || 'info';
      if (lastLog && lastLog.message === incomingMessage && lastLog.type === incomingType) {
        // non aggiungere duplicato consecutivo
        return state;
      }
      const newLog = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        message: incomingMessage,
        type: incomingType
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
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const payloadCopy = { ...action.payload };
        delete payloadCopy.source;
        return {
          ...state,
          kOptimizationBySource: {
            ...state.kOptimizationBySource,
            [src]: { ...state.kOptimizationBySource[src], status: payloadCopy }
          }
        };
      }
      return {
        ...state,
        kOptimization: { ...state.kOptimization, status: action.payload }
      };
      
    case ML_ACTIONS.SET_K_OPTIMIZATION_PROGRESS:
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const payloadCopy = { ...action.payload };
        delete payloadCopy.source;
        return {
          ...state,
          kOptimizationBySource: {
            ...state.kOptimizationBySource,
            [src]: { ...state.kOptimizationBySource[src], progress: payloadCopy }
          }
        };
      }
      return {
        ...state,
        kOptimization: { ...state.kOptimization, progress: action.payload }
      };
      
    case ML_ACTIONS.SET_K_OPTIMIZATION_PHASE:
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const payloadCopy = { ...action.payload };
        delete payloadCopy.source;
        return {
          ...state,
          kOptimizationBySource: {
            ...state.kOptimizationBySource,
            [src]: { ...state.kOptimizationBySource[src], currentPhase: payloadCopy }
          }
        };
      }
      return {
        ...state,
        kOptimization: { ...state.kOptimization, currentPhase: action.payload }
      };
      
    case ML_ACTIONS.ADD_K_SVD_RESULT:
      // Se payload contiene source -> aggiorna namespace per sorgente, altrimenti aggiorna il namespace globale
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const entry = { ...action.payload };
        delete entry.source;
        return {
          ...state,
          kOptimizationBySource: {
            ...state.kOptimizationBySource,
            [src]: {
              ...state.kOptimizationBySource[src],
              kSvdResults: [...state.kOptimizationBySource[src].kSvdResults, entry]
            }
          }
        };
      }
      return {
        ...state,
        kOptimization: {
          ...state.kOptimization,
          kSvdResults: [...state.kOptimization.kSvdResults, action.payload]
        }
      };
      
    case ML_ACTIONS.ADD_K_CLUSTER_RESULT:
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const entry = { ...action.payload };
        delete entry.source;
        return {
          ...state,
          kOptimizationBySource: {
            ...state.kOptimizationBySource,
            [src]: {
              ...state.kOptimizationBySource[src],
              kClusterResults: [...state.kOptimizationBySource[src].kClusterResults, entry]
            }
          }
        };
      }
      return {
        ...state,
        kOptimization: {
          ...state.kOptimization,
          kClusterResults: [...state.kOptimization.kClusterResults, action.payload]
        }
      };
      
    case ML_ACTIONS.SET_OPTIMAL_RESULTS:
      // Supporta payload.source per aggiornare risultati per sorgente
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const payloadCopy = { ...action.payload };
        delete payloadCopy.source;
        return {
          ...state,
          kOptimizationBySource: {
            ...state.kOptimizationBySource,
            [src]: {
              ...state.kOptimizationBySource[src],
              optimalResults: { ...state.kOptimizationBySource[src].optimalResults, ...payloadCopy }
            }
          }
        };
      }
      return {
        ...state,
        kOptimization: {
          ...state.kOptimization,
          optimalResults: { ...state.kOptimization.optimalResults, ...action.payload }
        }
      };
      
    case ML_ACTIONS.SET_MATRIX_INFO:
      if (action.payload && action.payload.source) {
        const src = action.payload.source;
        const payloadCopy = { ...action.payload };
        delete payloadCopy.source;
        return {
          ...state,
          kOptimizationBySource: {
            ...state.kOptimizationBySource,
            [src]: { ...state.kOptimizationBySource[src], matrixInfo: payloadCopy }
          }
        };
      }
      return {
        ...state,
        kOptimization: { ...state.kOptimization, matrixInfo: action.payload }
      };
      
    case ML_ACTIONS.SET_K_OPTIMIZING:
      // action.meta.source is also supported inside payload.source
      if (action.payload && typeof action.payload === 'object' && action.payload.source) {
        const src = action.payload.source;
        const flag = action.payload.flag !== undefined ? action.payload.flag : action.payload;
        return {
          ...state,
          kOptimizationBySource: {
            ...state.kOptimizationBySource,
            [src]: { ...state.kOptimizationBySource[src], isOptimizing: flag }
          }
        };
      }
      return {
        ...state,
        kOptimization: { ...state.kOptimization, isOptimizing: action.payload }
      };
      
    case ML_ACTIONS.RESET_K_OPTIMIZATION:
      // Resetta sia namespace globale che namespaces per sorgente
      return {
        ...state,
        kOptimization: {
          ...initialState.kOptimization,
          optimalResults: state.kOptimization.optimalResults,
          isOptimizing: false
        },
        kOptimizationBySource: {
          tmdb: { ...initialState.kOptimization },
          omdb: { ...initialState.kOptimization }
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
  const fetchModelStatus = useCallback(async (source = 'tmdb') => {
    try {
      const response = await recommendationsAPI.getModelStatus(source);
      // Includi la sorgente nel payload così il reducer può salvarlo per-source
      const payload = { ...response.data, source };
      dispatch({ type: ML_ACTIONS.SET_MODEL_STATUS, payload });
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

  const trainModel = useCallback(async (source = null) => {
    dispatch({ type: ML_ACTIONS.SET_TRAINING, payload: true });
    dispatch({ type: ML_ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ML_ACTIONS.CLEAR_ERROR });
    addLog('🚀 Avvio training del modello...', 'info');
    
    try {
  const resp = await recommendationsAPI.trainModel(source || 'tmdb');
  const result = resp.data;
      
      addLog('✅ Training completato con successo!', 'success');
      
      // Aggiorna immediatamente lo stato del modello per la sorgente specificata e mostra metriche
      try {
        const status = await fetchModelStatus(source || 'tmdb');
        if (status) {
          const srcLabel = source || 'tmdb';
          addLog(`📊 Statistiche (${srcLabel}): ${status.total_ratings || 0} rating processati`, 'info');
          addLog(`🎯 K utilizzato: ${status.actual_k_used || 'N/A'}`, 'info');
          addLog(`📈 Varianza spiegata: ${(status.explained_variance * 100 || 0).toFixed(1)}%`, 'info');
        }
      } catch (e) {
        // se fetchModelStatus fallisce, non blocchiamo il flusso
        addLog('⚠️ Non è stato possibile aggiornare immediatamente lo stato del modello', 'warning');
      }
      
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

  const fetchRecommendations = useCallback(async (userId, topN = 10, source = 'tmdb') => {
    dispatch({ type: ML_ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ML_ACTIONS.CLEAR_ERROR });
    
    try {
      const response = await recommendationsAPI.getRecommendations(userId, topN, source);
      dispatch({ type: ML_ACTIONS.SET_RECOMMENDATIONS, payload: response.data });
      addLog(`✅ Caricate ${response.data.length} raccomandazioni (${source})`, 'success');
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

  const fetchEvaluation = useCallback(async (source = 'tmdb') => {
    dispatch({ type: ML_ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ML_ACTIONS.CLEAR_ERROR });
    
    try {
      const response = await recommendationsAPI.getEvaluation(source);
      // Includiamo source nel payload per la memorizzazione per-sorgente
      const payload = { ...response.data, source };
      dispatch({ type: ML_ACTIONS.SET_EVALUATION, payload });
      addLog(`✅ Valutazione modello completata (${source})`, 'success');
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

  const fetchClustering = useCallback(async (source = 'tmdb') => {
    dispatch({ type: ML_ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ML_ACTIONS.CLEAR_ERROR });
    
    try {
      const response = await recommendationsAPI.getClustering(source);
      const payload = { ...response.data, source };
      dispatch({ type: ML_ACTIONS.SET_CLUSTERING, payload });
      addLog(`✅ Clustering (${source}) caricato con successo`, 'success');
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
  const startKOptimization = useCallback((source = null) => {
    // Resetta gli stati di ottimizzazione (global e per-sorgente) per partire puliti
    dispatch({ type: ML_ACTIONS.RESET_K_OPTIMIZATION });
    // Imposta lo stato isOptimizing per la sorgente specificata o globalmente
    if (source) {
      dispatch({ type: ML_ACTIONS.SET_K_OPTIMIZING, payload: { source, flag: true } });
      dispatch({ type: ML_ACTIONS.SET_K_OPTIMIZATION_STATUS, payload: { source, status: 'running' } });
      addLog(`🚀 Avvio ottimizzazione K-values (${source.toUpperCase()})...`, 'info');
    } else {
      dispatch({ type: ML_ACTIONS.SET_K_OPTIMIZING, payload: true });
      dispatch({ type: ML_ACTIONS.SET_K_OPTIMIZATION_STATUS, payload: 'running' });
      addLog('🚀 Avvio ottimizzazione K-values (globale)...', 'info');
    }
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
    // Helpers per-source
    getKOptimizationForSource: (source) => state.kOptimizationBySource && state.kOptimizationBySource[source] ? state.kOptimizationBySource[source] : null,
    getAllSources: () => Object.keys(state.kOptimizationBySource || {}),
    
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
  // Helpers per-sorgente
  getEvaluationForSource: (source) => state.evaluationBySource ? state.evaluationBySource[source] : null,
  getClusteringForSource: (source) => state.clusteringBySource ? state.clusteringBySource[source] : null,
    
    // K-Optimization Functions
    startKOptimization,
    stopKOptimization,
    setKOptimizationProgress: (progress, source = null) => dispatch({ 
      type: ML_ACTIONS.SET_K_OPTIMIZATION_PROGRESS, 
      payload: source ? { progress, source } : progress 
    }),
    setKOptimizationPhase: (phase, source = null) => dispatch({ 
      type: ML_ACTIONS.SET_K_OPTIMIZATION_PHASE, 
      payload: source ? { ...phase, source } : phase 
    }),
    addKSvdResult: (result, source = null) => dispatch({ 
      type: ML_ACTIONS.ADD_K_SVD_RESULT, 
      payload: source ? { ...result, source } : result 
    }),
    addKClusterResult: (result, source = null) => dispatch({ 
      type: ML_ACTIONS.ADD_K_CLUSTER_RESULT, 
      payload: source ? { ...result, source } : result 
    }),
    setOptimalResults: (results, source = null) => dispatch({ 
      type: ML_ACTIONS.SET_OPTIMAL_RESULTS, 
      payload: source ? { ...results, source } : results 
    }),
    setMatrixInfo: (info, source = null) => dispatch({ 
      type: ML_ACTIONS.SET_MATRIX_INFO, 
      payload: source ? { ...info, source } : info 
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