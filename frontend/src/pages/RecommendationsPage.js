import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import api from '../services/api';
import { theme } from '../styles/theme';
import { FaBrain, FaStar, FaCog, FaPlay, FaHistory, FaChartLine } from 'react-icons/fa';
import ClusteringChart from '../components/ClusteringChart';

const PageContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, ${theme.colors.primary} 0%, ${theme.colors.primaryDark} 100%);
  padding: ${theme.spacing.xl};
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: ${theme.spacing.xl};
  color: white;
`;

const Title = styled.h1`
  font-size: ${theme.fontSize['3xl']};
  margin-bottom: ${theme.spacing.md};
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${theme.spacing.md};
`;

const Subtitle = styled.p`
  font-size: ${theme.fontSize.lg};
  opacity: 0.9;
`;

const SectionContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${theme.spacing.xl};
  max-width: 1400px;
  margin: 0 auto;
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const Section = styled.div`
  background: white;
  border-radius: ${theme.borderRadius.lg};
  padding: ${theme.spacing.xl};
  box-shadow: ${theme.boxShadow.lg};
  margin-bottom: ${theme.spacing.lg};
`;

const SectionHeader = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.md};
  margin-bottom: ${theme.spacing.lg};
  border-bottom: 2px solid ${theme.colors.primaryLighter};
  padding-bottom: ${theme.spacing.md};
`;

const SectionTitle = styled.h2`
  color: ${theme.colors.primary};
  font-size: ${theme.fontSize.xl};
  margin: 0;
`;

const Button = styled.button`
  background: ${theme.colors.secondary};
  color: white;
  border: none;
  padding: ${theme.spacing.md} ${theme.spacing.lg};
  border-radius: ${theme.borderRadius.md};
  font-size: ${theme.fontSize.base};
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
  transition: all 0.3s ease;
  
  &:hover {
    background: ${theme.colors.secondaryDark};
    transform: translateY(-2px);
  }
  
  &:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
  }
`;

const StatusBadge = styled.span`
  padding: ${theme.spacing.xs} ${theme.spacing.sm};
  background: ${props => props.isReady ? theme.colors.secondaryLighter : '#fef3c7'};
  color: ${props => props.isReady ? theme.colors.secondary : '#92400e'};
  border-radius: ${theme.borderRadius.sm};
  font-size: ${theme.fontSize.sm};
  font-weight: bold;
`;

const MovieCard = styled.div`
  display: flex;
  gap: ${theme.spacing.md};
  padding: ${theme.spacing.md};
  border: 1px solid ${theme.colors.primaryLighter};
  border-radius: ${theme.borderRadius.md};
  margin-bottom: ${theme.spacing.md};
  transition: transform 0.2s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: ${theme.boxShadow.base};
  }
`;

const MoviePoster = styled.img`
  width: 80px;
  height: 120px;
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
  font-size: ${theme.fontSize.base};
`;

const MovieRating = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.xs};
  color: ${theme.colors.secondary};
  font-weight: bold;
  margin-bottom: ${theme.spacing.xs};
`;

const MovieGenres = styled.div`
  font-size: ${theme.fontSize.sm};
  color: #666;
`;

const MetricCard = styled.div`
  background: ${theme.colors.primaryLighter};
  padding: ${theme.spacing.lg};
  border-radius: ${theme.borderRadius.md};
  text-align: center;
  margin-bottom: ${theme.spacing.md};
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

const LoadingSpinner = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: ${theme.spacing.xl};
  color: ${theme.colors.primary};
`;

const ErrorMessage = styled.div`
  background: #fee2e2;
  color: #dc2626;
  padding: ${theme.spacing.md};
  border-radius: ${theme.borderRadius.md};
  text-align: center;
  margin: ${theme.spacing.md} 0;
`;

const ClusterChart = styled.div`
  background: transparent;
  border-radius: ${theme.borderRadius.md};
  padding: 0;
  width: 100%;
  min-height: 450px;
  display: block;
`;

const RecommendationsPage = ({ user }) => {
  const [modelStatus, setModelStatus] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [userHistory, setUserHistory] = useState([]);
  const [evaluation, setEvaluation] = useState(null);
  const [clustering, setClustering] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const getTMDBImageUrl = (posterPath) => {
    if (!posterPath) {
      const placeholderSVG = `data:image/svg+xml,${encodeURIComponent(`
        <svg width="80" height="120" xmlns="http://www.w3.org/2000/svg">
          <rect width="100%" height="100%" fill="#2a5298"/>
          <text x="50%" y="50%" text-anchor="middle" fill="white" font-size="12" font-family="Arial">
            No Image
          </text>
        </svg>
      `)}`;
      return placeholderSVG;
    }
    return posterPath.startsWith('http') ? posterPath : `https://image.tmdb.org/t/p/w200${posterPath}`;
  };

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      const response = await api.get('/recommendations/status');
      setModelStatus(response.data);
    } catch (error) {
      console.error('Error fetching model status:', error);
      setError('Errore nel recupero dello stato del modello');
    }
  };

  const trainModel = async () => {
    setLoading(true);
    setError('');
    try {
      await api.get('/recommendations/train-sync');
      setError('');
      fetchModelStatus();
      alert('Modello addestrato con successo!');
    } catch (error) {
      console.error('Error training model:', error);
      setError('Errore nel training del modello: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await api.get(`/recommendations/user/${user.id}?top_n=10`);
      setRecommendations(response.data);
      setError('');
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError('Errore nel recupero delle raccomandazioni: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const fetchUserHistory = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await api.get(`/recommendations/user/${user.id}/history`);
      setUserHistory(response.data);
      setError('');
    } catch (error) {
      console.error('Error fetching user history:', error);
      setError('Errore nel recupero dello storico: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const fetchEvaluation = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await api.get('/recommendations/evaluation');
      setEvaluation(response.data);
      setError('');
    } catch (error) {
      console.error('Error fetching evaluation:', error);
      setError('Errore nel recupero della valutazione: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const fetchClustering = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await api.get('/recommendations/clustering');
      setClustering(response.data);
      setError('');
    } catch (error) {
      console.error('Error fetching clustering:', error);
      setError('Errore nel recupero del clustering: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer>
      <Header>
        <Title>
          <FaBrain />
          Sistema di Raccomandazioni ML
        </Title>
        <Subtitle>
          Raccomandazioni personalizzate basate su Machine Learning con algoritmo SVD
        </Subtitle>
      </Header>

      {error && <ErrorMessage>{error}</ErrorMessage>}

      <SectionContainer>
        {/* Sezione Stato Modello e Training */}
        <Section>
          <SectionHeader>
            <FaCog />
            <SectionTitle>Stato Modello</SectionTitle>
          </SectionHeader>
          
          {modelStatus && (
            <>
              <MetricCard>
                <MetricValue>
                  <StatusBadge isReady={modelStatus.is_trained}>
                    {modelStatus.is_trained ? 'ADDESTRATO' : 'NON ADDESTRATO'}
                  </StatusBadge>
                </MetricValue>
                <MetricLabel>Stato del Modello</MetricLabel>
              </MetricCard>

              {modelStatus.is_trained && (
                <>
                  <MetricCard>
                    <MetricValue>{(modelStatus.explained_variance * 100).toFixed(2)}%</MetricValue>
                    <MetricLabel>Varianza Spiegata</MetricLabel>
                  </MetricCard>
                  
                  <MetricCard>
                    <MetricValue>{modelStatus.n_components}</MetricValue>
                    <MetricLabel>Componenti SVD</MetricLabel>
                  </MetricCard>
                </>
              )}

              <Button onClick={trainModel} disabled={loading}>
                <FaCog />
                {loading ? 'Training in corso...' : 'Ri-addestra Modello'}
              </Button>
            </>
          )}
        </Section>

        {/* Sezione Raccomandazioni */}
        <Section>
          <SectionHeader>
            <FaStar />
            <SectionTitle>Le Tue Raccomandazioni</SectionTitle>
          </SectionHeader>
          
          <Button onClick={fetchRecommendations} disabled={loading || !modelStatus?.is_trained}>
            <FaPlay />
            Genera Raccomandazioni
          </Button>

          {recommendations.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              {recommendations.slice(0, 5).map((movie, index) => (
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
                      {movie.tmdb_rating && ` • TMDB: ${movie.tmdb_rating}`}
                    </MovieRating>
                    <MovieGenres>
                      {movie.genres.join(', ')} • Cluster: {movie.cluster}
                    </MovieGenres>
                  </MovieInfo>
                </MovieCard>
              ))}
            </div>
          )}
        </Section>

        {/* Sezione Storico Utente */}
        <Section>
          <SectionHeader>
            <FaChartLine />
            <SectionTitle>Il Tuo Storico</SectionTitle>
          </SectionHeader>
          
          <Button onClick={fetchUserHistory} disabled={loading}>
            <FaChartLine />
            Visualizza Storico Voti
          </Button>

          {userHistory.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <MetricCard>
                <MetricValue>{userHistory.length}</MetricValue>
                <MetricLabel>Film Votati</MetricLabel>
              </MetricCard>

              {userHistory.slice(0, 5).map((movie, index) => (
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
                      {movie.rating}/5 (il tuo voto)
                      {movie.tmdb_rating && ` • TMDB: ${movie.tmdb_rating}`}
                    </MovieRating>
                    <MovieGenres>{movie.genres.join(', ')}</MovieGenres>
                  </MovieInfo>
                </MovieCard>
              ))}
            </div>
          )}
        </Section>

        {/* Sezione Valutazione Modello */}
        <Section>
          <SectionHeader>
            <FaChartLine />
            <SectionTitle>Metriche del Modello</SectionTitle>
          </SectionHeader>
          
          <Button onClick={fetchEvaluation} disabled={loading || !modelStatus?.is_trained}>
            <FaChartLine />
            Valuta Performance
          </Button>

          {evaluation && !evaluation.error && (
            <div style={{ marginTop: '20px' }}>
              <MetricCard>
                <MetricValue>{evaluation.rmse?.toFixed(4) || 'N/A'}</MetricValue>
                <MetricLabel>RMSE (Root Mean Square Error)</MetricLabel>
              </MetricCard>

              <MetricCard>
                <MetricValue>{evaluation.mae?.toFixed(4) || 'N/A'}</MetricValue>
                <MetricLabel>MAE (Mean Absolute Error)</MetricLabel>
              </MetricCard>

              <MetricCard>
                <MetricValue>{evaluation.test_samples || 'N/A'}</MetricValue>
                <MetricLabel>Campioni di Test</MetricLabel>
              </MetricCard>
            </div>
          )}

          {evaluation?.error && (
            <ErrorMessage style={{ marginTop: '20px' }}>
              {evaluation.error}
            </ErrorMessage>
          )}
        </Section>

        {/* Sezione Clustering */}
        <Section>
          <SectionHeader>
            <FaBrain />
            <SectionTitle>Clustering dei Film</SectionTitle>
          </SectionHeader>
          
          <Button onClick={fetchClustering} disabled={loading || !modelStatus?.is_trained}>
            <FaBrain />
            Visualizza Clustering
          </Button>

          {clustering && !clustering.error && (
            <div style={{ marginTop: '20px' }}>
              <MetricCard>
                <MetricValue>{clustering.n_clusters}</MetricValue>
                <MetricLabel>Cluster Identificati</MetricLabel>
              </MetricCard>

              <MetricCard>
                <MetricValue>{clustering.points?.length || 0}</MetricValue>
                <MetricLabel>Film nel Dataset</MetricLabel>
              </MetricCard>

              <ClusterChart>
                <ClusteringChart data={clustering} />
              </ClusterChart>
            </div>
          )}

          {clustering?.error && (
            <ErrorMessage style={{ marginTop: '20px' }}>
              {clustering.error}
            </ErrorMessage>
          )}
        </Section>
      </SectionContainer>

      {loading && (
        <LoadingSpinner>
          <div>Caricamento in corso...</div>
        </LoadingSpinner>
      )}
    </PageContainer>
  );
};

export default RecommendationsPage;