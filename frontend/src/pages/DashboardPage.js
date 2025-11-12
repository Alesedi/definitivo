import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { Container, Section, Title, Subtitle, Text, Button, Grid, LoadingSpinner, Flex, Badge } from '../styles/components';
import MovieCard from '../components/MovieCard';
import { recommendationsAPI, ratingsAPI } from '../services/api';
import { FaSync, FaStar, FaFilm, FaHeart } from 'react-icons/fa';

const DashboardContainer = styled.div`
  min-height: calc(100vh - 80px);
  background: ${props => props.theme.colors.background};
  padding: ${props => props.theme.spacing.xl} 0;
`;

const WelcomeSection = styled.div`
  background: linear-gradient(135deg, ${props => props.theme.colors.primary}, ${props => props.theme.colors.primaryLight});
  color: ${props => props.theme.colors.textWhite};
  padding: ${props => props.theme.spacing.xl};
  border-radius: ${props => props.theme.borderRadius.xl};
  margin-bottom: ${props => props.theme.spacing.xl};
  text-align: center;
`;

const StatsSection = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const StatCard = styled.div`
  background: ${props => props.theme.colors.surface};
  padding: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.borderRadius.lg};
  box-shadow: ${props => props.theme.boxShadow.base};
  text-align: center;
  
  .icon {
    font-size: ${props => props.theme.fontSize['2xl']};
    color: ${props => props.theme.colors.secondary};
    margin-bottom: ${props => props.theme.spacing.sm};
  }
  
  .number {
    font-size: ${props => props.theme.fontSize['2xl']};
    font-weight: ${props => props.theme.fontWeight.bold};
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.xs};
  }
`;

const FilterSection = styled(Flex)`
  margin-bottom: ${props => props.theme.spacing.lg};
  flex-wrap: wrap;
  gap: ${props => props.theme.spacing.md};
`;

const RecommendationsHeader = styled(Flex)`
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.lg};
  flex-wrap: wrap;
  gap: ${props => props.theme.spacing.md};
`;

const DashboardPage = ({ user }) => {
  const [recommendations, setRecommendations] = useState([]);
  const [userStats, setUserStats] = useState(null);
  const [mlSource, setMlSource] = useState('tmdb');
  const [evaluation, setEvaluation] = useState(null);
  const [clusteringData, setClusteringData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      // Prova prima a ottenere raccomandazioni ML per la sorgente selezionata
      const modelStatus = await recommendationsAPI.getModelStatus(mlSource);
      
      let recommendations = [];
      if (modelStatus.data.is_trained) {
        const mlRecsResponse = await recommendationsAPI.getRecommendations(user.id, 20, mlSource);
        recommendations = mlRecsResponse.data.map(rec => ({
          id: rec.tmdb_id,
          title: rec.title,
          genres: rec.genres.join('|'),
          poster_path: rec.poster_url ? rec.poster_url.replace('https://image.tmdb.org/t/p/w500', '') : null,
          vote_average: rec.tmdb_rating || 0,
          popularity: rec.predicted_rating * 20,
          recommendation_score: rec.predicted_rating
        }));
      } else {
        const popularResponse = await recommendationsAPI.getPopularRecommendations(20);
        recommendations = popularResponse.data.map(rec => ({
          id: rec.tmdb_id,
          title: rec.title,
          genres: rec.genres.join('|'),
          poster_path: rec.poster_url ? rec.poster_url.replace('https://image.tmdb.org/t/p/w500', '') : null,
          vote_average: rec.tmdb_rating || 0,
          popularity: rec.predicted_rating * 20,
          recommendation_score: rec.predicted_rating
        }));
      }

      // Ottieni storico utente per statistiche
      const historyResponse = await recommendationsAPI.getUserHistory(user.id);
      const userStatsLocal = {
        total_votes: historyResponse.data.length,
        average_rating: historyResponse.data.length > 0 
          ? historyResponse.data.reduce((sum, item) => sum + item.rating, 0) / historyResponse.data.length 
          : 0,
        declared_preferences: user.generi_preferiti || []
      };

      // Prova a ottenere evaluation e clustering per la sorgente selezionata
      try {
        const evalResp = await recommendationsAPI.getEvaluation(mlSource);
        setEvaluation(evalResp.data);
      } catch (e) {
        setEvaluation(null);
      }

      try {
        const clustResp = await recommendationsAPI.getClustering(mlSource);
        setClusteringData(clustResp.data);
      } catch (e) {
        setClusteringData(null);
      }

      setRecommendations(recommendations);
      setUserStats(userStatsLocal);

    } catch (mlError) {
      console.warn('ML endpoints not available, usando fallback:', mlError);
      // Fallback: crea dati fittizi per evitare errori
      setRecommendations([]);
      setUserStats({
        total_votes: 0,
        average_rating: 0,
        declared_preferences: user.generi_preferiti || []
      });
      setEvaluation(null);
      setClusteringData(null);
    } finally {
      setLoading(false);
    }
  }, [mlSource, user.id]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      // Prova prima le raccomandazioni ML per la sorgente selezionata
      const modelStatus = await recommendationsAPI.getModelStatus(mlSource);
      
      let recommendations = [];
      if (modelStatus.data.is_trained) {
        const mlRecsResponse = await recommendationsAPI.getRecommendations(user.id, 20, mlSource);
        recommendations = mlRecsResponse.data.map(rec => ({
          id: rec.tmdb_id,
          title: rec.title,
          genres: rec.genres.join('|'),
          poster_path: rec.poster_url ? rec.poster_url.replace('https://image.tmdb.org/t/p/w500', '') : null,
          vote_average: rec.tmdb_rating || 0,
          popularity: rec.predicted_rating * 20,
          recommendation_score: rec.predicted_rating
        }));
      } else {
        const popularResponse = await recommendationsAPI.getPopularRecommendations(20);
        recommendations = popularResponse.data.map(rec => ({
          id: rec.tmdb_id,
          title: rec.title,
          genres: rec.genres.join('|'),
          poster_path: rec.poster_url ? rec.poster_url.replace('https://image.tmdb.org/t/p/w500', '') : null,
          vote_average: rec.tmdb_rating || 0,
          popularity: rec.predicted_rating * 20,
          recommendation_score: rec.predicted_rating
        }));
      }

      // Anche qui aggiorniamo evaluation/clustering
      try {
        const evalResp = await recommendationsAPI.getEvaluation(mlSource);
        setEvaluation(evalResp.data);
      } catch (e) { setEvaluation(null); }

      try {
        const clustResp = await recommendationsAPI.getClustering(mlSource);
        setClusteringData(clustResp.data);
      } catch (e) { setClusteringData(null); }

      setRecommendations(recommendations);
    } catch (error) {
      console.error('Error refreshing recommendations:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const handleMovieRating = async (movie, rating) => {
    try {
      // Gestisci genres come stringa o array
      let genresList = [];
      if (movie.genres) {
        if (typeof movie.genres === 'string') {
          genresList = movie.genres.split('|');
        } else if (Array.isArray(movie.genres)) {
          genresList = movie.genres.map(g => typeof g === 'string' ? g : g.name);
        }
      }
      
      const voteData = {
        tmdb_id: movie.id,
        rating: rating,
        title: movie.title,
        genres: genresList,
        poster_path: movie.poster_path || null,
        release_date: movie.release_date || "2023-01-01",
        tmdb_rating: movie.vote_average || 0.0
      };
      
      console.log('🔍 Dati voto da inviare:', JSON.stringify(voteData, null, 2));
      
      await ratingsAPI.voteMovie(user.id, voteData);
      
      console.log(`✅ Voto salvato: ${movie.title} - ${rating} stelle`);
      
      // Refresh recommendations after rating
      handleRefresh();
    } catch (error) {
      console.error('❌ Error rating movie:', error);
      console.error('❌ Error details:', error.response?.data);
      if (error.response?.data?.detail) {
        console.error('❌ Detailed validation errors:', error.response.data.detail);
        error.response.data.detail.forEach((err, index) => {
          console.error(`❌ Validation Error ${index + 1}:`, err);
        });
      }
      alert('Errore durante la valutazione del film: ' + (error.response?.data?.detail?.[0]?.msg || error.message));
    }
  };

  if (loading) {
    return (
      <DashboardContainer>
        <Container>
          <LoadingSpinner />
        </Container>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      <Container>
        <WelcomeSection>
          <Title style={{ color: 'white', marginBottom: '0.5rem' }}>
            Benvenuto, {user.username}!
          </Title>
          <Text style={{ color: 'rgba(255,255,255,0.9)', fontSize: '1.1rem' }}>
            Ecco i film consigliati per te basati sui tuoi gusti
          </Text>
        </WelcomeSection>

        {userStats && (
          <StatsSection>
            <StatCard>
              <div className="icon"><FaFilm /></div>
              <div className="number">{userStats.total_votes || 0}</div>
              <Text>Film Votati</Text>
            </StatCard>
            <StatCard>
              <div className="icon"><FaStar /></div>
              <div className="number">{userStats.average_rating || '0.0'}</div>
              <Text>Voto Medio</Text>
            </StatCard>
            <StatCard>
              <div className="icon"><FaHeart /></div>
              <div className="number">{userStats.high_rated_movies || 0}</div>
              <Text>Film Amati (4-5★)</Text>
            </StatCard>
            <StatCard>
              <div className="icon"><FaSync /></div>
              <div className="number">{recommendations.length}</div>
              <Text>Nuovi Consigli</Text>
            </StatCard>
          </StatsSection>
        )}

        <Section>
          <RecommendationsHeader>
            <div>
              <Subtitle>I tuoi film consigliati</Subtitle>
              <div style={{ marginTop: '6px' }}>
                <Button variant={mlSource === 'tmdb' ? 'primary' : 'secondary'} onClick={() => setMlSource('tmdb')} style={{ marginRight: '8px' }}>TMDB</Button>
                <Button variant={mlSource === 'omdb' ? 'primary' : 'secondary'} onClick={() => setMlSource('omdb')}>OMDb</Button>
                <Text style={{ display: 'inline-block', marginLeft: '12px', color: '#666' }}>Sorgente: {mlSource.toUpperCase()}</Text>
              </div>
              <Text style={{ marginTop: '0.5rem' }}>
                Basati sui tuoi generi preferiti e voti precedenti
              </Text>
            </div>
            <Button
              variant="secondary"
              onClick={handleRefresh}
              disabled={refreshing}
            >
              <FaSync />
              {refreshing ? 'Aggiornamento...' : 'Aggiorna'}
            </Button>
          </RecommendationsHeader>

          {userStats?.declared_preferences && (
            <FilterSection>
              <Text style={{ marginRight: '1rem' }}>I tuoi generi:</Text>
              {userStats.declared_preferences.map(genre => (
                <Badge key={genre}>{genre}</Badge>
              ))}
            </FilterSection>
          )}

          {/* Brief evaluation/clustering summary for selected source */}
          <div style={{ marginTop: '1rem', marginBottom: '1rem' }}>
            {evaluation ? (
              <div style={{ color: '#333', marginBottom: '0.5rem' }}>
                <strong>Evaluation ({mlSource.toUpperCase()}):</strong>
                <span style={{ marginLeft: '8px' }}>RMSE: {evaluation.rmse ?? 'N/A'}</span>
                <span style={{ marginLeft: '12px' }}>MAE: {evaluation.mae ?? 'N/A'}</span>
                <span style={{ marginLeft: '12px' }}>Samples: {evaluation.test_samples ?? '-'}</span>
              </div>
            ) : (
              <div style={{ color: '#888', marginBottom: '0.5rem' }}>Nessuna valutazione disponibile per {mlSource.toUpperCase()}</div>
            )}

            {clusteringData ? (
              <div style={{ color: '#333' }}>
                <strong>Clustering ({mlSource.toUpperCase()}):</strong>
                <span style={{ marginLeft: '8px' }}>Clusters: {clusteringData.n_clusters ?? 'N/A'}</span>
                <span style={{ marginLeft: '12px' }}>Total AFIX movies: {clusteringData.total_aflix_movies ?? '-'}</span>
              </div>
            ) : (
              <div style={{ color: '#888' }}>Nessun dato di clustering disponibile per {mlSource.toUpperCase()}</div>
            )}
          </div>

          {recommendations.length > 0 ? (
            <Grid>
              {recommendations.map(movie => (
                <MovieCard
                  key={movie.id}
                  movie={movie}
                  showRating={true}
                  onRate={handleMovieRating}
                />
              ))}
            </Grid>
          ) : (
            <div style={{ textAlign: 'center', padding: '3rem 0' }}>
              <Text>Nessuna raccomandazione disponibile al momento.</Text>
              <Button 
                variant="primary" 
                onClick={handleRefresh}
                style={{ marginTop: '1rem' }}
              >
                Riprova
              </Button>
            </div>
          )}
        </Section>
      </Container>
    </DashboardContainer>
  );
};

export default DashboardPage;