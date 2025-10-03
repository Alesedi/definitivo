import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { Container, Section, Title, Text, Button, Grid, LoadingSpinner } from '../styles/components';
import MovieCard from '../components/MovieCard';
import { onboardingAPI, ratingsAPI } from '../services/api';
import { FaArrowRight, FaStepForward } from 'react-icons/fa';

const RatingContainer = styled.div`
  min-height: calc(100vh - 80px);
  background: ${props => props.theme.colors.background};
  padding: ${props => props.theme.spacing.xl} 0;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background-color: ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.full};
  margin-bottom: ${props => props.theme.spacing.xl};
  overflow: hidden;
  
  .progress {
    height: 100%;
    background: linear-gradient(90deg, ${props => props.theme.colors.secondary}, ${props => props.theme.colors.secondaryLight});
    transition: width 0.3s ease;
    border-radius: ${props => props.theme.borderRadius.full};
  }
`;

const StepIndicator = styled.div`
  text-align: center;
  margin-bottom: ${props => props.theme.spacing.xl};
  
  .step {
    color: ${props => props.theme.colors.secondary};
    font-weight: ${props => props.theme.fontWeight.bold};
  }
`;

const ActionButtons = styled.div`
  display: flex;
  justify-content: center;
  gap: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.xl};
`;

const RatingPage = ({ user }) => {
  const [movies, setMovies] = useState([]);
  const [ratings, setRatings] = useState({});
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchMovies();
  }, []);

  const fetchMovies = async () => {
    try {
      const response = await onboardingAPI.getMoviesForRating(user.id);
      setMovies(response.data.movies);
    } catch (error) {
      console.error('Error fetching movies:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMovieRating = async (movie, rating) => {
    console.log(`Tentativo di votare: ${movie.title} - ${rating} stelle`);
    
    // Salva il voto immediatamente nel backend
    try {
      const voteData = {
        tmdb_id: movie.id,
        rating: rating,
        title: movie.title,
        genres: movie.genre_ids ? movie.genre_ids.map(id => `Genre${id}`) : [],
        poster_path: movie.poster_path,
        release_date: movie.release_date,
        tmdb_rating: movie.vote_average
      };
      
      console.log('Invio voto al backend:', voteData);
      await ratingsAPI.voteMovie(user.id, voteData);
      console.log(`✅ Voto salvato: ${movie.title} - ${rating} stelle`);
      
      // Aggiorna anche lo stato locale
      setRatings(prev => ({
        ...prev,
        [movie.id]: voteData
      }));
    } catch (error) {
      console.error('❌ Error saving rating:', error);
      alert('Errore nel salvare il voto!');
    }
  };

  const handleSubmit = async () => {
    const ratedMovies = Object.values(ratings);
    
    if (ratedMovies.length === 0) {
      alert('Vota almeno un film per continuare!');
      return;
    }

    setSubmitting(true);
    try {
      await ratingsAPI.voteMultipleMovies(user.id, { votes: ratedMovies });
      navigate('/dashboard');
    } catch (error) {
      console.error('Error submitting ratings:', error);
    } finally {
      setSubmitting(false);
    }
  };

  const handleSkip = () => {
    navigate('/dashboard');
  };

  const ratedCount = Object.keys(ratings).length;
  const progressPercentage = (ratedCount / movies.length) * 100;

  if (loading) {
    return (
      <RatingContainer>
        <Container>
          <LoadingSpinner />
        </Container>
      </RatingContainer>
    );
  }

  return (
    <RatingContainer>
      <Container>
        <Section>
          <StepIndicator>
            <Text className="step">PASSO 2 DI 2</Text>
          </StepIndicator>
          
          <Title>Vota i film che hai già visto</Title>
          <Text style={{ textAlign: 'center', marginBottom: '2rem' }}>
            Vota i film che riconosci per aiutarci a capire i tuoi gusti.
            Più film voti, migliori saranno le nostre raccomandazioni!
          </Text>
          
          <ProgressBar>
            <div className="progress" style={{ width: `${progressPercentage}%` }} />
          </ProgressBar>
          
          <Text style={{ textAlign: 'center', marginBottom: '2rem' }}>
            Film votati: {ratedCount} / {movies.length}
          </Text>
          
          <Grid>
            {movies.map(movie => (
              <MovieCard
                key={movie.id}
                movie={movie}
                showRating={true}
                userRating={ratings[movie.id]?.rating}
                onRate={handleMovieRating}
              />
            ))}
          </Grid>
          
          <ActionButtons>
            <Button
              variant="outline"
              onClick={handleSkip}
            >
              <FaStepForward />
              Salta per ora
            </Button>
            <Button
              variant="primary"
              onClick={handleSubmit}
              disabled={submitting}
            >
              <FaArrowRight />
              {submitting ? 'Completamento...' : `Completa Setup (${ratedCount} voti)`}
            </Button>
          </ActionButtons>
        </Section>
      </Container>
    </RatingContainer>
  );
};

export default RatingPage;