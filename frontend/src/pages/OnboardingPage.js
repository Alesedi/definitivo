import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { Container, Section, Title, Subtitle, Text, Button, Card, CardContent, Grid, Flex } from '../styles/components';
import { onboardingAPI } from '../services/api';
import { FaCheck, FaArrowRight } from 'react-icons/fa';

const OnboardingContainer = styled.div`
  min-height: calc(100vh - 80px);
  background: ${props => props.theme.colors.background};
  padding: ${props => props.theme.spacing.xl} 0;
`;

const GenreCard = styled(Card)`
  cursor: pointer;
  transition: all 0.3s ease;
  
  ${props => props.selected && `
    background: linear-gradient(135deg, ${props.theme.colors.primary}, ${props.theme.colors.primaryLight});
    color: ${props.theme.colors.textWhite};
    transform: scale(1.05);
  `}
  
  &:hover {
    transform: scale(1.02);
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

const OnboardingPage = ({ user }) => {
  const [genres, setGenres] = useState([]);
  const [selectedGenres, setSelectedGenres] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchGenres();
  }, []);

  const fetchGenres = async () => {
    try {
      const response = await onboardingAPI.getGenres();
      setGenres(response.data.genres);
    } catch (error) {
      console.error('Error fetching genres:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleGenreToggle = (genreName) => {
    setSelectedGenres(prev => {
      if (prev.includes(genreName)) {
        return prev.filter(g => g !== genreName);
      } else {
        return [...prev, genreName];
      }
    });
  };

  const handleSubmit = async () => {
    if (selectedGenres.length < 3) {
      alert('Seleziona almeno 3 generi!');
      return;
    }

    setSubmitting(true);
    try {
      await onboardingAPI.selectGenres(user.id, { generi: selectedGenres });
      navigate('/rating');
    } catch (error) {
      console.error('Error saving genres:', error);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <OnboardingContainer>
        <Container>
          <div style={{ textAlign: 'center', padding: '4rem 0' }}>
            Loading...
          </div>
        </Container>
      </OnboardingContainer>
    );
  }

  return (
    <OnboardingContainer>
      <Container>
        <Section>
          <StepIndicator>
            <Text className="step">PASSO 1 DI 2</Text>
          </StepIndicator>
          
          <Title>Seleziona i tuoi generi preferiti</Title>
          <Text style={{ textAlign: 'center', marginBottom: '2rem' }}>
            Scegli almeno 3 generi cinematografici che ti piacciono di più.
            Questo ci aiuterà a consigliarti i film perfetti per te!
          </Text>
          
          <Grid>
            {genres.map(genre => (
              <GenreCard
                key={genre.id}
                selected={selectedGenres.includes(genre.name)}
                onClick={() => handleGenreToggle(genre.name)}
              >
                <CardContent>
                  <Flex justify="space-between" align="center">
                    <Subtitle>{genre.name}</Subtitle>
                    {selectedGenres.includes(genre.name) && <FaCheck />}
                  </Flex>
                </CardContent>
              </GenreCard>
            ))}
          </Grid>
          
          <div style={{ textAlign: 'center', marginTop: '3rem' }}>
            <Text style={{ marginBottom: '1rem' }}>
              Generi selezionati: {selectedGenres.length}/3 (minimo)
            </Text>
            <Button
              variant="primary"
              onClick={handleSubmit}
              disabled={selectedGenres.length < 3 || submitting}
            >
              <FaArrowRight />
              {submitting ? 'Salvataggio...' : 'Continua'}
            </Button>
          </div>
        </Section>
      </Container>
    </OnboardingContainer>
  );
};

export default OnboardingPage;