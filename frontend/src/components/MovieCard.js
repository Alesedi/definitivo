import React, { useState } from 'react';
import styled from 'styled-components';
import { Card, CardContent, CardTitle, CardSubtitle, CardDescription, Button, Flex, Badge } from '../styles/components';
import { FaStar, FaCalendar, FaHeart, FaPlay } from 'react-icons/fa';
import { getTMDBImageUrl } from '../services/api';

const MovieCardContainer = styled(Card)`
  height: 100%;
  display: flex;
  flex-direction: column;
`;

const MoviePoster = styled.div`
  width: 100%;
  height: 400px;
  background-image: url(${props => props.posterUrl});
  background-size: cover;
  background-position: center;
  background-color: ${props => props.theme.colors.border};
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
      to bottom,
      transparent 0%,
      transparent 60%,
      rgba(0, 0, 0, 0.8) 100%
    );
    opacity: 0;
    transition: opacity 0.3s ease;
  }
  
  &:hover::before {
    opacity: 1;
  }
`;

const PosterOverlay = styled.div`
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: ${props => props.theme.spacing.lg};
  color: ${props => props.theme.colors.textWhite};
  transform: translateY(100%);
  transition: transform 0.3s ease;
  
  ${MoviePoster}:hover & {
    transform: translateY(0);
  }
`;

const MovieInfo = styled(CardContent)`
  flex: 1;
  display: flex;
  flex-direction: column;
`;

const MovieTitle = styled(CardTitle)`
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const MovieMeta = styled(Flex)`
  margin-bottom: ${props => props.theme.spacing.md};
  flex-wrap: wrap;
`;

const RatingInfo = styled(Flex)`
  gap: ${props => props.theme.spacing.xs};
  align-items: center;
  
  .tmdb-rating {
    color: ${props => props.theme.colors.secondary};
    font-weight: ${props => props.theme.fontWeight.semibold};
  }
  
  .user-rating {
    color: ${props => props.theme.colors.primary};
    font-weight: ${props => props.theme.fontWeight.semibold};
  }
`;

const GenreList = styled(Flex)`
  gap: ${props => props.theme.spacing.xs};
  margin-bottom: ${props => props.theme.spacing.md};
  flex-wrap: wrap;
`;

const GenreBadge = styled(Badge)`
  background-color: ${props => props.theme.colors.secondaryLighter};
  color: ${props => props.theme.colors.secondary};
  font-size: ${props => props.theme.fontSize.xs};
`;

const MovieOverview = styled(CardDescription)`
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const ExpandedOverview = styled(CardDescription)`
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const ActionButtons = styled(Flex)`
  margin-top: auto;
  gap: ${props => props.theme.spacing.sm};
`;

const RatingButton = styled(Button)`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${props => props.theme.spacing.xs};
  padding: ${props => props.theme.spacing.sm};
  
  ${props => props.selected && `
    background-color: ${props.theme.colors.secondary};
    
    &:hover {
      background-color: ${props.theme.colors.secondaryLight};
    }
  `}
`;

const MovieCard = ({ 
  movie, 
  showRating = false, 
  userRating = null, 
  onRate = null,
  showGenres = true,
  showOverview = true 
}) => {
  const [showFullOverview, setShowFullOverview] = useState(false);
  const [currentRating, setCurrentRating] = useState(userRating);

  const posterUrl = getTMDBImageUrl(movie.poster_path);
  const releaseYear = movie.release_date ? new Date(movie.release_date).getFullYear() : 'TBA';
  const tmdbRating = movie.vote_average ? movie.vote_average.toFixed(1) : 'N/A';
  
  // Handle movie genres - dal CSV è una stringa separata da |
  const genres = movie.genres || movie.genre || [];
  let genreNames = [];
  
  if (Array.isArray(genres)) {
    genreNames = genres.map(g => typeof g === 'string' ? g : g.name);
  } else if (typeof genres === 'string') {
    genreNames = genres.split('|').slice(0, 3); // Max 3 generi
  }

  const handleRating = (rating) => {
    setCurrentRating(rating);
    if (onRate) {
      onRate(movie, rating);
    }
  };

  const toggleOverview = () => {
    setShowFullOverview(!showFullOverview);
  };

  return (
    <MovieCardContainer>
      <MoviePoster posterUrl={posterUrl}>
        <PosterOverlay>
          <Flex justify="space-between" align="center">
            <RatingInfo>
              <FaStar />
              <span className="tmdb-rating">{tmdbRating}/10</span>
            </RatingInfo>
            <Button variant="outline" size="sm">
              <FaPlay />
              Trailer
            </Button>
          </Flex>
        </PosterOverlay>
      </MoviePoster>
      
      <MovieInfo>
        <MovieTitle>{movie.title || movie.titolo}</MovieTitle>
        
        <MovieMeta>
          <Flex gap="xs" align="center">
            <FaCalendar />
            <span>{releaseYear}</span>
          </Flex>
          
          <RatingInfo>
            <FaStar />
            <span className="tmdb-rating">{tmdbRating}</span>
            {currentRating && (
              <>
                <span>•</span>
                <span className="user-rating">Il tuo: {currentRating}/5</span>
              </>
            )}
          </RatingInfo>
        </MovieMeta>
        
        {showGenres && genreNames.length > 0 && (
          <GenreList>
            {genreNames.slice(0, 3).map((genre, index) => (
              <GenreBadge key={index}>{genre}</GenreBadge>
            ))}
            {genreNames.length > 3 && (
              <GenreBadge>+{genreNames.length - 3}</GenreBadge>
            )}
          </GenreList>
        )}
        
        {showOverview && movie.overview && (
          <>
            {showFullOverview ? (
              <ExpandedOverview>
                {movie.overview}
                {movie.overview.length > 150 && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={toggleOverview}
                    style={{ marginTop: '8px', fontSize: '12px' }}
                  >
                    Mostra meno
                  </Button>
                )}
              </ExpandedOverview>
            ) : (
              <MovieOverview>
                {movie.overview}
                {movie.overview.length > 150 && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={toggleOverview}
                    style={{ marginTop: '8px', fontSize: '12px' }}
                  >
                    Leggi tutto
                  </Button>
                )}
              </MovieOverview>
            )}
          </>
        )}
        
        {showRating && (
          <ActionButtons>
            {[1, 2, 3, 4, 5].map(rating => (
              <RatingButton
                key={rating}
                variant={currentRating === rating ? 'secondary' : 'outline'}
                selected={currentRating === rating}
                onClick={() => handleRating(rating)}
              >
                <FaStar />
                {rating}
              </RatingButton>
            ))}
          </ActionButtons>
        )}
      </MovieInfo>
    </MovieCardContainer>
  );
};

export default MovieCard;