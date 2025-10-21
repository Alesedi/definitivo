import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://127.0.0.1:8005',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// API methods
export const authAPI = {
  register: (userData) => api.post('/api/auth/register', userData),
  login: (credentials) => api.post('/api/auth/login', credentials),
};

export const onboardingAPI = {
  getGenres: () => api.get('/api/onboarding/genres'),
  selectGenres: (userId, genres) => api.post(`/api/onboarding/select-genres/${userId}`, genres),
  getMoviesForRating: (userId) => api.get(`/api/onboarding/movies-for-rating/${userId}`),
};

export const ratingsAPI = {
  voteMovie: (userId, voteData) => api.post(`/api/ratings/vote/${userId}`, voteData),
  voteMultipleMovies: (userId, votesData) => api.post(`/api/ratings/vote-multiple/${userId}`, votesData),
  getUserRatings: (userId) => api.get(`/api/ratings/user-ratings/${userId}`),
};

export const recommendationsAPI = {
  // Nuovi endpoint ML
  getRecommendations: (userId, topN = 10) => api.get(`/api/recommendations/user/${userId}?top_n=${topN}`),
  getUserHistory: (userId) => api.get(`/api/recommendations/user/${userId}/history`),
  getModelStatus: () => api.get('/api/recommendations/status'),
  trainModel: () => api.get('/api/recommendations/train-sync'),
  trainModelAsync: () => api.post('/api/recommendations/train'),
  getEvaluation: () => api.get('/api/recommendations/evaluation'),
  getClustering: () => api.get('/api/recommendations/clustering'),
  getPopularRecommendations: (topN = 10) => api.get(`/api/recommendations/popular?top_n=${topN}`),
  
  // Backward compatibility (deprecated)
  getUserStats: (userId) => api.get(`/api/recommendations/user/${userId}/history`),
};

// Utility functions
export const setAuthToken = (token) => {
  if (token) {
    localStorage.setItem('token', token);
  } else {
    localStorage.removeItem('token');
  }
};

export const getAuthToken = () => {
  return localStorage.getItem('token');
};

export const removeAuthToken = () => {
  localStorage.removeItem('token');
};

// TMDB image helper
export const getTMDBImageUrl = (posterPath, size = 'w500') => {
  if (!posterPath) {
    // Creiamo un SVG inline come Data URI per il placeholder
    const placeholderSVG = `data:image/svg+xml,${encodeURIComponent(`
      <svg width="500" height="750" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#2a5298"/>
        <text x="50%" y="45%" text-anchor="middle" fill="white" font-size="24" font-family="Arial">
          Film
        </text>
        <text x="50%" y="55%" text-anchor="middle" fill="white" font-size="24" font-family="Arial">
          Poster
        </text>
      </svg>
    `)}`;
    return placeholderSVG;
  }
  return `https://image.tmdb.org/t/p/${size}${posterPath}`;
};

export default api;