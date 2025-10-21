import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import { theme } from './styles/theme';
import { GlobalStyle } from './styles/components';

// Import pages
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import OnboardingPage from './pages/OnboardingPage';
import DashboardPage from './pages/DashboardPage';
import RatingPage from './pages/RatingPage';
import RecommendationsPage from './pages/RecommendationsPage';

// Import components
import Navbar from './components/Navbar';
import MLMonitor from './components/MLMonitor';
import KOptimizationMonitor from './components/KOptimizationMonitor';
import { getAuthToken } from './services/api';

function App() {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in on app start
    const token = getAuthToken();
    const userData = localStorage.getItem('user');
    
    if (token && userData) {
      setUser(JSON.parse(userData));
    }
    setIsLoading(false);
  }, []);

  const handleLogin = (userData, token) => {
    setUser(userData);
    localStorage.setItem('user', JSON.stringify(userData));
    localStorage.setItem('token', token);
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
  };

  if (isLoading) {
    return (
      <ThemeProvider theme={theme}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh',
          backgroundColor: theme.colors.background 
        }}>
          Loading...
        </div>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <Router>
        <div className="App">
          {user && <Navbar user={user} onLogout={handleLogout} />}
          
          <Routes>
            {/* Public routes */}
            <Route 
              path="/login" 
              element={
                !user ? (
                  <LoginPage onLogin={handleLogin} />
                ) : (
                  <Navigate to="/dashboard" replace />
                )
              } 
            />
            <Route 
              path="/register" 
              element={
                !user ? (
                  <RegisterPage onLogin={handleLogin} />
                ) : (
                  <Navigate to="/dashboard" replace />
                )
              } 
            />
            
            {/* Protected routes */}
            <Route 
              path="/onboarding" 
              element={
                user ? (
                  <OnboardingPage user={user} />
                ) : (
                  <Navigate to="/login" replace />
                )
              } 
            />
            <Route 
              path="/rating" 
              element={
                user ? (
                  <RatingPage user={user} />
                ) : (
                  <Navigate to="/login" replace />
                )
              } 
            />
            <Route 
              path="/dashboard" 
              element={
                user ? (
                  <DashboardPage user={user} />
                ) : (
                  <Navigate to="/login" replace />
                )
              } 
            />
            <Route 
              path="/recommendations" 
              element={
                user ? (
                  <RecommendationsPage user={user} />
                ) : (
                  <Navigate to="/login" replace />
                )
              } 
            />
            <Route 
              path="/ml-monitor" 
              element={
                user ? (
                  <MLMonitor />
                ) : (
                  <Navigate to="/login" replace />
                )
              } 
            />
            <Route 
              path="/k-optimization" 
              element={
                user ? (
                  <KOptimizationMonitor />
                ) : (
                  <Navigate to="/login" replace />
                )
              } 
            />
            
            {/* Default redirect */}
            <Route 
              path="/" 
              element={
                user ? (
                  <Navigate to="/dashboard" replace />
                ) : (
                  <Navigate to="/login" replace />
                )
              } 
            />
          </Routes>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;