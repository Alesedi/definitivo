import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { Container, Card, CardContent, Form, FormGroup, Label, Input, Button, Title, Text, Flex } from '../styles/components';
import { authAPI } from '../services/api';
import { FaFilm, FaUser, FaLock, FaSignInAlt } from 'react-icons/fa';

const LoginContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, ${props => props.theme.colors.primary} 0%, ${props => props.theme.colors.primaryLight} 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${props => props.theme.spacing.lg};
`;

const LoginCard = styled(Card)`
  width: 100%;
  max-width: 400px;
  box-shadow: ${props => props.theme.boxShadow.xl};
`;

const LogoSection = styled.div`
  text-align: center;
  margin-bottom: ${props => props.theme.spacing.xl};
  
  .logo {
    font-size: ${props => props.theme.fontSize['4xl']};
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.sm};
  }
`;

const InputWithIcon = styled.div`
  position: relative;
  
  .icon {
    position: absolute;
    left: ${props => props.theme.spacing.md};
    top: 50%;
    transform: translateY(-50%);
    color: ${props => props.theme.colors.textLight};
  }
  
  input {
    padding-left: 40px;
  }
`;

const ErrorMessage = styled.div`
  background-color: ${props => props.theme.colors.error};
  color: ${props => props.theme.colors.textWhite};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius.md};
  margin-bottom: ${props => props.theme.spacing.md};
  font-size: ${props => props.theme.fontSize.sm};
`;

const SuccessMessage = styled.div`
  background-color: ${props => props.theme.colors.success};
  color: ${props => props.theme.colors.textWhite};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius.md};
  margin-bottom: ${props => props.theme.spacing.md};
  font-size: ${props => props.theme.fontSize.sm};
`;

const LoginFooter = styled.div`
  text-align: center;
  margin-top: ${props => props.theme.spacing.lg};
  
  a {
    color: ${props => props.theme.colors.primary};
    text-decoration: none;
    font-weight: ${props => props.theme.fontWeight.medium};
    
    &:hover {
      color: ${props => props.theme.colors.primaryLight};
    }
  }
`;

const LoginPage = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await authAPI.login(formData);
      const { access_token, utente, first_login } = response.data;
      
      setSuccess('Login effettuato con successo!');
      onLogin(utente, access_token);
      
      // Redirect based on first login status
      setTimeout(() => {
        if (first_login) {
          navigate('/onboarding');
        } else {
          navigate('/dashboard');
        }
      }, 1000);
      
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        'Errore durante il login. Verifica le credenziali.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <LoginContainer>
      <Container>
        <LoginCard>
          <CardContent>
            <LogoSection>
              <div className="logo">
                <FaFilm />
              </div>
              <Title>AFlix</Title>
              <Text>Accedi al tuo account</Text>
            </LogoSection>

            {error && <ErrorMessage>{error}</ErrorMessage>}
            {success && <SuccessMessage>{success}</SuccessMessage>}

            <Form onSubmit={handleSubmit}>
              <FormGroup>
                <Label htmlFor="username">Username</Label>
                <InputWithIcon>
                  <FaUser className="icon" />
                  <Input
                    type="text"
                    id="username"
                    name="username"
                    value={formData.username}
                    onChange={handleChange}
                    placeholder="Inserisci il tuo username"
                    required
                  />
                </InputWithIcon>
              </FormGroup>

              <FormGroup>
                <Label htmlFor="password">Password</Label>
                <InputWithIcon>
                  <FaLock className="icon" />
                  <Input
                    type="password"
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    placeholder="Inserisci la tua password"
                    required
                  />
                </InputWithIcon>
              </FormGroup>

              <Button 
                type="submit" 
                variant="primary" 
                disabled={loading}
                style={{ width: '100%', marginTop: '1rem' }}
              >
                <FaSignInAlt />
                {loading ? 'Accesso in corso...' : 'Accedi'}
              </Button>
            </Form>

            <LoginFooter>
              <Text>
                Non hai un account?{' '}
                <Link to="/register">Registrati qui</Link>
              </Text>
            </LoginFooter>
          </CardContent>
        </LoginCard>
      </Container>
    </LoginContainer>
  );
};

export default LoginPage;