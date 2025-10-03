import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { Container, Card, CardContent, Form, FormGroup, Label, Input, Button, Title, Text } from '../styles/components';
import { authAPI } from '../services/api';
import { FaFilm, FaUser, FaEnvelope, FaLock, FaUserPlus } from 'react-icons/fa';

const RegisterContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, ${props => props.theme.colors.primary} 0%, ${props => props.theme.colors.primaryLight} 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${props => props.theme.spacing.lg};
`;

const RegisterCard = styled(Card)`
  width: 100%;
  max-width: 450px;
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

const RegisterFooter = styled.div`
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

const PasswordHint = styled.div`
  font-size: ${props => props.theme.fontSize.xs};
  color: ${props => props.theme.colors.textLight};
  margin-top: ${props => props.theme.spacing.xs};
`;

const RegisterPage = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    conferma_password: ''
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

  const validateForm = () => {
    if (formData.password !== formData.conferma_password) {
      setError('Le password non coincidono');
      return false;
    }
    if (formData.password.length < 6) {
      setError('La password deve essere di almeno 6 caratteri');
      return false;
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setLoading(true);
    setError('');

    try {
      const response = await authAPI.register(formData);
      setSuccess('Registrazione completata! Effettua il login...');
      
      // Auto-login after successful registration
      setTimeout(async () => {
        try {
          const loginResponse = await authAPI.login({
            username: formData.username,
            password: formData.password
          });
          
          const { access_token, utente } = loginResponse.data;
          onLogin(utente, access_token);
          navigate('/onboarding'); // New users always go to onboarding
          
        } catch (loginErr) {
          navigate('/login');
        }
      }, 1500);
      
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        'Errore durante la registrazione. Riprova.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <RegisterContainer>
      <Container>
        <RegisterCard>
          <CardContent>
            <LogoSection>
              <div className="logo">
                <FaFilm />
              </div>
              <Title>AFlix</Title>
              <Text>Crea il tuo account</Text>
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
                    placeholder="Scegli un username"
                    required
                  />
                </InputWithIcon>
              </FormGroup>

              <FormGroup>
                <Label htmlFor="email">Email</Label>
                <InputWithIcon>
                  <FaEnvelope className="icon" />
                  <Input
                    type="text"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="La tua email (anche fittizia)"
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
                    placeholder="Crea una password"
                    required
                  />
                </InputWithIcon>
                <PasswordHint>Minimo 6 caratteri</PasswordHint>
              </FormGroup>

              <FormGroup>
                <Label htmlFor="conferma_password">Conferma Password</Label>
                <InputWithIcon>
                  <FaLock className="icon" />
                  <Input
                    type="password"
                    id="conferma_password"
                    name="conferma_password"
                    value={formData.conferma_password}
                    onChange={handleChange}
                    placeholder="Ripeti la password"
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
                <FaUserPlus />
                {loading ? 'Registrazione in corso...' : 'Registrati'}
              </Button>
            </Form>

            <RegisterFooter>
              <Text>
                Hai già un account?{' '}
                <Link to="/login">Accedi qui</Link>
              </Text>
            </RegisterFooter>
          </CardContent>
        </RegisterCard>
      </Container>
    </RegisterContainer>
  );
};

export default RegisterPage;