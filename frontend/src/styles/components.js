import styled, { createGlobalStyle } from 'styled-components';

export const GlobalStyle = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
      'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
      sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background-color: ${props => props.theme.colors.background};
    color: ${props => props.theme.colors.textPrimary};
    line-height: 1.6;
  }

  code {
    font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
      monospace;
  }
`;

// Layout Components
export const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 ${props => props.theme.spacing.md};
  
  @media (min-width: ${props => props.theme.breakpoints.tablet}) {
    padding: 0 ${props => props.theme.spacing.lg};
  }
`;

export const Section = styled.section`
  padding: ${props => props.theme.spacing.xl} 0;
`;

export const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin-top: ${props => props.theme.spacing.lg};
`;

// Button Components
export const Button = styled.button`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.lg};
  border: none;
  border-radius: ${props => props.theme.borderRadius.md};
  font-size: ${props => props.theme.fontSize.base};
  font-weight: ${props => props.theme.fontWeight.medium};
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none;
  
  ${props => props.variant === 'primary' && `
    background-color: ${props.theme.colors.primary};
    color: ${props.theme.colors.textWhite};
    
    &:hover {
      background-color: ${props.theme.colors.primaryLight};
      transform: translateY(-1px);
      box-shadow: ${props.theme.boxShadow.lg};
    }
  `}
  
  ${props => props.variant === 'secondary' && `
    background-color: ${props.theme.colors.secondary};
    color: ${props.theme.colors.textWhite};
    
    &:hover {
      background-color: ${props.theme.colors.secondaryLight};
      transform: translateY(-1px);
      box-shadow: ${props.theme.boxShadow.lg};
    }
  `}
  
  ${props => props.variant === 'outline' && `
    background-color: transparent;
    color: ${props.theme.colors.primary};
    border: 2px solid ${props.theme.colors.primary};
    
    &:hover {
      background-color: ${props.theme.colors.primary};
      color: ${props.theme.colors.textWhite};
    }
  `}
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

// Card Components
export const Card = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.lg};
  box-shadow: ${props => props.theme.boxShadow.base};
  overflow: hidden;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: ${props => props.theme.boxShadow.xl};
  }
`;

export const CardImage = styled.div`
  width: 100%;
  height: 400px;
  background-image: url(${props => props.src});
  background-size: cover;
  background-position: center;
  background-color: ${props => props.theme.colors.border};
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 50%;
    background: linear-gradient(transparent, rgba(0,0,0,0.7));
  }
`;

export const CardContent = styled.div`
  padding: ${props => props.theme.spacing.lg};
`;

export const CardTitle = styled.h3`
  font-size: ${props => props.theme.fontSize.xl};
  font-weight: ${props => props.theme.fontWeight.bold};
  margin-bottom: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.colors.textPrimary};
`;

export const CardSubtitle = styled.p`
  font-size: ${props => props.theme.fontSize.sm};
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.md};
`;

export const CardDescription = styled.p`
  font-size: ${props => props.theme.fontSize.base};
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.6;
  margin-bottom: ${props => props.theme.spacing.lg};
`;

// Form Components
export const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.lg};
`;

export const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.sm};
`;

export const Label = styled.label`
  font-size: ${props => props.theme.fontSize.sm};
  font-weight: ${props => props.theme.fontWeight.medium};
  color: ${props => props.theme.colors.textPrimary};
`;

export const Input = styled.input`
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border: 2px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  font-size: ${props => props.theme.fontSize.base};
  transition: border-color 0.2s ease;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }
  
  &::placeholder {
    color: ${props => props.theme.colors.textLight};
  }
`;

// Typography
export const Title = styled.h1`
  font-size: ${props => props.theme.fontSize['4xl']};
  font-weight: ${props => props.theme.fontWeight.bold};
  color: ${props => props.theme.colors.textPrimary};
  margin-bottom: ${props => props.theme.spacing.lg};
  text-align: center;
`;

export const Subtitle = styled.h2`
  font-size: ${props => props.theme.fontSize['2xl']};
  font-weight: ${props => props.theme.fontWeight.semibold};
  color: ${props => props.theme.colors.textPrimary};
  margin-bottom: ${props => props.theme.spacing.md};
`;

export const Text = styled.p`
  font-size: ${props => props.theme.fontSize.base};
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.6;
`;

// Utility Components
export const Flex = styled.div`
  display: flex;
  align-items: ${props => props.align || 'center'};
  justify-content: ${props => props.justify || 'flex-start'};
  gap: ${props => props.gap || props.theme.spacing.md};
  flex-wrap: ${props => props.wrap ? 'wrap' : 'nowrap'};
`;

export const Badge = styled.span`
  display: inline-block;
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  background-color: ${props => props.theme.colors.primaryLighter};
  color: ${props => props.theme.colors.primary};
  border-radius: ${props => props.theme.borderRadius.full};
  font-size: ${props => props.theme.fontSize.xs};
  font-weight: ${props => props.theme.fontWeight.medium};
`;

export const LoadingSpinner = styled.div`
  border: 3px solid ${props => props.theme.colors.border};
  border-top: 3px solid ${props => props.theme.colors.primary};
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: ${props => props.theme.spacing.lg} auto;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;