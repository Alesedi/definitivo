import React from 'react';
import styled from 'styled-components';
import { theme } from '../styles/theme';
import { FaBrain } from 'react-icons/fa';
import { MLProvider } from '../contexts/MLContext';
import MLUnifiedDashboard from '../components/MLUnifiedDashboard';

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

const DashboardContainer = styled.div`
  max-width: 1600px;
  margin: 0 auto;
  background: white;
  border-radius: ${theme.borderRadius.lg};
  box-shadow: ${theme.boxShadow.xl};
  overflow: hidden;
`;

const MLDashboard = ({ user }) => {
  return (
    <MLProvider>
      <PageContainer>
        <Header>
          <Title>
            <FaBrain />
            ML Dashboard - Sistema Raccomandazioni AFlix
          </Title>
          <Subtitle>
            Centro di controllo unificato per Machine Learning, Raccomandazioni e Ottimizzazione K-Values
          </Subtitle>
        </Header>

        <DashboardContainer>
          <MLUnifiedDashboard user={user} />
        </DashboardContainer>
      </PageContainer>
    </MLProvider>
  );
};

export default MLDashboard;