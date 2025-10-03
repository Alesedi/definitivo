import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { Button, Flex, Container } from '../styles/components';
import { FaFilm, FaStar, FaUser, FaSignOutAlt, FaBrain } from 'react-icons/fa';

const NavbarContainer = styled.nav`
  background: linear-gradient(135deg, ${props => props.theme.colors.primary} 0%, ${props => props.theme.colors.primaryLight} 100%);
  box-shadow: ${props => props.theme.boxShadow.lg};
  padding: ${props => props.theme.spacing.md} 0;
  position: sticky;
  top: 0;
  z-index: 100;
`;

const NavContent = styled(Container)`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Logo = styled(Link)`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.colors.textWhite};
  text-decoration: none;
  font-size: ${props => props.theme.fontSize['2xl']};
  font-weight: ${props => props.theme.fontWeight.bold};
  
  &:hover {
    color: ${props => props.theme.colors.secondaryLight};
  }
`;

const NavLinks = styled(Flex)`
  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    display: none;
  }
`;

const NavLink = styled(Link)`
  color: ${props => props.theme.colors.textWhite};
  text-decoration: none;
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius.md};
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  
  &:hover {
    background-color: rgba(255, 255, 255, 0.1);
    transform: translateY(-1px);
  }
`;

const UserSection = styled(Flex)`
  gap: ${props => props.theme.spacing.md};
`;

const UserInfo = styled.div`
  color: ${props => props.theme.colors.textWhite};
  font-weight: ${props => props.theme.fontWeight.medium};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  
  @media (max-width: ${props => props.theme.breakpoints.mobile}) {
    display: none;
  }
`;

const LogoutButton = styled(Button)`
  background-color: ${props => props.theme.colors.secondary};
  color: ${props => props.theme.colors.textWhite};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  
  &:hover {
    background-color: ${props => props.theme.colors.secondaryLight};
  }
`;

const Navbar = ({ user, onLogout }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout();
    navigate('/login');
  };

  return (
    <NavbarContainer>
      <NavContent>
        <Logo to="/dashboard">
          <FaFilm />
          AFlix
        </Logo>
        
        <NavLinks>
          <NavLink to="/dashboard">
            <FaStar />
            Dashboard
          </NavLink>
          <NavLink to="/rating">
            <FaFilm />
            Vota Film
          </NavLink>
          <NavLink to="/recommendations">
            <FaBrain />
            ML Raccomandazioni
          </NavLink>
        </NavLinks>
        
        <UserSection>
          <UserInfo>
            <FaUser />
            {user?.username}
          </UserInfo>
          <LogoutButton onClick={handleLogout}>
            <FaSignOutAlt />
            Logout
          </LogoutButton>
        </UserSection>
      </NavContent>
    </NavbarContainer>
  );
};

export default Navbar;