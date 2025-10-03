// Theme with blue and orange color scheme
export const theme = {
  colors: {
    // Primary colors (Blue)
    primary: '#1E3A8A',      // Dark blue
    primaryDark: '#1E40AF',  // Darker blue
    primaryLight: '#3B82F6', // Medium blue
    primaryLighter: '#DBEAFE', // Light blue

    // Secondary colors (Orange)
    secondary: '#EA580C',     // Dark orange
    secondaryDark: '#C2410C', // Darker orange
    secondaryLight: '#FB923C', // Medium orange
    secondaryLighter: '#FED7AA', // Light orange

    // Neutral colors
    background: '#F8FAFC',    // Very light gray
    surface: '#FFFFFF',       // White
    surfaceHover: '#F1F5F9',  // Light gray
    
    // Text colors
    textPrimary: '#1E293B',   // Dark gray
    textSecondary: '#64748B', // Medium gray
    textLight: '#94A3B8',     // Light gray
    textWhite: '#FFFFFF',

    // Status colors
    success: '#10B981',
    error: '#EF4444',
    warning: '#F59E0B',
    info: '#3B82F6',

    // Card and border colors
    border: '#E2E8F0',
    borderHover: '#CBD5E1',
    shadow: 'rgba(30, 58, 138, 0.1)',
  },
  
  // Spacing
  spacing: {
    xs: '0.25rem',   // 4px
    sm: '0.5rem',    // 8px
    md: '1rem',      // 16px
    lg: '1.5rem',    // 24px
    xl: '2rem',      // 32px
    xxl: '3rem',     // 48px
  },

  // Border radius
  borderRadius: {
    sm: '0.25rem',   // 4px
    md: '0.5rem',    // 8px
    lg: '0.75rem',   // 12px
    xl: '1rem',      // 16px
    full: '9999px',  // Full round
  },

  // Font sizes
  fontSize: {
    xs: '0.75rem',   // 12px
    sm: '0.875rem',  // 14px
    base: '1rem',    // 16px
    lg: '1.125rem',  // 18px
    xl: '1.25rem',   // 20px
    '2xl': '1.5rem', // 24px
    '3xl': '1.875rem', // 30px
    '4xl': '2.25rem',  // 36px
  },

  // Font weights
  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },

  // Shadows
  boxShadow: {
    sm: '0 1px 2px 0 rgba(30, 58, 138, 0.05)',
    base: '0 4px 6px -1px rgba(30, 58, 138, 0.1), 0 2px 4px -1px rgba(30, 58, 138, 0.06)',
    lg: '0 10px 15px -3px rgba(30, 58, 138, 0.1), 0 4px 6px -2px rgba(30, 58, 138, 0.05)',
    xl: '0 20px 25px -5px rgba(30, 58, 138, 0.1), 0 10px 10px -5px rgba(30, 58, 138, 0.04)',
  },

  // Breakpoints for responsive design
  breakpoints: {
    mobile: '480px',
    tablet: '768px',
    desktop: '1024px',
    wide: '1280px',
  },
};