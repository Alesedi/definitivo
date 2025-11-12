import React from 'react';
import RecommendationsComponent from '../components/RecommendationsComponent';

const RecommendationsPage = ({ user }) => {
  // Thin wrapper: reuse the centralized RecommendationsComponent to avoid duplicated UI code
  return (
    <div style={{ minHeight: '100vh', padding: '24px', background: '#f4f6fb' }}>
      <RecommendationsComponent user={user} />
    </div>
  );
};

export default RecommendationsPage;