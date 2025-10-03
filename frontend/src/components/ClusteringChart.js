import React from 'react';
import styled from 'styled-components';

const ChartContainer = styled.div`
  width: 100%;
  max-width: 100%;
  height: 450px;
  margin: 0;
  background: ${props => props.theme.colors.secondary};
  border-radius: 12px;
  padding: 15px;
  box-sizing: border-box;
  box-shadow: ${props => props.theme.boxShadow.medium};
  overflow: hidden;
  
  @media (max-width: 768px) {
    height: 350px;
    padding: 10px;
  }
`;

const SVGChart = styled.svg`
  width: 100%;
  height: calc(100% - 60px); /* Spazio per titolo e legenda */
  max-width: 100%;
  background: ${props => props.theme.colors.primary};
  border-radius: 8px;
  display: block;
  margin: 0 auto;
`;

const ChartTitle = styled.h3`
  color: ${props => props.theme.colors.text};
  margin: 0 0 10px 0;
  text-align: center;
  font-size: 16px;
  font-weight: 600;
`;

const Legend = styled.div`
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 10px;
  font-size: 12px;
`;

const LegendItem = styled.div`
  display: flex;
  align-items: center;
  gap: 5px;
  color: ${props => props.theme.colors.text};
  font-size: 14px;
`;

const LegendColor = styled.div`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: ${props => props.color};
`;

// Colori per i cluster
const CLUSTER_COLORS = [
  '#FF6B6B', // Rosso
  '#4ECDC4', // Turchese  
  '#45B7D1', // Blu
  '#96CEB4', // Verde
  '#FFEAA7', // Giallo
  '#DDA0DD', // Lavanda
  '#FFB347', // Arancione
  '#98D8C8'  // Verde acqua
];

const ClusteringChart = ({ data }) => {
  if (!data || !data.points || data.points.length === 0) {
    return (
      <ChartContainer>
        <ChartTitle>📊 Clustering dei Film</ChartTitle>
        <div style={{ textAlign: 'center', color: '#666', padding: '50px' }}>
          Nessun dato di clustering disponibile
        </div>
      </ChartContainer>
    );
  }

  // Calcola i limiti del grafico
  const xValues = data.points.map(p => p.x);
  const yValues = data.points.map(p => p.y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  
  // Aggiungi margini
  const marginX = (maxX - minX) * 0.1;
  const marginY = (maxY - minY) * 0.1;
  const chartMinX = minX - marginX;
  const chartMaxX = maxX + marginX;
  const chartMinY = minY - marginY;
  const chartMaxY = maxY + marginY;

  // Dimensioni del grafico
  const chartWidth = 500;
  const chartHeight = 300;
  const padding = 40;

  // Funzioni di scaling
  const scaleX = (x) => padding + ((x - chartMinX) / (chartMaxX - chartMinX)) * (chartWidth - 2 * padding);
  const scaleY = (y) => chartHeight - padding - ((y - chartMinY) / (chartMaxY - chartMinY)) * (chartHeight - 2 * padding);

  // Raggruppa punti per cluster
  const clusters = {};
  data.points.forEach(point => {
    if (!clusters[point.cluster]) {
      clusters[point.cluster] = [];
    }
    clusters[point.cluster].push(point);
  });

  return (
    <ChartContainer>
      <ChartTitle>📊 Clustering Film - Spazio SVD</ChartTitle>
      
      <SVGChart viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
        {/* Griglia di sfondo */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#333" strokeWidth="0.5" opacity="0.3"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        {/* Assi */}
        <line 
          x1={padding} 
          y1={chartHeight - padding} 
          x2={chartWidth - padding} 
          y2={chartHeight - padding} 
          stroke="#666" 
          strokeWidth="2"
        />
        <line 
          x1={padding} 
          y1={padding} 
          x2={padding} 
          y2={chartHeight - padding} 
          stroke="#666" 
          strokeWidth="2"
        />
        
        {/* Etichette assi */}
        <text 
          x={chartWidth / 2} 
          y={chartHeight - 10} 
          textAnchor="middle" 
          fill="#888" 
          fontSize="12"
        >
          Componente SVD 1
        </text>
        <text 
          x={15} 
          y={chartHeight / 2} 
          textAnchor="middle" 
          fill="#888" 
          fontSize="12" 
          transform={`rotate(-90, 15, ${chartHeight / 2})`}
        >
          Componente SVD 2
        </text>

        {/* Centroidi (se disponibili) */}
        {data.centroids && data.centroids.map((centroid, index) => (
          <g key={`centroid-${index}`}>
            <circle
              cx={scaleX(centroid.x)}
              cy={scaleY(centroid.y)}
              r="8"
              fill={CLUSTER_COLORS[index % CLUSTER_COLORS.length]}
              stroke="#000"
              strokeWidth="2"
              opacity="0.8"
            />
            <text
              x={scaleX(centroid.x)}
              y={scaleY(centroid.y)}
              textAnchor="middle"
              dy="0.3em"
              fill="#000"
              fontSize="10"
              fontWeight="bold"
            >
              C{index}
            </text>
          </g>
        ))}

        {/* Punti dei film */}
        {data.points.map((point, index) => (
          <circle
            key={`point-${index}`}
            cx={scaleX(point.x)}
            cy={scaleY(point.y)}
            r="4"
            fill={CLUSTER_COLORS[point.cluster % CLUSTER_COLORS.length]}
            stroke="#fff"
            strokeWidth="1"
            opacity="0.7"
          >
            <title>Film #{index} - Cluster {point.cluster}</title>
          </circle>
        ))}
      </SVGChart>

      {/* Legenda */}
      <Legend>
        {Object.keys(clusters).map(clusterId => (
          <LegendItem key={clusterId}>
            <LegendColor color={CLUSTER_COLORS[parseInt(clusterId) % CLUSTER_COLORS.length]} />
            Cluster {clusterId} ({clusters[clusterId].length} film)
          </LegendItem>
        ))}
        {data.centroids && (
          <LegendItem>
            <LegendColor color="#000" />
            Centroidi
          </LegendItem>
        )}
      </Legend>
    </ChartContainer>
  );
};

export default ClusteringChart;