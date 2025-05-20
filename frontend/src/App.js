import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [userId, setUserId] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [interaction, setInteraction] = useState({ productId: '', rating: '', timestamp: '' });
  const [modelType, setModelType] = useState('ncf');

  const fetchRecommendations = async () => {
    try {
      const response = await axios.get(`http://localhost:5000/recommend/${userId}`);
      setRecommendations(response.data.recommendations);
      setModelType(response.data.model);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    }
  };

  const submitInteraction = async () => {
    try {
      await axios.post('http://localhost:5000/interaction', {
        user_id: parseInt(userId),
        product_id: parseInt(interaction.productId),
        rating: parseInt(interaction.rating),
        timestamp: interaction.timestamp,
        model_type: modelType
      });
      alert('Interaction submitted!');
    } catch (error) {
      console.error('Error submitting interaction:', error);
    }
  };

  return (
    <div className="App">
      <h1>Recommender System</h1>
      <div>
        <label>User ID: </label>
        <input value={userId} onChange={(e) => setUserId(e.target.value)} />
        <button onClick={fetchRecommendations}>Get Recommendations</button>
      </div>
      <h2>Recommendations</h2>
      <ul>
        {recommendations.map((item) => (
          <li key={item.product_id}>{item.name} (ID: {item.product_id})</li>
        ))}
      </ul>
      <h2>Submit Interaction</h2>
      <div>
        <label>Product ID: </label>
        <input value={interaction.productId} onChange={(e) => setInteraction({ ...interaction, productId: e.target.value })} />
        <label>Rating: </label>
        <input value={interaction.rating} onChange={(e) => setInteraction({ ...interaction, rating: e.target.value })} />
        <label>Timestamp: </label>
        <input value={interaction.timestamp} onChange={(e) => setInteraction({ ...interaction, timestamp: e.target.value })} />
        <button onClick={submitInteraction}>Submit</button>
      </div>
    </div>
  );
}

export default App;