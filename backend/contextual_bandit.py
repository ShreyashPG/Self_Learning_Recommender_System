import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import OneHotEncoder
import flask

# Load data from databases
conn_interactions = sqlite3.connect('interactions.db')
interactions_df = pd.read_sql('SELECT * FROM interactions', conn_interactions)

conn_products = sqlite3.connect('products.db')
products_df = pd.read_sql('SELECT * FROM products', conn_products)

# Encode contextual features
encoder = OneHotEncoder(sparse_output=True)
context_features = encoder.fit_transform(interactions_df[['device', 'hour']])
item_ids = products_df['product_id'].tolist()

# Contextual bandit class (unchanged)
class ContextualBandit:
    def __init__(self, n_arms, context_dim):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.mu = [np.zeros(context_dim) for _ in range(n_arms)]  # Mean reward
        self.cov = [np.eye(context_dim) for _ in range(n_arms)]  # Covariance
        self.n = [0] * n_arms  # Number of pulls

    def select_arm(self, context):
        scores = []
        for i in range(self.n_arms):
            mu = self.mu[i]
            cov = self.cov[i]
            theta = np.random.multivariate_normal(mu, cov)
            score = context @ theta
            scores.append(score)
        return np.argmax(scores)

    def update(self, arm, context, reward):
        self.n[arm] += 1
        context = context.reshape(-1, 1)
        self.cov[arm] = np.linalg.inv(np.linalg.inv(self.cov[arm]) + context @ context.T)
        self.mu[arm] += (self.cov[arm] @ context * reward).ravel()

# Initialize bandit
bandit = ContextualBandit(n_arms=len(item_ids), context_dim=context_features.shape[1])

def recommend_bandit(user_id, context, n=5):
    context_vec = encoder.transform([[context['device'], context['hour']]]).toarray().ravel()
    selected_arms = []
    for _ in range(n):
        arm = bandit.select_arm(context_vec)
        if arm not in selected_arms:
            selected_arms.append(arm)
    top_item_ids = [item_ids[i] for i in selected_arms]
    return products_df[products_df['product_id'].isin(top_item_ids)][['product_id', 'name']].to_dict('records')


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    data = request.json
    user_id = data.get('user_id')
    device = data.get('device')
    hour = data.get('hour', pd.Timestamp.now().hour)  # Default to current hour

    if not user_id or not device:
        return jsonify({"error": "user_id and device are required"}), 400

    context = {'device': device, 'hour': int(hour)}
    recommendations = recommend_bandit(user_id, context, n=3)
    
    if "error" in recommendations:
        return jsonify(recommendations), 400
    
    return jsonify({
        "user_id": user_id,
        "context": context,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# # Example usage
# context = {'device': 'mobile', 'hour': 10}
# recommendations = recommend_bandit(user_id=1, context=context, n=3)
# print(f"Contextual Recommendations for User 1:")
# for item in recommendations:
#     print(f"- {item['name']} (ID: {item['product_id']})")