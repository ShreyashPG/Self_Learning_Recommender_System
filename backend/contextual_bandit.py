import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import OneHotEncoder

# Load data
conn = sqlite3.connect('interactions.db')
interactions_df = pd.read_sql('SELECT * FROM interactions', conn)

# Encode contextual features
encoder = OneHotEncoder(sparse=True)
context_features = encoder.fit_transform(interactions_df[['device', 'hour']])
products_df = pd.read_csv('products.csv')
item_ids = products_df['product_id'].tolist()

# Contextual bandit class
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

# Example usage
context = {'device': 'mobile', 'hour': 10}
recommendations = recommend_bandit(user_id=1, context=context, n=3)
print(f"Contextual Recommendations for User 1:")
for item in recommendations:
    print(f"- {item['name']} (ID: {item['product_id']})")