import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib

# Load data and model
products_df = pd.read_csv('products.csv')
interaction_matrix = sp.load_npz('interaction_matrix.npz')
item_features = sp.load_npz('item_features.npz')
model = joblib.load('recommender_model.pkl')

# Load user and product IDs
user_ids = pd.read_csv('interactions.csv')['user_id'].unique()
product_ids = products_df['product_id'].tolist()

def recommend(user_id, n=5):
    # Check if user exists
    if user_id not in user_ids:
        # Cold-start: Recommend popular items
        popular_items = products_df.sort_values(by='product_id').head(n)[['product_id', 'name']].to_dict('records')
        return popular_items
    
    # Get user index
    user_index = list(user_ids).index(user_id)
    
    # Predict scores for all items
    scores = model.predict(user_index, np.arange(len(product_ids)), item_features=item_features)
    
    # Get top-N items
    top_items_idx = np.argsort(-scores)[:n]
    top_items = products_df.iloc[top_items_idx][['product_id', 'name']].to_dict('records')
    
    return top_items

# Example usage
user_id = 1
recommendations = recommend(user_id, n=3)
print(f"Recommendations for User {user_id}:")
for item in recommendations:
    print(f"- {item['name']} (ID: {item['product_id']})")