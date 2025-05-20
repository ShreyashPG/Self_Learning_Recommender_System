import pandas as pd
import torch
import numpy as np
import joblib
from ncf_model import NCF
from sklearn.metrics.pairwise import cosine_similarity

# Load data and mappings
products_df = pd.read_csv('products.csv')
mappings = joblib.load('ncf_mappings.pkl')
user2idx = mappings['user2idx']
item2idx = mappings['item2idx']

# Load NCF model
model = NCF(num_users=len(user2idx), num_items=len(item2idx))
model.load_state_dict(torch.load('ncf_model.pth'))
model.eval()

# Compute item similarity based on category
category_matrix = pd.get_dummies(products_df['category']).values
item_similarity = cosine_similarity(category_matrix)

def recommend_diverse(user_id, n=5, alpha=0.7):
    if user_id not in user2idx:
        popular_items = products_df.sort_values(by='product_id').head(n)[['product_id', 'name']].to_dict('records')
        return popular_items
    
    user_idx = user2idx[user_id]
    item_indices = torch.tensor(list(item2idx.values()), dtype=torch.long)
    user_tensor = torch.tensor([user_idx] * len(item_indices), dtype=torch.long)
    
    with torch.no_grad():
        scores = model(user_tensor, item_indices).numpy()
    
    # Greedy selection for diversity
    selected_items = []
    candidate_indices = np.argsort(-scores).tolist()
    
    while len(selected_items) < n and candidate_indices:
        if not selected_items:
            selected_items.append(candidate_indices.pop(0))
        else:
            # Compute diversity score for remaining candidates
            diversity_scores = []
            for idx in candidate_indices:
                sim = np.mean([item_similarity[idx, s] for s in selected_items])
                score = alpha * scores[idx] - (1 - alpha) * sim
                diversity_scores.append(score)
            next_item = candidate_indices[np.argmax(diversity_scores)]
            selected_items.append(next_item)
            candidate_indices.remove(next_item)
    
    top_item_ids = [list(item2idx.keys())[i] for i in selected_items]
    top_items = products_df[products_df['product_id'].isin(top_item_ids)][['product_id', 'name']].to_dict('records')
    
    return top_items

# Example usage
user_id = 1
recommendations = recommend_diverse(user_id, n=3)
print(f"Diverse Recommendations for User {user_id}:")
for item in recommendations:
    print(f"- {item['name']} (ID: {item['product_id']})")