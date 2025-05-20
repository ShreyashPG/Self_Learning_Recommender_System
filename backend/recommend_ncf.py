import pandas as pd
import torch
import numpy as np
import joblib
from ncf_model import NCF

# Load data and mappings
products_df = pd.read_csv('products.csv')
mappings = joblib.load('ncf_mappings.pkl')
user2idx = mappings['user2idx']
item2idx = mappings['item2idx']

# Load NCF model
model = NCF(num_users=len(user2idx), num_items=len(item2idx))
model.load_state_dict(torch.load('ncf_model.pth'))
model.eval()

def recommend_ncf(user_id, n=5):
    if user_id not in user2idx:
        # Cold-start: Recommend popular items
        popular_items = products_df.sort_values(by='product_id').head(n)[['product_id', 'name']].to_dict('records')
        return popular_items
    
    user_idx = user2idx[user_id]
    item_indices = torch.tensor(list(item2idx.values()), dtype=torch.long)
    user_tensor = torch.tensor([user_idx] * len(item_indices), dtype=torch.long)
    
    with torch.no_grad():
        scores = model(user_tensor, item_indices)
    
    top_item_indices = torch.topk(scores, n).indices.numpy()
    top_item_ids = [list(item2idx.keys())[i] for i in top_item_indices]
    top_items = products_df[products_df['product_id'].isin(top_item_ids)][['product_id', 'name']].to_dict('records')
    
    return top_items

# Example usage
user_id = 1
recommendations = recommend_ncf(user_id, n=3)
print(f"Recommendations for User {user_id}:")
for item in recommendations:
    print(f"- {item['name']} (ID: {item['product_id']})")