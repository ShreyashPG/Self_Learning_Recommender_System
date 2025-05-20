import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# Load data
interactions_df = pd.read_csv('interactions.csv')
products_df = pd.read_csv('products.csv')

# Create user-item interaction matrix
interaction_matrix = interactions_df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
interaction_matrix_sparse = sp.csr_matrix(interaction_matrix.values)

# Save user and product IDs for mapping
user_ids = interaction_matrix.index.tolist()
product_ids = interaction_matrix.columns.tolist()

# Content-based features: TF-IDF for product descriptions
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
tfidf_matrix = tfidf.fit_transform(products_df['description'])
item_features = sp.csr_matrix(tfidf_matrix)

# Save matrices for model training
sp.save_npz('interaction_matrix.npz', interaction_matrix_sparse)
sp.save_npz('item_features.npz', item_features)

print("Interaction Matrix Shape:", interaction_matrix_sparse.shape)
print("Item Features Shape:", item_features.shape)