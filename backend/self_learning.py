import pandas as pd
import sqlite3
import scipy.sparse as sp
import joblib
from lightfm import LightFM

# Initialize SQLite database
conn = sqlite3.connect('interactions.db')
cursor = conn.cursor()

# Create interactions table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        user_id INTEGER,
        product_id INTEGER,
        rating INTEGER,
        timestamp TEXT
    )
''')
conn.commit()

# Load existing interactions to database
interactions_df = pd.read_csv('interactions.csv')
interactions_df.to_sql('interactions', conn, if_exists='replace', index=False)

def add_interaction(user_id, product_id, rating, timestamp):
    # Add new interaction to database
    cursor.execute('''
        INSERT INTO interactions (user_id, product_id, rating, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (user_id, product_id, rating, timestamp))
    conn.commit()

def update_model():
    # Load model and matrices
    model = joblib.load('recommender_model.pkl')
    item_features = sp.load_npz('item_features.npz')
    
    # Load all interactions from database
    interactions_df = pd.read_sql('SELECT * FROM interactions', conn)
    
    # Recreate interaction matrix
    interaction_matrix = interactions_df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
    interaction_matrix_sparse = sp.csr_matrix(interaction_matrix.values)
    
    # Fine-tune model with new data
    model.fit_partial(interactions=interaction_matrix_sparse, item_features=item_features, epochs=10, num_threads=2)
    
    # Save updated model
    joblib.dump(model, 'recommender_model.pkl')
    
    print("Model updated with new interactions.")

# Example: Add a new interaction and update model
add_interaction(user_id=1, product_id=105, rating=4, timestamp='2025-05-06')
update_model()

# Close database connection
conn.close()