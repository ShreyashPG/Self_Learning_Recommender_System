import pandas as pd
import numpy as np
import sqlite3

# Simulated user-item interaction data with added 'device' column
interactions_data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'product_id': [101, 102, 101, 103, 102, 104, 103, 105, 104, 105],
    'rating': [5, 3, 4, 5, 4, 2, 3, 5, 4, 3],
    'timestamp': ['2025-05-01 08:00:00', '2025-05-02 12:00:00', '2025-05-01 15:00:00', 
                  '2025-05-03 09:00:00', '2025-05-02 18:00:00', '2025-05-04 10:00:00', 
                  '2025-05-03 14:00:00', '2025-05-05 16:00:00', '2025-05-04 11:00:00', 
                  '2025-05-05 13:00:00'],
    'device': ['mobile', 'desktop', 'mobile', 'tablet', 'mobile', 'desktop', 'tablet', 
               'mobile', 'desktop', 'mobile']  # Added device column
}

# Create interactions DataFrame
interactions_df = pd.DataFrame(interactions_data)

# Extract hour from timestamp
interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
interactions_df['hour'] = interactions_df['timestamp'].dt.hour

# Simulated product metadata (unchanged)
products_data = {
    'product_id': [101, 102, 103, 104, 105],
    'name': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch'],
    'category': ['Electronics', 'Electronics', 'Accessories', 'Electronics', 'Accessories'],
    'description': [
        'High-performance laptop with 16GB RAM',
        'Latest smartphone with 5G support',
        'Noise-canceling wireless headphones',
        '10-inch tablet with stylus support',
        'Fitness-tracking smartwatch with heart rate monitor'
    ]
}

# Create products DataFrame
products_df = pd.DataFrame(products_data)

# Save to CSV
interactions_df.to_csv('interactions.csv', index=False)
products_df.to_csv('products.csv', index=False)

# Save to SQLite
conn = sqlite3.connect('interactions.db')
interactions_df.to_sql('interactions', conn, if_exists='replace', index=False)
products_df.to_sql('products', sqlite3.connect('products.db'), if_exists='replace', index=False)

print("Interactions DataFrame:")
print(interactions_df.head())
print("\nProducts DataFrame:")
print products_df.head())