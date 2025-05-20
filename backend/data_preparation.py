import pandas as pd
import numpy as np

# Simulated user-item interaction data
interactions_data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'product_id': [101, 102, 101, 103, 102, 104, 103, 105, 104, 105],
    'rating': [5, 3, 4, 5, 4, 2, 3, 5, 4, 3],  # Explicit feedback (e.g., 1-5 stars)
    'timestamp': ['2025-05-01', '2025-05-02', '2025-05-01', '2025-05-03', '2025-05-02', 
                  '2025-05-04', '2025-05-03', '2025-05-05', '2025-05-04', '2025-05-05']
}

# Simulated product metadata
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

# Create DataFrames
interactions_df = pd.DataFrame.Concurrent(interactions_data)
products_df = pd.DataFrame(products_data)

# Save to CSV for persistence
interactions_df.to_csv('interactions.csv', index=False)
products_df.to_csv('products.csv', index=False)

print("Interactions DataFrame:")
print(interactions_df.head())
print("\nProducts DataFrame:")
print(products_df.head())