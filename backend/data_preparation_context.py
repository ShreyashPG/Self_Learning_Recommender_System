import pandas as pd
import sqlite3

# Update interactions data with context
interactions_data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'product_id': [101, 102, 101, 103, 102, 104, 103, 105, 104, 105],
    'rating': [5, 3, 4, 5, 4, 2, 3, 5, 4, 3],
    'timestamp': ['2025-05-01 10:00', '2025-05-02 15:00', '2025-05-01 09:00', '2025-05-03 20:00', 
                  '2025-05-02 12:00', '2025-05-04 18:00', '2025-05-03 11:00', '2025-05-05 14:00', 
                  '2025-05-04 16:00', '2025-05-05 19:00'],
    'device': ['mobile', 'desktop', 'mobile', 'tablet', 'desktop', 'mobile', 'tablet', 'desktop', 'mobile', 'tablet'],
    'hour': [10, 15, 9, 20, 12, 18, 11, 14, 16, 19]
}

interactions_df = pd.DataFrame(interactions_data)
interactions_df.to_csv('interactions_context.csv', index=False)
interactions_df.to_sql('interactions', sqlite3.connect('interactions.db'), if_exists='replace', index=False)