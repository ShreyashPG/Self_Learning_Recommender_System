import pandas as pd

# Load A/B testing logs
with open('ab_testing.log', 'r') as f:
    logs = f.readlines()

# Parse logs
interactions = []
for log in logs:
    if 'Interaction' in log:
        parts = log.split(',')
        user_id = int(parts[0].split('User ')[1])
        model_type = parts[1].split(': ')[1].strip()
        product_id = int(parts[2].split('Product ')[1])
        rating = int(parts[3].split('Rating ')[1])
        interactions.append({'user_id': user_id, 'model_type': model_type, 'rating': rating})

# Analyze performance
df = pd.DataFrame(interactions)
performance = df.groupby('model_type')['rating'].mean().to_dict()
print("Average Rating by Model:")
for model, rating in performance.items():
    print(f"{model}: {rating:.4f}")