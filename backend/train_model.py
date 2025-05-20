from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import scipy.sparse as sp
import joblib

# Load matrices
interaction_matrix = sp.load_npz('interaction_matrix.npz')
item_features = sp.load_npz('item_features.npz')

# Initialize LightFM model
model = LightFM(no_components=30, learning_rate=0.05, loss='warp')

# Train model
model.fit(interactions=interaction_matrix, item_features=item_features, epochs=30, num_threads=2)

# Evaluate model
train_precision = precision_at_k(model, interaction_matrix, item_features=item_features, k=5).mean()
print(f"Precision@5: {train_precision:.4f}")

# Save model
joblib.dump(model, 'recommender_model.pkl')