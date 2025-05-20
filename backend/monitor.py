import logging
from lightfm.evaluation import precision_at_k
import scipy.sparse as sp
import joblib

# Configure logging
logging.basicConfig(filename='recommender.log', level=logging.INFO, 
                    format='%(asctime)s %(message)s')

def evaluate_model():
    # Load model and data
    model = joblib.load('recommender_model.pkl')
    interaction_matrix = sp.load_npz('interaction_matrix.npz')
    item_features = sp.load_npz('item_features.npz')
    
    # Compute precision@5
    precision = precision_at_k(model, interaction_matrix, item_features=item_features, k=5).mean()
    
    # Log result
    logging.info(f"Precision@5: {precision:.4f}")
    print(f"Precision@5: {precision:.4f}")

if __name__ == '__main__':
    evaluate_model()