from flask import Flask, request, jsonify
import random
import logging
from recommend_diverse import recommend_diverse
from contextual_bandit import recommend_bandit
from self_learning import add_interaction, update_model

app = Flask(__name__)

# Configure logging for A/B testing
logging.basicConfig(filename='ab_testing.log', level=logging.INFO, 
                    format='%(asctime)s %(message)s')

@app.route('/recommend/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    n = request.args.get('n', default=5, type=int)
    # Randomly assign model (50% chance for each)
    model_type = 'ncf' if random.random() < 0.5 else 'bandit'
    context = {'device': request.args.get('device', 'mobile'), 'hour': int(request.args.get('hour', 10))}
    
    if model_type == 'ncf':
        recommendations = recommend_diverse(user_id, n)
    else:
        recommendations = recommend_bandit(user_id, context, n)
    
    # Log model used
    logging.info(f"User {user_id}, Model: {model_type}, Recommendations: {[r['product_id'] for r in recommendations]}")
    
    return jsonify({'user_id': user_id, 'model': model_type, 'recommendations': recommendations})

@app.route('/interaction', methods=['POST'])
def add_new_interaction():
    data = request.get_json()
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    rating = data.get('rating')
    timestamp = data.get('timestamp')
    model_type = data.get('model_type')
    
    if not all([user_id, product_id, rating, timestamp]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    add_interaction(user_id, product_id, rating, timestamp)
    update_model()
    
    # Log interaction for A/B testing
    logging.info(f"User {user_id}, Model: {model_type}, Interaction: Product {product_id}, Rating {rating}")
    
    return jsonify({'message': 'Interaction added and model updated'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)