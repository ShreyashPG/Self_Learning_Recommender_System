from flask import Flask, request, jsonify
import pandas as pd
import sqlite3
from recommend import recommend
from self_learning import add_interaction, update_model

app = Flask(__name__)

@app.route('/recommend/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    n = request.args.get('n', default=5, type=int)
    recommendations = recommend(user_id, n)
    return jsonify({'user_id': user_id, 'recommendations': recommendations})

@app.route('/interaction', methods=['POST'])
def add_new_interaction():
    data = request.get_json()
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    rating = data.get('rating')
    timestamp = data.get('timestamp')
    
    if not all([user_id, product_id, rating, timestamp]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    add_interaction(user_id, product_id, rating, timestamp)
    update_model()  # Update model after new interaction
    
    return jsonify({'message': 'Interaction added and model updated'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)