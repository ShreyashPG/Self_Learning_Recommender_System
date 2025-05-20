from flask import Flask, request, jsonify
from recommend_ncf import recommend_ncf
from contextual_bandit import recommend_bandit
from self_learning import add_interaction, update_model

app = Flask(__name__)

@app.route('/recommend/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    n = request.args.get('n', default=5, type=int)
    recommendations = recommend_ncf(user_id, n)
    return jsonify({'user_id': user_id, 'recommendations': recommendations})

@app.route('/recommend_context/<int:user_id>', methods=['POST'])
def get_contextual_recommendations(user_id):
    data = request.get_json()
    context = {'device': data.get('device'), 'hour': data.get('hour')}
    n = data.get('n', 5)
    recommendations = recommend_bandit(user_id, context, n)
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
    update_model()
    return jsonify({'message': 'Interaction added and model updated'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)