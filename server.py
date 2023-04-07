from flask import Flask, request, jsonify
from predict import predict_intent

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    intent = predict_intent(data['text'])
    return jsonify({'intent': intent})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
