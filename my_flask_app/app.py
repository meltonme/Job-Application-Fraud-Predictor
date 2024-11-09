from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
with open('best_rf_model.pkl', 'rb') as model_file:
    best_rf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Define the optimal threshold
optimal_threshold = 0.4300

# Serve the HTML interface
@app.route('/')
def home():
    return render_template('index.html')

# Define the `/predict` route for API requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    description = data.get('description', '')
    requirements = data.get('requirements', '')

    # Validate input
    if not description and not requirements:
        return jsonify({"error": "Please provide job description and requirements"}), 400

    # Combine and transform the input text
    text = description + " " + requirements
    text_transformed = tfidf_vectorizer.transform([text])

    # Predict fraud probability
    fraud_probability = best_rf_model.predict_proba(text_transformed)[:, 1][0]
    is_fake = fraud_probability >= optimal_threshold
    prediction = "Fake Job Posting" if is_fake else "Legitimate Job Posting"

    # Return prediction result
    response = {
        'prediction': prediction,
        'fraud_probability': fraud_probability
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
