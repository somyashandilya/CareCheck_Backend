from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from utils import preprocess_image

# Define a Blueprint for routes
api_routes = Blueprint('api_routes', __name__)

# Load pre-trained models
cnn_model = load_model('models/cnn_model.h5', compile=False)
rf_model = joblib.load('models/random_forest_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
lr_model = joblib.load('models/logistic_regression_model.pkl')

@api_routes.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = preprocess_image(file)

    # Choose model based on input (default: cnn)
    model_name = request.form.get('model', 'cnn')

    if model_name == 'cnn':
        prediction = cnn_model.predict(image)
        predicted_class = int(np.argmax(prediction))
    elif model_name == 'random_forest':
        predicted_class = int(rf_model.predict(image.reshape(1, -1))[0])
    elif model_name == 'knn':
        predicted_class = int(knn_model.predict(image.reshape(1, -1))[0])
    elif model_name == 'svm':
        predicted_class = int(svm_model.predict(image.reshape(1, -1))[0])
    elif model_name == 'logistic_regression':
        predicted_class = int(lr_model.predict(image.reshape(1, -1))[0])
    else:
        return jsonify({'error': 'Invalid model name'}), 400

    return jsonify({'prediction': predicted_class})
